//! ControlNet for Qwen-Image.
//!
//! This module implements the ControlNet architecture for Qwen-Image, enabling
//! conditional image generation guided by control signals (edges, depth, pose, etc.).
//!
//! # Architecture
//!
//! The ControlNet is a lightweight copy of the main transformer that:
//! 1. Processes control conditions through a zero-initialized embedder
//! 2. Runs through a subset of transformer blocks (typically 5 layers)
//! 3. Produces residuals through zero-initialized projection layers
//!
//! The "zero initialization" is key: it ensures that at the start of training,
//! the ControlNet adds nothing to the base model, providing a stable starting point.
//!
//! # Usage
//!
//! ```ignore
//! let controlnet = QwenImageControlNetModel::new(&config, vb)?;
//! let residuals = controlnet.forward(
//!     &latents,
//!     &control_cond,
//!     0.8, // conditioning_scale
//!     &text_embeds,
//!     &timestep,
//!     &img_shapes,
//!     &txt_seq_lens,
//! )?;
//!
//! // Apply residuals to main transformer
//! let output = transformer.forward_with_controlnet(
//!     &latents,
//!     &text_embeds,
//!     &text_mask,
//!     &timestep,
//!     &img_shapes,
//!     &txt_seq_lens,
//!     Some(&residuals.block_residuals),
//! )?;
//! ```

use candle::{DType, Result, Tensor};
use candle_nn::{Linear, Module, RmsNorm, VarBuilder};

use super::blocks::QwenImageTransformerBlock;
use super::config::Config;
use super::rope::QwenEmbedRope;
use super::model::QwenTimestepProjEmbeddings;

/// Configuration for ControlNet.
#[derive(Debug, Clone)]
pub struct ControlNetConfig {
    /// Base transformer configuration.
    pub base_config: Config,
    /// Number of transformer layers in the ControlNet (default: 5).
    pub num_layers: usize,
    /// Extra condition channels for inpainting (masked image + mask).
    pub extra_condition_channels: usize,
}

impl ControlNetConfig {
    /// Default ControlNet configuration (5 layers).
    pub fn default_5_layers() -> Self {
        Self {
            base_config: Config::qwen_image(),
            num_layers: 5,
            extra_condition_channels: 0,
        }
    }

    /// ControlNet configuration for inpainting (includes mask channels).
    pub fn for_inpainting() -> Self {
        Self {
            base_config: Config::qwen_image(),
            num_layers: 5,
            extra_condition_channels: 16, // Mask is in latent space (16 channels)
        }
    }
}

/// Output from ControlNet forward pass.
#[derive(Debug, Clone)]
pub struct ControlNetOutput {
    /// Residuals to add at each transformer block.
    /// Length equals the number of ControlNet layers.
    pub block_residuals: Vec<Tensor>,
}

/// Zero-initialized linear layer.
///
/// This is the key innovation in ControlNet: outputs are zero at initialization,
/// so the ControlNet adds nothing to the base model before training.
#[derive(Debug, Clone)]
pub struct ZeroLinear {
    linear: Linear,
}

impl ZeroLinear {
    /// Create a new zero-initialized linear layer.
    ///
    /// Loads weights from the VarBuilder, which should contain zero-initialized weights.
    /// During training, these start at zero and gradually learn the control signal.
    pub fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear = candle_nn::linear(in_dim, out_dim, vb)?;
        Ok(Self { linear })
    }

    /// Create a zero-initialized linear layer from scratch (for building from base model).
    ///
    /// This initializes weights to zero, useful when creating a ControlNet from
    /// a pretrained base transformer.
    pub fn new_zeroed(in_dim: usize, out_dim: usize, device: &candle::Device, dtype: DType) -> Result<Self> {
        let weight = Tensor::zeros((out_dim, in_dim), dtype, device)?;
        let bias = Tensor::zeros(out_dim, dtype, device)?;
        let linear = Linear::new(weight, Some(bias));
        Ok(Self { linear })
    }
}

impl Module for ZeroLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear.forward(xs)
    }
}

/// Qwen-Image ControlNet Model.
///
/// A lightweight transformer that processes control conditions and produces
/// residuals to guide the main transformer's generation.
#[derive(Debug, Clone)]
pub struct QwenImageControlNetModel {
    /// Inner dimension = num_heads × head_dim = 3072
    inner_dim: usize,

    /// RoPE embeddings for 3D positioning
    pos_embed: QwenEmbedRope,

    /// Timestep embedding projection
    time_text_embed: QwenTimestepProjEmbeddings,

    /// Text input normalization
    txt_norm: RmsNorm,

    /// Image input projection: 64 -> 3072
    img_in: Linear,

    /// Text input projection: 3584 -> 3072
    txt_in: Linear,

    /// Control condition embedder (zero-initialized)
    /// Input: in_channels + extra_condition_channels
    controlnet_x_embedder: ZeroLinear,

    /// Stack of transformer blocks (subset of main model)
    transformer_blocks: Vec<QwenImageTransformerBlock>,

    /// Zero-initialized output projections for each block
    controlnet_blocks: Vec<ZeroLinear>,
}

impl QwenImageControlNetModel {
    /// Create a new ControlNet model from config and weights.
    pub fn new(config: &ControlNetConfig, vb: VarBuilder) -> Result<Self> {
        let inner_dim = config.base_config.inner_dim();
        let device = vb.device();
        let dtype = vb.dtype();

        // RoPE embeddings
        let pos_embed = QwenEmbedRope::new(
            config.base_config.theta,
            vec![
                config.base_config.axes_dims_rope.0,
                config.base_config.axes_dims_rope.1,
                config.base_config.axes_dims_rope.2,
            ],
            true, // scale_rope for center-aligned positioning
            device,
            dtype,
        )?;

        // Timestep embeddings
        let time_text_embed = QwenTimestepProjEmbeddings::new(inner_dim, vb.pp("time_text_embed"))?;

        // Text normalization
        let txt_norm_weight = vb.get(config.base_config.joint_attention_dim, "txt_norm.weight")?;
        let txt_norm = RmsNorm::new(txt_norm_weight, 1e-6);

        // Input projections
        let img_in = candle_nn::linear(config.base_config.in_channels, inner_dim, vb.pp("img_in"))?;
        let txt_in = candle_nn::linear(config.base_config.joint_attention_dim, inner_dim, vb.pp("txt_in"))?;

        // ControlNet-specific: zero-initialized control embedder
        let control_in_channels = config.base_config.in_channels + config.extra_condition_channels;
        let controlnet_x_embedder = ZeroLinear::new(control_in_channels, inner_dim, vb.pp("controlnet_x_embedder"))?;

        // Transformer blocks (subset)
        let mut transformer_blocks = Vec::with_capacity(config.num_layers);
        let vb_blocks = vb.pp("transformer_blocks");
        for idx in 0..config.num_layers {
            let block = QwenImageTransformerBlock::new(
                inner_dim,
                config.base_config.num_attention_heads,
                config.base_config.attention_head_dim,
                vb_blocks.pp(idx),
            )?;
            transformer_blocks.push(block);
        }

        // Zero-initialized output blocks
        let mut controlnet_blocks = Vec::with_capacity(config.num_layers);
        let vb_controlnet = vb.pp("controlnet_blocks");
        for idx in 0..config.num_layers {
            let block = ZeroLinear::new(inner_dim, inner_dim, vb_controlnet.pp(idx))?;
            controlnet_blocks.push(block);
        }

        Ok(Self {
            inner_dim,
            pos_embed,
            time_text_embed,
            txt_norm,
            img_in,
            txt_in,
            controlnet_x_embedder,
            transformer_blocks,
            controlnet_blocks,
        })
    }

    /// Get the inner dimension of the model.
    pub fn inner_dim(&self) -> usize {
        self.inner_dim
    }

    /// Get the number of ControlNet layers.
    pub fn num_layers(&self) -> usize {
        self.transformer_blocks.len()
    }

    /// Forward pass through the ControlNet.
    ///
    /// # Arguments
    /// * `hidden_states` - Packed image latents [batch, img_seq, in_channels]
    /// * `controlnet_cond` - Packed control condition [batch, img_seq, in_channels (+ extra)]
    /// * `conditioning_scale` - Scale factor for residuals (0.0 = no control, 1.0 = full)
    /// * `encoder_hidden_states` - Text embeddings [batch, txt_seq, joint_attention_dim]
    /// * `timestep` - Diffusion timestep [batch]
    /// * `img_shapes` - List of (frame, height, width) tuples for RoPE
    /// * `txt_seq_lens` - Text sequence lengths per batch item
    ///
    /// # Returns
    /// ControlNetOutput containing residuals for each transformer block
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        controlnet_cond: &Tensor,
        conditioning_scale: f64,
        encoder_hidden_states: &Tensor,
        timestep: &Tensor,
        img_shapes: &[(usize, usize, usize)],
        txt_seq_lens: &[usize],
    ) -> Result<ControlNetOutput> {
        let dtype = hidden_states.dtype();

        // Project image latents: [batch, seq, 64] -> [batch, seq, 3072]
        let mut hidden_states = hidden_states.apply(&self.img_in)?;

        // Add control condition (zero-initialized at start of training)
        let control_embed = self.controlnet_x_embedder.forward(controlnet_cond)?;
        hidden_states = (hidden_states + control_embed)?;

        // Normalize and project text: [batch, seq, 3584] -> [batch, seq, 3072]
        let mut encoder_hidden_states = encoder_hidden_states.apply(&self.txt_norm)?;
        encoder_hidden_states = encoder_hidden_states.apply(&self.txt_in)?;

        // Compute timestep embedding
        let timestep = timestep.to_dtype(dtype)?;
        let temb = self.time_text_embed.forward(&timestep, dtype)?;

        // Compute RoPE frequencies
        let image_rotary_emb = self.pos_embed.forward(img_shapes, txt_seq_lens)?;

        // Collect block outputs
        let mut block_samples = Vec::with_capacity(self.transformer_blocks.len());

        // Process through transformer blocks
        for block in &self.transformer_blocks {
            let (enc_out, hid_out) = block.forward(
                &hidden_states,
                &encoder_hidden_states,
                &temb,
                Some(&image_rotary_emb),
            )?;
            encoder_hidden_states = enc_out;
            hidden_states = hid_out;
            block_samples.push(hidden_states.clone());
        }

        // Apply controlnet output projections and scaling
        let mut block_residuals = Vec::with_capacity(self.controlnet_blocks.len());
        for (sample, controlnet_block) in block_samples.iter().zip(&self.controlnet_blocks) {
            let residual = controlnet_block.forward(sample)?;
            let scaled_residual = (residual * conditioning_scale)?;
            block_residuals.push(scaled_residual);
        }

        Ok(ControlNetOutput { block_residuals })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn test_zero_linear() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Create a zero-initialized linear layer
        let zero_linear = ZeroLinear::new_zeroed(64, 128, &device, dtype)?;

        // Input tensor
        let input = Tensor::randn(0f32, 1f32, (2, 10, 64), &device)?;

        // Output should be all zeros
        let output = zero_linear.forward(&input)?;
        let max_val = output.abs()?.max_all()?.to_scalar::<f32>()?;

        assert!(max_val < 1e-6, "Zero-initialized output should be zero, got max: {}", max_val);

        Ok(())
    }
}
