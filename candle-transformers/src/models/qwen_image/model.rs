//! Qwen-Image Transformer Model.
//!
//! This module implements the main QwenImageTransformer2DModel, a 20B parameter
//! dual-stream Multimodal Diffusion Transformer (MMDiT) for text-to-image generation.
//!
//! The architecture follows the flow:
//! 1. Project image latents and text embeddings to inner dimension
//! 2. Compute timestep embeddings and RoPE positional encodings
//! 3. Process through 60 dual-stream transformer blocks
//! 4. Apply final normalization and project to output channels

use candle::{DType, Result, Tensor};
use candle_nn::{LayerNorm, Linear, RmsNorm, VarBuilder};

use super::blocks::QwenImageTransformerBlock;
use super::config::Config;
use super::rope::{timestep_embedding, QwenEmbedRope};

/// Create a parameter-free LayerNorm (equivalent to PyTorch's elementwise_affine=False).
fn layer_norm_no_affine(
    size: usize,
    eps: f64,
    device: &candle::Device,
    dtype: DType,
) -> Result<LayerNorm> {
    let weight = Tensor::ones(size, dtype, device)?;
    Ok(LayerNorm::new_no_bias(weight, eps))
}

/// Timestep projection embeddings.
///
/// Converts timesteps to sinusoidal embeddings, then projects to hidden dimension.
#[derive(Debug, Clone)]
pub struct QwenTimestepProjEmbeddings {
    /// Projects sinusoidal embeddings to hidden dimension
    timestep_embedder_linear1: Linear,
    timestep_embedder_linear2: Linear,
}

impl QwenTimestepProjEmbeddings {
    pub fn new(embedding_dim: usize, vb: VarBuilder) -> Result<Self> {
        let timestep_embedder_linear1 =
            candle_nn::linear(256, embedding_dim, vb.pp("timestep_embedder.linear_1"))?;
        let timestep_embedder_linear2 = candle_nn::linear(
            embedding_dim,
            embedding_dim,
            vb.pp("timestep_embedder.linear_2"),
        )?;
        Ok(Self {
            timestep_embedder_linear1,
            timestep_embedder_linear2,
        })
    }

    pub fn forward(&self, timestep: &Tensor, dtype: DType) -> Result<Tensor> {
        // Create sinusoidal embeddings (256-dim)
        let timesteps_proj = timestep_embedding(timestep, 256, dtype)?;

        // Project through MLP: 256 -> embedding_dim -> embedding_dim
        let timesteps_emb = timesteps_proj
            .apply(&self.timestep_embedder_linear1)?
            .silu()?
            .apply(&self.timestep_embedder_linear2)?;

        Ok(timesteps_emb)
    }
}

/// Adaptive Layer Normalization with continuous conditioning.
///
/// Used for the final output layer, modulated by timestep embedding.
#[derive(Debug, Clone)]
pub struct AdaLayerNormContinuous {
    norm: candle_nn::LayerNorm,
    linear: Linear,
}

impl AdaLayerNormContinuous {
    pub fn new(dim: usize, conditioning_dim: usize, vb: VarBuilder) -> Result<Self> {
        // Note: elementwise_affine=False in PyTorch (no learned params for norm)
        let norm = layer_norm_no_affine(dim, 1e-6, vb.device(), vb.dtype())?;
        let linear = candle_nn::linear(conditioning_dim, 2 * dim, vb.pp("linear"))?;
        Ok(Self { norm, linear })
    }

    pub fn forward(&self, xs: &Tensor, conditioning: &Tensor) -> Result<Tensor> {
        let chunks = conditioning.silu()?.apply(&self.linear)?.chunk(2, 1)?;
        if chunks.len() != 2 {
            candle::bail!("Expected 2 chunks for AdaLN, got {}", chunks.len());
        }
        // PyTorch: scale, shift = torch.chunk(emb, 2, dim=1)
        let scale = &chunks[0];
        let shift = &chunks[1];

        xs.apply(&self.norm)?
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)
    }
}

/// Qwen-Image Transformer 2D Model.
///
/// A 20B parameter dual-stream MMDiT that processes packed image latents
/// conditioned on text embeddings and timesteps.
///
/// # Architecture
///
/// - **Input**: Packed latents [batch, seq, 64] where seq = (H/2) × (W/2)
/// - **Text**: Qwen2.5-VL embeddings [batch, txt_seq, 3584]
/// - **Output**: Unpacked predictions [batch, seq, 16 × patch_size²]
///
/// # Edit Mode (zero_cond_t)
///
/// When `zero_cond_t` is enabled (edit mode), the model uses per-token modulation:
/// - Timestep is doubled: `[t, 0]` → creates two sets of modulation parameters
/// - `modulate_index` tensor marks which modulation to use per token:
///   - Index 0 (actual timestep): for noise latents being denoised
///   - Index 1 (zero timestep): for reference image latents (conditioning)
///
/// # Example
///
/// ```ignore
/// let model = QwenImageTransformer2DModel::new(&config, vb)?;
/// let output = model.forward(
///     &latents,           // [batch, seq, 64]
///     &text_embeds,       // [batch, txt_seq, 3584]
///     &text_mask,         // [batch, txt_seq]
///     &timestep,          // [batch]
///     &[(1, 64, 64)],     // [(frame, height, width)]
///     &[512],             // text sequence lengths
/// )?;
/// ```
#[derive(Debug, Clone)]
pub struct QwenImageTransformer2DModel {
    /// Inner dimension = num_heads × head_dim = 3072
    _inner_dim: usize,
    _out_channels: usize,
    _patch_size: usize,

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

    /// Stack of 60 dual-stream transformer blocks
    transformer_blocks: Vec<QwenImageTransformerBlock>,

    /// Output normalization with AdaLN
    norm_out: AdaLayerNormContinuous,

    /// Output projection: 3072 -> patch_size² × out_channels
    proj_out: Linear,

    /// Whether to use zero conditioning for timestep (edit mode).
    /// When true, doubles timestep and uses per-token modulation.
    zero_cond_t: bool,
}

impl QwenImageTransformer2DModel {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let inner_dim = config.inner_dim();
        let device = vb.device();
        let dtype = vb.dtype();

        // RoPE embeddings
        let pos_embed = QwenEmbedRope::new(
            config.theta,
            vec![
                config.axes_dims_rope.0,
                config.axes_dims_rope.1,
                config.axes_dims_rope.2,
            ],
            true, // scale_rope for center-aligned positioning
            device,
            dtype,
        )?;

        // Timestep embeddings
        let time_text_embed = QwenTimestepProjEmbeddings::new(inner_dim, vb.pp("time_text_embed"))?;

        // Text normalization (RMSNorm before projection)
        let txt_norm_weight = vb.get(config.joint_attention_dim, "txt_norm.weight")?;
        let txt_norm = RmsNorm::new(txt_norm_weight, 1e-6);

        // Input projections
        let img_in = candle_nn::linear(config.in_channels, inner_dim, vb.pp("img_in"))?;
        let txt_in = candle_nn::linear(config.joint_attention_dim, inner_dim, vb.pp("txt_in"))?;

        // Transformer blocks
        let mut transformer_blocks = Vec::with_capacity(config.num_layers);
        let vb_blocks = vb.pp("transformer_blocks");
        for idx in 0..config.num_layers {
            let block = QwenImageTransformerBlock::new(
                inner_dim,
                config.num_attention_heads,
                config.attention_head_dim,
                vb_blocks.pp(idx),
            )?;
            transformer_blocks.push(block);
        }

        // Output layers
        let norm_out = AdaLayerNormContinuous::new(inner_dim, inner_dim, vb.pp("norm_out"))?;
        let proj_out = candle_nn::linear(
            inner_dim,
            config.patch_size * config.patch_size * config.out_channels,
            vb.pp("proj_out"),
        )?;

        Ok(Self {
            _inner_dim: inner_dim,
            _out_channels: config.out_channels,
            _patch_size: config.patch_size,
            pos_embed,
            time_text_embed,
            txt_norm,
            img_in,
            txt_in,
            transformer_blocks,
            norm_out,
            proj_out,
            zero_cond_t: config.zero_cond_t,
        })
    }

    /// Compute the timestep embedding (temb) for a given timestep.
    ///
    /// This allows computing temb externally for debugging/substitution purposes.
    /// In zero_cond_t mode (edit), the timestep should already be doubled: [t, 0].
    pub fn compute_temb(&self, timestep: &Tensor, dtype: DType) -> Result<Tensor> {
        self.time_text_embed.forward(timestep, dtype)
    }

    /// Forward pass through the transformer.
    ///
    /// # Arguments
    /// * `hidden_states` - Packed image latents [batch, img_seq, in_channels]
    /// * `encoder_hidden_states` - Text embeddings [batch, txt_seq, joint_attention_dim]
    /// * `timestep` - Diffusion timestep [batch]
    /// * `img_shapes` - List of (frame, height, width) tuples for RoPE
    ///
    /// # Returns
    /// Output predictions [batch, img_seq, patch_size² × out_channels]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: &Tensor,
        img_shapes: &[(usize, usize, usize)],
    ) -> Result<Tensor> {
        self.forward_with_controlnet(hidden_states, encoder_hidden_states, timestep, img_shapes, None)
    }

    /// Forward pass with optional ControlNet residuals.
    ///
    /// # Arguments
    /// * `hidden_states` - Packed image latents [batch, img_seq, in_channels]
    /// * `encoder_hidden_states` - Text embeddings [batch, txt_seq, joint_attention_dim]
    /// * `timestep` - Diffusion timestep [batch]
    /// * `img_shapes` - List of (frame, height, width) tuples for RoPE
    /// * `controlnet_residuals` - Optional residuals from ControlNet, one per block
    ///
    /// # Returns
    /// Output predictions [batch, img_seq, patch_size² × out_channels]
    pub fn forward_with_controlnet(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: &Tensor,
        img_shapes: &[(usize, usize, usize)],
        controlnet_residuals: Option<&[Tensor]>,
    ) -> Result<Tensor> {
        let dtype = hidden_states.dtype();
        let device = hidden_states.device();

        // Project image latents: [batch, seq, 64] -> [batch, seq, 3072]
        let mut hidden_states = hidden_states.apply(&self.img_in)?;

        // Normalize and project text: [batch, seq, 3584] -> [batch, seq, 3072]
        let mut encoder_hidden_states = encoder_hidden_states.apply(&self.txt_norm)?;
        encoder_hidden_states = encoder_hidden_states.apply(&self.txt_in)?;

        // Handle zero_cond_t for edit mode:
        // - Double the timestep: [t, 0] creates two modulation sets
        // - Create modulate_index: per-token mask for which modulation to use
        let timestep = timestep.to_dtype(dtype)?;
        let (timestep, modulate_index) = if self.zero_cond_t {
            // Double timestep: [t, 0] for two different modulation parameter sets
            let zero_timestep = (&timestep * 0.0)?;
            let doubled_timestep = Tensor::cat(&[&timestep, &zero_timestep], 0)?;

            // Create modulate_index: marks which tokens use which modulation
            // In edit mode, img_shapes is [(noise_f, noise_h, noise_w), (img_f, img_h, img_w), ...]
            // - First shape (index 0): noise latents -> use timestep modulation (index 0)
            // - Remaining shapes (index 1+): reference images -> use zero-timestep modulation (index 1)
            let modulate_idx = Self::create_modulate_index(img_shapes, device)?;

            (doubled_timestep, Some(modulate_idx))
        } else {
            (timestep, None)
        };

        // Compute timestep embedding (doubled if zero_cond_t)
        let temb = self.time_text_embed.forward(&timestep, dtype)?;

        // Derive text sequence length from actual encoder_hidden_states shape
        let txt_seq_len = encoder_hidden_states.dim(1)?;
        let txt_seq_lens = &[txt_seq_len];

        // Compute RoPE frequencies
        let image_rotary_emb = self.pos_embed.forward(img_shapes, txt_seq_lens)?;

        // Process through transformer blocks
        for (idx, block) in self.transformer_blocks.iter().enumerate() {
            let (enc_out, hid_out) = block.forward_with_modulate_index(
                &hidden_states,
                &encoder_hidden_states,
                &temb,
                Some(&image_rotary_emb),
                modulate_index.as_ref(),
            )?;
            encoder_hidden_states = enc_out;
            hidden_states = hid_out;

            // Apply ControlNet residual if available for this block
            if let Some(residuals) = controlnet_residuals {
                if idx < residuals.len() {
                    hidden_states = (&hidden_states + &residuals[idx])?;
                }
            }
        }

        // For zero_cond_t, use only the first half of temb for final normalization
        let temb_for_norm = if self.zero_cond_t {
            let batch_size = temb.dim(0)? / 2;
            temb.narrow(0, 0, batch_size)?
        } else {
            temb
        };

        // Final normalization with AdaLN
        let hidden_states = self.norm_out.forward(&hidden_states, &temb_for_norm)?;

        // Project to output: [batch, seq, patch_size² × out_channels]
        let output = hidden_states.apply(&self.proj_out)?;

        Ok(output)
    }

    /// Create modulate_index tensor for edit mode.
    ///
    /// In edit mode, img_shapes contains multiple shapes:
    /// - First shape: noise latents being denoised (use index 0 = actual timestep)
    /// - Remaining shapes: reference image latents (use index 1 = zero timestep)
    ///
    /// Returns a tensor of shape [1, total_seq_len] with 0s for noise tokens and 1s for image tokens.
    fn create_modulate_index(
        img_shapes: &[(usize, usize, usize)],
        device: &candle::Device,
    ) -> Result<Tensor> {
        if img_shapes.is_empty() {
            candle::bail!("img_shapes cannot be empty for modulate_index creation");
        }

        let mut indices: Vec<i64> = Vec::new();

        // First shape (noise latents) -> index 0
        let (f0, h0, w0) = img_shapes[0];
        let noise_seq_len = f0 * h0 * w0;
        indices.extend(std::iter::repeat_n(0i64, noise_seq_len));

        // Remaining shapes (reference images) -> index 1
        for &(f, h, w) in img_shapes.iter().skip(1) {
            let seq_len = f * h * w;
            indices.extend(std::iter::repeat_n(1i64, seq_len));
        }

        // Create tensor with shape [1, total_seq_len] (batch dim = 1 for now)
        let total_len = indices.len();
        Tensor::from_vec(indices, (1, total_len), device)
    }
}

/// Pack latents from [batch, 1, channels, height, width] to [batch, (H/2)×(W/2), channels×4].
///
/// Input uses (B, T, C, H, W) convention matching PyTorch's diffusers pipeline.
/// The view() operation interprets the flat data as [B, C, H/2, 2, W/2, 2] since T=1.
/// The 2×2 spatial patches are flattened into the channel dimension.
pub fn pack_latents(latents: &Tensor, height: usize, width: usize) -> Result<Tensor> {
    let (batch, _frames, channels, h, w) = latents.dims5()?;
    assert_eq!(h, height, "Height mismatch");
    assert_eq!(w, width, "Width mismatch");

    // [batch, 1, channels, height, width] -> [batch, channels, height/2, 2, width/2, 2]
    // This reinterprets the flat data (works because T=1)
    let latents = latents.reshape((batch, channels, height / 2, 2, width / 2, 2))?;

    // Permute to [batch, height/2, width/2, channels, 2, 2]
    let latents = latents.permute([0, 2, 4, 1, 3, 5])?;

    // Reshape to [batch, (height/2)×(width/2), channels×4]
    latents.reshape((batch, (height / 2) * (width / 2), channels * 4))
}

/// Unpack latents from [batch, num_patches, channels×4] to [batch, channels, 1, height, width].
///
/// Output uses (B, C, T, H, W) convention matching PyTorch's _unpack_latents.
/// Reverses the packing operation.
pub fn unpack_latents(
    latents: &Tensor,
    height: usize,
    width: usize,
    out_channels: usize,
) -> Result<Tensor> {
    let (batch, _num_patches, packed_channels) = latents.dims3()?;
    let channels = packed_channels / 4;
    assert_eq!(channels, out_channels, "Channel mismatch");

    // [batch, num_patches, channels×4] -> [batch, height/2, width/2, channels, 2, 2]
    let latents = latents.reshape((batch, height / 2, width / 2, channels, 2, 2))?;

    // Permute to [batch, channels, height/2, 2, width/2, 2]
    // From: [0, 1, 2, 3, 4, 5] = [B, H/2, W/2, C, 2, 2]
    // To:   [B, C, H/2, 2, W/2, 2]
    let latents = latents.permute([0, 3, 1, 4, 2, 5])?;

    // Reshape to [batch, channels, 1, height, width] - (B, C, T, H, W) convention
    latents.reshape((batch, channels, 1, height, width))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn test_pack_unpack_latents() -> Result<()> {
        let device = Device::Cpu;
        let batch = 2;
        let channels = 16;
        let height = 64;
        let width = 64;

        // Create random latents in [B, T, C, H, W] format (matching PyTorch diffusers)
        let latents = Tensor::randn(0f32, 1f32, (batch, 1, channels, height, width), &device)?;

        // Pack: [B, T, C, H, W] -> [B, num_patches, C*4]
        let packed = pack_latents(&latents, height, width)?;
        assert_eq!(
            packed.dims(),
            &[batch, (height / 2) * (width / 2), channels * 4]
        );

        // Unpack: [B, num_patches, C*4] -> [B, C, T, H, W]
        let unpacked = unpack_latents(&packed, height, width, channels)?;
        assert_eq!(unpacked.dims(), &[batch, channels, 1, height, width]);

        // PyTorch: pack takes [B, T, C, H, W], unpack gives [B, C, T, H, W]
        // Verify data is preserved by comparing with permuted input
        let latents_bcthw = latents.permute([0, 2, 1, 3, 4])?;
        let diff = (&latents_bcthw - &unpacked)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            diff < 1e-5,
            "Pack/unpack should preserve data, diff: {}",
            diff
        );

        Ok(())
    }
}
