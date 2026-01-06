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
use super::debug::debug_tensor;
use super::rope::{timestep_embedding, QwenEmbedRope};

/// Create a parameter-free LayerNorm (equivalent to PyTorch's elementwise_affine=False).
fn layer_norm_no_affine(size: usize, eps: f64, device: &candle::Device, dtype: DType) -> Result<LayerNorm> {
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
        let timestep_embedder_linear2 =
            candle_nn::linear(embedding_dim, embedding_dim, vb.pp("timestep_embedder.linear_2"))?;
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
    inner_dim: usize,
    out_channels: usize,
    patch_size: usize,

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
            inner_dim,
            out_channels: config.out_channels,
            patch_size: config.patch_size,
            pos_embed,
            time_text_embed,
            txt_norm,
            img_in,
            txt_in,
            transformer_blocks,
            norm_out,
            proj_out,
        })
    }

    /// Forward pass through the transformer.
    ///
    /// # Arguments
    /// * `hidden_states` - Packed image latents [batch, img_seq, in_channels]
    /// * `encoder_hidden_states` - Text embeddings [batch, txt_seq, joint_attention_dim]
    /// * `encoder_hidden_states_mask` - Text attention mask [batch, txt_seq]
    /// * `timestep` - Diffusion timestep [batch]
    /// * `img_shapes` - List of (frame, height, width) tuples for RoPE
    /// * `txt_seq_lens` - Text sequence lengths per batch item
    ///
    /// # Returns
    /// Output predictions [batch, img_seq, patch_size² × out_channels]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_hidden_states_mask: &Tensor,
        timestep: &Tensor,
        img_shapes: &[(usize, usize, usize)],
        txt_seq_lens: &[usize],
    ) -> Result<Tensor> {
        self.forward_with_controlnet(
            hidden_states,
            encoder_hidden_states,
            encoder_hidden_states_mask,
            timestep,
            img_shapes,
            txt_seq_lens,
            None,
        )
    }

    /// Forward pass with optional ControlNet residuals.
    ///
    /// This method allows applying ControlNet-generated residuals at each
    /// transformer block, enabling conditional control over the generation.
    ///
    /// # Arguments
    /// * `hidden_states` - Packed image latents [batch, img_seq, in_channels]
    /// * `encoder_hidden_states` - Text embeddings [batch, txt_seq, joint_attention_dim]
    /// * `encoder_hidden_states_mask` - Text attention mask [batch, txt_seq]
    /// * `timestep` - Diffusion timestep [batch]
    /// * `img_shapes` - List of (frame, height, width) tuples for RoPE
    /// * `txt_seq_lens` - Text sequence lengths per batch item
    /// * `controlnet_residuals` - Optional residuals from ControlNet, one per block
    ///
    /// # Returns
    /// Output predictions [batch, img_seq, patch_size² × out_channels]
    ///
    /// # ControlNet Integration
    ///
    /// When `controlnet_residuals` is provided, each residual is added to the
    /// corresponding block's output. Typically, ControlNet only provides residuals
    /// for a subset of blocks (e.g., first 5 of 60), so residuals are applied
    /// to blocks at matching indices.
    pub fn forward_with_controlnet(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        _encoder_hidden_states_mask: &Tensor,
        timestep: &Tensor,
        img_shapes: &[(usize, usize, usize)],
        _txt_seq_lens: &[usize],
        controlnet_residuals: Option<&[Tensor]>,
    ) -> Result<Tensor> {
        let dtype = hidden_states.dtype();

        debug_tensor("input_hidden_states", hidden_states);
        debug_tensor("input_encoder_hidden_states", encoder_hidden_states);
        debug_tensor("input_timestep", timestep);

        // Project image latents: [batch, seq, 64] -> [batch, seq, 3072]
        let mut hidden_states = hidden_states.apply(&self.img_in)?;
        debug_tensor("after_img_in", &hidden_states);

        // Normalize and project text: [batch, seq, 3584] -> [batch, seq, 3072]
        let mut encoder_hidden_states = encoder_hidden_states.apply(&self.txt_norm)?;
        debug_tensor("after_txt_norm", &encoder_hidden_states);
        encoder_hidden_states = encoder_hidden_states.apply(&self.txt_in)?;
        debug_tensor("after_txt_in", &encoder_hidden_states);

        // Compute timestep embedding
        let timestep = timestep.to_dtype(dtype)?;
        let temb = self.time_text_embed.forward(&timestep, dtype)?;
        debug_tensor("temb", &temb);

        // Derive text sequence length from actual encoder_hidden_states shape
        let txt_seq_len = encoder_hidden_states.dim(1)?;
        let txt_seq_lens = &[txt_seq_len];

        // Compute RoPE frequencies
        let image_rotary_emb = self.pos_embed.forward(img_shapes, txt_seq_lens)?;
        debug_tensor("rope_img_freqs", &image_rotary_emb.0);
        debug_tensor("rope_txt_freqs", &image_rotary_emb.1);

        // Process through transformer blocks
        let num_blocks = self.transformer_blocks.len();
        for (idx, block) in self.transformer_blocks.iter().enumerate() {
            // Use debug mode for block 0 to see internal values
            let (enc_out, hid_out) = if idx == 0 {
                block.forward_with_debug(
                    &hidden_states,
                    &encoder_hidden_states,
                    &temb,
                    Some(&image_rotary_emb),
                    true,  // Enable debug for block 0
                )?
            } else {
                block.forward(
                    &hidden_states,
                    &encoder_hidden_states,
                    &temb,
                    Some(&image_rotary_emb),
                )?
            };
            encoder_hidden_states = enc_out;
            hidden_states = hid_out;

            // Apply ControlNet residual if available for this block
            if let Some(residuals) = controlnet_residuals {
                if idx < residuals.len() {
                    hidden_states = (&hidden_states + &residuals[idx])?;
                }
            }

            // Debug at key blocks: 0, 1, 10, 30, last
            if idx == 0 || idx == 1 || idx == 10 || idx == 30 || idx == num_blocks - 1 {
                debug_tensor(&format!("after_block_{}", idx), &hidden_states);
            }
        }

        // Final normalization with AdaLN
        let hidden_states = self.norm_out.forward(&hidden_states, &temb)?;
        debug_tensor("after_norm_out", &hidden_states);

        // Project to output: [batch, seq, patch_size² × out_channels]
        let output = hidden_states.apply(&self.proj_out)?;
        debug_tensor("output", &output);

        Ok(output)
    }
}

/// Pack latents from [batch, channels, 1, height, width] to [batch, (H/2)×(W/2), channels×4].
///
/// This converts VAE latents into the packed format expected by the transformer.
/// The 2×2 spatial patches are flattened into the channel dimension.
pub fn pack_latents(latents: &Tensor, height: usize, width: usize) -> Result<Tensor> {
    let (batch, channels, _frames, h, w) = latents.dims5()?;
    assert_eq!(h, height, "Height mismatch");
    assert_eq!(w, width, "Width mismatch");

    // [batch, channels, 1, height, width] -> [batch, channels, height/2, 2, width/2, 2]
    let latents = latents.reshape((batch, channels, height / 2, 2, width / 2, 2))?;

    // Permute to [batch, height/2, width/2, channels, 2, 2]
    let latents = latents.permute([0, 2, 4, 1, 3, 5])?;

    // Reshape to [batch, (height/2)×(width/2), channels×4]
    latents.reshape((batch, (height / 2) * (width / 2), channels * 4))
}

/// Unpack latents from [batch, num_patches, channels×4] to [batch, channels, 1, height, width].
///
/// Reverses the packing operation for VAE decoding.
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
    let latents = latents.permute([0, 3, 1, 4, 2, 5])?;

    // Reshape to [batch, channels, 1, height, width]
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

        // Create random latents
        let latents = Tensor::randn(0f32, 1f32, (batch, channels, 1, height, width), &device)?;

        // Pack
        let packed = pack_latents(&latents, height, width)?;
        assert_eq!(packed.dims(), &[batch, (height / 2) * (width / 2), channels * 4]);

        // Unpack
        let unpacked = unpack_latents(&packed, height, width, channels)?;
        assert_eq!(unpacked.dims(), &[batch, channels, 1, height, width]);

        // Values should match (approximately due to reshaping)
        let diff = (&latents - &unpacked)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5, "Pack/unpack should be lossless, diff: {}", diff);

        Ok(())
    }
}
