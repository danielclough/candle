//! Quantized Qwen-Image Transformer Model.
//!
//! This module provides GGUF-quantized loading for the Qwen-Image diffusion transformer.
//! The quantization follows the patterns established in other quantized models:
//! - Weight matrices use QMatMul for memory-efficient quantized inference
//! - Normalization layers are dequantized (small memory footprint)
//! - RoPE computations stay in FP16/FP32 for numerical stability
//!
//! # Usage
//!
//! ```ignore
//! use candle::quantized::gguf_file;
//!
//! let mut file = std::fs::File::open("qwen-image-q4_k.gguf")?;
//! let content = gguf_file::Content::read(&mut file)?;
//! let model = QwenImageTransformer2DModelQuantized::from_gguf(content, &mut file, &device)?;
//! ```

use candle::{quantized::gguf_file, DType, Device, Result, Tensor};
use candle_nn::{Module, RmsNorm};
use std::io::{Read, Seek};
use std::sync::Arc;

use crate::models::with_tracing::QMatMul;

use super::blocks::{
    layer_norm_no_affine, FeedForward, Modulation, QkNorm, QwenDoubleStreamAttention,
    QwenImageTransformerBlock,
};
use super::config::InferenceConfig;
use super::model::{AdaLayerNormContinuous, QwenTimestepProjEmbeddings};
use super::rope::QwenEmbedRope;

// ============================================================================
// QLinear: Quantized Linear with Bias Support
// ============================================================================

/// A quantized linear layer that combines QMatMul weights with an optional bias.
/// This mirrors candle_nn::Linear but uses quantized weights.
#[derive(Debug, Clone)]
pub struct QLinear {
    weight: QMatMul,
    bias: Option<Tensor>,
}

impl QLinear {
    pub fn new(weight: QMatMul, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.weight.forward(x)?;
        match &self.bias {
            Some(b) => out.broadcast_add(b),
            None => Ok(out),
        }
    }
}

impl Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.weight.forward(x)?;
        match &self.bias {
            Some(b) => out.broadcast_add(b),
            None => Ok(out),
        }
    }
}

// ============================================================================
// Quantized Timestep Embeddings (Type Alias)
// ============================================================================

/// Quantized timestep projection embeddings - uses the generic type with QLinear.
pub type QwenTimestepProjEmbeddingsQuantized = QwenTimestepProjEmbeddings<QLinear>;

// ============================================================================
// Quantized AdaLayerNorm (Type Alias)
// ============================================================================

/// Quantized adaptive layer normalization - uses the generic type with QLinear.
pub type AdaLayerNormContinuousQuantized = AdaLayerNormContinuous<QLinear>;

// ============================================================================
// Quantized Modulation (Type Alias)
// ============================================================================

/// Quantized modulation layer - uses the generic Modulation with QLinear.
pub type ModulationQuantized = Modulation<QLinear>;

// ============================================================================
// Quantized Attention (Type Alias)
// ============================================================================

/// Quantized dual-stream attention - uses the generic QwenDoubleStreamAttention.
pub type QwenDoubleStreamAttentionQuantized = QwenDoubleStreamAttention<QLinear>;

// ============================================================================
// Quantized MLP (Type Alias)
// ============================================================================

/// Quantized feed-forward MLP - uses the generic FeedForward with QLinear.
pub type MlpQuantized = FeedForward<QLinear>;

// ============================================================================
// Quantized Transformer Block (Type Alias)
// ============================================================================

/// Quantized dual-stream transformer block - uses the generic QwenImageTransformerBlock.
pub type QwenImageTransformerBlockQuantized = QwenImageTransformerBlock<QLinear>;

// ============================================================================
// Quantized Main Model
// ============================================================================

/// Quantized Qwen-Image Transformer 2D Model.
///
/// A quantized version of the 20B parameter dual-stream MMDiT.
///
/// # Edit Mode (zero_cond_t)
///
/// When `zero_cond_t` is enabled (edit mode), the model uses per-token modulation:
/// - Timestep is doubled: `[t, 0]` → creates two sets of modulation parameters
/// - `modulate_index` tensor marks which modulation to use per token:
///   - Index 0 (actual timestep): for noise latents being denoised
///   - Index 1 (zero timestep): for reference image latents (conditioning)
#[derive(Debug, Clone)]
pub struct QwenImageTransformer2DModelQuantized {
    #[allow(dead_code)]
    inner_dim: usize,
    #[allow(dead_code)]
    out_channels: usize,
    #[allow(dead_code)]
    patch_size: usize,

    pos_embed: QwenEmbedRope,
    time_text_embed: QwenTimestepProjEmbeddingsQuantized,
    txt_norm: RmsNorm,
    img_in: QLinear,
    txt_in: QLinear,
    transformer_blocks: Vec<QwenImageTransformerBlockQuantized>,
    norm_out: AdaLayerNormContinuousQuantized,
    proj_out: QLinear,

    /// Whether to use zero conditioning for timestep (edit mode).
    /// When true, doubles timestep and uses per-token modulation.
    zero_cond_t: bool,
}

impl QwenImageTransformer2DModelQuantized {
    /// Load model from GGUF file.
    ///
    /// # Arguments
    /// * `ct` - GGUF file content
    /// * `reader` - Reader for tensor data
    /// * `device` - Device to load model on
    /// * `dtype` - Working dtype for biases and normalization layers
    /// * `inference_config` - Runtime inference configuration (attention behavior, etc.)
    /// * `zero_cond_t` - Enable edit mode with per-token modulation
    ///
    /// # Note
    /// The GGUF tensor names must follow the llama.cpp convention for Qwen-Image.
    /// Use `debug_print_gguf_tensors()` to inspect actual tensor names in your file.
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        dtype: DType,
        inference_config: &InferenceConfig,
        zero_cond_t: bool,
    ) -> Result<Self> {
        // Extract config from metadata
        let get_u32 = |keys: &[&str]| -> Result<u32> {
            for key in keys {
                if let Some(v) = ct.metadata.get(*key) {
                    return v.to_u32();
                }
            }
            candle::bail!("cannot find any of {:?} in metadata", keys);
        };

        // Model dimensions
        let inner_dim =
            get_u32(&["qwen_image.inner_dim", "transformer.inner_dim"]).unwrap_or(3072) as usize;

        let num_heads = get_u32(&[
            "qwen_image.num_attention_heads",
            "transformer.num_attention_heads",
        ])
        .unwrap_or(24) as usize;

        let head_dim = inner_dim / num_heads;

        let num_layers =
            get_u32(&["qwen_image.num_layers", "transformer.num_layers"]).unwrap_or(60) as usize;

        let patch_size = get_u32(&["qwen_image.patch_size"]).unwrap_or(2) as usize;

        let out_channels = get_u32(&["qwen_image.out_channels"]).unwrap_or(16) as usize;

        let _joint_attention_dim =
            get_u32(&["qwen_image.joint_attention_dim"]).unwrap_or(3584) as usize;

        let _in_channels = get_u32(&["qwen_image.in_channels"]).unwrap_or(64) as usize;

        let theta = 10000usize;
        let axes_dims = (16usize, 56usize, 56usize);

        // Create RoPE embeddings
        let pos_embed = QwenEmbedRope::new(
            theta,
            vec![axes_dims.0, axes_dims.1, axes_dims.2],
            true,
            device,
            DType::F32,
        )?;

        // Macro to load and dequantize tensor (for small params)
        macro_rules! load_dequant {
            ($name:expr, $dtype:expr) => {{
                let name: &str = &$name;
                let qt = ct.tensor(reader, name, device)?;
                qt.dequantize(device)?.to_dtype($dtype)?
            }};
        }

        // Macro to load QLinear with weight and optional bias
        macro_rules! load_qlinear {
            ($weight_name:expr, $bias_name:expr, $dtype:expr) => {{
                let weight_name: &str = &$weight_name;
                let bias_name: &str = &$bias_name;
                let qt = ct.tensor(reader, weight_name, device)?;
                let weight = QMatMul::from_weights(Arc::new(qt))?;
                let bias = if ct.tensor_infos.contains_key(bias_name) {
                    let bt = ct.tensor(reader, bias_name, device)?;
                    Some(bt.dequantize(device)?.to_dtype($dtype)?)
                } else {
                    None
                };
                QLinear::new(weight, bias)
            }};
        }

        // Use the passed dtype for biases to match the working precision
        let bias_dtype = dtype;

        // Load timestep embeddings
        let time_text_embed = QwenTimestepProjEmbeddings::from_linears(
            load_qlinear!(
                "time_text_embed.timestep_embedder.linear_1.weight",
                "time_text_embed.timestep_embedder.linear_1.bias",
                bias_dtype
            ),
            load_qlinear!(
                "time_text_embed.timestep_embedder.linear_2.weight",
                "time_text_embed.timestep_embedder.linear_2.bias",
                bias_dtype
            ),
        );

        // Load text norm
        let txt_norm_weight = load_dequant!("txt_norm.weight", DType::F32);
        let txt_norm = RmsNorm::new(txt_norm_weight, 1e-6);

        // Load input projections
        let img_in = load_qlinear!("img_in.weight", "img_in.bias", bias_dtype);
        let txt_in = load_qlinear!("txt_in.weight", "txt_in.bias", bias_dtype);

        // Load transformer blocks
        let mut transformer_blocks = Vec::with_capacity(num_layers);
        for idx in 0..num_layers {
            let prefix = format!("transformer_blocks.{idx}");

            // Modulation (GGUF uses img_mod/txt_mod naming from Flux)
            let img_mod = Modulation::from_linear(load_qlinear!(
                format!("{prefix}.img_mod.1.weight"),
                format!("{prefix}.img_mod.1.bias"),
                bias_dtype
            ));
            let txt_mod = Modulation::from_linear(load_qlinear!(
                format!("{prefix}.txt_mod.1.weight"),
                format!("{prefix}.txt_mod.1.bias"),
                bias_dtype
            ));

            // Attention - using from_parts constructor
            let img_norm = QkNorm::from_rms_norms(
                RmsNorm::new(
                    load_dequant!(format!("{prefix}.attn.norm_q.weight"), DType::F32),
                    1e-6,
                ),
                RmsNorm::new(
                    load_dequant!(format!("{prefix}.attn.norm_k.weight"), DType::F32),
                    1e-6,
                ),
            );
            let txt_norm = QkNorm::from_rms_norms(
                RmsNorm::new(
                    load_dequant!(format!("{prefix}.attn.norm_added_q.weight"), DType::F32),
                    1e-6,
                ),
                RmsNorm::new(
                    load_dequant!(format!("{prefix}.attn.norm_added_k.weight"), DType::F32),
                    1e-6,
                ),
            );

            let attn = QwenDoubleStreamAttention::from_parts(
                load_qlinear!(
                    format!("{prefix}.attn.to_q.weight"),
                    format!("{prefix}.attn.to_q.bias"),
                    bias_dtype
                ),
                load_qlinear!(
                    format!("{prefix}.attn.to_k.weight"),
                    format!("{prefix}.attn.to_k.bias"),
                    bias_dtype
                ),
                load_qlinear!(
                    format!("{prefix}.attn.to_v.weight"),
                    format!("{prefix}.attn.to_v.bias"),
                    bias_dtype
                ),
                load_qlinear!(
                    format!("{prefix}.attn.to_out.0.weight"),
                    format!("{prefix}.attn.to_out.0.bias"),
                    bias_dtype
                ),
                load_qlinear!(
                    format!("{prefix}.attn.add_q_proj.weight"),
                    format!("{prefix}.attn.add_q_proj.bias"),
                    bias_dtype
                ),
                load_qlinear!(
                    format!("{prefix}.attn.add_k_proj.weight"),
                    format!("{prefix}.attn.add_k_proj.bias"),
                    bias_dtype
                ),
                load_qlinear!(
                    format!("{prefix}.attn.add_v_proj.weight"),
                    format!("{prefix}.attn.add_v_proj.bias"),
                    bias_dtype
                ),
                load_qlinear!(
                    format!("{prefix}.attn.to_add_out.weight"),
                    format!("{prefix}.attn.to_add_out.bias"),
                    bias_dtype
                ),
                img_norm,
                txt_norm,
                num_heads,
                head_dim,
                inference_config.upcast_attention,
            );

            // MLPs (GGUF uses img_mlp/txt_mlp naming from Flux)
            let img_mlp = FeedForward::from_linears(
                load_qlinear!(
                    format!("{prefix}.img_mlp.net.0.proj.weight"),
                    format!("{prefix}.img_mlp.net.0.proj.bias"),
                    bias_dtype
                ),
                load_qlinear!(
                    format!("{prefix}.img_mlp.net.2.weight"),
                    format!("{prefix}.img_mlp.net.2.bias"),
                    bias_dtype
                ),
            );
            let txt_mlp = FeedForward::from_linears(
                load_qlinear!(
                    format!("{prefix}.txt_mlp.net.0.proj.weight"),
                    format!("{prefix}.txt_mlp.net.0.proj.bias"),
                    bias_dtype
                ),
                load_qlinear!(
                    format!("{prefix}.txt_mlp.net.2.weight"),
                    format!("{prefix}.txt_mlp.net.2.bias"),
                    bias_dtype
                ),
            );

            // Layer norms (parameter-free, reused for both attention and MLP phases)
            let img_norm1 = layer_norm_no_affine(inner_dim, 1e-6, device, DType::F32)?;
            let txt_norm1 = layer_norm_no_affine(inner_dim, 1e-6, device, DType::F32)?;

            transformer_blocks.push(QwenImageTransformerBlockQuantized {
                img_mod,
                img_norm1,
                img_mlp,
                txt_mod,
                txt_norm1,
                txt_mlp,
                attn,
            });
        }

        // Output layers
        let norm_out = AdaLayerNormContinuous::from_parts(
            layer_norm_no_affine(inner_dim, 1e-6, device, DType::F32)?,
            load_qlinear!("norm_out.linear.weight", "norm_out.linear.bias", bias_dtype),
        );
        let proj_out = load_qlinear!("proj_out.weight", "proj_out.bias", bias_dtype);

        Ok(Self {
            inner_dim,
            out_channels,
            patch_size,
            pos_embed,
            time_text_embed,
            txt_norm,
            img_in,
            txt_in,
            transformer_blocks,
            norm_out,
            proj_out,
            zero_cond_t,
        })
    }

    /// Compute timestep embeddings externally.
    ///
    /// This is useful for edit mode where temb needs to be computed once
    /// and potentially substituted for debugging.
    pub fn compute_temb(&self, timestep: &Tensor, dtype: DType) -> Result<Tensor> {
        self.time_text_embed.forward(timestep, dtype)
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `img` - Packed image latents [batch, seq, 64]
    /// * `txt` - Text embeddings [batch, txt_seq, 3584]
    /// * `timestep` - Timestep values [batch]
    /// * `img_shapes` - Image shapes [(frames, height, width), ...]
    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        timestep: &Tensor,
        img_shapes: &[(usize, usize, usize)],
    ) -> Result<Tensor> {
        let dtype = img.dtype();
        let device = img.device();

        // Project inputs (QMatMul handles dtype conversion automatically)
        let mut img = self.img_in.forward(img)?;

        let txt_normed = self.txt_norm.forward(txt)?;
        let mut txt = self.txt_in.forward(&txt_normed)?;

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

        // Timestep embeddings (doubled if zero_cond_t)
        let temb = self.time_text_embed.forward(&timestep, dtype)?;

        // Compute RoPE frequencies (derive txt_lens from tensor shape)
        let txt_lens = &[txt.dim(1)?];
        let (img_freqs, txt_freqs) = self.pos_embed.forward(img_shapes, txt_lens)?;
        let image_rotary_emb = (img_freqs, txt_freqs);

        // Process through transformer blocks
        for block in self.transformer_blocks.iter() {
            let (new_txt, new_img) = block.forward_with_modulate_index(
                &img,
                &txt,
                &temb,
                Some(&image_rotary_emb),
                modulate_index.as_ref(),
            )?;
            img = new_img;
            txt = new_txt;
        }

        // For zero_cond_t, use only the first half of temb for final normalization
        let temb_for_norm = if self.zero_cond_t {
            let batch_size = temb.dim(0)? / 2;
            temb.narrow(0, 0, batch_size)?
        } else {
            temb
        };

        // Output projection
        let img = self.norm_out.forward(&img, &temb_for_norm)?;
        self.proj_out.forward(&img)
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
