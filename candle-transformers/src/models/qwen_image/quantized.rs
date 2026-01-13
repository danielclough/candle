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
use candle_nn::{LayerNorm, Module, RmsNorm};
use std::io::{Read, Seek};
use std::sync::Arc;

use crate::models::with_tracing::QMatMul;

use super::blocks::{
    apply_modulation_with_index, layer_norm_no_affine, DirectRmsNormPair, FeedForward, Modulation,
    QwenDoubleStreamAttention,
};
use super::config::InferenceConfig;
use super::rope::{timestep_embedding, QwenEmbedRope};

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
// Quantized Timestep Embeddings
// ============================================================================

/// Quantized timestep projection embeddings.
#[derive(Debug, Clone)]
pub struct QwenTimestepProjEmbeddingsQuantized {
    linear1: QLinear,
    linear2: QLinear,
}

impl QwenTimestepProjEmbeddingsQuantized {
    pub fn forward(&self, timestep: &Tensor, dtype: DType) -> Result<Tensor> {
        let timesteps_proj = timestep_embedding(timestep, 256, dtype)?;
        let x = self.linear1.forward(&timesteps_proj)?;
        let x = x.silu()?;
        self.linear2.forward(&x)
    }
}

// ============================================================================
// Quantized AdaLayerNorm
// ============================================================================

/// Quantized adaptive layer normalization.
#[derive(Debug, Clone)]
pub struct AdaLayerNormContinuousQuantized {
    norm: LayerNorm,
    linear: QLinear,
}

impl AdaLayerNormContinuousQuantized {
    pub fn forward(&self, xs: &Tensor, conditioning: &Tensor) -> Result<Tensor> {
        let emb = conditioning.silu()?;
        let emb = self.linear.forward(&emb)?;
        let chunks = emb.chunk(2, 1)?;
        if chunks.len() != 2 {
            candle::bail!("Expected 2 chunks for AdaLN, got {}", chunks.len());
        }
        let scale = &chunks[0];
        let shift = &chunks[1];

        xs.apply(&self.norm)?
            .broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift.unsqueeze(1)?)
    }
}

// ============================================================================
// Quantized Modulation (Type Alias)
// ============================================================================

/// Quantized modulation layer - uses the generic Modulation with QLinear.
pub type ModulationQuantized = Modulation<QLinear>;

// ============================================================================
// Quantized Attention (Type Alias)
// ============================================================================

/// Quantized dual-stream attention - uses the generic QwenDoubleStreamAttention.
pub type QwenDoubleStreamAttentionQuantized =
    QwenDoubleStreamAttention<QLinear, DirectRmsNormPair>;

// ============================================================================
// Quantized MLP (Type Alias)
// ============================================================================

/// Quantized feed-forward MLP - uses the generic FeedForward with QLinear.
pub type MlpQuantized = FeedForward<QLinear>;

// ============================================================================
// Quantized Transformer Block
// ============================================================================

/// Quantized dual-stream transformer block.
#[derive(Debug, Clone)]
pub struct QwenImageTransformerBlockQuantized {
    // Layer norms (parameter-free)
    norm1: LayerNorm,
    norm1_context: LayerNorm,

    // Modulation
    modulation: ModulationQuantized,
    modulation_context: ModulationQuantized,

    // Attention
    attn: QwenDoubleStreamAttentionQuantized,

    // MLPs
    mlp: MlpQuantized,
    mlp_context: MlpQuantized,
}

impl QwenImageTransformerBlockQuantized {
    /// Forward pass through the dual-stream block (standard mode, no per-token modulation).
    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        temb: &Tensor,
        img_freqs: &Tensor,
        txt_freqs: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        self.forward_with_modulate_index(img, txt, temb, img_freqs, txt_freqs, None)
    }

    /// Forward pass with modulate_index for edit mode (zero_cond_t).
    ///
    /// When modulate_index is provided:
    /// - `temb` is doubled: [2*batch, dim] with [actual_timestep, zero_timestep]
    /// - Image modulation uses per-token selection based on modulate_index
    /// - Text modulation uses only actual timestep (first half of temb)
    pub fn forward_with_modulate_index(
        &self,
        img: &Tensor,
        txt: &Tensor,
        temb: &Tensor,
        img_freqs: &Tensor,
        txt_freqs: &Tensor,
        modulate_index: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let batch_size = img.dim(0)?;

        // Get modulation parameters for image stream (uses full temb if doubled)
        let (img_mod1, img_mod2) = self.modulation.forward(temb)?;

        // For text stream, use only first half of temb if modulate_index is provided (zero_cond_t mode)
        let txt_temb = if modulate_index.is_some() {
            // In zero_cond_t mode, temb is doubled [2*batch, dim]
            // Text uses only actual timestep (first half)
            temb.narrow(0, 0, batch_size)?
        } else {
            temb.clone()
        };
        let (txt_mod1, txt_mod2) = self.modulation_context.forward(&txt_temb)?;

        // === Attention phase ===

        // Image: norm1 + modulate (with per-token selection if modulate_index provided)
        let img_normed = self.norm1.forward(img)?;

        let (img_modulated, img_gate1) = if let Some(mod_idx) = modulate_index {
            // Per-token modulation for edit mode
            apply_modulation_with_index(&img_normed, &img_mod1, mod_idx)?
        } else {
            // Standard modulation
            let modulated = img_mod1.scale_shift(&img_normed)?;
            let gate = img_mod1.gate.clone();
            (modulated, gate)
        };

        // Text: norm1 + modulate (always standard, no per-token selection)
        let txt_normed = self.norm1_context.forward(txt)?;
        let txt_modulated = txt_mod1.scale_shift(&txt_normed)?;
        let txt_gate1 = txt_mod1.gate.clone();

        // Joint attention - bundle freqs into tuple for generic forward
        let freqs = (img_freqs.clone(), txt_freqs.clone());
        let (img_attn, txt_attn) = self.attn.forward(&img_modulated, &txt_modulated, Some(&freqs))?;

        // Gated residual for attention (use per-token gate for image if in edit mode)
        let gated_img_attn = img_gate1.broadcast_mul(&img_attn)?;
        let gated_txt_attn = txt_gate1.broadcast_mul(&txt_attn)?;

        // Residual add in F32 to avoid overflow, then cast back
        let img = (img.to_dtype(DType::F32)? + gated_img_attn.to_dtype(DType::F32)?)?
            .to_dtype(img.dtype())?;
        let txt = (txt.to_dtype(DType::F32)? + gated_txt_attn.to_dtype(DType::F32)?)?
            .to_dtype(txt.dtype())?;

        // === MLP phase ===

        // Image: norm1 + modulate + MLP + gated residual
        let img_normed2 = self.norm1.forward(&img)?;

        let (img_modulated2, img_gate2) = if let Some(mod_idx) = modulate_index {
            apply_modulation_with_index(&img_normed2, &img_mod2, mod_idx)?
        } else {
            let modulated = img_mod2.scale_shift(&img_normed2)?;
            let gate = img_mod2.gate.clone();
            (modulated, gate)
        };

        let img_mlp_out = self.mlp.forward(&img_modulated2)?;
        let gated_img_mlp = img_gate2.broadcast_mul(&img_mlp_out)?;

        // Residual add in F32
        let img = (img.to_dtype(DType::F32)? + gated_img_mlp.to_dtype(DType::F32)?)?
            .to_dtype(img.dtype())?;

        // Text: norm1_context + modulate + MLP + gated residual (always standard)
        let txt_normed2 = self.norm1_context.forward(&txt)?;
        let txt_modulated2 = txt_mod2.scale_shift(&txt_normed2)?;
        let txt_mlp_out = self.mlp_context.forward(&txt_modulated2)?;
        let gated_txt_mlp = txt_mod2.gate(&txt_mlp_out)?;

        // Residual add in F32
        let txt = (txt.to_dtype(DType::F32)? + gated_txt_mlp.to_dtype(DType::F32)?)?
            .to_dtype(txt.dtype())?;

        Ok((img, txt))
    }
}

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
        let time_text_embed = QwenTimestepProjEmbeddingsQuantized {
            linear1: load_qlinear!(
                "time_text_embed.timestep_embedder.linear_1.weight",
                "time_text_embed.timestep_embedder.linear_1.bias",
                bias_dtype
            ),
            linear2: load_qlinear!(
                "time_text_embed.timestep_embedder.linear_2.weight",
                "time_text_embed.timestep_embedder.linear_2.bias",
                bias_dtype
            ),
        };

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
            let modulation = Modulation::from_linear(load_qlinear!(
                format!("{prefix}.img_mod.1.weight"),
                format!("{prefix}.img_mod.1.bias"),
                bias_dtype
            ));
            let modulation_context = Modulation::from_linear(load_qlinear!(
                format!("{prefix}.txt_mod.1.weight"),
                format!("{prefix}.txt_mod.1.bias"),
                bias_dtype
            ));

            // Attention - using from_parts constructor
            let img_norm = DirectRmsNormPair {
                query_norm: RmsNorm::new(
                    load_dequant!(format!("{prefix}.attn.norm_q.weight"), DType::F32),
                    1e-6,
                ),
                key_norm: RmsNorm::new(
                    load_dequant!(format!("{prefix}.attn.norm_k.weight"), DType::F32),
                    1e-6,
                ),
            };
            let txt_norm = DirectRmsNormPair {
                query_norm: RmsNorm::new(
                    load_dequant!(format!("{prefix}.attn.norm_added_q.weight"), DType::F32),
                    1e-6,
                ),
                key_norm: RmsNorm::new(
                    load_dequant!(format!("{prefix}.attn.norm_added_k.weight"), DType::F32),
                    1e-6,
                ),
            };

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
            let mlp = FeedForward::from_linears(
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
            let mlp_context = FeedForward::from_linears(
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

            // Layer norms (parameter-free)
            let norm1 = layer_norm_no_affine(inner_dim, 1e-6, device, DType::F32)?;
            let norm1_context = layer_norm_no_affine(inner_dim, 1e-6, device, DType::F32)?;

            transformer_blocks.push(QwenImageTransformerBlockQuantized {
                norm1,
                norm1_context,
                modulation,
                modulation_context,
                attn,
                mlp,
                mlp_context,
            });
        }

        // Output layers
        let norm_out = AdaLayerNormContinuousQuantized {
            norm: layer_norm_no_affine(inner_dim, 1e-6, device, DType::F32)?,
            linear: load_qlinear!("norm_out.linear.weight", "norm_out.linear.bias", bias_dtype),
        };
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

        // Process through transformer blocks
        for block in self.transformer_blocks.iter() {
            let (new_img, new_txt) = block.forward_with_modulate_index(
                &img,
                &txt,
                &temb,
                &img_freqs,
                &txt_freqs,
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
