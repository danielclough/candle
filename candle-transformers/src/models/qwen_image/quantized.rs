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

use candle::{quantized::gguf_file, DType, Device, Result, Tensor, D};
use candle_nn::{LayerNorm, Module, RmsNorm};
use std::io::{Read, Seek};
use std::sync::Arc;

use crate::models::with_tracing::QMatMul;

use super::rope::{timestep_embedding, apply_rotary_emb_qwen, QwenEmbedRope};

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
// Quantized Modulation
// ============================================================================

/// Modulation output.
pub struct ModulationOut {
    pub shift: Tensor,
    pub scale: Tensor,
    pub gate: Tensor,
}

impl ModulationOut {
    pub fn scale_shift(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&(&self.scale + 1.0)?)?
            .broadcast_add(&self.shift)
    }

    pub fn gate(&self, xs: &Tensor) -> Result<Tensor> {
        self.gate.broadcast_mul(xs)
    }
}

/// Quantized modulation layer.
#[derive(Debug, Clone)]
pub struct ModulationQuantized {
    lin: QLinear,
}

impl ModulationQuantized {
    pub fn forward(&self, timestep_emb: &Tensor) -> Result<(ModulationOut, ModulationOut)> {
        let emb = timestep_emb.silu()?;
        // Apply linear and unsqueeze to [batch, 1, 6*dim] for proper broadcasting over sequence
        let out = self.lin.forward(&emb)?.unsqueeze(1)?;

        // Split into 6 chunks for shift1, scale1, gate1, shift2, scale2, gate2
        let chunks = out.chunk(6, D::Minus1)?;
        if chunks.len() != 6 {
            candle::bail!("Expected 6 chunks from modulation, got {}", chunks.len());
        }

        let mod1 = ModulationOut {
            shift: chunks[0].clone(),
            scale: chunks[1].clone(),
            gate: chunks[2].clone(),
        };
        let mod2 = ModulationOut {
            shift: chunks[3].clone(),
            scale: chunks[4].clone(),
            gate: chunks[5].clone(),
        };

        Ok((mod1, mod2))
    }
}

// ============================================================================
// Quantized Attention
// ============================================================================

/// Quantized dual-stream attention.
#[derive(Debug, Clone)]
pub struct QwenDoubleStreamAttentionQuantized {
    // Image stream projections
    to_q: QLinear,
    to_k: QLinear,
    to_v: QLinear,
    to_out: QLinear,

    // Text stream projections
    add_q_proj: QLinear,
    add_k_proj: QLinear,
    add_v_proj: QLinear,
    to_add_out: QLinear,

    // QK normalization (dequantized)
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    norm_add_q: RmsNorm,
    norm_add_k: RmsNorm,

    num_heads: usize,
    head_dim: usize,
}

impl QwenDoubleStreamAttentionQuantized {
    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        img_freqs: &Tensor,
        txt_freqs: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, img_seq, _) = img.dims3()?;
        let txt_seq = txt.dim(1)?;

        // Image stream Q/K/V
        let img_q = self.to_q.forward(img)?;
        let img_k = self.to_k.forward(img)?;
        let img_v = self.to_v.forward(img)?;

        // Text stream Q/K/V
        let txt_q = self.add_q_proj.forward(txt)?;
        let txt_k = self.add_k_proj.forward(txt)?;
        let txt_v = self.add_v_proj.forward(txt)?;

        // Reshape for multi-head attention: [batch, seq, heads, head_dim]
        let img_q = img_q.reshape((batch, img_seq, self.num_heads, self.head_dim))?;
        let img_k = img_k.reshape((batch, img_seq, self.num_heads, self.head_dim))?;
        let img_v = img_v.reshape((batch, img_seq, self.num_heads, self.head_dim))?;

        let txt_q = txt_q.reshape((batch, txt_seq, self.num_heads, self.head_dim))?;
        let txt_k = txt_k.reshape((batch, txt_seq, self.num_heads, self.head_dim))?;
        let txt_v = txt_v.reshape((batch, txt_seq, self.num_heads, self.head_dim))?;

        // QK normalization (on [batch, seq, heads, head_dim])
        let img_q = self.norm_q.forward(&img_q)?;
        let img_k = self.norm_k.forward(&img_k)?;
        let txt_q = self.norm_add_q.forward(&txt_q)?;
        let txt_k = self.norm_add_k.forward(&txt_k)?;

        // Apply RoPE (expects [batch, seq, heads, head_dim])
        let img_q = apply_rotary_emb_qwen(&img_q, img_freqs)?;
        let img_k = apply_rotary_emb_qwen(&img_k, img_freqs)?;
        let txt_q = apply_rotary_emb_qwen(&txt_q, txt_freqs)?;
        let txt_k = apply_rotary_emb_qwen(&txt_k, txt_freqs)?;

        // Concatenate for joint attention on dim 1: [txt, img] -> [batch, txt+img, heads, head_dim]
        let q = Tensor::cat(&[&txt_q, &img_q], 1)?;
        let k = Tensor::cat(&[&txt_k, &img_k], 1)?;
        let v = Tensor::cat(&[&txt_v, &img_v], 1)?;

        // Now transpose for attention: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Scaled dot-product attention (in F32 for numerical stability)
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Transpose back: [B, H, S, D] -> [B, S, H, D]
        let attn_output = attn_output.transpose(1, 2)?;
        let attn_output = attn_output.to_dtype(img.dtype())?;

        // Flatten: [B, S, H, D] -> [B, S, H*D]
        let attn_output = attn_output.flatten_from(2)?;

        // Split back to text and image on dim 1
        let txt_attn = attn_output.narrow(1, 0, txt_seq)?;
        let img_attn = attn_output.narrow(1, txt_seq, img_seq)?;

        let txt_out = self.to_add_out.forward(&txt_attn)?;
        let img_out = self.to_out.forward(&img_attn)?;

        Ok((img_out, txt_out))
    }
}

// ============================================================================
// Quantized MLP
// ============================================================================

/// Quantized feed-forward MLP.
#[derive(Debug, Clone)]
pub struct MlpQuantized {
    proj_in: QLinear,
    proj_out: QLinear,
}

impl MlpQuantized {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.proj_in.forward(xs)?;
        // Use gelu() (tanh approximation) to match FP16 model
        let x = x.gelu()?;
        self.proj_out.forward(&x)
    }
}

// ============================================================================
// Quantized Transformer Block
// ============================================================================

/// Helper to create parameter-free LayerNorm.
fn layer_norm_no_affine(size: usize, eps: f64, device: &Device, dtype: DType) -> Result<LayerNorm> {
    let weight = Tensor::ones(size, dtype, device)?;
    Ok(LayerNorm::new_no_bias(weight, eps))
}

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
    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        temb: &Tensor,
        img_freqs: &Tensor,
        txt_freqs: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Get modulation parameters
        let (img_mod1, img_mod2) = self.modulation.forward(temb)?;
        let (txt_mod1, txt_mod2) = self.modulation_context.forward(temb)?;

        // Pre-attention norm + modulation
        let img_normed = self.norm1.forward(img)?;
        let img_modulated = img_mod1.scale_shift(&img_normed)?;

        let txt_normed = self.norm1_context.forward(txt)?;
        let txt_modulated = txt_mod1.scale_shift(&txt_normed)?;

        // Joint attention
        let (img_attn, txt_attn) =
            self.attn.forward(&img_modulated, &txt_modulated, img_freqs, txt_freqs)?;

        // Gated residual (in F32 for numerical stability due to large values)
        let img_gated = img_mod1.gate(&img_attn)?;
        let img = img.to_dtype(DType::F32)?
            .add(&img_gated.to_dtype(DType::F32)?)?
            .to_dtype(img.dtype())?;

        let txt_gated = txt_mod1.gate(&txt_attn)?;
        let txt = txt.to_dtype(DType::F32)?
            .add(&txt_gated.to_dtype(DType::F32)?)?
            .to_dtype(txt.dtype())?;

        // MLP
        let img_mlp_in = self.norm1.forward(&img)?;
        let img_mlp_in = img_mod2.scale_shift(&img_mlp_in)?;
        let img_mlp_out = self.mlp.forward(&img_mlp_in)?;
        let img_mlp_gated = img_mod2.gate(&img_mlp_out)?;

        let txt_mlp_in = self.norm1_context.forward(&txt)?;
        let txt_mlp_in = txt_mod2.scale_shift(&txt_mlp_in)?;
        let txt_mlp_out = self.mlp_context.forward(&txt_mlp_in)?;
        let txt_mlp_gated = txt_mod2.gate(&txt_mlp_out)?;

        // Final residual (in F32)
        let img = img.to_dtype(DType::F32)?
            .add(&img_mlp_gated.to_dtype(DType::F32)?)?
            .to_dtype(img.dtype())?;

        let txt = txt.to_dtype(DType::F32)?
            .add(&txt_mlp_gated.to_dtype(DType::F32)?)?
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
}

impl QwenImageTransformer2DModelQuantized {
    /// Load model from GGUF file.
    ///
    /// # Arguments
    /// * `ct` - GGUF file content
    /// * `reader` - Reader for tensor data
    /// * `device` - Device to load model on
    ///
    /// # Note
    /// The GGUF tensor names must follow the llama.cpp convention for Qwen-Image.
    /// Use `debug_print_gguf_tensors()` to inspect actual tensor names in your file.
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
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
        let inner_dim = get_u32(&[
            "qwen_image.inner_dim",
            "transformer.inner_dim",
        ])
        .unwrap_or(3072) as usize;

        let num_heads = get_u32(&[
            "qwen_image.num_attention_heads",
            "transformer.num_attention_heads",
        ])
        .unwrap_or(24) as usize;

        let head_dim = inner_dim / num_heads;

        let num_layers = get_u32(&[
            "qwen_image.num_layers",
            "transformer.num_layers",
        ])
        .unwrap_or(60) as usize;

        let patch_size = get_u32(&["qwen_image.patch_size"])
            .unwrap_or(2) as usize;

        let out_channels = get_u32(&["qwen_image.out_channels"])
            .unwrap_or(16) as usize;

        let _joint_attention_dim = get_u32(&["qwen_image.joint_attention_dim"])
            .unwrap_or(3584) as usize;

        let _in_channels = get_u32(&["qwen_image.in_channels"])
            .unwrap_or(64) as usize;

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

        // Helper to get dtype for biases (match input dtype)
        let bias_dtype = DType::BF16;

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
            let modulation = ModulationQuantized {
                lin: load_qlinear!(
                    format!("{prefix}.img_mod.1.weight"),
                    format!("{prefix}.img_mod.1.bias"),
                    bias_dtype
                ),
            };
            let modulation_context = ModulationQuantized {
                lin: load_qlinear!(
                    format!("{prefix}.txt_mod.1.weight"),
                    format!("{prefix}.txt_mod.1.bias"),
                    bias_dtype
                ),
            };

            // Attention
            let attn = QwenDoubleStreamAttentionQuantized {
                to_q: load_qlinear!(
                    format!("{prefix}.attn.to_q.weight"),
                    format!("{prefix}.attn.to_q.bias"),
                    bias_dtype
                ),
                to_k: load_qlinear!(
                    format!("{prefix}.attn.to_k.weight"),
                    format!("{prefix}.attn.to_k.bias"),
                    bias_dtype
                ),
                to_v: load_qlinear!(
                    format!("{prefix}.attn.to_v.weight"),
                    format!("{prefix}.attn.to_v.bias"),
                    bias_dtype
                ),
                to_out: load_qlinear!(
                    format!("{prefix}.attn.to_out.0.weight"),
                    format!("{prefix}.attn.to_out.0.bias"),
                    bias_dtype
                ),
                add_q_proj: load_qlinear!(
                    format!("{prefix}.attn.add_q_proj.weight"),
                    format!("{prefix}.attn.add_q_proj.bias"),
                    bias_dtype
                ),
                add_k_proj: load_qlinear!(
                    format!("{prefix}.attn.add_k_proj.weight"),
                    format!("{prefix}.attn.add_k_proj.bias"),
                    bias_dtype
                ),
                add_v_proj: load_qlinear!(
                    format!("{prefix}.attn.add_v_proj.weight"),
                    format!("{prefix}.attn.add_v_proj.bias"),
                    bias_dtype
                ),
                to_add_out: load_qlinear!(
                    format!("{prefix}.attn.to_add_out.weight"),
                    format!("{prefix}.attn.to_add_out.bias"),
                    bias_dtype
                ),
                norm_q: RmsNorm::new(
                    load_dequant!(format!("{prefix}.attn.norm_q.weight"), DType::F32),
                    1e-6,
                ),
                norm_k: RmsNorm::new(
                    load_dequant!(format!("{prefix}.attn.norm_k.weight"), DType::F32),
                    1e-6,
                ),
                norm_add_q: RmsNorm::new(
                    load_dequant!(format!("{prefix}.attn.norm_added_q.weight"), DType::F32),
                    1e-6,
                ),
                norm_add_k: RmsNorm::new(
                    load_dequant!(format!("{prefix}.attn.norm_added_k.weight"), DType::F32),
                    1e-6,
                ),
                num_heads,
                head_dim,
            };

            // MLPs (GGUF uses img_mlp/txt_mlp naming from Flux)
            let mlp = MlpQuantized {
                proj_in: load_qlinear!(
                    format!("{prefix}.img_mlp.net.0.proj.weight"),
                    format!("{prefix}.img_mlp.net.0.proj.bias"),
                    bias_dtype
                ),
                proj_out: load_qlinear!(
                    format!("{prefix}.img_mlp.net.2.weight"),
                    format!("{prefix}.img_mlp.net.2.bias"),
                    bias_dtype
                ),
            };
            let mlp_context = MlpQuantized {
                proj_in: load_qlinear!(
                    format!("{prefix}.txt_mlp.net.0.proj.weight"),
                    format!("{prefix}.txt_mlp.net.0.proj.bias"),
                    bias_dtype
                ),
                proj_out: load_qlinear!(
                    format!("{prefix}.txt_mlp.net.2.weight"),
                    format!("{prefix}.txt_mlp.net.2.bias"),
                    bias_dtype
                ),
            };

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
    /// * `txt_lens` - Text sequence lengths
    /// * `dtype` - Working dtype
    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        timestep: &Tensor,
        img_shapes: &[(usize, usize, usize)],
        txt_lens: &[usize],
        dtype: DType,
    ) -> Result<Tensor> {
        // Project inputs (QMatMul handles dtype conversion automatically)
        let mut img = self.img_in.forward(img)?;

        let txt_normed = self.txt_norm.forward(txt)?;
        let mut txt = self.txt_in.forward(&txt_normed)?;

        // Timestep embeddings
        let temb = self.time_text_embed.forward(timestep, dtype)?;

        // Compute RoPE frequencies
        let (img_freqs, txt_freqs) = self.pos_embed.forward(img_shapes, txt_lens)?;

        // Process through transformer blocks
        for block in self.transformer_blocks.iter() {
            let (new_img, new_txt) = block.forward(&img, &txt, &temb, &img_freqs, &txt_freqs)?;
            img = new_img;
            txt = new_txt;
        }

        // Output projection
        let img = self.norm_out.forward(&img, &temb)?;
        self.proj_out.forward(&img)
    }
}
