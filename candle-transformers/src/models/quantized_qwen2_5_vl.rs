//! Qwen2.5-VL model implementation with quantization support.
//!
//! Qwen2.5-VL is a multimodal vision-language model with:
//! - Grouped Query Attention (GQA) with M-RoPE (Multimodal RoPE)
//! - RMSNorm for layer normalization
//! - SwiGLU MLP activation
//! - Support for GGUF quantization
//!
//! This quantized version supports:
//! - Loading quantized text decoder from GGUF files
//! - Loading FP16 vision encoder from mmproj GGUF files
//!
//! # Usage
//!
//! ```ignore
//! use candle_transformers::models::quantized_qwen2_5_vl::{ModelWeights, Qwen25VLQuantized};
//! use candle::quantized::gguf_file;
//!
//! // Load quantized text model
//! let mut file = std::fs::File::open("model-q4_k.gguf")?;
//! let ct = gguf_file::Content::read(&mut file)?;
//! let text_model = ModelWeights::from_gguf(ct, &mut file, &device)?;
//!
//! // Load FP16 vision encoder from mmproj
//! let vision_model = Qwen25VLVisionModel::new(&vision_config, vb)?;
//!
//! // Create combined model
//! let model = Qwen25VLQuantized::new(text_model, vision_model, config);
//! ```
//!
//! References:
//! - [Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
//! - [GGUF Models](https://huggingface.co/Mungert/Qwen2.5-VL-7B-Instruct-GGUF)

use crate::{quantized_nn::RmsNorm, utils::repeat_kv};
use candle::{
    quantized::{gguf_file, QMatMul},
    DType, Device, IndexOp, Result, Tensor, D,
};
use candle_nn::{Embedding, Module};
use std::collections::HashMap;

// ============================================================================
// MLP
// ============================================================================

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_w1: QMatMul, // gate_proj
    feed_forward_w2: QMatMul, // down_proj
    feed_forward_w3: QMatMul, // up_proj
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

// ============================================================================
// Layer Weights
// ============================================================================

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_bq: Tensor,
    attention_bk: Tensor,
    attention_bv: Tensor,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    mlp: Mlp,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    mrope_section: Vec<usize>,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
    span_attn: tracing::Span,
    span_rot: tracing::Span,
    span_mlp: tracing::Span,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

impl LayerWeights {
    /// Apply Multimodal RoPE (M-RoPE) to queries and keys.
    ///
    /// M-RoPE uses 3D position IDs [temporal, height, width] and applies
    /// different position embeddings to different sections of the head dimension.
    fn apply_mrope(&self, q: &Tensor, k: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let _enter = self.span_rot.enter();
        let (three, _batch, _seq_len) = position_ids.dims3()?;
        assert_eq!(three, 3, "position_ids must have 3 dimensions [t, h, w]");

        // Compute 3D rope embeddings for each position dimension
        let (cos_3d, sin_3d) = self.compute_3d_rope_embeddings(position_ids)?;

        // Apply mrope_section to select bands from each dimension
        let (cos, sin) = self.apply_mrope_sections(&cos_3d, &sin_3d)?;

        // Reshape for broadcasting: [batch, 1, seq_len, head_dim]
        let cos = cos.unsqueeze(1)?;
        let sin = sin.unsqueeze(1)?;

        // Apply RoPE to q and k
        let q_embed = self.apply_rope_to_tensor(q, &cos, &sin)?;
        let k_embed = self.apply_rope_to_tensor(k, &cos, &sin)?;

        Ok((q_embed, k_embed))
    }

    /// Compute cos/sin embeddings for 3D position IDs.
    fn compute_3d_rope_embeddings(&self, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (three, batch, seq_len) = position_ids.dims3()?;
        let half_dim = self.head_dim / 2;

        let mut cos_parts = Vec::new();
        let mut sin_parts = Vec::new();

        for dim_idx in 0..three {
            let pos = position_ids.i(dim_idx)?;
            let pos_flat = pos.flatten_all()?;

            // Gather from precomputed cos/sin
            let cos_gathered = self.cos.index_select(&pos_flat, 0)?;
            let sin_gathered = self.sin.index_select(&pos_flat, 0)?;

            // Reshape to [batch, seq_len, half_dim]
            let cos_dim = cos_gathered.reshape((batch, seq_len, half_dim))?;
            let sin_dim = sin_gathered.reshape((batch, seq_len, half_dim))?;

            // Duplicate to full head_dim: [batch, seq_len, head_dim]
            let cos_full = Tensor::cat(&[&cos_dim, &cos_dim], D::Minus1)?;
            let sin_full = Tensor::cat(&[&sin_dim, &sin_dim], D::Minus1)?;

            cos_parts.push(cos_full);
            sin_parts.push(sin_full);
        }

        // Stack to [3, batch, seq_len, head_dim]
        let cos_3d = Tensor::stack(&cos_parts, 0)?;
        let sin_3d = Tensor::stack(&sin_parts, 0)?;

        Ok((cos_3d, sin_3d))
    }

    /// Apply mrope_section to select bands from each dimension.
    ///
    /// CRITICAL: In Python, `mrope_section * 2` is **list repetition**!
    /// `[16, 24, 24] * 2 = [16, 24, 24, 16, 24, 24]` (6 chunks totaling 128)
    fn apply_mrope_sections(&self, cos_3d: &Tensor, sin_3d: &Tensor) -> Result<(Tensor, Tensor)> {
        // List repetition: [16, 24, 24] * 2 = [16, 24, 24, 16, 24, 24]
        let mut sections_repeated: Vec<usize> = Vec::new();
        sections_repeated.extend_from_slice(&self.mrope_section);
        sections_repeated.extend_from_slice(&self.mrope_section);

        let mut cos_parts = Vec::new();
        let mut sin_parts = Vec::new();
        let mut offset = 0;

        for (i, &sec_size) in sections_repeated.iter().enumerate() {
            let dim_idx = i % 3; // Cycles: temporal(0), height(1), width(2)
            let cos_slice = cos_3d.i(dim_idx)?.narrow(D::Minus1, offset, sec_size)?;
            let sin_slice = sin_3d.i(dim_idx)?.narrow(D::Minus1, offset, sec_size)?;
            cos_parts.push(cos_slice);
            sin_parts.push(sin_slice);
            offset += sec_size;
        }

        // Concatenate along head_dim: [batch, seq_len, head_dim]
        let cos = Tensor::cat(&cos_parts, D::Minus1)?;
        let sin = Tensor::cat(&sin_parts, D::Minus1)?;

        Ok((cos, sin))
    }

    /// Apply rotary embedding to a tensor using rotate_half.
    fn apply_rope_to_tensor(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let x = x.contiguous()?;
        let head_dim = x.dim(D::Minus1)?;
        let half_dim = head_dim / 2;

        let x1 = x.narrow(D::Minus1, 0, half_dim)?;
        let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;

        // rotate_half: [-x2, x1]
        let x_rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;

        // x * cos + rotate_half(x) * sin
        x.broadcast_mul(cos)? + x_rotated.broadcast_mul(sin)?
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, n_embd) = x.dims3()?;

        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        // Add biases (Qwen2.5-VL has biases on Q, K, V)
        let q = q.broadcast_add(&self.attention_bq)?;
        let k = k.broadcast_add(&self.attention_bk)?;
        let v = v.broadcast_add(&self.attention_bv)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply M-RoPE
        let (q, k) = self.apply_mrope(&q, &k, position_ids)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                let k = Tensor::cat(&[k_cache, &k], 2)?;
                let v = Tensor::cat(&[v_cache, &v], 2)?;
                (k, v)
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // GQA: repeat K/V heads
        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = match mask {
            None => att,
            Some(mask) => {
                let mask = mask.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, &self.neg_inf)?
            }
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v.contiguous()?)?;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let y = self.attention_wo.forward(&y)?;
        Ok(y)
    }
}

// ============================================================================
// Model Weights
// ============================================================================

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    masks: HashMap<usize, Tensor>,
    span: tracing::Span,
    span_output: tracing::Span,
}

fn precompute_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, context_length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

impl ModelWeights {
    /// Load quantized text decoder from GGUF file.
    ///
    /// # Arguments
    /// * `ct` - GGUF file content with metadata
    /// * `reader` - Reader for tensor data
    /// * `device` - Device to load tensors on
    ///
    /// # Returns
    /// Model weights ready for inference
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        // Try multiple prefixes for GGUF compatibility:
        // - qwen2_5_vl: explicit version naming
        // - qwen2vl: llama.cpp style naming (used by Mungert GGUFs)
        // - qwen2: generic fallback
        let get_with_fallback = |key: &str| -> Result<&gguf_file::Value> {
            let prefixes = ["qwen2_5_vl", "qwen2vl", "qwen2"];
            for prefix in prefixes {
                let full_key = format!("{prefix}.{key}");
                if let Some(v) = ct.metadata.get(&full_key) {
                    return Ok(v);
                }
            }
            candle::bail!("cannot find {key} in metadata (tried prefixes: {:?})", prefixes)
        };

        let head_count = get_with_fallback("attention.head_count")?.to_u32()? as usize;
        let head_count_kv = get_with_fallback("attention.head_count_kv")?.to_u32()? as usize;
        let embedding_length = get_with_fallback("embedding_length")?.to_u32()? as usize;
        let context_length = get_with_fallback("context_length")?.to_u32()? as usize;
        let block_count = get_with_fallback("block_count")?.to_u32()? as usize;
        let rms_norm_eps = get_with_fallback("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = get_with_fallback("rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(1_000_000f32); // Qwen2.5-VL uses higher theta

        let head_dim = embedding_length / head_count;

        // M-RoPE section sizes (default for Qwen2.5-VL: [16, 24, 24])
        let mrope_section = vec![16usize, 24, 24];

        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(v) => QMatMul::from_qtensor(v)?,
            _ => {
                // use tie_word_embeddings
                QMatMul::from_qtensor(ct.tensor(reader, "token_embd.weight", device)?)?
            }
        };

        let (cos, sin) = precompute_freqs_cis(head_dim, rope_freq_base, context_length, device)?;

        let mut layers = Vec::with_capacity(block_count);

        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");

            // Attention weights (quantized)
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo = ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;

            // Attention biases (dequantized) - Qwen2.5-VL has biases on Q, K, V
            let attention_bq = ct.tensor(reader, &format!("{prefix}.attn_q.bias"), device)?;
            let attention_bk = ct.tensor(reader, &format!("{prefix}.attn_k.bias"), device)?;
            let attention_bv = ct.tensor(reader, &format!("{prefix}.attn_v.bias"), device)?;

            // MLP weights (quantized, no bias)
            let mlp = {
                let feed_forward_w1 =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
                let feed_forward_w2 =
                    ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
                let feed_forward_w3 =
                    ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                }
            };

            // Layer norms (dequantized)
            let attention_norm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;

            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");

            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_bq: attention_bq.dequantize(device)?,
                attention_bk: attention_bk.dequantize(device)?,
                attention_bv: attention_bv.dequantize(device)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                cos: cos.clone(),
                sin: sin.clone(),
                mrope_section: mrope_section.clone(),
                mlp,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                neg_inf: neg_inf.clone(),
                kv_cache: None,
                span_attn,
                span_rot,
                span_mlp,
            });
        }

        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output,
            masks: HashMap::new(),
            span,
            span_output,
        })
    }

    fn mask(&mut self, t: usize, device: &Device) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), device)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }

    /// Forward pass with M-RoPE position IDs.
    ///
    /// # Arguments
    /// * `x` - Input token IDs of shape (batch, seq_len)
    /// * `position_ids` - 3D M-RoPE position IDs of shape (3, batch, seq_len)
    ///
    /// # Returns
    /// Logits for the last token, shape (batch, vocab_size)
    pub fn forward(&mut self, x: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, x.device())?)
        };
        let _enter = self.span.enter();
        let mut layer_in = self.tok_embeddings.forward(x)?;
        for layer in self.layers.iter_mut() {
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(&x, mask.as_ref(), position_ids)?;
            let x = (attn + residual)?;

            // MLP
            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            let x = (x + residual)?;
            layer_in = x
        }
        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        let _enter = self.span_output.enter();
        self.output.forward(&x)
    }

    /// Clear all KV caches for fresh generation.
    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.kv_cache = None;
        }
    }

    /// Forward pass for text-only input, returning hidden states (not logits).
    ///
    /// This is used by diffusion models like Qwen-Image for text encoding,
    /// where we need the raw hidden states from the transformer.
    ///
    /// Unlike `forward()`, this method:
    /// - Uses simple sequential position IDs (same for all 3 M-RoPE dimensions)
    /// - Returns all hidden states, not just the last token
    /// - Skips the lm_head projection
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs of shape (batch, seq_len)
    /// * `attention_mask` - Optional attention mask of shape (batch, seq_len)
    ///
    /// # Returns
    /// Hidden states tensor of shape (batch, seq_len, hidden_size)
    pub fn forward_text_only(
        &mut self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Clear KV cache to ensure fresh forward pass
        self.clear_kv_cache();

        let (b_sz, seq_len) = input_ids.dims2()?;
        let device = input_ids.device();

        // Get token embeddings
        let mut hidden_states = self.tok_embeddings.forward(input_ids)?;

        // Create simple sequential position IDs: [3, batch, seq_len]
        // For text-only, all three M-RoPE dimensions use the same sequential positions
        let positions: Vec<i64> = (0..seq_len as i64).collect();
        let pos_tensor = Tensor::from_vec(positions, seq_len, device)?;
        let pos_tensor = pos_tensor.unsqueeze(0)?.expand((b_sz, seq_len))?;
        let position_ids = Tensor::stack(&[&pos_tensor, &pos_tensor, &pos_tensor], 0)?;

        // Create causal attention mask if sequence length > 1
        let mask = if seq_len <= 1 {
            None
        } else {
            let base_mask = self.mask(seq_len, device)?;
            // Combine with provided attention mask if any
            if let Some(attn_mask) = attention_mask {
                // Expand mask to match causal mask shape
                let attn_mask = attn_mask.unsqueeze(1)?.unsqueeze(1)?;
                // Convert 0/1 mask to 0/1 (inverted for additive mask)
                let attn_mask = attn_mask.to_dtype(DType::F32)?;
                let attn_mask = (1.0 - attn_mask)?;
                // Broadcast and combine
                let attn_mask = attn_mask.broadcast_as((b_sz, 1, seq_len, seq_len))?;
                let base_f32 = base_mask.to_dtype(DType::F32)?;
                // Combined mask: 1 where either mask blocks
                Some((base_f32.broadcast_as((b_sz, 1, seq_len, seq_len))? + attn_mask)?.clamp(0.0, 1.0)?.to_dtype(DType::U8)?)
            } else {
                Some(base_mask)
            }
        };

        // Forward through all transformer layers
        for layer in self.layers.iter_mut() {
            let residual = &hidden_states;
            let x = layer.attention_norm.forward(&hidden_states)?;
            let attn = layer.forward_attn(&x, mask.as_ref(), &position_ids)?;
            let x = (attn + residual)?;

            // MLP
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            hidden_states = (x + residual)?;
        }

        // Apply final norm (but NOT lm_head - we want hidden states)
        self.norm.forward(&hidden_states)
    }

    /// Forward pass with vision embeddings, returning hidden states (not logits).
    ///
    /// This is used by multimodal diffusion models like Qwen-Image for
    /// image-conditional text encoding.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs of shape (batch, seq_len)
    /// * `vision_embeds` - Vision embeddings to inject at image token positions
    /// * `image_grid_thw` - Grid dimensions for each image (t, h, w before merge)
    /// * `attention_mask` - Optional attention mask
    /// * `spatial_merge_size` - Spatial merge factor from vision config (typically 2)
    /// * `image_token_id` - Token ID for image placeholders (typically 151655)
    ///
    /// # Returns
    /// Hidden states tensor of shape (batch, seq_len, hidden_size)
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_vision(
        &mut self,
        input_ids: &Tensor,
        vision_embeds: &Tensor,
        image_grid_thw: &Tensor,
        attention_mask: Option<&Tensor>,
        spatial_merge_size: usize,
        image_token_id: u32,
    ) -> Result<Tensor> {
        // Clear KV cache to ensure fresh forward pass
        self.clear_kv_cache();

        let (batch_size, seq_len) = input_ids.dims2()?;
        let device = input_ids.device();
        let hidden_dim = self.tok_embeddings.embeddings().dim(1)?;

        // 1. Get base token embeddings
        let mut input_embeds = self.tok_embeddings.forward(input_ids)?;

        // 2. Replace image placeholder tokens with vision embeddings
        let input_ids_flat: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
        let vision_embeds = vision_embeds.to_dtype(input_embeds.dtype())?;
        let mut vision_offset = 0usize;

        for batch_idx in 0..batch_size {
            let batch_start = batch_idx * seq_len;
            let mut token_idx = 0usize;

            while token_idx < seq_len {
                if input_ids_flat[batch_start + token_idx] == image_token_id {
                    // Find contiguous image tokens
                    let start = token_idx;
                    while token_idx < seq_len
                        && input_ids_flat[batch_start + token_idx] == image_token_id
                    {
                        token_idx += 1;
                    }
                    let len = token_idx - start;

                    // Replace with vision embeddings
                    let vision_chunk = vision_embeds.narrow(0, vision_offset, len)?;
                    input_embeds = input_embeds.slice_assign(
                        &[batch_idx..batch_idx + 1, start..start + len, 0..hidden_dim],
                        &vision_chunk.unsqueeze(0)?,
                    )?;
                    vision_offset += len;
                } else {
                    token_idx += 1;
                }
            }
        }

        // 3. Compute M-RoPE position IDs from image grid dimensions
        let grid_thw_vec = image_grid_thw.to_vec2::<u32>()?;
        let image_grids: Vec<ImageGrid> = grid_thw_vec
            .iter()
            .map(|g| {
                let h = g[1] as usize;
                let w = g[2] as usize;
                ImageGrid {
                    grid_h: h / spatial_merge_size,
                    grid_w: w / spatial_merge_size,
                }
            })
            .collect();

        let position_ids =
            compute_mrope_position_ids_multi(input_ids, image_token_id, &image_grids, device)?;

        // 4. Create causal attention mask
        let mask = if seq_len <= 1 {
            None
        } else {
            let base_mask = self.mask(seq_len, device)?;
            if let Some(attn_mask) = attention_mask {
                let attn_mask = attn_mask.unsqueeze(1)?.unsqueeze(1)?;
                let attn_mask = attn_mask.to_dtype(DType::F32)?;
                let attn_mask = (1.0 - attn_mask)?;
                let attn_mask = attn_mask.broadcast_as((batch_size, 1, seq_len, seq_len))?;
                let base_f32 = base_mask.to_dtype(DType::F32)?;
                Some((base_f32.broadcast_as((batch_size, 1, seq_len, seq_len))? + attn_mask)?.clamp(0.0, 1.0)?.to_dtype(DType::U8)?)
            } else {
                Some(base_mask)
            }
        };

        // 5. Forward through all transformer layers
        let mut hidden_states = input_embeds;
        for layer in self.layers.iter_mut() {
            let residual = &hidden_states;
            let x = layer.attention_norm.forward(&hidden_states)?;
            let attn = layer.forward_attn(&x, mask.as_ref(), &position_ids)?;
            let x = (attn + residual)?;

            // MLP
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            hidden_states = (x + residual)?;
        }

        // 6. Apply final norm (but NOT lm_head - we want hidden states)
        self.norm.forward(&hidden_states)
    }
}

// ============================================================================
// M-RoPE Position ID Computation
// ============================================================================

/// Image grid specification for M-RoPE position computation.
#[derive(Debug, Clone)]
pub struct ImageGrid {
    /// Grid height (patches after spatial merge)
    pub grid_h: usize,
    /// Grid width (patches after spatial merge)
    pub grid_w: usize,
}

/// Compute 3D M-RoPE position IDs for single image input.
///
/// # Arguments
/// * `input_ids` - Token IDs of shape (batch, seq_len)
/// * `image_token_id` - The token ID for image placeholders
/// * `grid_h` - Image grid height after spatial merge
/// * `grid_w` - Image grid width after spatial merge
/// * `device` - Device to create tensors on
///
/// # Returns
/// Position IDs tensor of shape [3, batch, seq_len]
pub fn compute_mrope_position_ids(
    input_ids: &Tensor,
    image_token_id: u32,
    grid_h: usize,
    grid_w: usize,
    device: &Device,
) -> Result<Tensor> {
    let grids = vec![ImageGrid { grid_h, grid_w }];
    compute_mrope_position_ids_multi(input_ids, image_token_id, &grids, device)
}

/// Compute 3D M-RoPE position IDs for multi-image input.
///
/// Position encoding:
/// - Text tokens: all 3 dims same (t=h=w=sequential_pos)
/// - Image tokens: 2D grid positions offset by preceding text
///   - t = offset (temporal is 0 for images)
///   - h = row_in_grid + offset
///   - w = col_in_grid + offset
pub fn compute_mrope_position_ids_multi(
    input_ids: &Tensor,
    image_token_id: u32,
    image_grids: &[ImageGrid],
    device: &Device,
) -> Result<Tensor> {
    let (batch, seq_len) = input_ids.dims2()?;
    let input_ids_vec: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;

    let mut pos_t = vec![0i64; batch * seq_len];
    let mut pos_h = vec![0i64; batch * seq_len];
    let mut pos_w = vec![0i64; batch * seq_len];

    for b in 0..batch {
        let batch_start = b * seq_len;

        // Find all image token ranges
        let mut image_ranges: Vec<(usize, usize)> = Vec::new();
        let mut in_image = false;
        let mut image_start = 0usize;

        for s in 0..seq_len {
            let token_id = input_ids_vec[batch_start + s];
            if token_id == image_token_id {
                if !in_image {
                    in_image = true;
                    image_start = s;
                }
            } else if in_image {
                image_ranges.push((image_start, s));
                in_image = false;
            }
        }
        if in_image {
            image_ranges.push((image_start, seq_len));
        }

        // Verify image count matches
        if image_ranges.len() != image_grids.len() {
            return Err(candle::Error::Msg(format!(
                "Mismatch: found {} image ranges but {} grids provided",
                image_ranges.len(),
                image_grids.len()
            )));
        }

        // Compute positions
        let mut current_pos = 0i64;
        let mut range_idx = 0usize;

        for s in 0..seq_len {
            let idx = batch_start + s;

            // At start of image range?
            if range_idx < image_ranges.len() && s == image_ranges[range_idx].0 {
                let (img_start, img_end) = image_ranges[range_idx];
                let grid = &image_grids[range_idx];
                let num_vision_tokens = grid.grid_h * grid.grid_w;

                let actual_tokens = img_end - img_start;
                if actual_tokens != num_vision_tokens {
                    return Err(candle::Error::Msg(format!(
                        "Image {} has {} tokens but grid {}x{} = {} expected",
                        range_idx, actual_tokens, grid.grid_h, grid.grid_w, num_vision_tokens
                    )));
                }

                // Assign spatial positions
                let offset = current_pos;
                for vision_idx in 0..num_vision_tokens {
                    let token_idx = batch_start + img_start + vision_idx;
                    let t_pos = 0i64; // Temporal is 0 for images
                    let h_pos = (vision_idx / grid.grid_w) as i64;
                    let w_pos = (vision_idx % grid.grid_w) as i64;

                    pos_t[token_idx] = t_pos + offset;
                    pos_h[token_idx] = h_pos + offset;
                    pos_w[token_idx] = w_pos + offset;
                }

                // Update position to max + 1
                let max_h = (grid.grid_h - 1) as i64;
                let max_w = (grid.grid_w - 1) as i64;
                current_pos = offset + max_h.max(max_w) + 1;

                range_idx += 1;
                continue;
            }

            // Skip if inside image range
            if range_idx > 0 {
                let prev_range = image_ranges[range_idx - 1];
                if s >= prev_range.0 && s < prev_range.1 {
                    continue;
                }
            }
            if range_idx < image_ranges.len() {
                let curr_range = image_ranges[range_idx];
                if s >= curr_range.0 && s < curr_range.1 {
                    continue;
                }
            }

            // Text token: all dimensions same
            pos_t[idx] = current_pos;
            pos_h[idx] = current_pos;
            pos_w[idx] = current_pos;
            current_pos += 1;
        }
    }

    let pos_t = Tensor::from_vec(pos_t, (batch, seq_len), device)?;
    let pos_h = Tensor::from_vec(pos_h, (batch, seq_len), device)?;
    let pos_w = Tensor::from_vec(pos_w, (batch, seq_len), device)?;

    Tensor::stack(&[pos_t, pos_h, pos_w], 0)
}

// ============================================================================
// mmproj GGUF Loading for Vision Encoder
// ============================================================================

use crate::models::qwen2_5_vl::{Qwen25VLVisionModel, VisionConfig};
use candle_nn::{Activation, VarBuilder};
use std::io::{Read, Seek};

/// Load vision encoder from mmproj GGUF file (FP16).
///
/// The mmproj GGUF files from llama.cpp contain the vision encoder weights
/// with tensor names like `v.patch_embd.weight`, `v.blk.0.attn_qkv.weight`, etc.
/// This function maps them to the expected safetensors names.
///
/// # Arguments
/// * `ct` - GGUF file content with metadata
/// * `reader` - Reader for tensor data
/// * `device` - Device to load tensors on
/// * `dtype` - Target dtype (typically F16 or BF16)
///
/// # Returns
/// Vision encoder ready for inference
pub fn load_vision_from_mmproj<R: Seek + Read>(
    ct: &gguf_file::Content,
    reader: &mut R,
    device: &Device,
    dtype: DType,
) -> Result<(Qwen25VLVisionModel, VisionConfig)> {
    // Helper to try multiple metadata key variants and return first match
    let get_u32 = |keys: &[&str]| -> Result<u32> {
        for key in keys {
            if let Some(v) = ct.metadata.get(*key) {
                return v.to_u32();
            }
        }
        candle::bail!("cannot find any of {:?} in metadata", keys);
    };

    // Vision config from metadata
    let hidden_size = get_u32(&[
        "clip.vision.embedding_length",
        "vision.embedding_length",
        "qwen2vl.vision.hidden_size",
    ])? as usize;

    let num_heads = get_u32(&[
        "clip.vision.attention.head_count",
        "vision.attention.head_count",
        "qwen2vl.vision.num_heads",
    ])? as usize;

    let depth = get_u32(&[
        "clip.vision.block_count",
        "vision.block_count",
        "qwen2vl.vision.depth",
    ])? as usize;

    let intermediate_size = get_u32(&[
        "clip.vision.feed_forward_length",
        "vision.feed_forward_length",
        "qwen2vl.vision.intermediate_size",
    ])
    .unwrap_or((hidden_size * 27 / 10) as u32) as usize; // Default ~2.7x hidden

    let patch_size = get_u32(&["clip.vision.patch_size", "vision.patch_size"])
        .unwrap_or(14) as usize;

    let spatial_merge_size = get_u32(&[
        "qwen2vl.vision.spatial_merge_size",
        "vision.spatial_merge_size",
    ])
    .unwrap_or(2) as usize;

    let temporal_patch_size = get_u32(&[
        "qwen2vl.vision.temporal_patch_size",
        "vision.temporal_patch_size",
    ])
    .unwrap_or(2) as usize;

    // Output hidden size (text model hidden size)
    let out_hidden_size = get_u32(&[
        "qwen2vl.vision.out_hidden_size",
        "vision.out_hidden_size",
        "clip.vision.projection_dim",
    ])
    .unwrap_or(3584) as usize; // Default for 7B

    let window_size = get_u32(&["qwen2vl.vision.window_size", "vision.window_size"])
        .unwrap_or(112) as usize;

    // Full attention block indexes (default for Qwen2.5-VL: [7, 15, 23, 31])
    let fullatt_block_indexes = vec![7, 15, 23, 31];

    let vision_config = VisionConfig {
        hidden_size,
        num_heads,
        depth,
        intermediate_size,
        in_chans: 3,
        patch_size,
        spatial_merge_size,
        temporal_patch_size,
        out_hidden_size,
        window_size,
        fullatt_block_indexes,
        tokens_per_second: 4, // Default for Qwen2.5-VL
        hidden_act: Activation::Silu,
    };

    // Load tensors from GGUF and map to expected names
    let mut tensors: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();

    // Helper function to try loading tensor with multiple name variants
    fn try_load_tensor<R: Seek + Read>(
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
        dtype: DType,
        names: &[&str],
    ) -> Option<Tensor> {
        for name in names {
            if ct.tensor_infos.contains_key(*name) {
                if let Ok(qt) = ct.tensor(reader, name, device) {
                    if let Ok(t) = qt.dequantize(device) {
                        if let Ok(t) = t.to_dtype(dtype) {
                            return Some(t);
                        }
                    }
                }
            }
        }
        None
    }

    // Patch embedding
    if let Some(t) = try_load_tensor(ct, reader, device, dtype, &["v.patch_embd.weight", "vision.patch_embed.proj.weight"]) {
        tensors.insert("patch_embed.proj.weight".to_string(), t);
    }

    // Vision blocks
    for i in 0..depth {
        // Attention norm (norm1)
        let norm1_names: Vec<String> = vec![
            format!("v.blk.{i}.attn_norm.weight"),
            format!("v.blk.{i}.ln1.weight"),
            format!("vision.blocks.{i}.norm1.weight"),
        ];
        let norm1_refs: Vec<&str> = norm1_names.iter().map(|s| s.as_str()).collect();
        if let Some(t) = try_load_tensor(ct, reader, device, dtype, &norm1_refs) {
            tensors.insert(format!("blocks.{i}.norm1.weight"), t);
        }

        // FFN norm (norm2)
        let norm2_names: Vec<String> = vec![
            format!("v.blk.{i}.ffn_norm.weight"),
            format!("v.blk.{i}.ln2.weight"),
            format!("vision.blocks.{i}.norm2.weight"),
        ];
        let norm2_refs: Vec<&str> = norm2_names.iter().map(|s| s.as_str()).collect();
        if let Some(t) = try_load_tensor(ct, reader, device, dtype, &norm2_refs) {
            tensors.insert(format!("blocks.{i}.norm2.weight"), t);
        }

        // Attention QKV
        let qkv_w_names: Vec<String> = vec![
            format!("v.blk.{i}.attn_qkv.weight"),
            format!("vision.blocks.{i}.attn.qkv.weight"),
        ];
        let qkv_w_refs: Vec<&str> = qkv_w_names.iter().map(|s| s.as_str()).collect();
        if let Some(t) = try_load_tensor(ct, reader, device, dtype, &qkv_w_refs) {
            tensors.insert(format!("blocks.{i}.attn.qkv.weight"), t);
        }

        let qkv_b_names: Vec<String> = vec![
            format!("v.blk.{i}.attn_qkv.bias"),
            format!("vision.blocks.{i}.attn.qkv.bias"),
        ];
        let qkv_b_refs: Vec<&str> = qkv_b_names.iter().map(|s| s.as_str()).collect();
        if let Some(t) = try_load_tensor(ct, reader, device, dtype, &qkv_b_refs) {
            tensors.insert(format!("blocks.{i}.attn.qkv.bias"), t);
        }

        // Attention output projection
        let proj_w_names: Vec<String> = vec![
            format!("v.blk.{i}.attn_out.weight"),
            format!("v.blk.{i}.attn_output.weight"),
            format!("vision.blocks.{i}.attn.proj.weight"),
        ];
        let proj_w_refs: Vec<&str> = proj_w_names.iter().map(|s| s.as_str()).collect();
        if let Some(t) = try_load_tensor(ct, reader, device, dtype, &proj_w_refs) {
            tensors.insert(format!("blocks.{i}.attn.proj.weight"), t);
        }

        let proj_b_names: Vec<String> = vec![
            format!("v.blk.{i}.attn_out.bias"),
            format!("v.blk.{i}.attn_output.bias"),
            format!("vision.blocks.{i}.attn.proj.bias"),
        ];
        let proj_b_refs: Vec<&str> = proj_b_names.iter().map(|s| s.as_str()).collect();
        if let Some(t) = try_load_tensor(ct, reader, device, dtype, &proj_b_refs) {
            tensors.insert(format!("blocks.{i}.attn.proj.bias"), t);
        }

        // MLP - SwiGLU (gate, up, down)
        let gate_w_names: Vec<String> = vec![
            format!("v.blk.{i}.ffn_gate.weight"),
            format!("vision.blocks.{i}.mlp.gate_proj.weight"),
        ];
        let gate_w_refs: Vec<&str> = gate_w_names.iter().map(|s| s.as_str()).collect();
        if let Some(t) = try_load_tensor(ct, reader, device, dtype, &gate_w_refs) {
            tensors.insert(format!("blocks.{i}.mlp.gate_proj.weight"), t);
        }

        let gate_b_names: Vec<String> = vec![
            format!("v.blk.{i}.ffn_gate.bias"),
            format!("vision.blocks.{i}.mlp.gate_proj.bias"),
        ];
        let gate_b_refs: Vec<&str> = gate_b_names.iter().map(|s| s.as_str()).collect();
        if let Some(t) = try_load_tensor(ct, reader, device, dtype, &gate_b_refs) {
            tensors.insert(format!("blocks.{i}.mlp.gate_proj.bias"), t);
        }

        let up_w_names: Vec<String> = vec![
            format!("v.blk.{i}.ffn_up.weight"),
            format!("vision.blocks.{i}.mlp.up_proj.weight"),
        ];
        let up_w_refs: Vec<&str> = up_w_names.iter().map(|s| s.as_str()).collect();
        if let Some(t) = try_load_tensor(ct, reader, device, dtype, &up_w_refs) {
            tensors.insert(format!("blocks.{i}.mlp.up_proj.weight"), t);
        }

        let up_b_names: Vec<String> = vec![
            format!("v.blk.{i}.ffn_up.bias"),
            format!("vision.blocks.{i}.mlp.up_proj.bias"),
        ];
        let up_b_refs: Vec<&str> = up_b_names.iter().map(|s| s.as_str()).collect();
        if let Some(t) = try_load_tensor(ct, reader, device, dtype, &up_b_refs) {
            tensors.insert(format!("blocks.{i}.mlp.up_proj.bias"), t);
        }

        let down_w_names: Vec<String> = vec![
            format!("v.blk.{i}.ffn_down.weight"),
            format!("vision.blocks.{i}.mlp.down_proj.weight"),
        ];
        let down_w_refs: Vec<&str> = down_w_names.iter().map(|s| s.as_str()).collect();
        if let Some(t) = try_load_tensor(ct, reader, device, dtype, &down_w_refs) {
            tensors.insert(format!("blocks.{i}.mlp.down_proj.weight"), t);
        }

        let down_b_names: Vec<String> = vec![
            format!("v.blk.{i}.ffn_down.bias"),
            format!("vision.blocks.{i}.mlp.down_proj.bias"),
        ];
        let down_b_refs: Vec<&str> = down_b_names.iter().map(|s| s.as_str()).collect();
        if let Some(t) = try_load_tensor(ct, reader, device, dtype, &down_b_refs) {
            tensors.insert(format!("blocks.{i}.mlp.down_proj.bias"), t);
        }
    }

    // Patch merger
    if let Some(t) = try_load_tensor(
        ct, reader, device, dtype,
        &["v.post_ln.weight", "v.mm.ln_q.weight", "vision.merger.ln_q.weight"],
    ) {
        tensors.insert("merger.ln_q.weight".to_string(), t);
    }

    // Merger MLP (2-layer with GELU)
    if let Some(t) = try_load_tensor(
        ct, reader, device, dtype,
        &["v.mm.0.weight", "v.mm_proj.0.weight", "vision.merger.mlp.0.weight"],
    ) {
        tensors.insert("merger.mlp.0.weight".to_string(), t);
    }
    if let Some(t) = try_load_tensor(
        ct, reader, device, dtype,
        &["v.mm.0.bias", "v.mm_proj.0.bias", "vision.merger.mlp.0.bias"],
    ) {
        tensors.insert("merger.mlp.0.bias".to_string(), t);
    }

    if let Some(t) = try_load_tensor(
        ct, reader, device, dtype,
        &["v.mm.2.weight", "v.mm_proj.2.weight", "vision.merger.mlp.2.weight"],
    ) {
        tensors.insert("merger.mlp.2.weight".to_string(), t);
    }
    if let Some(t) = try_load_tensor(
        ct, reader, device, dtype,
        &["v.mm.2.bias", "v.mm_proj.2.bias", "vision.merger.mlp.2.bias"],
    ) {
        tensors.insert("merger.mlp.2.bias".to_string(), t);
    }

    // Debug: list loaded tensors
    tracing::debug!(
        "Loaded {} vision tensors from mmproj GGUF",
        tensors.len()
    );

    // Create VarBuilder from the tensor map
    let vb = VarBuilder::from_tensors(tensors, dtype, device);

    // Construct the vision model
    let vision_model = Qwen25VLVisionModel::new(&vision_config, vb)?;

    Ok((vision_model, vision_config))
}

/// Print all tensor names in the GGUF file (for debugging tensor name mapping).
pub fn debug_print_gguf_tensors(ct: &gguf_file::Content) {
    println!("GGUF tensors ({} total):", ct.tensor_infos.len());
    let mut names: Vec<_> = ct.tensor_infos.keys().collect();
    names.sort();
    for name in names {
        println!("  {}", name);
    }
}

// ============================================================================
// Combined Quantized Model
// ============================================================================

/// Qwen2.5-VL model with quantized text decoder and FP16 vision encoder.
///
/// This combines:
/// - Quantized text decoder loaded from GGUF
/// - FP16 vision encoder loaded from mmproj GGUF (via safetensors VarBuilder)
pub struct Qwen25VLQuantized {
    /// Quantized text decoder
    pub text: ModelWeights,
    /// FP16 vision encoder
    pub vision: Qwen25VLVisionModel,
    /// Image token ID for placeholder detection
    pub image_token_id: u32,
    /// Video token ID for video placeholder detection
    pub video_token_id: u32,
    /// Spatial merge size (typically 2)
    pub spatial_merge_size: usize,
    /// DType for vision embeddings
    pub dtype: DType,
}

impl Qwen25VLQuantized {
    /// Create a new combined quantized model.
    ///
    /// # Arguments
    /// * `text` - Quantized text decoder loaded from GGUF
    /// * `vision` - FP16 vision encoder
    /// * `image_token_id` - Token ID for image placeholders (typically 151655)
    /// * `video_token_id` - Token ID for video placeholders (typically 151656)
    /// * `spatial_merge_size` - Spatial merge size (typically 2)
    /// * `dtype` - DType for computations (typically BF16 or F16)
    pub fn new(
        text: ModelWeights,
        vision: Qwen25VLVisionModel,
        image_token_id: u32,
        video_token_id: u32,
        spatial_merge_size: usize,
        dtype: DType,
    ) -> Self {
        Self {
            text,
            vision,
            image_token_id,
            video_token_id,
            spatial_merge_size,
            dtype,
        }
    }

    /// Forward pass for image understanding.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs with image placeholders, shape (batch, seq_len)
    /// * `pixel_values` - Preprocessed image pixels, shape (num_patches, channels * temporal * patch * patch)
    /// * `image_grid_thw` - Grid dimensions for each image, shape (num_images, 3)
    ///
    /// # Returns
    /// Logits for the last token, shape (batch, vocab_size)
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        image_grid_thw: &Tensor,
    ) -> Result<Tensor> {
        let device = input_ids.device();

        // 1. Process images through vision encoder
        let vision_embeds = self.vision.forward(pixel_values, image_grid_thw)?;
        let vision_embeds = vision_embeds.to_dtype(self.dtype)?;

        // 2. Get text embeddings
        let (batch_size, seq_len) = input_ids.dims2()?;
        let hidden_dim = self.text.tok_embeddings.hidden_size();
        let mut input_embeds = self.text.tok_embeddings.forward(input_ids)?;

        // 3. Compute M-RoPE position IDs
        let grid_thw_vec = image_grid_thw.to_vec2::<u32>()?;
        let image_grids: Vec<ImageGrid> = grid_thw_vec
            .iter()
            .map(|g| {
                let h = g[1] as usize;
                let w = g[2] as usize;
                ImageGrid {
                    grid_h: h / self.spatial_merge_size,
                    grid_w: w / self.spatial_merge_size,
                }
            })
            .collect();

        let position_ids = compute_mrope_position_ids_multi(
            input_ids,
            self.image_token_id,
            &image_grids,
            device,
        )?;

        // 4. Find image placeholder positions and replace with vision embeddings
        let input_ids_flat: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
        let mut vision_offset = 0usize;

        for batch_idx in 0..batch_size {
            let batch_start = batch_idx * seq_len;
            let mut token_idx = 0usize;
            while token_idx < seq_len {
                if input_ids_flat[batch_start + token_idx] == self.image_token_id {
                    // Find contiguous image tokens
                    let start = token_idx;
                    while token_idx < seq_len
                        && input_ids_flat[batch_start + token_idx] == self.image_token_id
                    {
                        token_idx += 1;
                    }
                    let len = token_idx - start;

                    // Replace with vision embeddings
                    let vision_chunk = vision_embeds.narrow(0, vision_offset, len)?;
                    input_embeds = input_embeds.slice_assign(
                        &[batch_idx..batch_idx + 1, start..start + len, 0..hidden_dim],
                        &vision_chunk.unsqueeze(0)?,
                    )?;
                    vision_offset += len;
                } else {
                    token_idx += 1;
                }
            }
        }

        // 5. Forward through text model layers
        self.forward_with_embeds(input_embeds, &position_ids)
    }

    /// Forward pass with pre-computed embeddings and position IDs.
    fn forward_with_embeds(&mut self, xs: Tensor, position_ids: &Tensor) -> Result<Tensor> {
        let (_b_sz, seq_len, _hidden) = xs.dims3()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.text.mask(seq_len, xs.device())?)
        };

        let _enter = self.text.span.enter();
        let mut layer_in = xs;
        for layer in self.text.layers.iter_mut() {
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(&x, mask.as_ref(), position_ids)?;
            let x = (attn + residual)?;

            // MLP
            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            let x = (x + residual)?;
            layer_in = x
        }
        let x = self.text.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        let _enter = self.text.span_output.enter();
        self.text.output.forward(&x)
    }

    /// Generate tokens autoregressively using greedy decoding.
    ///
    /// # Arguments
    /// * `input_ids` - Initial token IDs with image placeholders
    /// * `pixel_values` - Preprocessed image pixels
    /// * `image_grid_thw` - Grid dimensions for each image
    /// * `max_length` - Maximum number of tokens to generate
    /// * `eos_token_id` - End of sequence token ID
    ///
    /// # Returns
    /// Vector of generated token IDs (excluding input tokens)
    pub fn generate(
        &mut self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        image_grid_thw: &Tensor,
        max_length: usize,
        eos_token_id: u32,
    ) -> Result<Vec<u32>> {
        self.text.clear_kv_cache();
        let device = input_ids.device();
        let input_len = input_ids.dim(1)?;

        // Compute image grids for M-RoPE delta
        let grid_thw_vec = image_grid_thw.to_vec2::<u32>()?;
        let image_grids: Vec<ImageGrid> = grid_thw_vec
            .iter()
            .map(|g| {
                let h = g[1] as usize;
                let w = g[2] as usize;
                ImageGrid {
                    grid_h: h / self.spatial_merge_size,
                    grid_w: w / self.spatial_merge_size,
                }
            })
            .collect();

        // Compute M-RoPE delta for generation
        // (max position seen minus sequence length)
        let input_ids_vec: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
        let mut current_pos = 0i64;
        let mut range_idx = 0usize;
        let seq_len = input_len;

        // Find image ranges
        let mut image_ranges: Vec<(usize, usize)> = Vec::new();
        let mut in_image = false;
        let mut image_start = 0usize;
        for (s, &token_id) in input_ids_vec.iter().enumerate() {
            if token_id == self.image_token_id {
                if !in_image {
                    in_image = true;
                    image_start = s;
                }
            } else if in_image {
                image_ranges.push((image_start, s));
                in_image = false;
            }
        }
        if in_image {
            image_ranges.push((image_start, seq_len));
        }

        // Compute max position
        for s in 0..seq_len {
            if range_idx < image_ranges.len() && s == image_ranges[range_idx].0 {
                let grid = &image_grids[range_idx];
                let offset = current_pos;
                let max_h = (grid.grid_h - 1) as i64;
                let max_w = (grid.grid_w - 1) as i64;
                current_pos = offset + max_h.max(max_w) + 1;
                range_idx += 1;
            } else if range_idx > 0 {
                let prev = image_ranges[range_idx - 1];
                if s < prev.1 {
                    continue;
                }
                current_pos += 1;
            } else {
                current_pos += 1;
            }
        }
        let mrope_delta = current_pos - seq_len as i64;

        // Prefill
        let logits = self.forward(input_ids, pixel_values, image_grid_thw)?.squeeze(0)?;
        let mut generated: Vec<u32> = Vec::new();
        let mut next_token = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
        generated.push(next_token);

        // Decode loop
        for _ in 1..max_length {
            if next_token == eos_token_id {
                break;
            }

            let next_input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
            let seq_len = input_len + generated.len();
            let pos = (seq_len as i64 - 1) + mrope_delta;
            let position_ids = Tensor::new(&[[[pos]], [[pos]], [[pos]]], device)?;

            let next_embeds = self.text.tok_embeddings.forward(&next_input)?;
            let logits = self.forward_with_embeds(next_embeds, &position_ids)?.squeeze(0)?;

            next_token = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
            generated.push(next_token);
        }

        Ok(generated)
    }

    /// Clear all KV caches for fresh generation.
    pub fn clear_kv_cache(&mut self) {
        self.text.clear_kv_cache();
    }
}
