//! Transformer blocks for Qwen-Image.
//!
//! This module implements the dual-stream transformer blocks used in Qwen-Image:
//! - **AdaLN Modulation**: Timestep-conditioned shift, scale, and gate parameters
//! - **Dual-Stream Attention**: Joint attention where text and image streams are concatenated
//! - **Separate MLPs**: Independent feed-forward networks for each stream
//!
//! The key innovation is that text and image are processed together in attention
//! (allowing cross-modal information flow) but separately in MLPs (preserving
//! modality-specific representations).

use candle::{DType, Result, Tensor, D};
use candle_nn::{LayerNorm, Linear, Module, RmsNorm, VarBuilder};

use super::debug::{
    debug_tensor, BlockOverrides,
    is_attention_debug, debug_attention_internals,
    debug_per_head_stats, save_attention_tensors,
    save_qk_pipeline_tensors, is_qk_save_enabled,
};
use super::rope::apply_rotary_emb_qwen;

/// Create a parameter-free LayerNorm (equivalent to PyTorch's elementwise_affine=False).
///
/// This is used in Qwen-Image transformer blocks where the modulation provides
/// the scale and shift instead of learned LayerNorm parameters.
fn layer_norm_no_affine(size: usize, eps: f64, device: &candle::Device, dtype: DType) -> Result<LayerNorm> {
    let weight = Tensor::ones(size, dtype, device)?;
    Ok(LayerNorm::new_no_bias(weight, eps))
}

/// Output of modulation: shift, scale, and gate tensors.
#[derive(Debug, Clone)]
pub struct ModulationOut {
    pub shift: Tensor,
    pub scale: Tensor,
    pub gate: Tensor,
}

impl ModulationOut {
    /// Apply scale and shift to input: x * (1 + scale) + shift
    pub fn scale_shift(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&(&self.scale + 1.0)?)?
            .broadcast_add(&self.shift)
    }

    /// Apply gating: gate * x
    pub fn gate(&self, xs: &Tensor) -> Result<Tensor> {
        self.gate.broadcast_mul(xs)
    }

    /// Convert modulation parameters to F32 for mixed precision
    pub fn to_f32(&self) -> Result<Self> {
        Ok(Self {
            shift: self.shift.to_dtype(DType::F32)?,
            scale: self.scale.to_dtype(DType::F32)?,
            gate: self.gate.to_dtype(DType::F32)?,
        })
    }
}

/// Apply per-token modulation based on modulate_index.
///
/// When modulate_index is provided, we have doubled modulation params (from doubled timestep).
/// The index tensor indicates which modulation to use per token:
/// - Index 0: use modulation from actual timestep (for noise latents)
/// - Index 1: use modulation from zero timestep (for reference image latents)
///
/// When `debug` is true, prints detailed trace information about the selection.
///
/// # Arguments
/// * `xs` - Input tensor [batch, seq, dim]
/// * `mod_out` - Modulation output with shape [2*batch, 1, dim] (doubled for zero_cond_t)
/// * `modulate_index` - Per-token index tensor [batch, seq] with values 0 or 1
/// * `debug` - If true, print detailed debug output
///
/// # Returns
/// Tuple of (modulated_x, gate) where modulated_x = x * (1 + scale) + shift
pub fn apply_modulation_with_index(
    xs: &Tensor,
    mod_out: &ModulationOut,
    modulate_index: &Tensor,
    debug: bool,
) -> Result<(Tensor, Tensor)> {
    let dtype = xs.dtype();
    let batch_size = xs.dim(0)?;

    // mod_out has shape [2*batch, 1, dim] from doubled timestep
    // Split into two halves: [batch, 1, dim] each
    let shift_all = &mod_out.shift; // [2*batch, 1, dim]
    let scale_all = &mod_out.scale;
    let gate_all = &mod_out.gate;

    // First half (index 0): actual timestep modulation
    let shift_0 = shift_all.narrow(0, 0, batch_size)?; // [batch, 1, dim]
    let scale_0 = scale_all.narrow(0, 0, batch_size)?;
    let gate_0 = gate_all.narrow(0, 0, batch_size)?;

    // Second half (index 1): zero timestep modulation
    let shift_1 = shift_all.narrow(0, batch_size, batch_size)?;
    let scale_1 = scale_all.narrow(0, batch_size, batch_size)?;
    let gate_1 = gate_all.narrow(0, batch_size, batch_size)?;

    // Expand index to match feature dimension: [batch, seq] -> [batch, seq, 1]
    let index_expanded = modulate_index.unsqueeze(D::Minus1)?.to_dtype(dtype)?;

    // For broadcasting, we need shift_0/1 to be [batch, 1, dim] which they already are
    // Then torch.where will broadcast to [batch, seq, dim]

    // Create masks for selection (index == 0 means use _0, else use _1)
    // In Candle, we can use: result = (1 - index) * val_0 + index * val_1
    let one_minus_index = (1.0 - &index_expanded)?;

    // Select shift per token
    let shift_result = (&shift_0.broadcast_mul(&one_minus_index)?
        + &shift_1.broadcast_mul(&index_expanded)?)?;

    // Select scale per token
    let scale_result = (&scale_0.broadcast_mul(&one_minus_index)?
        + &scale_1.broadcast_mul(&index_expanded)?)?;

    // Select gate per token
    let gate_result = (&gate_0.broadcast_mul(&one_minus_index)?
        + &gate_1.broadcast_mul(&index_expanded)?)?;

    // Apply modulation: x * (1 + scale) + shift
    let modulated = xs.broadcast_mul(&(&scale_result + 1.0)?)?
        .broadcast_add(&shift_result)?;

    if debug {
        use super::debug::debug_modulation_with_index;
        let _ = debug_modulation_with_index(
            xs, shift_all, scale_all, modulate_index,
            &shift_0, &shift_1, &scale_0, &scale_1,
            &shift_result, &scale_result, &modulated,
        );
    }

    Ok((modulated, gate_result))
}

/// Modulation layer for dual-stream blocks.
///
/// Projects the timestep embedding to 6 × dim parameters:
/// (shift1, scale1, gate1, shift2, scale2, gate2) for norm1 and norm2.
#[derive(Debug, Clone)]
pub struct Modulation {
    lin: Linear,
}

impl Modulation {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        // Linear: dim -> 6 * dim (for 2 modulations × 3 params each)
        let lin = candle_nn::linear(dim, 6 * dim, vb.pp("1"))?;
        Ok(Self { lin })
    }

    /// Forward pass: compute modulation parameters for both norm1 and norm2.
    pub fn forward(&self, vec_: &Tensor) -> Result<(ModulationOut, ModulationOut)> {
        // SiLU activation then linear projection
        let ys = vec_.silu()?.apply(&self.lin)?.unsqueeze(1)?;

        // Split into 6 parts
        let chunks = ys.chunk(6, D::Minus1)?;
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

/// Feed-forward network with GELU activation.
#[derive(Debug, Clone)]
pub struct FeedForward {
    proj_in: Linear,
    proj_out: Linear,
}

impl FeedForward {
    pub fn new(dim: usize, dim_out: usize, vb: VarBuilder) -> Result<Self> {
        // MLP: dim -> 4*dim -> dim_out (using GELU-approximate)
        let hidden_dim = dim * 4;
        let proj_in = candle_nn::linear(dim, hidden_dim, vb.pp("net.0.proj"))?;
        let proj_out = candle_nn::linear(hidden_dim, dim_out, vb.pp("net.2"))?;
        Ok(Self { proj_in, proj_out })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // GELU-approximate activation (tanh approximation, matches PyTorch gelu-approximate)
        xs.apply(&self.proj_in)?.gelu()?.apply(&self.proj_out)
    }
}

impl FeedForward {
    /// Forward with debug output to capture intermediate tensors.
    pub fn forward_with_debug(&self, xs: &Tensor, prefix: &str) -> Result<Tensor> {
        let proj_in_out = xs.apply(&self.proj_in)?;
        debug_tensor(&format!("{}_proj_in_output", prefix), &proj_in_out);

        let gelu_out = proj_in_out.gelu()?;
        debug_tensor(&format!("{}_gelu_output", prefix), &gelu_out);

        let proj_out = gelu_out.apply(&self.proj_out)?;
        debug_tensor(&format!("{}_proj_out_output", prefix), &proj_out);

        Ok(proj_out)
    }
}

/// QK Normalization using RMSNorm.
///
/// Applies separate normalization to queries and keys before attention,
/// which helps stabilize training with large hidden dimensions.
#[derive(Debug, Clone)]
pub struct QkNorm {
    query_norm: RmsNorm,
    key_norm: RmsNorm,
}

impl QkNorm {
    pub fn new(dim: usize, vb: VarBuilder, eps: f64) -> Result<Self> {
        let query_norm_weight = vb.get(dim, "norm_q.weight")?;
        let query_norm = RmsNorm::new(query_norm_weight, eps);

        let key_norm_weight = vb.get(dim, "norm_k.weight")?;
        let key_norm = RmsNorm::new(key_norm_weight, eps);

        Ok(Self {
            query_norm,
            key_norm,
        })
    }

    pub fn forward_q(&self, q: &Tensor) -> Result<Tensor> {
        q.apply(&self.query_norm)
    }

    pub fn forward_k(&self, k: &Tensor) -> Result<Tensor> {
        k.apply(&self.key_norm)
    }
}

/// QK Normalization for the added (text) stream.
#[derive(Debug, Clone)]
pub struct AddedQkNorm {
    query_norm: RmsNorm,
    key_norm: RmsNorm,
}

impl AddedQkNorm {
    pub fn new(dim: usize, vb: VarBuilder, eps: f64) -> Result<Self> {
        let query_norm_weight = vb.get(dim, "norm_added_q.weight")?;
        let query_norm = RmsNorm::new(query_norm_weight, eps);

        let key_norm_weight = vb.get(dim, "norm_added_k.weight")?;
        let key_norm = RmsNorm::new(key_norm_weight, eps);

        Ok(Self {
            query_norm,
            key_norm,
        })
    }

    pub fn forward_q(&self, q: &Tensor) -> Result<Tensor> {
        q.apply(&self.query_norm)
    }

    pub fn forward_k(&self, k: &Tensor) -> Result<Tensor> {
        k.apply(&self.key_norm)
    }
}

/// Scaled dot-product attention with optional overrides for debugging.
///
/// Input shapes: Q, K, V are [batch, heads, seq, head_dim]
fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    overrides: Option<&BlockOverrides>,
    debug: bool,
) -> Result<Tensor> {
    let head_dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (head_dim as f64).sqrt();

    // Upcast to F32 for numerical stability
    // Note: Must call contiguous() because input tensors are non-contiguous after transpose.
    // When already F32, to_dtype is a no-op preserving non-contiguous layout, but Metal
    // matmul requires contiguous tensors.
    let q = q.to_dtype(DType::F32)?.contiguous()?;
    let k = k.to_dtype(DType::F32)?.contiguous()?;
    let v = v.to_dtype(DType::F32)?.contiguous()?;

    let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
    let attn_logits = q.matmul(&k_t)?;
    let attn_weights = (&attn_logits * scale_factor)?;

    if debug {
        debug_tensor("[BLOCK0.ATTN] attn_weights (Q@K.T/sqrt(d))", &attn_weights);
    }

    // Override attention weights if provided
    let attn_weights = if let Some(BlockOverrides { attn_weights: Some(ref ovr), .. }) = overrides {
        if debug {
            eprintln!("[BLOCK0.ATTN] SUBSTITUTING attn_weights from override");
        }
        ovr.to_dtype(DType::F32)?.contiguous()?
    } else {
        attn_weights
    };

    let attn_probs = candle_nn::ops::softmax_last_dim(&attn_weights)?;

    if debug {
        debug_tensor("[BLOCK0.ATTN] attn_probs (softmax)", &attn_probs);
    }

    // Deep attention debug: full internals comparison + per-head analysis (only for block 0)
    if debug && is_attention_debug() {
        debug_attention_internals(
            "block0",
            &attn_logits,
            &attn_weights,
            &attn_probs,
            Some("debug_tensors/pytorch_edit"),
        )?;
        debug_per_head_stats(&attn_probs, "block0")?;
        save_attention_tensors("block0", &attn_logits, &attn_weights, &attn_probs)?;
    }

    // Override attention probs if provided
    let attn_probs = if let Some(BlockOverrides { attn_probs: Some(ref ovr), .. }) = overrides {
        if debug {
            eprintln!("[BLOCK0.ATTN] SUBSTITUTING attn_probs from override");
        }
        ovr.to_dtype(DType::F32)?.contiguous()?
    } else {
        attn_probs
    };

    let attn_output = attn_probs.matmul(&v)?;

    if debug {
        debug_tensor("[BLOCK0.ATTN] attn_output (probs @ V)", &attn_output);
    }

    Ok(attn_output)
}

/// Dual-stream attention for Qwen-Image.
///
/// This implements joint attention where:
/// 1. Both image and text streams compute Q, K, V
/// 2. Streams are concatenated for joint attention computation
/// 3. Results are split back to separate streams
#[derive(Debug, Clone)]
pub struct QwenDoubleStreamAttention {
    // Image stream projections
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,

    // Text stream projections
    add_q_proj: Linear,
    add_k_proj: Linear,
    add_v_proj: Linear,
    to_add_out: Linear,

    // QK normalization
    img_norm: QkNorm,
    txt_norm: AddedQkNorm,

    num_heads: usize,
    head_dim: usize,
}

impl QwenDoubleStreamAttention {
    pub fn new(
        dim: usize,
        num_heads: usize,
        head_dim: usize,
        vb: VarBuilder,
        eps: f64,
    ) -> Result<Self> {
        let inner_dim = num_heads * head_dim;

        // Image stream projections
        let to_q = candle_nn::linear(dim, inner_dim, vb.pp("to_q"))?;
        let to_k = candle_nn::linear(dim, inner_dim, vb.pp("to_k"))?;
        let to_v = candle_nn::linear(dim, inner_dim, vb.pp("to_v"))?;
        let to_out = candle_nn::linear(inner_dim, dim, vb.pp("to_out.0"))?;

        // Text stream projections
        let add_q_proj = candle_nn::linear(dim, inner_dim, vb.pp("add_q_proj"))?;
        let add_k_proj = candle_nn::linear(dim, inner_dim, vb.pp("add_k_proj"))?;
        let add_v_proj = candle_nn::linear(dim, inner_dim, vb.pp("add_v_proj"))?;
        let to_add_out = candle_nn::linear(inner_dim, dim, vb.pp("to_add_out"))?;

        // QK normalization
        let img_norm = QkNorm::new(head_dim, vb.clone(), eps)?;
        let txt_norm = AddedQkNorm::new(head_dim, vb, eps)?;

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            add_q_proj,
            add_k_proj,
            add_v_proj,
            to_add_out,
            img_norm,
            txt_norm,
            num_heads,
            head_dim,
        })
    }

    /// Forward pass for dual-stream attention.
    ///
    /// # Arguments
    /// * `hidden_states` - Image stream [batch, img_seq, dim]
    /// * `encoder_hidden_states` - Text stream [batch, txt_seq, dim]
    /// * `image_rotary_emb` - RoPE frequencies (img_freqs, txt_freqs)
    ///
    /// # Returns
    /// Tuple of (img_attn_output, txt_attn_output)
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        image_rotary_emb: Option<&(Tensor, Tensor)>,
    ) -> Result<(Tensor, Tensor)> {
        self.forward_impl(hidden_states, encoder_hidden_states, image_rotary_emb, false, None)
    }

    /// Forward pass with debug logging for attention internals.
    pub fn forward_with_debug(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        image_rotary_emb: Option<&(Tensor, Tensor)>,
    ) -> Result<(Tensor, Tensor)> {
        self.forward_impl(hidden_states, encoder_hidden_states, image_rotary_emb, true, None)
    }

    /// Forward pass with optional Q/K/V overrides for debugging.
    ///
    /// # Arguments
    /// * `overrides` - Optional Q/K/V tensor overrides from PyTorch for debugging
    pub fn forward_with_overrides(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        image_rotary_emb: Option<&(Tensor, Tensor)>,
        debug: bool,
        overrides: Option<&BlockOverrides>,
    ) -> Result<(Tensor, Tensor)> {
        self.forward_impl(hidden_states, encoder_hidden_states, image_rotary_emb, debug, overrides)
    }

    fn forward_impl(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        image_rotary_emb: Option<&(Tensor, Tensor)>,
        debug: bool,
        overrides: Option<&BlockOverrides>,
    ) -> Result<(Tensor, Tensor)> {
        let (b_sz, img_seq, _) = hidden_states.dims3()?;
        let txt_seq = encoder_hidden_states.dim(1)?;

        // Compute QKV for image stream
        let img_q = hidden_states.apply(&self.to_q)?;
        let img_k = hidden_states.apply(&self.to_k)?;
        let img_v = hidden_states.apply(&self.to_v)?;

        if debug {
            // Debug: Print weight statistics to verify loading
            let to_q_weight = self.to_q.weight();
            let w_mean = to_q_weight.mean_all()?.to_scalar::<f32>().unwrap_or(0.0);
            let w_flat = to_q_weight.flatten_all()?.to_dtype(DType::F32)?;
            let w_std = (w_flat.sqr()?.mean_all()?.to_scalar::<f32>().unwrap_or(0.0) - w_mean.powi(2)).sqrt();
            eprintln!("[BLOCK0.ATTN] to_q.weight: shape={:?}, mean={:.6}, std={:.6}",
                to_q_weight.dims(), w_mean, w_std);
            if let Some(bias) = self.to_q.bias() {
                let b_mean = bias.mean_all()?.to_scalar::<f32>().unwrap_or(0.0);
                let b_flat = bias.flatten_all()?.to_dtype(DType::F32)?;
                let b_std = (b_flat.sqr()?.mean_all()?.to_scalar::<f32>().unwrap_or(0.0) - b_mean.powi(2)).sqrt();
                eprintln!("[BLOCK0.ATTN] to_q.bias: shape={:?}, mean={:.6}, std={:.6}",
                    bias.dims(), b_mean, b_std);
            } else {
                eprintln!("[BLOCK0.ATTN] to_q.bias: NONE");
            }
            debug_tensor("[BLOCK0.ATTN] img_q_proj", &img_q);
            debug_tensor("[BLOCK0.ATTN] img_k_proj", &img_k);
            debug_tensor("[BLOCK0.ATTN] img_v_proj", &img_v);
        }

        // Apply Q/K/V overrides for image stream if provided
        let img_q = if let Some(BlockOverrides { img_q: Some(ref ovr), .. }) = overrides {
            if debug {
                eprintln!("[BLOCK0.ATTN] SUBSTITUTING img_q from override");
                debug_tensor("[BLOCK0.ATTN] img_q (original)", &img_q);
                debug_tensor("[BLOCK0.ATTN] img_q (override)", ovr);
            }
            ovr.clone()
        } else {
            img_q
        };
        let img_k = if let Some(BlockOverrides { img_k: Some(ref ovr), .. }) = overrides {
            if debug {
                eprintln!("[BLOCK0.ATTN] SUBSTITUTING img_k from override");
            }
            ovr.clone()
        } else {
            img_k
        };
        let img_v = if let Some(BlockOverrides { img_v: Some(ref ovr), .. }) = overrides {
            if debug {
                eprintln!("[BLOCK0.ATTN] SUBSTITUTING img_v from override");
            }
            ovr.clone()
        } else {
            img_v
        };

        // Compute QKV for text stream
        let txt_q = encoder_hidden_states.apply(&self.add_q_proj)?;
        let txt_k = encoder_hidden_states.apply(&self.add_k_proj)?;
        let txt_v = encoder_hidden_states.apply(&self.add_v_proj)?;

        if debug {
            debug_tensor("[BLOCK0.ATTN] txt_q_proj", &txt_q);
            debug_tensor("[BLOCK0.ATTN] txt_k_proj", &txt_k);
            debug_tensor("[BLOCK0.ATTN] txt_v_proj", &txt_v);
        }

        // Save Q/K at projection stage (before reshape, before overrides)
        // Note: This captures the raw projection output for comparison with PyTorch
        // Only save for first attention call (block 0) by checking debug flag
        if debug && is_qk_save_enabled() {
            save_qk_pipeline_tensors("block0", "proj", &img_q, &img_k, &txt_q, &txt_k)?;
        }

        // Apply Q/K/V overrides for text stream if provided
        let txt_q = if let Some(BlockOverrides { txt_q: Some(ref ovr), .. }) = overrides {
            if debug {
                eprintln!("[BLOCK0.ATTN] SUBSTITUTING txt_q from override");
            }
            ovr.clone()
        } else {
            txt_q
        };
        let txt_k = if let Some(BlockOverrides { txt_k: Some(ref ovr), .. }) = overrides {
            if debug {
                eprintln!("[BLOCK0.ATTN] SUBSTITUTING txt_k from override");
            }
            ovr.clone()
        } else {
            txt_k
        };
        let txt_v = if let Some(BlockOverrides { txt_v: Some(ref ovr), .. }) = overrides {
            if debug {
                eprintln!("[BLOCK0.ATTN] SUBSTITUTING txt_v from override");
            }
            ovr.clone()
        } else {
            txt_v
        };

        // Reshape for multi-head attention: [batch, seq, heads, head_dim]
        let img_q = img_q.reshape((b_sz, img_seq, self.num_heads, self.head_dim))?;
        let img_k = img_k.reshape((b_sz, img_seq, self.num_heads, self.head_dim))?;
        let img_v = img_v.reshape((b_sz, img_seq, self.num_heads, self.head_dim))?;

        let txt_q = txt_q.reshape((b_sz, txt_seq, self.num_heads, self.head_dim))?;
        let txt_k = txt_k.reshape((b_sz, txt_seq, self.num_heads, self.head_dim))?;
        let txt_v = txt_v.reshape((b_sz, txt_seq, self.num_heads, self.head_dim))?;

        // Apply QK normalization
        let img_q = self.img_norm.forward_q(&img_q)?;
        let img_k = self.img_norm.forward_k(&img_k)?;
        let txt_q = self.txt_norm.forward_q(&txt_q)?;
        let txt_k = self.txt_norm.forward_k(&txt_k)?;

        if debug {
            debug_tensor("[BLOCK0.ATTN] img_q_after_norm", &img_q);
            debug_tensor("[BLOCK0.ATTN] img_k_after_norm", &img_k);
            debug_tensor("[BLOCK0.ATTN] txt_q_after_norm", &txt_q);
            debug_tensor("[BLOCK0.ATTN] txt_k_after_norm", &txt_k);
        }

        // Save Q/K after normalization (before RoPE)
        // Only save for block 0 (debug flag is only true for block 0)
        if debug && is_qk_save_enabled() {
            save_qk_pipeline_tensors("block0", "norm", &img_q, &img_k, &txt_q, &txt_k)?;
        }

        // Apply RoPE
        let (img_q, img_k, txt_q, txt_k) = if let Some((img_freqs, txt_freqs)) = image_rotary_emb {
            let img_q = apply_rotary_emb_qwen(&img_q, img_freqs)?;
            let img_k = apply_rotary_emb_qwen(&img_k, img_freqs)?;
            let txt_q = apply_rotary_emb_qwen(&txt_q, txt_freqs)?;
            let txt_k = apply_rotary_emb_qwen(&txt_k, txt_freqs)?;
            (img_q, img_k, txt_q, txt_k)
        } else {
            (img_q, img_k, txt_q, txt_k)
        };

        if debug {
            debug_tensor("[BLOCK0.ATTN] img_q_after_rope", &img_q);
            debug_tensor("[BLOCK0.ATTN] img_k_after_rope", &img_k);
            debug_tensor("[BLOCK0.ATTN] txt_q_after_rope", &txt_q);
            debug_tensor("[BLOCK0.ATTN] txt_k_after_rope", &txt_k);
        }

        // Save Q/K after RoPE (final Q/K used in attention)
        // Only save for block 0 (debug flag is only true for block 0)
        if debug && is_qk_save_enabled() {
            save_qk_pipeline_tensors("block0", "rope", &img_q, &img_k, &txt_q, &txt_k)?;
        }

        // Concatenate for joint attention in [B, S, H, D] format: order is [text, image]
        // This matches Python which concatenates before permuting
        let joint_q = Tensor::cat(&[&txt_q, &img_q], 1)?;
        let joint_k = Tensor::cat(&[&txt_k, &img_k], 1)?;
        let joint_v = Tensor::cat(&[&txt_v, &img_v], 1)?;

        if debug {
            debug_tensor("[BLOCK0.ATTN] joint_q", &joint_q);
            debug_tensor("[BLOCK0.ATTN] joint_k", &joint_k);
            debug_tensor("[BLOCK0.ATTN] joint_v", &joint_v);
        }

        // Transpose for attention: [B, S, H, D] -> [B, H, S, D]
        let joint_q = joint_q.transpose(1, 2)?;
        let joint_k = joint_k.transpose(1, 2)?;
        let joint_v = joint_v.transpose(1, 2)?;

        // Compute attention (pass overrides for debugging)
        let joint_attn = scaled_dot_product_attention(&joint_q, &joint_k, &joint_v, overrides, debug)?;

        if debug {
            debug_tensor("[BLOCK0.ATTN] joint_attn_output", &joint_attn);
        }

        // Transpose back: [B, H, S, D] -> [B, S, H, D]
        let joint_attn = joint_attn.transpose(1, 2)?;

        // Override attention output if provided (shape: [B, S, H, D])
        let joint_attn = if let Some(BlockOverrides { attn_output: Some(ref ovr), .. }) = overrides {
            if debug {
                eprintln!("[BLOCK0.ATTN] SUBSTITUTING attn_output from override");
                debug_tensor("[BLOCK0.ATTN] attn_output (original)", &joint_attn);
                debug_tensor("[BLOCK0.ATTN] attn_output (override)", ovr);
            }
            ovr.to_dtype(joint_attn.dtype())?.to_device(joint_attn.device())?
        } else {
            joint_attn
        };

        // Flatten: [B, S, H, D] -> [B, S, H*D]
        let joint_attn = joint_attn.flatten_from(2)?;
        let joint_attn = joint_attn.to_dtype(hidden_states.dtype())?;

        // Split back to text and image
        let txt_attn = joint_attn.narrow(1, 0, txt_seq)?;
        let img_attn = joint_attn.narrow(1, txt_seq, img_seq)?;

        if debug {
            debug_tensor("[BLOCK0.ATTN] img_attn_pre_proj", &img_attn);
        }

        // Output projections
        let img_out = img_attn.apply(&self.to_out)?;
        let txt_out = txt_attn.apply(&self.to_add_out)?;

        if debug {
            debug_tensor("[BLOCK0.ATTN] img_out_proj", &img_out);
            debug_tensor("[BLOCK0.ATTN] txt_out_proj", &txt_out);
        }

        Ok((img_out, txt_out))
    }
}

/// Qwen-Image Transformer Block.
///
/// Implements the dual-stream architecture where:
/// 1. Both streams get modulation parameters from timestep embedding
/// 2. Norm1 + modulation + joint attention + gated residual
/// 3. Norm2 + modulation + MLP + gated residual
#[derive(Debug, Clone)]
pub struct QwenImageTransformerBlock {
    // Image stream
    img_mod: Modulation,
    img_norm1: candle_nn::LayerNorm,
    img_norm2: candle_nn::LayerNorm,
    img_mlp: FeedForward,

    // Text stream
    txt_mod: Modulation,
    txt_norm1: candle_nn::LayerNorm,
    txt_norm2: candle_nn::LayerNorm,
    txt_mlp: FeedForward,

    // Shared attention
    attn: QwenDoubleStreamAttention,
}

impl QwenImageTransformerBlock {
    pub fn new(
        dim: usize,
        num_attention_heads: usize,
        attention_head_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let eps = 1e-6;
        let device = vb.device();
        let dtype = vb.dtype();

        // Image stream
        // Note: LayerNorm uses elementwise_affine=False in PyTorch (no learned params)
        // The modulation provides scale/shift instead
        let img_mod = Modulation::new(dim, vb.pp("img_mod"))?;
        let img_norm1 = layer_norm_no_affine(dim, eps, device, dtype)?;
        let img_norm2 = layer_norm_no_affine(dim, eps, device, dtype)?;
        let img_mlp = FeedForward::new(dim, dim, vb.pp("img_mlp"))?;

        // Text stream
        let txt_mod = Modulation::new(dim, vb.pp("txt_mod"))?;
        let txt_norm1 = layer_norm_no_affine(dim, eps, device, dtype)?;
        let txt_norm2 = layer_norm_no_affine(dim, eps, device, dtype)?;
        let txt_mlp = FeedForward::new(dim, dim, vb.pp("txt_mlp"))?;

        // Shared attention
        let attn = QwenDoubleStreamAttention::new(
            dim,
            num_attention_heads,
            attention_head_dim,
            vb.pp("attn"),
            eps,
        )?;

        Ok(Self {
            img_mod,
            img_norm1,
            img_norm2,
            img_mlp,
            txt_mod,
            txt_norm1,
            txt_norm2,
            txt_mlp,
            attn,
        })
    }

    /// Forward pass through the dual-stream block (standard mode, no per-token modulation).
    ///
    /// # Arguments
    /// * `hidden_states` - Image stream [batch, img_seq, dim]
    /// * `encoder_hidden_states` - Text stream [batch, txt_seq, dim]
    /// * `temb` - Timestep embedding [batch, dim]
    /// * `image_rotary_emb` - RoPE frequencies
    ///
    /// # Returns
    /// Tuple of (updated_encoder_hidden_states, updated_hidden_states)
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        temb: &Tensor,
        image_rotary_emb: Option<&(Tensor, Tensor)>,
    ) -> Result<(Tensor, Tensor)> {
        self.forward_impl(hidden_states, encoder_hidden_states, temb, image_rotary_emb, None, false, None)
    }

    /// Forward pass with modulate_index for edit mode (zero_cond_t).
    ///
    /// When modulate_index is provided:
    /// - `temb` is doubled: [2*batch, dim] with [actual_timestep, zero_timestep]
    /// - Image modulation uses per-token selection based on modulate_index
    /// - Text modulation uses only actual timestep (first half of temb)
    pub fn forward_with_modulate_index(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        temb: &Tensor,
        image_rotary_emb: Option<&(Tensor, Tensor)>,
        modulate_index: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        self.forward_impl(hidden_states, encoder_hidden_states, temb, image_rotary_emb, modulate_index, false, None)
    }

    /// Forward pass with optional debug logging (for block 0 analysis).
    pub fn forward_with_debug(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        temb: &Tensor,
        image_rotary_emb: Option<&(Tensor, Tensor)>,
        modulate_index: Option<&Tensor>,
        debug: bool,
    ) -> Result<(Tensor, Tensor)> {
        self.forward_impl(hidden_states, encoder_hidden_states, temb, image_rotary_emb, modulate_index, debug, None)
    }

    /// Forward pass with debug AND optional tensor overrides for debugging substitution.
    ///
    /// # Arguments
    /// * `overrides` - Optional overrides for intermediate tensors:
    ///   - `img_modulated`: Override the modulated image tensor before attention
    ///   - `img_gate2`: Override the MLP gate tensor
    pub fn forward_with_overrides(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        temb: &Tensor,
        image_rotary_emb: Option<&(Tensor, Tensor)>,
        modulate_index: Option<&Tensor>,
        debug: bool,
        overrides: Option<&BlockOverrides>,
    ) -> Result<(Tensor, Tensor)> {
        self.forward_impl(hidden_states, encoder_hidden_states, temb, image_rotary_emb, modulate_index, debug, overrides)
    }

    /// Internal forward implementation handling both standard and edit modes.
    fn forward_impl(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        temb: &Tensor,
        image_rotary_emb: Option<&(Tensor, Tensor)>,
        modulate_index: Option<&Tensor>,
        debug: bool,
        overrides: Option<&BlockOverrides>,
    ) -> Result<(Tensor, Tensor)> {
        let orig_dtype = hidden_states.dtype();
        let batch_size = hidden_states.dim(0)?;

        // Get modulation parameters for image stream (uses full temb if doubled)
        let (img_mod1, img_mod2) = self.img_mod.forward(temb)?;

        // For text stream, use only first half of temb if modulate_index is provided (zero_cond_t mode)
        let txt_temb = if modulate_index.is_some() {
            // In zero_cond_t mode, temb is doubled [2*batch, dim]
            // Text uses only actual timestep (first half)
            temb.narrow(0, 0, batch_size)?
        } else {
            temb.clone()
        };
        let (txt_mod1, txt_mod2) = self.txt_mod.forward(&txt_temb)?;

        if debug {
            debug_tensor("[BLOCK0] img_mod1.shift", &img_mod1.shift);
            debug_tensor("[BLOCK0] img_mod1.scale", &img_mod1.scale);
            debug_tensor("[BLOCK0] img_mod1.gate", &img_mod1.gate);
            debug_tensor("[BLOCK0] txt_mod1.shift", &txt_mod1.shift);
            debug_tensor("[BLOCK0] txt_mod1.scale", &txt_mod1.scale);
            debug_tensor("[BLOCK0] txt_mod1.gate", &txt_mod1.gate);
        }

        // === Attention phase ===

        // Image: norm1 + modulate (with per-token selection if modulate_index provided)
        let img_normed = hidden_states.apply(&self.img_norm1)?;
        if debug {
            debug_tensor("[BLOCK0] img_norm1_output", &img_normed);
        }

        let (img_modulated, img_gate1) = if let Some(mod_idx) = modulate_index {
            // Per-token modulation for edit mode
            apply_modulation_with_index(&img_normed, &img_mod1, mod_idx, debug)?
        } else {
            // Standard modulation
            let modulated = img_mod1.scale_shift(&img_normed)?;
            let gate = img_mod1.gate.clone();
            (modulated, gate)
        };

        // Allow override for debugging
        let img_modulated = if let Some(BlockOverrides { img_modulated: Some(ref ovr), .. }) = overrides {
            if debug {
                eprintln!("[BLOCK0] SUBSTITUTING img_modulated from override");
                debug_tensor("[BLOCK0] img_modulated (original)", &img_modulated);
                debug_tensor("[BLOCK0] img_modulated (override)", ovr);
            }
            ovr.clone()
        } else {
            img_modulated
        };

        if debug {
            debug_tensor("[BLOCK0] img_modulated", &img_modulated);
        }

        // Text: norm1 + modulate (always standard, no per-token selection)
        let txt_normed = encoder_hidden_states.apply(&self.txt_norm1)?;
        if debug {
            debug_tensor("[BLOCK0] txt_norm1_output", &txt_normed);
        }
        let txt_modulated = txt_mod1.scale_shift(&txt_normed)?;
        let txt_gate1 = txt_mod1.gate.clone();
        if debug {
            debug_tensor("[BLOCK0] txt_modulated", &txt_modulated);
        }

        // Joint attention (pass overrides for Q/K/V substitution debugging)
        let (img_attn_out, txt_attn_out) = if debug || overrides.map(|o| o.has_qkv_overrides()).unwrap_or(false) {
            self.attn.forward_with_overrides(&img_modulated, &txt_modulated, image_rotary_emb, debug, overrides)?
        } else {
            self.attn.forward(&img_modulated, &txt_modulated, image_rotary_emb)?
        };

        if debug {
            debug_tensor("[BLOCK0] img_attn_output", &img_attn_out);
            debug_tensor("[BLOCK0] txt_attn_output", &txt_attn_out);
        }

        // === F32 ACCUMULATION FOR RESIDUAL ADDITIONS ===
        // The MLP outputs have std > 2000, and accumulated values reach ±2.3M
        // This exceeds BF16's max of ±65536, so we MUST compute residuals in F32
        // then cast back to original dtype.

        // Gated residual for attention (use per-token gate for image if in edit mode)
        let gated_img_attn = img_gate1.broadcast_mul(&img_attn_out)?;
        let gated_txt_attn = txt_gate1.broadcast_mul(&txt_attn_out)?;
        if debug {
            debug_tensor("[BLOCK0] gated_img_attn", &gated_img_attn);
            debug_tensor("[BLOCK0] gated_txt_attn", &gated_txt_attn);
        }

        // Residual add in F32 to avoid overflow, then cast back
        let hidden_states = (hidden_states.to_dtype(DType::F32)?
            + gated_img_attn.to_dtype(DType::F32)?)?
            .to_dtype(orig_dtype)?;
        let encoder_hidden_states = (encoder_hidden_states.to_dtype(DType::F32)?
            + gated_txt_attn.to_dtype(DType::F32)?)?
            .to_dtype(orig_dtype)?;

        if debug {
            debug_tensor("[BLOCK0] after_attn_residual_img", &hidden_states);
            debug_tensor("[BLOCK0] after_attn_residual_txt", &encoder_hidden_states);
        }

        // === MLP phase ===

        // Image: norm2 + modulate + MLP + gated residual
        let img_normed2 = hidden_states.apply(&self.img_norm2)?;
        if debug {
            debug_tensor("[BLOCK0] img_norm2_output", &img_normed2);
        }

        let (img_modulated2, img_gate2) = if let Some(mod_idx) = modulate_index {
            apply_modulation_with_index(&img_normed2, &img_mod2, mod_idx, debug)?
        } else {
            let modulated = img_mod2.scale_shift(&img_normed2)?;
            let gate = img_mod2.gate.clone();
            (modulated, gate)
        };
        if debug {
            debug_tensor("[BLOCK0] img_modulated2", &img_modulated2);
            debug_tensor("[BLOCK0] img_mod2.shift", &img_mod2.shift);
            debug_tensor("[BLOCK0] img_mod2.scale", &img_mod2.scale);
            debug_tensor("[BLOCK0] img_mod2.gate", &img_mod2.gate);
        }

        // Allow override for debugging
        let img_gate2 = if let Some(BlockOverrides { img_gate2: Some(ref ovr), .. }) = overrides {
            if debug {
                eprintln!("[BLOCK0] SUBSTITUTING img_gate2 from override");
                debug_tensor("[BLOCK0] img_gate2 (original)", &img_gate2);
                debug_tensor("[BLOCK0] img_gate2 (override)", ovr);
            }
            ovr.clone()
        } else {
            img_gate2
        };

        let img_mlp_out = if debug {
            self.img_mlp.forward_with_debug(&img_modulated2, "[BLOCK0] img_mlp")?
        } else {
            self.img_mlp.forward(&img_modulated2)?
        };
        if debug {
            debug_tensor("[BLOCK0] img_mlp_output", &img_mlp_out);
            debug_tensor("[BLOCK0] img_gate2", &img_gate2);
        }
        let gated_img_mlp = img_gate2.broadcast_mul(&img_mlp_out)?;
        if debug {
            debug_tensor("[BLOCK0] gated_img_mlp", &gated_img_mlp);
        }

        // Residual add in F32
        let hidden_states = (hidden_states.to_dtype(DType::F32)?
            + gated_img_mlp.to_dtype(DType::F32)?)?
            .to_dtype(orig_dtype)?;
        if debug {
            debug_tensor("[BLOCK0] after_mlp_residual_img", &hidden_states);
        }

        // Text: norm2 + modulate + MLP + gated residual (always standard)
        let txt_normed2 = encoder_hidden_states.apply(&self.txt_norm2)?;
        let txt_modulated2 = txt_mod2.scale_shift(&txt_normed2)?;
        let txt_mlp_out = self.txt_mlp.forward(&txt_modulated2)?;
        if debug {
            debug_tensor("[BLOCK0] txt_mlp_output", &txt_mlp_out);
        }
        let gated_txt_mlp = txt_mod2.gate(&txt_mlp_out)?;

        // Residual add in F32
        let encoder_hidden_states = (encoder_hidden_states.to_dtype(DType::F32)?
            + gated_txt_mlp.to_dtype(DType::F32)?)?
            .to_dtype(orig_dtype)?;

        Ok((encoder_hidden_states, hidden_states))
    }
}
