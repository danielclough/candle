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

use super::rope::apply_rotary_emb_qwen;

/// Create a parameter-free LayerNorm (equivalent to PyTorch's elementwise_affine=False).
///
/// This is used in Qwen-Image transformer blocks where the modulation provides
/// the scale and shift instead of learned LayerNorm parameters.
fn layer_norm_no_affine(
    size: usize,
    eps: f64,
    device: &candle::Device,
    dtype: DType,
) -> Result<LayerNorm> {
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
    pub fn _to_f32(&self) -> Result<Self> {
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
/// # Arguments
/// * `xs` - Input tensor [batch, seq, dim]
/// * `mod_out` - Modulation output with shape [2*batch, 1, dim] (doubled for zero_cond_t)
/// * `modulate_index` - Per-token index tensor [batch, seq] with values 0 or 1
///
/// # Returns
/// Tuple of (modulated_x, gate) where modulated_x = x * (1 + scale) + shift
pub fn apply_modulation_with_index(
    xs: &Tensor,
    mod_out: &ModulationOut,
    modulate_index: &Tensor,
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
    let shift_result =
        (&shift_0.broadcast_mul(&one_minus_index)? + &shift_1.broadcast_mul(&index_expanded)?)?;

    // Select scale per token
    let scale_result =
        (&scale_0.broadcast_mul(&one_minus_index)? + &scale_1.broadcast_mul(&index_expanded)?)?;

    // Select gate per token
    let gate_result =
        (&gate_0.broadcast_mul(&one_minus_index)? + &gate_1.broadcast_mul(&index_expanded)?)?;

    // Apply modulation: x * (1 + scale) + shift
    let modulated = xs
        .broadcast_mul(&(&scale_result + 1.0)?)?
        .broadcast_add(&shift_result)?;

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

/// Scaled dot-product attention.
///
/// Input shapes: Q, K, V are [batch, heads, seq, head_dim]
///
/// # Arguments
/// * `q` - Query tensor
/// * `k` - Key tensor
/// * `v` - Value tensor
/// * `upcast_attention` - If true, upcast Q, K, V to F32 for numerical stability.
///   If false, compute in the input dtype (e.g., BF16) for faster inference.
fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    upcast_attention: bool,
) -> Result<Tensor> {
    let head_dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (head_dim as f64).sqrt();
    let orig_dtype = q.dtype();

    // Optionally upcast to F32 for numerical stability
    // Note: Must call contiguous() because input tensors are non-contiguous after transpose.
    // When already F32, to_dtype is a no-op preserving non-contiguous layout, but Metal
    // matmul requires contiguous tensors.
    let (q, k, v) = if upcast_attention {
        (
            q.to_dtype(DType::F32)?.contiguous()?,
            k.to_dtype(DType::F32)?.contiguous()?,
            v.to_dtype(DType::F32)?.contiguous()?,
        )
    } else {
        (q.contiguous()?, k.contiguous()?, v.contiguous()?)
    };

    let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
    let attn_logits = q.matmul(&k_t)?;
    let attn_weights = (&attn_logits * scale_factor)?;

    let attn_probs = candle_nn::ops::softmax_last_dim(&attn_weights)?;

    let attn_output = attn_probs.matmul(&v)?;

    // Cast back to original dtype if we upcasted
    if upcast_attention && orig_dtype != DType::F32 {
        attn_output.to_dtype(orig_dtype)
    } else {
        Ok(attn_output)
    }
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

    /// Whether to upcast attention to F32 for numerical stability
    upcast_attention: bool,
}

impl QwenDoubleStreamAttention {
    pub fn new(
        dim: usize,
        num_heads: usize,
        head_dim: usize,
        vb: VarBuilder,
        eps: f64,
        upcast_attention: bool,
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
            upcast_attention,
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
        let (b_sz, img_seq, _) = hidden_states.dims3()?;
        let txt_seq = encoder_hidden_states.dim(1)?;

        // Compute QKV for image stream
        let img_q = hidden_states.apply(&self.to_q)?;
        let img_k = hidden_states.apply(&self.to_k)?;
        let img_v = hidden_states.apply(&self.to_v)?;

        // Compute QKV for text stream
        let txt_q = encoder_hidden_states.apply(&self.add_q_proj)?;
        let txt_k = encoder_hidden_states.apply(&self.add_k_proj)?;
        let txt_v = encoder_hidden_states.apply(&self.add_v_proj)?;

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

        // Concatenate for joint attention in [B, S, H, D] format: order is [text, image]
        // This matches Python which concatenates before permuting
        let joint_q = Tensor::cat(&[&txt_q, &img_q], 1)?;
        let joint_k = Tensor::cat(&[&txt_k, &img_k], 1)?;
        let joint_v = Tensor::cat(&[&txt_v, &img_v], 1)?;

        // Transpose for attention: [B, S, H, D] -> [B, H, S, D]
        let joint_q = joint_q.transpose(1, 2)?;
        let joint_k = joint_k.transpose(1, 2)?;
        let joint_v = joint_v.transpose(1, 2)?;

        // Compute attention
        let joint_attn =
            scaled_dot_product_attention(&joint_q, &joint_k, &joint_v, self.upcast_attention)?;

        // Transpose back: [B, H, S, D] -> [B, S, H, D]
        let joint_attn = joint_attn.transpose(1, 2)?;

        // Flatten: [B, S, H, D] -> [B, S, H*D]
        let joint_attn = joint_attn.flatten_from(2)?;

        // Split back to text and image
        let txt_attn = joint_attn.narrow(1, 0, txt_seq)?;
        let img_attn = joint_attn.narrow(1, txt_seq, img_seq)?;

        // Output projections
        let img_out = img_attn.apply(&self.to_out)?;
        let txt_out = txt_attn.apply(&self.to_add_out)?;

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
        upcast_attention: bool,
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
            upcast_attention,
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
        self.forward_impl(
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb,
            None,
        )
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
        self.forward_impl(
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb,
            modulate_index,
        )
    }

    /// Internal forward implementation handling both standard and edit modes.
    fn forward_impl(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        temb: &Tensor,
        image_rotary_emb: Option<&(Tensor, Tensor)>,
        modulate_index: Option<&Tensor>,
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

        // === Attention phase ===

        // Image: norm1 + modulate (with per-token selection if modulate_index provided)
        let img_normed = hidden_states.apply(&self.img_norm1)?;

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
        let txt_normed = encoder_hidden_states.apply(&self.txt_norm1)?;
        let txt_modulated = txt_mod1.scale_shift(&txt_normed)?;
        let txt_gate1 = txt_mod1.gate.clone();

        // Joint attention
        let (img_attn_out, txt_attn_out) =
            self.attn
                .forward(&img_modulated, &txt_modulated, image_rotary_emb)?;

        // === F32 ACCUMULATION FOR RESIDUAL ADDITIONS ===
        // The MLP outputs have std > 2000, and accumulated values reach ±2.3M
        // This exceeds BF16's max of ±65536, so we MUST compute residuals in F32
        // then cast back to original dtype.

        // Gated residual for attention (use per-token gate for image if in edit mode)
        let gated_img_attn = img_gate1.broadcast_mul(&img_attn_out)?;
        let gated_txt_attn = txt_gate1.broadcast_mul(&txt_attn_out)?;

        // Residual add in F32 to avoid overflow, then cast back
        let hidden_states = (hidden_states.to_dtype(DType::F32)?
            + gated_img_attn.to_dtype(DType::F32)?)?
        .to_dtype(orig_dtype)?;
        let encoder_hidden_states = (encoder_hidden_states.to_dtype(DType::F32)?
            + gated_txt_attn.to_dtype(DType::F32)?)?
        .to_dtype(orig_dtype)?;

        // === MLP phase ===

        // Image: norm2 + modulate + MLP + gated residual
        let img_normed2 = hidden_states.apply(&self.img_norm2)?;

        let (img_modulated2, img_gate2) = if let Some(mod_idx) = modulate_index {
            apply_modulation_with_index(&img_normed2, &img_mod2, mod_idx)?
        } else {
            let modulated = img_mod2.scale_shift(&img_normed2)?;
            let gate = img_mod2.gate.clone();
            (modulated, gate)
        };

        let img_mlp_out = self.img_mlp.forward(&img_modulated2)?;
        let gated_img_mlp = img_gate2.broadcast_mul(&img_mlp_out)?;

        // Residual add in F32
        let hidden_states = (hidden_states.to_dtype(DType::F32)?
            + gated_img_mlp.to_dtype(DType::F32)?)?
        .to_dtype(orig_dtype)?;

        // Text: norm2 + modulate + MLP + gated residual (always standard)
        let txt_normed2 = encoder_hidden_states.apply(&self.txt_norm2)?;
        let txt_modulated2 = txt_mod2.scale_shift(&txt_normed2)?;
        let txt_mlp_out = self.txt_mlp.forward(&txt_modulated2)?;
        let gated_txt_mlp = txt_mod2.gate(&txt_mlp_out)?;

        // Residual add in F32
        let encoder_hidden_states = (encoder_hidden_states.to_dtype(DType::F32)?
            + gated_txt_mlp.to_dtype(DType::F32)?)?
        .to_dtype(orig_dtype)?;

        Ok((encoder_hidden_states, hidden_states))
    }
}
