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
        // GELU-approximate activation
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
fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();

    // Flatten batch dimensions for efficient computation
    let mut batch_dims = q.dims().to_vec();
    batch_dims.pop();
    batch_dims.pop();
    let q = q.flatten_to(batch_dims.len() - 1)?;
    let k = k.flatten_to(batch_dims.len() - 1)?;
    let v = v.flatten_to(batch_dims.len() - 1)?;

    // Attention: softmax(Q @ K^T / sqrt(d)) @ V
    let attn_weights = (q.matmul(&k.t()?)? * scale_factor)?;
    let attn_scores = candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(&v)?;

    // Restore original batch dimensions
    batch_dims.push(attn_scores.dim(D::Minus2)?);
    batch_dims.push(attn_scores.dim(D::Minus1)?);
    attn_scores.reshape(batch_dims)
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

        // Transpose for attention: [batch, heads, seq, head_dim]
        let img_q = img_q.transpose(1, 2)?;
        let img_k = img_k.transpose(1, 2)?;
        let img_v = img_v.transpose(1, 2)?;
        let txt_q = txt_q.transpose(1, 2)?;
        let txt_k = txt_k.transpose(1, 2)?;
        let txt_v = txt_v.transpose(1, 2)?;

        // Concatenate for joint attention: order is [text, image]
        let joint_q = Tensor::cat(&[&txt_q, &img_q], 2)?;
        let joint_k = Tensor::cat(&[&txt_k, &img_k], 2)?;
        let joint_v = Tensor::cat(&[&txt_v, &img_v], 2)?;

        // Compute attention
        let joint_attn = scaled_dot_product_attention(&joint_q, &joint_k, &joint_v)?;

        // Reshape: [batch, heads, seq, head_dim] -> [batch, seq, heads * head_dim]
        let joint_attn = joint_attn.transpose(1, 2)?.flatten_from(2)?;
        let joint_attn = joint_attn.to_dtype(hidden_states.dtype())?;

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

    /// Forward pass through the dual-stream block.
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
        // Get modulation parameters for both streams
        let (img_mod1, img_mod2) = self.img_mod.forward(temb)?;
        let (txt_mod1, txt_mod2) = self.txt_mod.forward(temb)?;

        // === Attention phase ===

        // Image: norm1 + modulate
        let img_normed = hidden_states.apply(&self.img_norm1)?;
        let img_modulated = img_mod1.scale_shift(&img_normed)?;

        // Text: norm1 + modulate
        let txt_normed = encoder_hidden_states.apply(&self.txt_norm1)?;
        let txt_modulated = txt_mod1.scale_shift(&txt_normed)?;

        // Joint attention
        let (img_attn_out, txt_attn_out) =
            self.attn
                .forward(&img_modulated, &txt_modulated, image_rotary_emb)?;

        // Gated residual
        let hidden_states = (hidden_states + img_mod1.gate(&img_attn_out)?)?;
        let encoder_hidden_states = (encoder_hidden_states + txt_mod1.gate(&txt_attn_out)?)?;

        // === MLP phase ===

        // Image: norm2 + modulate + MLP + gated residual
        let img_normed2 = hidden_states.apply(&self.img_norm2)?;
        let img_modulated2 = img_mod2.scale_shift(&img_normed2)?;
        let img_mlp_out = self.img_mlp.forward(&img_modulated2)?;
        let hidden_states = (&hidden_states + img_mod2.gate(&img_mlp_out)?)?;

        // Text: norm2 + modulate + MLP + gated residual
        let txt_normed2 = encoder_hidden_states.apply(&self.txt_norm2)?;
        let txt_modulated2 = txt_mod2.scale_shift(&txt_normed2)?;
        let txt_mlp_out = self.txt_mlp.forward(&txt_modulated2)?;
        let encoder_hidden_states = (&encoder_hidden_states + txt_mod2.gate(&txt_mlp_out)?)?;

        // Clip for fp16 stability
        let (hidden_states, encoder_hidden_states) =
            if hidden_states.dtype() == DType::F16 || hidden_states.dtype() == DType::BF16 {
                let hidden_states = hidden_states.clamp(-65504f32, 65504f32)?;
                let encoder_hidden_states = encoder_hidden_states.clamp(-65504f32, 65504f32)?;
                (hidden_states, encoder_hidden_states)
            } else {
                (hidden_states, encoder_hidden_states)
            };

        Ok((encoder_hidden_states, hidden_states))
    }
}
