//! FastConformer Encoder for Parakeet TDT
//!
//! FastConformer is ~2.4x faster than standard Conformer due to:
//! - 8x depthwise conv subsampling (vs 4x standard)
//! - Depthwise separable convolutions
//! - Reduced kernel size (9 vs 31)

use candle::{Device, IndexOp, Module, Result, Tensor};
use candle_nn::{
    batch_norm, linear_no_bias, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, LayerNorm,
    Linear, ModuleT, VarBuilder,
};

use super::EncoderConfig;

fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    let weight = vb.get(size, "weight")?;
    let bias = vb.get(size, "bias")?;
    Ok(LayerNorm::new(weight, bias, eps))
}

fn linear_with_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    let bias = vb.get(out_dim, "bias")?;
    Ok(Linear::new(weight, Some(bias)))
}

/// GLU activation - splits input along dim and applies sigmoid gate
fn glu(x: &Tensor, dim: usize) -> Result<Tensor> {
    let chunks = x.chunk(2, dim)?;
    &chunks[0] * candle_nn::ops::sigmoid(&chunks[1])?
}

/// Convolutional subsampling module using Conv2d (NeMo style)
/// Uses depthwise striding with 8x downsampling
#[derive(Debug, Clone)]
pub struct ConvSubsampling {
    // NeMo uses: conv.0, conv.2(dw), conv.3(pw), conv.5(dw), conv.6(pw)
    conv0: Conv2d,       // [256, 1, 3, 3] - first conv
    conv2: Conv2d,       // [256, 1, 3, 3] - depthwise
    conv3: Conv2d,       // [256, 256, 1, 1] - pointwise
    conv5: Conv2d,       // [256, 1, 3, 3] - depthwise
    conv6: Conv2d,       // [256, 256, 1, 1] - pointwise
    out_weight: Tensor,  // [1024, 4096]
    out_bias: Tensor,    // [1024]
}

impl ConvSubsampling {
    pub fn load(vb: VarBuilder, cfg: &EncoderConfig) -> Result<Self> {
        let channels = cfg.subsampling_conv_channels; // 256

        // First conv: [256, 1, 3, 3] with stride 2
        let cfg_stride2 = Conv2dConfig {
            padding: 1,
            stride: 2,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let conv0 = {
            let weight = vb.pp("conv.0").get((channels, 1, 3, 3), "weight")?;
            let bias = vb.pp("conv.0").get(channels, "bias")?;
            Conv2d::new(weight, Some(bias), cfg_stride2)
        };

        // Depthwise conv: [256, 1, 3, 3] with groups=256, stride 2
        let cfg_dw_stride2 = Conv2dConfig {
            padding: 1,
            stride: 2,
            dilation: 1,
            groups: channels,
            cudnn_fwd_algo: None,
        };
        let conv2 = {
            let weight = vb.pp("conv.2").get((channels, 1, 3, 3), "weight")?;
            let bias = vb.pp("conv.2").get(channels, "bias")?;
            Conv2d::new(weight, Some(bias), cfg_dw_stride2)
        };

        // Pointwise conv: [256, 256, 1, 1]
        let cfg_pw = Conv2dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let conv3 = {
            let weight = vb.pp("conv.3").get((channels, channels, 1, 1), "weight")?;
            let bias = vb.pp("conv.3").get(channels, "bias")?;
            Conv2d::new(weight, Some(bias), cfg_pw)
        };

        // Depthwise conv: [256, 1, 3, 3] with groups=256, stride 2
        let conv5 = {
            let weight = vb.pp("conv.5").get((channels, 1, 3, 3), "weight")?;
            let bias = vb.pp("conv.5").get(channels, "bias")?;
            Conv2d::new(weight, Some(bias), cfg_dw_stride2)
        };

        // Pointwise conv: [256, 256, 1, 1]
        let conv6 = {
            let weight = vb.pp("conv.6").get((channels, channels, 1, 1), "weight")?;
            let bias = vb.pp("conv.6").get(channels, "bias")?;
            Conv2d::new(weight, Some(bias), cfg_pw)
        };

        // Output projection: [hidden_size, flattened_features]
        // flattened_features = (num_mel_bins / subsampling_factor) * channels
        // XL (128 mel bins): 16 * 256 = 4096
        // XXL (80 mel bins): 10 * 256 = 2560
        let flattened_features = (cfg.num_mel_bins / cfg.subsampling_factor) * channels;
        let out_weight = vb.pp("out").get((cfg.hidden_size, flattened_features), "weight")?;
        let out_bias = vb.pp("out").get(cfg.hidden_size, "bias")?;

        Ok(Self {
            conv0,
            conv2,
            conv3,
            conv5,
            conv6,
            out_weight,
            out_bias,
        })
    }
}

impl Module for ConvSubsampling {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, 1, mel_bins, time] from caller
        // NeMo expects [batch, 1, time, mel_bins] - transpose the spatial dims
        let x = x.transpose(2, 3)?; // [batch, 1, time, mel_bins]

        // NeMo activation pattern:
        //   conv0 → ReLU → conv2(dw) → conv3(pw) → ReLU → conv5(dw) → conv6(pw) → ReLU
        // Note: NO ReLU between depthwise and pointwise convs!
        let x = self.conv0.forward(&x)?.relu()?;
        let x = self.conv2.forward(&x)?; // No ReLU after depthwise
        let x = self.conv3.forward(&x)?.relu()?; // ReLU after pointwise
        let x = self.conv5.forward(&x)?; // No ReLU after depthwise
        let x = self.conv6.forward(&x)?.relu()?; // ReLU after pointwise

        // x: [batch, 256, time/8, mel_bins/8]
        // Reshape to [batch, time/8, 256 * mel_bins/8]
        let (b, c, t, m) = x.dims4()?;
        let x = x.permute((0, 2, 1, 3))?; // [batch, time/8, 256, mel_bins/8]
        // CRITICAL: Must make contiguous before reshape to preserve correct memory order
        let x = x.contiguous()?;
        let x = x.reshape((b, t, c * m))?; // [batch, time/8, 256 * mel_bins/8]

        // Project to hidden_size
        let out = x
            .broadcast_matmul(&self.out_weight.t()?)?
            .broadcast_add(&self.out_bias)?;

        Ok(out)
    }
}

/// Feed-forward module with Macaron-style 0.5 scaling
#[derive(Debug, Clone)]
pub struct FeedForward {
    layer_norm: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    scale: f64,
}

impl FeedForward {
    pub fn load(vb: VarBuilder, cfg: &EncoderConfig, scale: f64) -> Result<Self> {
        let layer_norm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("norm"))?;
        let linear1 = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("linear1"))?;
        let linear2 = linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("linear2"))?;

        Ok(Self {
            layer_norm,
            linear1,
            linear2,
            scale,
        })
    }
}

impl FeedForward {
    /// Forward with optional debug output
    pub fn forward_debug(&self, x: &Tensor, debug: bool, name: &str) -> Result<Tensor> {
        let h = self.layer_norm.forward(x)?;
        if debug {
            let ln_rms: f32 = h.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            println!("    {} LN output: rms={:.4}", name, ln_rms);
        }
        let h = self.linear1.forward(&h)?.silu()?;
        if debug {
            let l1_rms: f32 = h.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            println!("    {} Linear1+SiLU: rms={:.4}", name, l1_rms);
        }
        let h = self.linear2.forward(&h)?;
        if debug {
            let l2_rms: f32 = h.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            println!("    {} Linear2 (before scale): rms={:.4}", name, l2_rms);
        }
        // Scale and add residual
        (h * self.scale)? + x
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_debug(x, false, "FFN")
    }
}

/// Conformer convolution module with GLU and BatchNorm
/// NeMo structure: norm_conv -> pointwise1 -> GLU -> depthwise -> batchnorm -> silu -> pointwise2
#[derive(Debug, Clone)]
pub struct ConvModule {
    layer_norm: LayerNorm,
    pointwise_conv1: candle_nn::Conv1d,
    depthwise_conv: candle_nn::Conv1d,
    batch_norm: BatchNorm,
    pointwise_conv2: candle_nn::Conv1d,
}

impl ConvModule {
    pub fn forward_debug(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, time, hidden]
        let h = self.layer_norm.forward(x)?;
        let h_ln_rms: f32 = h.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;

        let h = h.transpose(1, 2)?; // [batch, hidden, time] for conv

        // Pointwise conv1
        let h = self.pointwise_conv1.forward(&h)?;
        let h_pw1_rms: f32 = h.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;

        let h = glu(&h, 1)?; // [batch, hidden, time]
        let h_glu_rms: f32 = h.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;

        // Depthwise conv
        let h = self.depthwise_conv.forward(&h)?;
        let h_dw_rms: f32 = h.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;

        let h = self.batch_norm.forward_t(&h, false)?;
        let h_bn_rms: f32 = h.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;

        let h = h.silu()?;
        let h_silu_rms: f32 = h.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;

        // Pointwise conv2
        let h = self.pointwise_conv2.forward(&h)?;
        let h_pw2_rms: f32 = h.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;

        let h = h.transpose(1, 2)?; // back to [batch, time, hidden]

        let x_rms: f32 = x.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
        let out = (h.clone() + x)?; // residual
        let out_rms: f32 = out.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;

        println!("    Conv debug: input={:.2} → ln={:.2} → pw1={:.2} → glu={:.2} → dw={:.2} → bn={:.2} → silu={:.2} → pw2={:.2} + res({:.2}) → out={:.2}",
                 x_rms, h_ln_rms, h_pw1_rms, h_glu_rms, h_dw_rms, h_bn_rms, h_silu_rms, h_pw2_rms, x_rms, out_rms);
        Ok(out)
    }

    pub fn load(vb: VarBuilder, norm_vb: VarBuilder, cfg: &EncoderConfig) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let kernel = cfg.conv_kernel_size;

        // NeMo: norm_conv is at the block level, not inside conv module
        let layer_norm = layer_norm(hidden, cfg.layer_norm_eps, norm_vb)?;

        // Pointwise conv1: [2048, 1024, 1] -> for GLU
        // Note: XXL models (CTC-1B, RNN-T-1B) have biases, XL models (TDT-v3) don't
        let pw1_vb = vb.pp("pointwise_conv1");
        let pw1_weight = pw1_vb.get((hidden * 2, hidden, 1), "weight")?;
        let pw1_bias = pw1_vb.get(hidden * 2, "bias").ok();
        let pw1_cfg = candle_nn::Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let pointwise_conv1 = candle_nn::Conv1d::new(pw1_weight, pw1_bias, pw1_cfg);

        // Depthwise conv: [1024, 1, 9] with groups=1024
        let dw_vb = vb.pp("depthwise_conv");
        let dw_weight = dw_vb.get((hidden, 1, kernel), "weight")?;
        let dw_bias = dw_vb.get(hidden, "bias").ok();
        let dw_cfg = candle_nn::Conv1dConfig {
            padding: kernel / 2,
            stride: 1,
            dilation: 1,
            groups: hidden,
            cudnn_fwd_algo: None,
        };
        let depthwise_conv = candle_nn::Conv1d::new(dw_weight, dw_bias, dw_cfg);

        // BatchNorm
        let bn_cfg = BatchNormConfig {
            eps: 1e-5,
            remove_mean: true,
            affine: true,
            momentum: 0.1,
        };
        let batch_norm = batch_norm(hidden, bn_cfg, vb.pp("batch_norm"))?;

        // Pointwise conv2: [1024, 1024, 1]
        let pw2_vb = vb.pp("pointwise_conv2");
        let pw2_weight = pw2_vb.get((hidden, hidden, 1), "weight")?;
        let pw2_bias = pw2_vb.get(hidden, "bias").ok();
        let pointwise_conv2 = candle_nn::Conv1d::new(pw2_weight, pw2_bias, pw1_cfg);

        Ok(Self {
            layer_norm,
            pointwise_conv1,
            depthwise_conv,
            batch_norm,
            pointwise_conv2,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, time, hidden]
        let h = self.layer_norm.forward(x)?;
        let h = h.transpose(1, 2)?; // [batch, hidden, time] for conv

        // Pointwise conv1
        let h = self.pointwise_conv1.forward(&h)?;
        let h = glu(&h, 1)?; // [batch, hidden, time]

        // Depthwise conv
        let h = self.depthwise_conv.forward(&h)?;
        let h = self.batch_norm.forward_t(&h, false)?.silu()?;

        // Pointwise conv2
        let h = self.pointwise_conv2.forward(&h)?;

        let h = h.transpose(1, 2)?; // back to [batch, time, hidden]
        h + x // residual
    }
}

/// Multi-head self-attention with relative position bias
#[derive(Debug, Clone)]
pub struct MultiHeadSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    pos_proj: Linear, // linear_pos for position encoding
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadSelfAttention {
    pub fn load(vb: VarBuilder, cfg: &EncoderConfig) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim();

        // NeMo uses weight-only (no bias) for attention projections
        let q_proj = linear_no_bias(hidden, hidden, vb.pp("linear_q"))?;
        let k_proj = linear_no_bias(hidden, hidden, vb.pp("linear_k"))?;
        let v_proj = linear_no_bias(hidden, hidden, vb.pp("linear_v"))?;
        let out_proj = linear_no_bias(hidden, hidden, vb.pp("linear_out"))?;
        let pos_proj = linear_no_bias(hidden, hidden, vb.pp("linear_pos"))?;

        // Learnable position biases
        let pos_bias_u = vb.get((num_heads, head_dim), "pos_bias_u")?;
        let pos_bias_v = vb.get((num_heads, head_dim), "pos_bias_v")?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            pos_proj,
            pos_bias_u,
            pos_bias_v,
            num_heads,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, pos_emb: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        self.forward_internal(x, pos_emb, mask, false)
    }

    pub fn forward_debug(&self, x: &Tensor, pos_emb: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        self.forward_internal(x, pos_emb, mask, true)
    }

    fn forward_internal(&self, x: &Tensor, pos_emb: &Tensor, mask: Option<&Tensor>, debug: bool) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let rel_len = 2 * seq_len - 1;

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        if debug {
            let q_rms: f32 = q.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            let k_rms: f32 = k.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            let v_rms: f32 = v.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            println!("    Attn Q proj: rms={:.4}", q_rms);
            println!("    Attn K proj: rms={:.4}", k_rms);
            println!("    Attn V proj: rms={:.4}", v_rms);
        }

        // Project position embedding
        // pos_emb: [1, 2*seq-1, hidden] (relative position encodings)
        if debug {
            let pos_emb_rms: f32 = pos_emb.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            let w_shape = self.pos_proj.weight().dims();
            let w_rms: f32 = self.pos_proj.weight().flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            println!("    Attn pos_emb input: shape={:?}, rms={:.4}", pos_emb.dims(), pos_emb_rms);
            println!("    Attn pos_proj weight: shape={:?}, rms={:.4}", w_shape, w_rms);
            // Expected: sqrt(in_features) * input_rms * weight_rms
            let in_features = pos_emb.dims()[2];
            let expected_out = (in_features as f64).sqrt() as f32 * pos_emb_rms * w_rms;
            println!("    Expected output rms: sqrt({}) * {:.4} * {:.4} = {:.4}", in_features, pos_emb_rms, w_rms, expected_out);
        }
        let pos = self.pos_proj.forward(pos_emb)?;

        if debug {
            let pos_proj_rms: f32 = pos.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            println!("    Attn pos_proj actual output: rms={:.4}", pos_proj_rms);
        }

        // Reshape to [batch, heads, seq, head_dim]
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Position biases: [1, heads, 1, head_dim]
        let pos_bias_u = self.pos_bias_u.unsqueeze(0)?.unsqueeze(2)?;
        let pos_bias_v = self.pos_bias_v.unsqueeze(0)?.unsqueeze(2)?;

        if debug {
            let u_rms: f32 = pos_bias_u.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            let v_rms: f32 = pos_bias_v.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            println!("    pos_bias_u: rms={:.4}", u_rms);
            println!("    pos_bias_v: rms={:.4}", v_rms);
        }

        // Content attention: (q + u) @ k^T -> [batch, heads, seq, seq]
        let q_with_u = q.broadcast_add(&pos_bias_u)?;
        let content_score = q_with_u.matmul(&k.transpose(2, 3)?)?;

        if debug {
            let content_rms: f32 = content_score.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            println!("    content_score (matrix_ac): rms={:.4}", content_rms);
        }

        // Position attention with relative encoding (NeMo-compatible):
        // pos: [batch, 2*seq-1, hidden] -> [batch, heads, head_dim, 2*seq-1]
        let q_with_v = q.broadcast_add(&pos_bias_v)?;
        let pos = pos
            .reshape((batch, rel_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?  // [batch, heads, 2*seq-1, head_dim]
            .transpose(2, 3)?; // [batch, heads, head_dim, 2*seq-1]

        // q_with_v @ pos -> [batch, heads, seq, 2*seq-1]
        let pos_score_raw = q_with_v.matmul(&pos)?;

        if debug {
            let pos_raw_rms: f32 = pos_score_raw.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            println!("    pos_score_raw (before shift): rms={:.4}", pos_raw_rms);
        }

        // Apply relative shift: [batch, heads, seq, 2*seq-1] -> [batch, heads, seq, 2*seq-1] (realigned)
        let pos_score_shifted = rel_shift(&pos_score_raw)?;

        // Truncate to first seq_len columns: [batch, heads, seq, seq]
        // This extracts the valid attention scores (NeMo: matrix_bd[:, :, :, :matrix_ac.size(-1)])
        let pos_score = pos_score_shifted.narrow(3, 0, seq_len)?;

        if debug {
            let pos_rms: f32 = pos_score.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            println!("    pos_score (matrix_bd after shift+trunc): rms={:.4}", pos_rms);
        }

        // Combine scores
        let scale = (self.head_dim as f64).sqrt();
        let scores = ((content_score + pos_score)? / scale)?;

        if debug {
            let scores_rms: f32 = scores.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            println!("    scores (combined/scaled): rms={:.4}, scale={:.4}", scores_rms, scale);
        }

        // Apply attention mask if provided
        let scores = match mask {
            Some(m) => scores.broadcast_add(m)?,
            None => scores,
        };

        // Softmax and apply to values
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let output = attn_weights.matmul(&v)?;

        if debug {
            let weights_mean: f32 = attn_weights.flatten_all()?.mean_all()?.to_scalar()?;
            let output_rms: f32 = output.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            println!("    attn_weights mean: {:.6} (expect ~{:.6})", weights_mean, 1.0 / seq_len as f32);
            println!("    output (before out_proj): rms={:.4}", output_rms);
        }

        // Reshape back and project
        let output = output
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?;
        let output = self.out_proj.forward(&output)?;

        if debug {
            let final_rms: f32 = output.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            println!("    Attn final output: rms={:.4}", final_rms);
        }

        Ok(output)
    }
}

/// Full Conformer block: FFN -> Attn -> Conv -> FFN -> LN
#[derive(Debug, Clone)]
pub struct ConformerBlock {
    ff1: FeedForward,
    self_attn: MultiHeadSelfAttention,
    self_attn_ln: LayerNorm,
    conv_module: ConvModule,
    ff2: FeedForward,
    final_ln: LayerNorm,
}

impl ConformerBlock {
    pub fn load(vb: VarBuilder, cfg: &EncoderConfig) -> Result<Self> {
        // NeMo naming: feed_forward1, norm_feed_forward1, etc.
        let ff1_norm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("norm_feed_forward1"))?;
        let ff1_linear1 = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("feed_forward1").pp("linear1"))?;
        let ff1_linear2 = linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("feed_forward1").pp("linear2"))?;

        let ff2_norm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("norm_feed_forward2"))?;
        let ff2_linear1 = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("feed_forward2").pp("linear1"))?;
        let ff2_linear2 = linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("feed_forward2").pp("linear2"))?;

        let self_attn = MultiHeadSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let self_attn_ln = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("norm_self_att"))?;

        // NeMo: norm_conv is at block level, conv module is under "conv"
        let conv_module = ConvModule::load(vb.pp("conv"), vb.pp("norm_conv"), cfg)?;
        let final_ln = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("norm_out"))?;

        Ok(Self {
            ff1: FeedForward {
                layer_norm: ff1_norm,
                linear1: ff1_linear1,
                linear2: ff1_linear2,
                scale: 0.5,
            },
            self_attn,
            self_attn_ln,
            conv_module,
            ff2: FeedForward {
                layer_norm: ff2_norm,
                linear1: ff2_linear1,
                linear2: ff2_linear2,
                scale: 0.5,
            },
            final_ln,
        })
    }

    pub fn forward(&self, x: &Tensor, pos_emb: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        self.forward_internal(x, pos_emb, mask, false, true)
    }

    pub fn forward_debug(&self, x: &Tensor, pos_emb: &Tensor, mask: Option<&Tensor>, debug: bool) -> Result<Tensor> {
        self.forward_internal(x, pos_emb, mask, debug, true)
    }

    pub fn forward_skip_final_ln(&self, x: &Tensor, pos_emb: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        self.forward_internal(x, pos_emb, mask, false, false)
    }

    fn forward_internal(&self, x: &Tensor, pos_emb: &Tensor, mask: Option<&Tensor>, debug: bool, apply_final_ln: bool) -> Result<Tensor> {
        // Macaron-Net style: FFN -> Attn -> Conv -> FFN -> LN

        // FFN1: Pre-norm, then feedforward, then scale + residual
        // NeMo: residual = residual + dropout(ff1(norm(x))) * 0.5
        let x = if debug {
            self.ff1.forward_debug(x, true, "FFN1")?
        } else {
            self.ff1.forward(x)?
        };
        if debug {
            let rms: f32 = x.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            println!("  After ff1+residual: rms={:.3}", rms);
        }

        // Self-Attention: Pre-norm on input, attention output added to residual
        // NeMo: x = norm(residual); attn = self_attn(x); residual = residual + dropout(attn)
        let attn_ln_out = self.self_attn_ln.forward(&x)?;
        if debug {
            let ln_rms: f32 = attn_ln_out.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            println!("  Attn LN output: rms={:.3}", ln_rms);
        }
        let attn = if debug {
            self.self_attn.forward_debug(&attn_ln_out, pos_emb, mask)?
        } else {
            self.self_attn.forward(&attn_ln_out, pos_emb, mask)?
        };
        if debug {
            let attn_rms: f32 = attn.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            println!("  Attn output (before residual): rms={:.3}", attn_rms);
        }
        let x = (x + attn)?;
        if debug {
            let rms: f32 = x.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            println!("  After attn+residual: rms={:.3}", rms);
        }

        let x = if debug {
            self.conv_module.forward_debug(&x)?
        } else {
            self.conv_module.forward(&x)?
        };
        if debug {
            let rms: f32 = x.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            println!("  After conv: rms={:.3}", rms);
        }

        let x = if debug {
            self.ff2.forward_debug(&x, true, "FFN2")?
        } else {
            self.ff2.forward(&x)?
        };
        if debug {
            let rms: f32 = x.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            println!("  After ff2+residual: rms={:.3}", rms);
        }

        let out = if apply_final_ln {
            // Debug: trace layer norm step by step for block 23
            if debug {
                let x_rms: f32 = x.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
                let hidden_size = x.dim(candle::D::Minus1)? as f64;

                // Compute mean
                let mean = (x.sum_keepdim(candle::D::Minus1)? / hidden_size)?;
                let x_centered = x.broadcast_sub(&mean)?;
                let centered_rms: f32 = x_centered.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;

                // Compute variance
                let var = (x_centered.sqr()?.sum_keepdim(candle::D::Minus1)? / hidden_size)?;
                let var_mean: f32 = var.flatten_all()?.mean_all()?.to_scalar()?;

                // Normalize
                let eps = 1e-5f64;
                let std = (var + eps)?.sqrt()?;
                let x_normed = x_centered.broadcast_div(&std)?;
                let normed_rms: f32 = x_normed.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;

                println!("  LayerNorm debug:");
                println!("    Input RMS: {:.4}", x_rms);
                println!("    After centering RMS: {:.4}", centered_rms);
                println!("    Variance mean: {:.4}", var_mean);
                println!("    After normalization RMS: {:.4} (should be ~1.0)", normed_rms);

                // Apply weight and bias manually to compare
                let ln_w = self.final_ln.weight();
                let ln_b = self.final_ln.bias().unwrap();
                let manual_out = x_normed.broadcast_mul(ln_w)?.broadcast_add(ln_b)?;
                let manual_rms: f32 = manual_out.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
                println!("    Manual weight+bias RMS: {:.4} (should be ~0.05)", manual_rms);

                // Print first 5 weight values
                let w_first5: Vec<f32> = ln_w.i(0..5)?.to_vec1()?;
                println!("    LN weight[:5]: {:?}", w_first5);
                // Expected: close to 0.05 each (since mean=0.05)
            }

            self.final_ln.forward(&x)?
        } else {
            x
        };

        if debug {
            let rms: f32 = out.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            // Also check layer norm weights
            let ln_weight_rms: f32 = self.final_ln.weight().sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            println!("  After final_ln: rms={:.3}, ln_weight_rms={:.3} (applied={})", rms, ln_weight_rms, apply_final_ln);
        }
        Ok(out)
    }
}

/// Sinusoidal position encoding for RELATIVE positions (NeMo-compatible)
/// Generates encodings for positions [length-1, length-2, ..., 0, -1, ..., -(length-1)]
/// This DESCENDING order is critical - it matches NeMo's implementation
///
/// CRITICAL: NeMo uses INTERLEAVED sin/cos pattern: [sin_0, cos_0, sin_1, cos_1, ...]
/// NOT concatenated [sin_0, sin_1, ..., cos_0, cos_1, ...]
fn sinusoidal_position_encoding(length: usize, d_model: usize, device: &Device) -> Result<Tensor> {
    let half_dim = d_model / 2;
    let emb_scale = -(10000.0_f64.ln()) / d_model as f64;

    // Generate 2*length - 1 positions in DESCENDING order: [length-1, length-2, ..., -(length-1)]
    // This matches NeMo: torch.arange(length - 1, -length, -1)
    let rel_length = 2 * length - 1;
    let positions: Vec<f32> = (0..rel_length)
        .map(|i| (length as i32 - 1 - i as i32) as f32)
        .collect();
    let positions = Tensor::new(positions.as_slice(), device)?.unsqueeze(1)?;

    // NeMo div_term: torch.exp(torch.arange(0, d_model, 2) * -(log(10000) / d_model))
    // This gives frequencies for dimension pairs (0,1), (2,3), (4,5), etc.
    let div_term: Vec<f32> = (0..half_dim)
        .map(|i| ((2 * i) as f64 * emb_scale).exp() as f32)
        .collect();
    let div_term = Tensor::new(div_term.as_slice(), device)?.unsqueeze(0)?;

    // angles shape: [2*seq-1, half_dim]
    let angles = positions.broadcast_mul(&div_term)?;
    let sin = angles.sin()?;
    let cos = angles.cos()?;

    // NeMo interleaves: pe[:, 0::2] = sin, pe[:, 1::2] = cos
    // Result: [sin_0, cos_0, sin_1, cos_1, ..., sin_{d/2-1}, cos_{d/2-1}]
    // We need to interleave sin and cos tensors
    let sin = sin.unsqueeze(2)?;  // [2*seq-1, half_dim, 1]
    let cos = cos.unsqueeze(2)?;  // [2*seq-1, half_dim, 1]
    let interleaved = Tensor::cat(&[&sin, &cos], 2)?;  // [2*seq-1, half_dim, 2]
    interleaved.reshape((rel_length, d_model))  // [2*seq-1, d_model]
}

/// Relative shift operation for relative position attention (NeMo-compatible)
/// Input: [batch, heads, seq, 2*seq-1]
/// Output: [batch, heads, seq, 2*seq-1] with positions realigned
/// After this, take first seq columns to get [batch, heads, seq, seq]
fn rel_shift(x: &Tensor) -> Result<Tensor> {
    let (batch, heads, seq_len, pos_len) = x.dims4()?;
    // pos_len should be 2*seq_len - 1

    // Pad with one column of zeros on the left: [batch, heads, seq, pos_len+1]
    let zero_pad = Tensor::zeros((batch, heads, seq_len, 1), x.dtype(), x.device())?;
    let x_padded = Tensor::cat(&[&zero_pad, x], 3)?;

    // Reshape to [batch, heads, pos_len+1, seq]: exposes diagonals
    let x_reshaped = x_padded.reshape((batch, heads, pos_len + 1, seq_len))?;

    // Drop first row: [batch, heads, pos_len, seq]
    let x_shifted = x_reshaped.narrow(2, 1, pos_len)?;

    // Reshape back to [batch, heads, seq, pos_len]
    x_shifted.reshape((batch, heads, seq_len, pos_len))
}

/// FastConformer Encoder - stack of Conformer blocks
#[derive(Debug, Clone)]
pub struct FastConformerEncoder {
    subsampling: ConvSubsampling,
    blocks: Vec<ConformerBlock>,
    num_mel_bins: usize,
    hidden_size: usize,
    /// Scale input by 1/sqrt(hidden_size) before conformer blocks (NeMo default: true)
    scale_input: bool,
}

impl FastConformerEncoder {
    pub fn load(vb: VarBuilder, cfg: &EncoderConfig) -> Result<Self> {
        let subsampling = ConvSubsampling::load(vb.pp("pre_encode"), cfg)?;

        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| ConformerBlock::load(vb.pp(format!("layers.{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            subsampling,
            blocks,
            num_mel_bins: cfg.num_mel_bins,
            hidden_size: cfg.hidden_size,
            scale_input: cfg.scale_input,
        })
    }

    pub fn forward(&self, mel: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // mel: [batch, features, time]
        // Add channel dimension for Conv2d: [batch, 1, features, time]
        let x = mel.unsqueeze(1)?;

        // Subsampling
        let x = self.subsampling.forward(&x)?;
        // x: [batch, time/8, hidden_size]

        // NeMo applies xscale (sqrt(d_model)) inside RelPositionalEncoding.forward() to the input.
        // When xscaling=true (default for XXL models), input is multiplied by sqrt(hidden_size).
        // TDT v3 (XL) has xscaling=false, while CTC-1B and RNN-T (XXL) have xscaling=true.
        let x = if self.scale_input {
            let scale = (self.hidden_size as f64).sqrt();
            (x * scale)?
        } else {
            x
        };

        let seq_len = x.dim(1)?;
        let device = x.device();

        // Generate RELATIVE position encoding
        // Creates 2*seq_len - 1 positions for relative distances [-(seq-1), ..., 0, ..., +(seq-1)]
        let pos_emb = sinusoidal_position_encoding(seq_len, self.hidden_size, device)?;
        let pos_emb = pos_emb.unsqueeze(0)?; // [1, 2*seq-1, hidden]

        // Apply conformer blocks
        let mut x = x;
        for block in self.blocks.iter() {
            x = block.forward(&x, &pos_emb, mask)?;
        }

        Ok(x)
    }

    /// Forward pass that returns all intermediate block outputs for comparison/substitution testing.
    ///
    /// Returns: (subsampling_output, block_outputs, final_output)
    /// - subsampling_output: Output after conv subsampling (before input scaling)
    /// - block_outputs: Vec of each block's output [block_0, block_1, ..., block_23]
    /// - final_output: Same as forward() output
    ///
    /// If substitute_block is Some(n), the block at index n will be replaced with the provided tensor.
    pub fn forward_with_block_outputs(
        &self,
        mel: &Tensor,
        mask: Option<&Tensor>,
        substitute_subsampling: Option<&Tensor>,
        substitute_block: Option<(usize, &Tensor)>,
    ) -> Result<(Tensor, Vec<Tensor>, Tensor)> {
        // mel: [batch, features, time]
        let x = mel.unsqueeze(1)?;

        // Subsampling - optionally substitute with NeMo output
        let subsampling_output = if let Some(nemo_sub) = substitute_subsampling {
            println!("SUBSTITUTION: Using NeMo subsampling output");
            nemo_sub.clone()
        } else {
            self.subsampling.forward(&x)?
        };

        // Apply input scaling if enabled (see notes in forward() method)
        let x = if self.scale_input {
            let scale = (self.hidden_size as f64).sqrt();
            (subsampling_output.clone() * scale)?
        } else {
            subsampling_output.clone()
        };

        let seq_len = x.dim(1)?;
        let device = x.device();

        // Generate RELATIVE position encoding
        let pos_emb = sinusoidal_position_encoding(seq_len, self.hidden_size, device)?;
        let pos_emb = pos_emb.unsqueeze(0)?;

        // Apply conformer blocks, collecting outputs
        let mut x = x;
        let mut block_outputs = Vec::with_capacity(self.blocks.len());

        for (i, block) in self.blocks.iter().enumerate() {
            // Check if we should substitute this block's output
            if let Some((sub_idx, sub_tensor)) = substitute_block {
                if i == sub_idx {
                    println!("SUBSTITUTION: Using NeMo block {} output", i);
                    x = sub_tensor.clone();
                    block_outputs.push(x.clone());
                    continue;
                }
            }

            // Enable debug for Block 0 (to investigate divergence) and blocks 20+
            let debug_block = i == 0 || i >= 20;
            if i == 0 {
                println!("\n=== Block 0 Detailed Debug ===");
                let input_rms: f32 = x.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
                println!("  Block 0 input: rms={:.4}", input_rms);
            }
            x = block.forward_debug(&x, &pos_emb, mask, debug_block)?;
            block_outputs.push(x.clone());
        }

        Ok((subsampling_output, block_outputs, x))
    }
}
