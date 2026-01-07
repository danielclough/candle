//! 3D Causal VAE for Qwen-Image.
//!
//! This module implements the AutoencoderKLQwenImage, a 3D causal VAE derived from
//! Wan Video VAE. Key features:
//!
//! - **Causal 3D Convolutions**: Only look at past frames (no future leakage)
//! - **Feature Caching**: Efficient iterative encoding/decoding for memory efficiency
//! - **8× Spatial Compression**: 1024×1024 image → 128×128 latent
//! - **16 Latent Channels**: Higher capacity than typical SD-VAE (4 channels)
//!
//! # Architecture
//!
//! ```text
//! Encoder: RGB [B,3,1,H,W] → Latent [B,32,1,H/8,W/8] → μ,σ [B,16,1,H/8,W/8]
//! Decoder: Latent [B,16,1,H/8,W/8] → RGB [B,3,1,H,W]
//! ```
//!
//! # Native Conv3d Support
//!
//! This implementation uses Candle's native `conv3d` operation for efficient 3D convolutions.
//! Causal temporal padding is applied manually before the convolution to ensure the model
//! only sees past frames (no future leakage).

use candle::{Device, IndexOp, Result, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder};

use super::config::VaeConfig;
use super::debug::debug_vae_tensor;

/// Cache size for temporal causality (number of frames to cache).
const CACHE_T: usize = 2;

/// 3D Causal Convolution with feature caching support.
///
/// Implements causal padding in the temporal dimension (only past frames)
/// while using symmetric padding for spatial dimensions. This uses Candle's
/// native `conv3d` operation for efficient GPU-accelerated convolutions.
///
/// # Causal Padding
///
/// Unlike standard convolutions that use symmetric padding, causal convolutions
/// only pad on the "past" side of the temporal dimension. This ensures the model
/// cannot see future frames during inference, which is critical for video generation
/// and autoregressive processing.
///
/// For a kernel with temporal size `kt` and temporal padding `p`, the causal padding is:
/// - Left (past): `2 * p` frames
/// - Right (future): `0` frames
#[derive(Debug, Clone)]
pub struct QwenImageCausalConv3d {
    /// Weight tensor: [out_channels, in_channels, kt, kh, kw]
    weight: Tensor,
    /// Optional bias: [out_channels]
    bias: Option<Tensor>,
    /// Kernel size: (temporal, height, width)
    kernel_size: (usize, usize, usize),
    /// Stride: (temporal, height, width)
    stride: (usize, usize, usize),
    /// Original padding (used to compute causal padding)
    padding: (usize, usize, usize),
}

impl QwenImageCausalConv3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight = vb.get(
            (
                out_channels,
                in_channels,
                kernel_size.0,
                kernel_size.1,
                kernel_size.2,
            ),
            "weight",
        )?;
        let bias = vb.get(out_channels, "bias").ok();
        Ok(Self {
            weight,
            bias,
            kernel_size,
            stride,
            padding,
        })
    }

    /// Forward pass with optional feature caching for causal inference.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, channels, time, height, width]
    /// * `cache_x` - Optional cached features from previous frames
    ///
    /// # Implementation Details
    ///
    /// Uses native `conv3d` with manual causal temporal padding:
    /// 1. Concatenate cached frames (if available) to reduce required padding
    /// 2. Apply causal padding: zeros on left (past), nothing on right (future)
    /// 3. Apply spatial padding symmetrically
    /// 4. Call native conv3d with padding=0 for temporal, spatial padding for h/w
    pub fn forward(&self, x: &Tensor, cache_x: Option<&Tensor>) -> Result<Tensor> {
        let (_b, _c, _t, _h, _w) = x.dims5()?;
        let device = x.device();
        let dtype = x.dtype();

        // Compute causal temporal padding: only pad on the left (past frames)
        // For causal conv, we need 2*padding on the left, 0 on the right
        let t_pad_left = 2 * self.padding.0;

        // Handle caching: if cache provided, concatenate and reduce required padding
        let (x, t_pad_left) = if let Some(cache) = cache_x {
            if t_pad_left > 0 && cache.dim(2)? > 0 {
                let cache = cache.to_device(device)?.to_dtype(dtype)?;
                let x = Tensor::cat(&[&cache, x], 2)?;
                let t_pad_left = t_pad_left.saturating_sub(cache.dim(2)?);
                (x, t_pad_left)
            } else {
                (x.clone(), t_pad_left)
            }
        } else {
            (x.clone(), t_pad_left)
        };

        let (b, c, _t_in, h_in, w_in) = x.dims5()?;

        // Apply causal temporal padding (zeros on left only)
        // NOTE: After cat, tensor may be non-contiguous which can cause issues
        // with Metal's im2col3d kernel that expects contiguous NCDHW layout
        let x = if t_pad_left > 0 {
            let zero_pad = Tensor::zeros((b, c, t_pad_left, h_in, w_in), dtype, device)?;
            Tensor::cat(&[&zero_pad, &x], 2)?.contiguous()?
        } else {
            x
        };

        // Apply spatial padding symmetrically
        let h_pad = self.padding.1;
        let w_pad = self.padding.2;
        let x = if h_pad > 0 || w_pad > 0 {
            // Pad height dimension (dim 3)
            let x = if h_pad > 0 {
                x.pad_with_zeros(3, h_pad, h_pad)?
            } else {
                x
            };
            // Pad width dimension (dim 4)
            if w_pad > 0 {
                x.pad_with_zeros(4, w_pad, w_pad)?
            } else {
                x
            }
        } else {
            x
        };

        // Use native conv3d with no padding (we've already applied it manually)
        // Native conv3d signature: conv3d(kernel, padding, stride, dilation, groups)
        let stride = self.stride;
        let dilation = (1, 1, 1);
        let groups = 1;

        let output = x.conv3d(&self.weight, (0, 0, 0), stride, dilation, groups)?;

        // Add bias if present
        match &self.bias {
            Some(bias) => {
                let bias = bias.reshape((1, bias.dim(0)?, 1, 1, 1))?;
                output.broadcast_add(&bias)
            }
            None => Ok(output),
        }
    }
}

/// RMS Normalization for VAE (channel-first).
#[derive(Debug, Clone)]
pub struct QwenImageRMSNorm {
    gamma: Tensor,
    scale: f64,
    channel_first: bool,
}

impl QwenImageRMSNorm {
    pub fn new(dim: usize, channel_first: bool, vb: VarBuilder) -> Result<Self> {
        // Try to load gamma with various shapes - the saved weights might have
        // trailing singleton dimensions for broadcasting:
        // - [dim] - flat 1D (ideal case)
        // - [dim, 1, 1, 1] - for 5D tensors in residual blocks
        // - [dim, 1, 1] - for 4D tensors in attention blocks
        let gamma = vb.get(dim, "gamma")
            .or_else(|_| vb.get((dim, 1, 1, 1), "gamma").and_then(|g| g.flatten_all()))
            .or_else(|_| vb.get((dim, 1, 1), "gamma").and_then(|g| g.flatten_all()))?;
        let scale = (dim as f64).sqrt();
        Ok(Self {
            gamma,
            scale,
            channel_first,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // L2 normalize along channel dimension
        // Matches PyTorch's F.normalize: v / max(||v||, eps)
        let dim = if self.channel_first { 1 } else { x.dims().len() - 1 };
        let eps = 1e-12; // Match PyTorch's F.normalize default

        // Compute L2 norm: sqrt(sum(x^2, dim=dim, keepdim=True))
        let norm = x.sqr()?.sum_keepdim(dim)?.sqrt()?;
        // Use max(norm, eps) instead of norm + eps to match F.normalize behavior
        let norm = norm.clamp(eps, f64::MAX)?;
        let normalized = x.broadcast_div(&norm)?;

        // Broadcast gamma appropriately
        let gamma = if self.channel_first {
            // For 5D: [1, C, 1, 1, 1]
            let ndim = x.dims().len();
            let mut shape = vec![1; ndim];
            shape[1] = self.gamma.dim(0)?;
            self.gamma.reshape(shape)?
        } else {
            self.gamma.clone()
        };

        (normalized * self.scale)?.broadcast_mul(&gamma)
    }
}

/// Residual block with causal 3D convolutions.
#[derive(Debug, Clone)]
pub struct QwenImageResidualBlock {
    norm1: QwenImageRMSNorm,
    conv1: QwenImageCausalConv3d,
    norm2: QwenImageRMSNorm,
    conv2: QwenImageCausalConv3d,
    conv_shortcut: Option<QwenImageCausalConv3d>,
}

impl QwenImageResidualBlock {
    pub fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let norm1 = QwenImageRMSNorm::new(in_dim, true, vb.pp("norm1"))?;
        let conv1 = QwenImageCausalConv3d::new(in_dim, out_dim, (3, 3, 3), (1, 1, 1), (1, 1, 1), vb.pp("conv1"))?;
        let norm2 = QwenImageRMSNorm::new(out_dim, true, vb.pp("norm2"))?;
        let conv2 = QwenImageCausalConv3d::new(out_dim, out_dim, (3, 3, 3), (1, 1, 1), (1, 1, 1), vb.pp("conv2"))?;

        let conv_shortcut = if in_dim != out_dim {
            Some(QwenImageCausalConv3d::new(
                in_dim,
                out_dim,
                (1, 1, 1),
                (1, 1, 1),
                (0, 0, 0),
                vb.pp("conv_shortcut"),
            )?)
        } else {
            None
        };

        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            conv_shortcut,
        })
    }

    pub fn forward(&self, x: &Tensor, cache: Option<&mut Vec<Option<Tensor>>>, cache_idx: &mut usize) -> Result<Tensor> {
        // Shortcut
        let h = if let Some(conv) = &self.conv_shortcut {
            conv.forward(x, None)?
        } else {
            x.clone()
        };

        // First norm + silu + conv
        let x = self.norm1.forward(x)?;
        let x = x.silu()?;

        let x = if let Some(cache) = cache.as_ref() {
            let cache_tensor = cache.get(*cache_idx).and_then(|c| c.as_ref());
            let out = self.conv1.forward(&x, cache_tensor)?;
            *cache_idx += 1;
            out
        } else {
            self.conv1.forward(&x, None)?
        };

        // Second norm + silu + conv
        let x = self.norm2.forward(&x)?;
        let x = x.silu()?;

        let x = if let Some(cache) = cache.as_ref() {
            let cache_tensor = cache.get(*cache_idx).and_then(|c| c.as_ref());
            let out = self.conv2.forward(&x, cache_tensor)?;
            *cache_idx += 1;
            out
        } else {
            self.conv2.forward(&x, None)?
        };

        // Residual
        x + h
    }
}

/// Spatial self-attention block.
#[derive(Debug, Clone)]
pub struct QwenImageAttentionBlock {
    norm: QwenImageRMSNorm,
    to_qkv: Conv2d,
    proj: Conv2d,
    dim: usize,
}

impl QwenImageAttentionBlock {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let norm = QwenImageRMSNorm::new(dim, true, vb.pp("norm"))?;

        // 1x1 convolutions for QKV and output projection
        let to_qkv_weight = vb.get((dim * 3, dim, 1, 1), "to_qkv.weight")?;
        let to_qkv_bias = vb.get(dim * 3, "to_qkv.bias").ok();
        let to_qkv = Conv2d::new(to_qkv_weight, to_qkv_bias, Default::default());

        let proj_weight = vb.get((dim, dim, 1, 1), "proj.weight")?;
        let proj_bias = vb.get(dim, "proj.bias").ok();
        let proj = Conv2d::new(proj_weight, proj_bias, Default::default());

        Ok(Self { norm, to_qkv, proj, dim })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = x.clone();
        let (b, c, t, h, w) = x.dims5()?;

        // Process each frame independently
        // Reshape to [batch*time, channels, height, width]
        let x = x.permute([0, 2, 1, 3, 4])?.reshape((b * t, c, h, w))?;
        let x = self.norm.forward(&x)?;
        debug_vae_tensor("attn after norm", &x);

        // Compute QKV
        let qkv = self.to_qkv.forward(&x)?;
        debug_vae_tensor("attn qkv", &qkv);

        // Reshape for attention: [batch*time, 1, h*w, 3*dim] -> split to q, k, v
        let qkv = qkv.reshape((b * t, 1, c * 3, h * w))?;
        let qkv = qkv.permute([0, 1, 3, 2])?; // [batch*time, 1, h*w, 3*dim]

        let chunks = qkv.chunk(3, D::Minus1)?;
        let (q, k, v) = (&chunks[0], &chunks[1], &chunks[2]);
        debug_vae_tensor("attn Q", q);
        debug_vae_tensor("attn K", k);
        debug_vae_tensor("attn V", v);

        // Scaled dot-product attention
        // NOTE: k.t() does a FULL transpose (reverses all dims), which is wrong for 4D tensors!
        // We need to transpose only the last two dimensions: [B*T, 1, H*W, C] -> [B*T, 1, C, H*W]
        let scale = (self.dim as f64).sqrt();
        let k_t = k.transpose(D::Minus2, D::Minus1)?;
        debug_vae_tensor("attn K^T", &k_t);
        let attn = (q.matmul(&k_t)? / scale)?;
        debug_vae_tensor("attn weights (pre-softmax)", &attn);
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        debug_vae_tensor("attn weights (post-softmax)", &attn);
        let out = attn.matmul(v)?;
        debug_vae_tensor("attn output", &out);

        // Reshape back
        let out = out.squeeze(1)?.permute([0, 2, 1])?; // [batch*time, dim, h*w]
        let out = out.reshape((b * t, c, h, w))?;

        // Output projection
        let out = self.proj.forward(&out)?;
        debug_vae_tensor("attn proj output", &out);

        // Reshape back to 5D and add residual
        let out = out.reshape((b, t, c, h, w))?.permute([0, 2, 1, 3, 4])?;

        out + identity
    }
}

/// Middle block with residual + attention.
#[derive(Debug, Clone)]
pub struct QwenImageMidBlock {
    resnets: Vec<QwenImageResidualBlock>,
    attentions: Vec<QwenImageAttentionBlock>,
}

impl QwenImageMidBlock {
    pub fn new(dim: usize, num_layers: usize, vb: VarBuilder) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers + 1);
        let mut attentions = Vec::with_capacity(num_layers);

        // First resnet
        resnets.push(QwenImageResidualBlock::new(dim, dim, vb.pp("resnets.0"))?);

        // Interleaved attention + resnet
        for i in 0..num_layers {
            attentions.push(QwenImageAttentionBlock::new(dim, vb.pp(format!("attentions.{}", i)))?);
            resnets.push(QwenImageResidualBlock::new(dim, dim, vb.pp(format!("resnets.{}", i + 1)))?);
        }

        Ok(Self { resnets, attentions })
    }

    pub fn forward(&self, x: &Tensor, mut cache: Option<&mut Vec<Option<Tensor>>>, cache_idx: &mut usize) -> Result<Tensor> {
        let mut x = self.resnets[0].forward(x, cache.as_deref_mut(), cache_idx)?;

        for (attn, resnet) in self.attentions.iter().zip(self.resnets[1..].iter()) {
            x = attn.forward(&x)?;
            x = resnet.forward(&x, cache.as_deref_mut(), cache_idx)?;
        }

        Ok(x)
    }
}

/// 3D Encoder for the VAE.
#[derive(Debug, Clone)]
pub struct QwenImageEncoder3d {
    conv_in: QwenImageCausalConv3d,
    down_blocks: Vec<DownBlock>,
    mid_block: QwenImageMidBlock,
    norm_out: QwenImageRMSNorm,
    conv_out: QwenImageCausalConv3d,
}

/// Downsampling block types.
#[derive(Debug, Clone)]
enum DownBlock {
    Residual(QwenImageResidualBlock),
    Downsample2D(Downsample2D),
    Downsample3D(Downsample3D),
}

#[derive(Debug, Clone)]
struct Downsample2D {
    conv: Conv2d,
}

impl Downsample2D {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let w = vb.get((dim, dim, 3, 3), "resample.1.weight")?;
        let b = vb.get(dim, "resample.1.bias").ok();
        let conv = Conv2d::new(
            w,
            b,
            Conv2dConfig {
                stride: 2,
                padding: 0, // We'll do manual padding
                ..Default::default()
            },
        );
        Ok(Self { conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;

        // Reshape to process each frame
        let x = x.permute([0, 2, 1, 3, 4])?.reshape((b * t, c, h, w))?;

        // ZeroPad2d: (0, 1, 0, 1) - pad right and bottom by 1
        let x = x.pad_with_zeros(D::Minus1, 0, 1)?; // width
        let x = x.pad_with_zeros(D::Minus2, 0, 1)?; // height

        // Apply conv
        let x = self.conv.forward(&x)?;

        // Reshape back
        let (_, c_out, h_out, w_out) = x.dims4()?;
        x.reshape((b, t, c_out, h_out, w_out))?.permute([0, 2, 1, 3, 4])
    }
}

#[derive(Debug, Clone)]
struct Downsample3D {
    conv_2d: Downsample2D,
    time_conv: QwenImageCausalConv3d,
}

impl Downsample3D {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let conv_2d = Downsample2D::new(dim, vb.clone())?;
        // Temporal padding of 1 ensures causal padding of 2 (t_pad_left = 2 * padding.0)
        // This is required for single-frame images where T=1, allowing the kernel (3,1,1)
        // to have enough padded input to produce valid output.
        let time_conv = QwenImageCausalConv3d::new(
            dim, dim, (3, 1, 1), (2, 1, 1), (1, 0, 0), vb.pp("time_conv")
        )?;
        Ok(Self { conv_2d, time_conv })
    }

    fn forward(&self, x: &Tensor, cache: Option<&Tensor>) -> Result<Tensor> {
        // First spatial downsample
        let x = self.conv_2d.forward(x)?;
        // Then temporal downsample
        self.time_conv.forward(&x, cache)
    }
}

impl QwenImageEncoder3d {
    pub fn new(config: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let dim = config.base_dim;
        let z_dim = config.z_dim;

        // Input conv
        let conv_in = QwenImageCausalConv3d::new(
            config.input_channels,
            dim,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            vb.pp("conv_in"),
        )?;

        // Build dimension progression
        let dims: Vec<usize> = std::iter::once(dim)
            .chain(config.dim_mult.iter().map(|&m| dim * m))
            .collect();

        // Down blocks
        let mut down_blocks = Vec::new();
        let vb_down = vb.pp("down_blocks");
        let mut block_idx = 0;

        for i in 0..(dims.len() - 1) {
            let in_dim = dims[i];
            let out_dim = dims[i + 1];

            // Residual blocks
            for j in 0..config.num_res_blocks {
                let block_in = if j == 0 { in_dim } else { out_dim };
                down_blocks.push(DownBlock::Residual(QwenImageResidualBlock::new(
                    block_in,
                    out_dim,
                    vb_down.pp(block_idx),
                )?));
                block_idx += 1;
            }

            // Downsample (not on last block)
            if i < dims.len() - 2 {
                if config.temporal_downsample[i] {
                    down_blocks.push(DownBlock::Downsample3D(Downsample3D::new(
                        out_dim,
                        vb_down.pp(block_idx),
                    )?));
                } else {
                    down_blocks.push(DownBlock::Downsample2D(Downsample2D::new(
                        out_dim,
                        vb_down.pp(block_idx),
                    )?));
                }
                block_idx += 1;
            }
        }

        // Mid block
        let out_dim = *dims.last().unwrap();
        let mid_block = QwenImageMidBlock::new(out_dim, 1, vb.pp("mid_block"))?;

        // Output
        let norm_out = QwenImageRMSNorm::new(out_dim, true, vb.pp("norm_out"))?;
        let conv_out = QwenImageCausalConv3d::new(
            out_dim,
            z_dim * 2, // Mean and log_var
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            vb.pp("conv_out"),
        )?;

        Ok(Self {
            conv_in,
            down_blocks,
            mid_block,
            norm_out,
            conv_out,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_in.forward(x, None)?;

        for block in &self.down_blocks {
            x = match block {
                DownBlock::Residual(res) => res.forward(&x, None, &mut 0)?,
                DownBlock::Downsample2D(down) => down.forward(&x)?,
                DownBlock::Downsample3D(down) => down.forward(&x, None)?,
            };
        }

        x = self.mid_block.forward(&x, None, &mut 0)?;

        x = self.norm_out.forward(&x)?;
        x = x.silu()?;
        self.conv_out.forward(&x, None)
    }
}

/// 3D Decoder for the VAE.
#[derive(Debug, Clone)]
pub struct QwenImageDecoder3d {
    conv_in: QwenImageCausalConv3d,
    mid_block: QwenImageMidBlock,
    up_blocks: Vec<UpBlock>,
    norm_out: QwenImageRMSNorm,
    conv_out: QwenImageCausalConv3d,
}

#[derive(Debug, Clone)]
struct UpBlock {
    resnets: Vec<QwenImageResidualBlock>,
    upsampler: Option<Upsample>,
}

#[derive(Debug, Clone)]
enum Upsample {
    Upsample2D(Upsample2D),
    Upsample3D(Upsample3D),
}

#[derive(Debug, Clone)]
struct Upsample2D {
    conv: Conv2d,
}

impl Upsample2D {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let w = vb.get((dim / 2, dim, 3, 3), "resample.1.weight")?;
        let b = vb.get(dim / 2, "resample.1.bias").ok();
        let conv = Conv2d::new(
            w,
            b,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
        );
        Ok(Self { conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;

        // Reshape to process each frame
        let x = x.permute([0, 2, 1, 3, 4])?.reshape((b * t, c, h, w))?;

        // Upsample 2x
        let x = x.upsample_nearest2d(h * 2, w * 2)?;

        // Apply conv
        let x = self.conv.forward(&x)?;

        // Reshape back
        let (_, c_out, h_out, w_out) = x.dims4()?;
        x.reshape((b, t, c_out, h_out, w_out))?.permute([0, 2, 1, 3, 4])
    }
}

#[derive(Debug, Clone)]
struct Upsample3D {
    upsample_2d: Upsample2D,
    time_conv: QwenImageCausalConv3d,
}

impl Upsample3D {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let upsample_2d = Upsample2D::new(dim, vb.clone())?;
        let time_conv = QwenImageCausalConv3d::new(
            dim, dim * 2, (3, 1, 1), (1, 1, 1), (1, 0, 0), vb.pp("time_conv")
        )?;
        Ok(Self { upsample_2d, time_conv })
    }

    fn forward(&self, x: &Tensor, cache: Option<&Tensor>) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;

        // Temporal upsampling is ONLY done when using caching (frame-by-frame processing)
        // When cache is None, we skip temporal upsample and only do spatial upsample
        // This matches PyTorch's behavior in QwenImageResample.forward()
        let x = if cache.is_some() {
            // Time upsample: apply time_conv then interleave
            let x_time = self.time_conv.forward(x, cache)?;

            // x_time is [b, 2*c, t, h, w] - split and interleave
            let x_time = x_time.reshape((b, 2, c, t, h, w))?;
            let x0 = x_time.i((.., 0, .., .., .., ..))?;
            let x1 = x_time.i((.., 1, .., .., .., ..))?;

            // Interleave: [b, c, t, h, w] + [b, c, t, h, w] -> [b, c, 2*t, h, w]
            let x = Tensor::stack(&[x0, x1], 3)?; // [b, c, t, 2, h, w]
            x.reshape((b, c, t * 2, h, w))?
        } else {
            // No caching - skip temporal upsample entirely
            x.clone()
        };

        // Spatial upsample
        self.upsample_2d.forward(&x)
    }
}

impl QwenImageDecoder3d {
    pub fn new(config: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let dim = config.base_dim;
        let z_dim = config.z_dim;
        let temporal_upsample = config.temporal_upsample();

        // Dimension progression (reversed from encoder)
        let dim_mult_rev: Vec<usize> = config.dim_mult.iter().rev().copied().collect();
        let mut dims: Vec<usize> = vec![dim * dim_mult_rev[0]];
        for &m in &dim_mult_rev {
            dims.push(dim * m);
        }

        // Input conv
        let conv_in = QwenImageCausalConv3d::new(
            z_dim,
            dims[0],
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            vb.pp("conv_in"),
        )?;

        // Mid block
        let mid_block = QwenImageMidBlock::new(dims[0], 1, vb.pp("mid_block"))?;

        // Up blocks
        let mut up_blocks = Vec::new();
        let vb_up = vb.pp("up_blocks");

        for i in 0..(dims.len() - 1) {
            let in_dim = if i == 0 { dims[i] } else { dims[i] / 2 };
            let out_dim = dims[i + 1];

            // Residual blocks
            let mut resnets = Vec::new();
            for j in 0..(config.num_res_blocks + 1) {
                let block_in = if j == 0 { in_dim } else { out_dim };
                resnets.push(QwenImageResidualBlock::new(
                    block_in,
                    out_dim,
                    vb_up.pp(format!("{}.resnets.{}", i, j)),
                )?);
            }

            // Upsampler (not on last block)
            let upsampler = if i < dims.len() - 2 {
                if temporal_upsample[i] {
                    Some(Upsample::Upsample3D(Upsample3D::new(
                        out_dim,
                        vb_up.pp(format!("{}.upsamplers.0", i)),
                    )?))
                } else {
                    Some(Upsample::Upsample2D(Upsample2D::new(
                        out_dim,
                        vb_up.pp(format!("{}.upsamplers.0", i)),
                    )?))
                }
            } else {
                None
            };

            up_blocks.push(UpBlock { resnets, upsampler });
        }

        // Output
        let out_dim = *dims.last().unwrap();
        let norm_out = QwenImageRMSNorm::new(out_dim, true, vb.pp("norm_out"))?;
        let conv_out = QwenImageCausalConv3d::new(
            out_dim,
            config.input_channels,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            vb.pp("conv_out"),
        )?;

        Ok(Self {
            conv_in,
            mid_block,
            up_blocks,
            norm_out,
            conv_out,
        })
    }

    pub fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_in.forward(z, None)?;
        debug_vae_tensor("after conv_in", &x);

        x = self.mid_block.forward(&x, None, &mut 0)?;
        debug_vae_tensor("after mid_block", &x);

        for (i, block) in self.up_blocks.iter().enumerate() {
            for (j, resnet) in block.resnets.iter().enumerate() {
                x = resnet.forward(&x, None, &mut 0)?;
                if j == 0 {
                    debug_vae_tensor(&format!("after up_block[{}].resnet[0]", i), &x);
                }
            }
            if let Some(upsampler) = &block.upsampler {
                x = match upsampler {
                    Upsample::Upsample2D(up) => up.forward(&x)?,
                    Upsample::Upsample3D(up) => up.forward(&x, None)?,
                };
                debug_vae_tensor(&format!("after up_block[{}].upsample", i), &x);
            }
        }

        x = self.norm_out.forward(&x)?;
        debug_vae_tensor("after norm_out", &x);

        x = x.silu()?;
        let x = self.conv_out.forward(&x, None)?;
        debug_vae_tensor("after conv_out (pre-clamp)", &x);

        // Clamp output to [-1, 1]
        x.clamp(-1.0, 1.0)
    }
}

/// Diagonal Gaussian distribution for VAE latent space.
#[derive(Debug, Clone)]
pub struct DiagonalGaussianDistribution {
    mean: Tensor,
    logvar: Tensor,
    std: Tensor,
}

impl DiagonalGaussianDistribution {
    pub fn new(parameters: &Tensor) -> Result<Self> {
        let chunks = parameters.chunk(2, 1)?;
        let mean = chunks[0].clone();
        let logvar = chunks[1].clamp(-30.0, 20.0)?;
        let std = (&logvar * 0.5)?.exp()?;
        Ok(Self { mean, logvar, std })
    }

    /// Sample from the distribution (reparameterization trick).
    pub fn sample(&self, device: &Device) -> Result<Tensor> {
        let noise = Tensor::randn(0f32, 1f32, self.mean.dims(), device)?;
        let noise = noise.to_dtype(self.mean.dtype())?;
        &self.mean + &self.std.broadcast_mul(&noise)?
    }

    /// Return the mode (mean) of the distribution.
    pub fn mode(&self) -> &Tensor {
        &self.mean
    }
}

/// Tiled decoding configuration for memory-efficient VAE decode.
#[derive(Debug, Clone, Copy)]
pub struct TiledDecodeConfig {
    /// Minimum tile size in pixel space (default: 256)
    pub tile_sample_min_size: usize,
    /// Stride between tiles in pixel space (default: 192, overlap = 64)
    pub tile_sample_stride: usize,
}

impl Default for TiledDecodeConfig {
    fn default() -> Self {
        Self {
            tile_sample_min_size: 256,
            tile_sample_stride: 192, // overlap = 256 - 192 = 64 pixels
        }
    }
}

/// Full VAE: AutoencoderKLQwenImage.
#[derive(Debug, Clone)]
pub struct AutoencoderKLQwenImage {
    encoder: QwenImageEncoder3d,
    decoder: QwenImageDecoder3d,
    quant_conv: QwenImageCausalConv3d,
    post_quant_conv: QwenImageCausalConv3d,
    config: VaeConfig,
}

impl AutoencoderKLQwenImage {
    pub fn new(config: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = QwenImageEncoder3d::new(config, vb.pp("encoder"))?;
        let decoder = QwenImageDecoder3d::new(config, vb.pp("decoder"))?;

        let quant_conv = QwenImageCausalConv3d::new(
            config.z_dim * 2,
            config.z_dim * 2,
            (1, 1, 1),
            (1, 1, 1),
            (0, 0, 0),
            vb.pp("quant_conv"),
        )?;

        let post_quant_conv = QwenImageCausalConv3d::new(
            config.z_dim,
            config.z_dim,
            (1, 1, 1),
            (1, 1, 1),
            (0, 0, 0),
            vb.pp("post_quant_conv"),
        )?;

        Ok(Self {
            encoder,
            decoder,
            quant_conv,
            post_quant_conv,
            config: config.clone(),
        })
    }

    /// Encode an image to latent distribution.
    pub fn encode(&self, x: &Tensor) -> Result<DiagonalGaussianDistribution> {
        let h = self.encoder.forward(x)?;
        let h = self.quant_conv.forward(&h, None)?;
        DiagonalGaussianDistribution::new(&h)
    }

    /// Decode latents to an image.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let z = self.post_quant_conv.forward(z, None)?;
        self.decoder.forward(&z)
    }

    /// Decode latents to a single image frame.
    ///
    /// For single-frame image generation, extracts frame 0 from the temporal dimension.
    /// This matches the PyTorch pipeline behavior: `vae.decode(z)[0][:, :, 0]`
    pub fn decode_image(&self, z: &Tensor) -> Result<Tensor> {
        let decoded = self.decode(z)?;
        // Shape: [B, C, T, H, W] -> [B, C, H, W] by taking first frame
        let (b, c, _t, h, w) = decoded.dims5()?;
        decoded.i((.., .., 0, .., ..))?.reshape((b, c, h, w))
    }

    /// Normalize latents before transformer (training normalization).
    pub fn normalize_latents(&self, latents: &Tensor) -> Result<Tensor> {
        let device = latents.device();
        let dtype = latents.dtype();

        // latents_mean: [1, 16, 1, 1, 1]
        let mean = Tensor::from_slice(&self.config.latents_mean, (1, 16, 1, 1, 1), device)?
            .to_dtype(dtype)?;
        // latents_std: [1, 16, 1, 1, 1]
        let std = Tensor::from_slice(&self.config.latents_std, (1, 16, 1, 1, 1), device)?
            .to_dtype(dtype)?;

        // normalized = (latents - mean) / std
        latents.broadcast_sub(&mean)?.broadcast_div(&std)
    }

    /// Denormalize latents after transformer (for decoding).
    pub fn denormalize_latents(&self, latents: &Tensor) -> Result<Tensor> {
        let device = latents.device();
        let dtype = latents.dtype();

        let mean = Tensor::from_slice(&self.config.latents_mean, (1, 16, 1, 1, 1), device)?
            .to_dtype(dtype)?;
        let std = Tensor::from_slice(&self.config.latents_std, (1, 16, 1, 1, 1), device)?
            .to_dtype(dtype)?;

        // denormalized = latents * std + mean
        latents.broadcast_mul(&std)?.broadcast_add(&mean)
    }

    /// Get the spatial compression ratio (8 for Qwen-Image).
    pub fn spatial_compression_ratio(&self) -> usize {
        self.config.spatial_compression_ratio()
    }

    /// Get the number of latent channels (16 for Qwen-Image).
    pub fn latent_channels(&self) -> usize {
        self.config.z_dim
    }

    /// Decode latents using tiled decoding for memory efficiency.
    ///
    /// This method splits large latents into overlapping tiles, decodes each tile
    /// separately, and blends them together to avoid seams. Use this when
    /// full decoding fails due to GPU memory constraints.
    ///
    /// # Arguments
    /// * `z` - Input latent tensor [B, C, T, H, W]
    /// * `config` - Tiled decode configuration (tile size, stride)
    ///
    /// # Returns
    /// Decoded image tensor [B, 3, T, H*8, W*8]
    pub fn tiled_decode(&self, z: &Tensor, config: &TiledDecodeConfig) -> Result<Tensor> {
        let (_, _, _num_frames, height, width) = z.dims5()?;
        let spatial_ratio = self.config.spatial_compression_ratio();

        // Convert pixel-space config to latent space
        let tile_latent_min = config.tile_sample_min_size / spatial_ratio;
        let tile_latent_stride = config.tile_sample_stride / spatial_ratio;

        // Blend extent in pixel space
        let blend_extent = config.tile_sample_min_size - config.tile_sample_stride;

        // Process tiles
        let mut rows: Vec<Vec<Tensor>> = Vec::new();

        let mut i = 0;
        while i < height {
            let mut row: Vec<Tensor> = Vec::new();
            let mut j = 0;

            while j < width {
                // Extract tile, handling boundary cases
                let tile_h = tile_latent_min.min(height - i);
                let tile_w = tile_latent_min.min(width - j);

                // Get tile from latent - need to handle case where tile is at boundary
                let tile = if tile_h == tile_latent_min && tile_w == tile_latent_min {
                    z.narrow(3, i, tile_h)?.narrow(4, j, tile_w)?
                } else {
                    // Pad the tile to minimum size for consistent decoding
                    let tile_part = z.narrow(3, i, tile_h)?.narrow(4, j, tile_w)?;
                    let device = z.device();
                    let dtype = z.dtype();
                    let b = z.dim(0)?;
                    let c = z.dim(1)?;

                    // Decode the partial tile as-is since padding zeros might cause artifacts
                    // The VAE can handle variable-sized inputs
                    let _ = (b, c, device, dtype); // suppress unused warnings
                    tile_part
                };

                // Decode this tile
                let tile_decoded = self.decode(&tile)?;
                row.push(tile_decoded);

                if j + tile_latent_min >= width {
                    break;
                }
                j += tile_latent_stride;
            }

            rows.push(row);

            if i + tile_latent_min >= height {
                break;
            }
            i += tile_latent_stride;
        }

        // Blend rows together
        let sample_height = height * spatial_ratio;
        let sample_width = width * spatial_ratio;
        let sample_stride = config.tile_sample_stride;

        let mut result_rows: Vec<Tensor> = Vec::new();

        for (i, row) in rows.iter().enumerate() {
            let mut blended_row: Vec<Tensor> = Vec::new();

            for (j, tile) in row.iter().enumerate() {
                let mut current_tile = tile.clone();

                // Blend with tile above
                if i > 0 && !result_rows.is_empty() {
                    // Get the corresponding tile from the previous row
                    let _above_row = &result_rows[result_rows.len() - 1];
                    // Blend vertically at the overlap region
                    current_tile = Self::blend_v(&rows[i - 1][j], &current_tile, blend_extent)?;
                }

                // Blend with tile to the left
                if j > 0 && !blended_row.is_empty() {
                    current_tile = Self::blend_h(&row[j - 1], &current_tile, blend_extent)?;
                }

                // Extract only the stride portion (non-overlapping part)
                let (_, _, _t, h, w) = current_tile.dims5()?;
                let extract_h = sample_stride.min(h);
                let extract_w = sample_stride.min(w);

                // For the last tiles in each direction, we need the full remaining size
                let final_h = if i + tile_latent_stride >= height {
                    h.min(sample_height - i * spatial_ratio)
                } else {
                    extract_h
                };
                let final_w = if j + tile_latent_stride >= width {
                    w.min(sample_width - j * spatial_ratio)
                } else {
                    extract_w
                };

                let extracted = current_tile.narrow(3, 0, final_h)?.narrow(4, 0, final_w)?;
                blended_row.push(extracted);
            }

            // Concatenate row horizontally
            let row_tensor = Tensor::cat(&blended_row, 4)?;
            result_rows.push(row_tensor);
        }

        // Concatenate all rows vertically
        let result = Tensor::cat(&result_rows, 3)?;

        // Trim to exact output size
        let result = result
            .narrow(3, 0, sample_height.min(result.dim(3)?))?
            .narrow(4, 0, sample_width.min(result.dim(4)?))?;

        // Clamp output to [-1, 1]
        result.clamp(-1.0, 1.0)
    }

    /// Blend two tiles vertically (top tile 'a' with bottom tile 'b').
    ///
    /// Linear interpolation in the overlap region: blend_extent pixels at the
    /// bottom of 'a' are blended with the top of 'b'.
    fn blend_v(a: &Tensor, b: &Tensor, blend_extent: usize) -> Result<Tensor> {
        let (_, _, _, h_a, _) = a.dims5()?;
        let (_, _, _, h_b, _) = b.dims5()?;
        let blend = blend_extent.min(h_a).min(h_b);

        if blend == 0 {
            return Ok(b.clone());
        }

        let device = b.device();
        let dtype = b.dtype();

        // Get the overlap regions
        let a_overlap = a.narrow(3, h_a - blend, blend)?;
        let b_overlap = b.narrow(3, 0, blend)?;

        // Create blend weights [1, 1, 1, blend, 1] for broadcasting
        let weights: Vec<f32> = (0..blend).map(|y| y as f32 / blend as f32).collect();
        let weights = Tensor::from_vec(weights, (1, 1, 1, blend, 1), device)?.to_dtype(dtype)?;

        // Blended overlap = a * (1 - weight) + b * weight
        let one_minus_w = (1.0 - &weights)?;
        let blended = (a_overlap.broadcast_mul(&one_minus_w)? + b_overlap.broadcast_mul(&weights)?)?;

        // Construct result: b with the overlap region replaced by blended values
        // For simplicity, we'll just return b with modified overlap region
        // This is done by concatenating: b_non_overlap | blended
        if blend < h_b {
            let b_rest = b.narrow(3, blend, h_b - blend)?;
            Tensor::cat(&[blended, b_rest], 3)
        } else {
            Ok(blended)
        }
    }

    /// Blend two tiles horizontally (left tile 'a' with right tile 'b').
    ///
    /// Linear interpolation in the overlap region: blend_extent pixels at the
    /// right of 'a' are blended with the left of 'b'.
    fn blend_h(a: &Tensor, b: &Tensor, blend_extent: usize) -> Result<Tensor> {
        let (_, _, _, _, w_a) = a.dims5()?;
        let (_, _, _, _, w_b) = b.dims5()?;
        let blend = blend_extent.min(w_a).min(w_b);

        if blend == 0 {
            return Ok(b.clone());
        }

        let device = b.device();
        let dtype = b.dtype();

        // Get the overlap regions
        let a_overlap = a.narrow(4, w_a - blend, blend)?;
        let b_overlap = b.narrow(4, 0, blend)?;

        // Create blend weights [1, 1, 1, 1, blend] for broadcasting
        let weights: Vec<f32> = (0..blend).map(|x| x as f32 / blend as f32).collect();
        let weights = Tensor::from_vec(weights, (1, 1, 1, 1, blend), device)?.to_dtype(dtype)?;

        // Blended overlap = a * (1 - weight) + b * weight
        let one_minus_w = (1.0 - &weights)?;
        let blended = (a_overlap.broadcast_mul(&one_minus_w)? + b_overlap.broadcast_mul(&weights)?)?;

        // Construct result: blended | b_rest
        if blend < w_b {
            let b_rest = b.narrow(4, blend, w_b - blend)?;
            Tensor::cat(&[blended, b_rest], 4)
        } else {
            Ok(blended)
        }
    }
}
