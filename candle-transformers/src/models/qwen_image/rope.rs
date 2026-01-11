//! 3D Rotary Position Embeddings for Qwen-Image.
//!
//! This module implements the RoPE (Rotary Position Embedding) system used by Qwen-Image,
//! which differs from standard implementations in several key ways:
//!
//! 1. **Complex Multiplication**: Uses true complex number multiplication rather than
//!    the real-valued cos/sin approach used by Flux and other models.
//!
//! 2. **3D Positioning**: Handles frame (temporal), height, and width dimensions
//!    with separate frequency components that are concatenated.
//!
//! 3. **Center-Aligned Coordinates**: When `scale_rope=true`, uses negative frequency
//!    indices for half of spatial positions, centering coordinates around 0.
//!
//! # Complex Number Representation
//!
//! Since Candle doesn't have native complex tensor support, we represent complex
//! numbers as tensors with shape `[..., 2]` where the last dimension contains
//! `[real, imag]` components.
//!
//! Complex multiplication: `(a + bi) × (c + di) = (ac - bd) + (ad + bc)i`

use candle::{DType, Device, IndexOp, Result, Tensor, D};

/// Maximum sequence length for precomputed frequencies.
const MAX_SEQ_LEN: usize = 4096;

/// 3D Rotary Position Embedding for Qwen-Image.
///
/// Precomputes frequency tensors for both positive and negative indices,
/// allowing efficient lookup during forward passes.
#[derive(Debug, Clone)]
pub struct QwenEmbedRope {
    _theta: usize,
    axes_dim: Vec<usize>,
    scale_rope: bool,
    /// Positive frequency cache: shape [MAX_SEQ_LEN, total_dim/2, 2]
    /// where total_dim = sum(axes_dim) and last dim is [cos, sin]
    pos_freqs: Tensor,
    /// Negative frequency cache for center-aligned positioning
    neg_freqs: Tensor,
}

impl QwenEmbedRope {
    /// Create a new RoPE embedder with precomputed frequencies.
    ///
    /// # Arguments
    /// * `theta` - Base frequency (typically 10000)
    /// * `axes_dim` - Dimensions for each axis [frame_dim, height_dim, width_dim]
    /// * `scale_rope` - Whether to use center-aligned positioning with negative frequencies
    /// * `device` - Device to create tensors on
    /// * `dtype` - Data type for frequency tensors
    pub fn new(
        theta: usize,
        axes_dim: Vec<usize>,
        scale_rope: bool,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        // Precompute positive indices: 0, 1, 2, ..., MAX_SEQ_LEN-1
        let pos_index: Vec<f32> = (0..MAX_SEQ_LEN).map(|i| i as f32).collect();
        let pos_index = Tensor::from_vec(pos_index, MAX_SEQ_LEN, device)?;

        // Precompute negative indices: -MAX_SEQ_LEN, -MAX_SEQ_LEN+1, ..., -1
        // This is equivalent to: flip(0..MAX_SEQ_LEN) * -1 - 1
        let neg_index: Vec<f32> = (0..MAX_SEQ_LEN)
            .rev()
            .map(|i| -((i + 1) as f32))
            .collect();
        let neg_index = Tensor::from_vec(neg_index, MAX_SEQ_LEN, device)?;

        // Compute frequencies for all axes concatenated
        let pos_freqs = Self::compute_all_freqs(&pos_index, &axes_dim, theta, device, dtype)?;
        let neg_freqs = Self::compute_all_freqs(&neg_index, &axes_dim, theta, device, dtype)?;

        Ok(Self {
            _theta: theta,
            axes_dim,
            scale_rope,
            pos_freqs,
            neg_freqs,
        })
    }

    /// Compute rope parameters for a single axis.
    ///
    /// Returns complex frequencies as [seq_len, dim/2, 2] where last dim is [cos, sin].
    fn rope_params(
        index: &Tensor,
        dim: usize,
        theta: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        assert!(dim.is_multiple_of(2), "RoPE dimension must be even");

        // Compute inverse frequencies: 1 / theta^(2i/dim) for i in 0..dim/2
        let inv_freq: Vec<f32> = (0..dim / 2)
            .map(|i| 1.0 / (theta as f64).powf(2.0 * i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq.clone(), (1, dim / 2), device)?;

        // freqs = outer(index, inv_freq) = index[:, None] * inv_freq[None, :]
        let index = index.unsqueeze(1)?; // [seq, 1]
        let freqs = index.broadcast_mul(&inv_freq)?; // [seq, dim/2]

        // polar(1, freqs) -> (cos(freqs), sin(freqs))
        let cos_freqs = freqs.cos()?.to_dtype(dtype)?;
        let sin_freqs = freqs.sin()?.to_dtype(dtype)?;

        // Stack to get [seq, dim/2, 2] where last dim is [cos, sin]
        Tensor::stack(&[cos_freqs, sin_freqs], D::Minus1)
    }

    /// Compute frequencies for all axes concatenated.
    ///
    /// Returns [seq_len, total_dim/2, 2] where total_dim = sum(axes_dim).
    fn compute_all_freqs(
        index: &Tensor,
        axes_dim: &[usize],
        theta: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let mut all_freqs = Vec::with_capacity(axes_dim.len());
        for &dim in axes_dim {
            let freqs = Self::rope_params(index, dim, theta, device, dtype)?;
            all_freqs.push(freqs);
        }
        // Concatenate along dim/2 axis: [seq, sum(dim_i/2), 2]
        Tensor::cat(&all_freqs, 1)
    }

    /// Compute video frequencies for a single region.
    ///
    /// # Arguments
    /// * `frame` - Number of frames (typically 1 for images)
    /// * `height` - Height in latent space (image_height / 8 / 2 for packed latents)
    /// * `width` - Width in latent space
    /// * `idx` - Region index (used to offset frame frequencies for edit mode)
    ///
    /// # Returns
    /// Tensor of shape [frame * height * width, total_dim/2, 2]
    fn compute_video_freqs(&self, frame: usize, height: usize, width: usize, idx: usize) -> Result<Tensor> {
        let seq_len = frame * height * width;

        // Compute split offsets for axis: [frame_dim/2, height_dim/2, width_dim/2]
        let half_dims: Vec<usize> = self.axes_dim.iter().map(|&d| d / 2).collect();

        // Manual split using narrow - compute cumulative offsets
        let offset0 = 0;
        let offset1 = half_dims[0];
        let offset2 = offset1 + half_dims[1];

        // Split pos_freqs by axis using narrow
        let pos_frame = self.pos_freqs.narrow(1, offset0, half_dims[0])?;
        let pos_height = self.pos_freqs.narrow(1, offset1, half_dims[1])?;
        let pos_width = self.pos_freqs.narrow(1, offset2, half_dims[2])?;

        // Split neg_freqs by axis using narrow
        let neg_height = self.neg_freqs.narrow(1, offset1, half_dims[1])?;
        let neg_width = self.neg_freqs.narrow(1, offset2, half_dims[2])?;

        // Frame frequencies: use region index to offset (matches PyTorch's idx parameter)
        // For edit mode: region 0 (noise) uses [0:frame], region 1 (image) uses [1:1+frame]
        let freqs_frame = pos_frame.narrow(0, idx, frame)?; // [frame, frame_dim/2, 2]
        let freqs_frame = freqs_frame
            .reshape((frame, 1, 1, half_dims[0], 2))?
            .broadcast_as((frame, height, width, half_dims[0], 2))?;

        // Height and width frequencies depend on scale_rope
        let (freqs_height, freqs_width) = if self.scale_rope {
            // Center-aligned: use negative freqs for first half, positive for second half
            // freqs_height = concat(neg_freqs[-(height - height/2):], pos_freqs[:height/2])
            let half_h = height / 2;
            let neg_start = MAX_SEQ_LEN - (height - half_h);
            let neg_h = neg_height.narrow(0, neg_start, height - half_h)?;
            let pos_h = pos_height.narrow(0, 0, half_h)?;
            let freqs_h = Tensor::cat(&[&neg_h, &pos_h], 0)?; // [height, height_dim/2, 2]

            let half_w = width / 2;
            let neg_start_w = MAX_SEQ_LEN - (width - half_w);
            let neg_w = neg_width.narrow(0, neg_start_w, width - half_w)?;
            let pos_w = pos_width.narrow(0, 0, half_w)?;
            let freqs_w = Tensor::cat(&[&neg_w, &pos_w], 0)?; // [width, width_dim/2, 2]

            (freqs_h, freqs_w)
        } else {
            // Standard: just take positive indices
            let freqs_h = pos_height.narrow(0, 0, height)?;
            let freqs_w = pos_width.narrow(0, 0, width)?;
            (freqs_h, freqs_w)
        };

        // Broadcast to [frame, height, width, dim/2, 2]
        let freqs_height = freqs_height
            .reshape((1, height, 1, half_dims[1], 2))?
            .broadcast_as((frame, height, width, half_dims[1], 2))?;
        let freqs_width = freqs_width
            .reshape((1, 1, width, half_dims[2], 2))?
            .broadcast_as((frame, height, width, half_dims[2], 2))?;

        // Concatenate along frequency dimension and reshape to [seq_len, total_dim/2, 2]
        let freqs = Tensor::cat(&[freqs_frame, freqs_height, freqs_width], 3)?;
        let total_half_dim: usize = half_dims.iter().sum();
        freqs
            .reshape((seq_len, total_half_dim, 2))?
            .contiguous()
    }

    /// Forward pass: compute frequencies for video and text sequences.
    ///
    /// # Arguments
    /// * `video_fhw` - List of (frame, height, width) tuples for video sequences
    /// * `txt_seq_lens` - List of text sequence lengths per batch
    ///
    /// # Returns
    /// Tuple of (video_freqs, text_freqs) tensors
    pub fn forward(
        &self,
        video_fhw: &[(usize, usize, usize)],
        txt_seq_lens: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        // Compute frequencies for all video regions (edit/layered modes may have multiple)
        let mut vid_freqs_list = Vec::with_capacity(video_fhw.len());
        let mut max_vid_index = 0usize;

        for (idx, &(frame, height, width)) in video_fhw.iter().enumerate() {
            // Pass idx to offset frame frequencies for each region (matches PyTorch behavior)
            let vid_freqs = self.compute_video_freqs(frame, height, width, idx)?;
            vid_freqs_list.push(vid_freqs);

            // Track max index for text offset
            let vid_max = if self.scale_rope {
                height.max(width) / 2
            } else {
                height.max(width)
            };
            max_vid_index = max_vid_index.max(vid_max);
        }

        // Concatenate all video region frequencies
        let vid_freqs = if vid_freqs_list.len() == 1 {
            vid_freqs_list.pop().unwrap()
        } else {
            Tensor::cat(&vid_freqs_list.iter().collect::<Vec<_>>(), 0)?
        };

        // Text frequencies start after max_vid_index
        let max_txt_len = txt_seq_lens.iter().copied().max().unwrap_or(0);
        let txt_freqs = self
            .pos_freqs
            .narrow(0, max_vid_index, max_txt_len)?
            .contiguous()?;

        Ok((vid_freqs, txt_freqs))
    }

    /// Move frequencies to a new device if needed.
    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        if self.pos_freqs.device().location() != device.location() {
            self.pos_freqs = self.pos_freqs.to_device(device)?;
            self.neg_freqs = self.neg_freqs.to_device(device)?;
        }
        Ok(())
    }
}

/// Apply rotary embeddings to input tensor using complex multiplication.
///
/// This implements the Qwen-Image RoPE which uses `use_real=False`, meaning
/// true complex multiplication rather than the real-valued approach.
///
/// # Arguments
/// * `x` - Input tensor of shape [batch, seq, heads, head_dim]
/// * `freqs_cis` - Frequency tensor of shape [seq, head_dim/2, 2] where last dim is [cos, sin]
///
/// # Returns
/// Tensor with rotary embeddings applied, same shape as input
pub fn apply_rotary_emb_qwen(x: &Tensor, freqs_cis: &Tensor) -> Result<Tensor> {
    let (b_sz, seq_len, n_heads, head_dim) = x.dims4()?;

    // Reshape x to pairs: [batch, seq, heads, head_dim/2, 2]
    let x_pairs = x.reshape((b_sz, seq_len, n_heads, head_dim / 2, 2))?;

    // Extract real and imaginary parts
    let x_real = x_pairs.i((.., .., .., .., 0))?; // [batch, seq, heads, head_dim/2]
    let x_imag = x_pairs.i((.., .., .., .., 1))?;

    // Extract cos and sin from freqs_cis: [seq, head_dim/2, 2]
    // Reshape for broadcasting: [1, seq, 1, head_dim/2]
    let freqs_cos = freqs_cis.i((.., .., 0))?.unsqueeze(0)?.unsqueeze(2)?;
    let freqs_sin = freqs_cis.i((.., .., 1))?.unsqueeze(0)?.unsqueeze(2)?;

    // Complex multiplication: (a + bi) × (c + di) = (ac - bd) + (ad + bc)i
    // where x = a + bi, freqs = c + di (c=cos, d=sin)
    let out_real = (x_real.broadcast_mul(&freqs_cos)? - x_imag.broadcast_mul(&freqs_sin)?)?; // ac - bd
    let out_imag = (x_real.broadcast_mul(&freqs_sin)? + x_imag.broadcast_mul(&freqs_cos)?)?; // ad + bc

    // Stack back to [batch, seq, heads, head_dim/2, 2] and reshape to [batch, seq, heads, head_dim]
    let out = Tensor::stack(&[&out_real, &out_imag], D::Minus1)?;
    out.reshape((b_sz, seq_len, n_heads, head_dim))
}

/// Compute sinusoidal timestep embeddings.
///
/// Creates embeddings for diffusion timesteps using sinusoidal functions,
/// similar to positional encodings in transformers.
///
/// # Arguments
/// * `timesteps` - 1D tensor of timestep values
/// * `embedding_dim` - Output embedding dimension
/// * `dtype` - Data type for output
///
/// # Returns
/// Tensor of shape [batch, embedding_dim]
pub fn timestep_embedding(timesteps: &Tensor, embedding_dim: usize, dtype: DType) -> Result<Tensor> {
    const SCALE: f64 = 1000.0;
    const MAX_PERIOD: f64 = 10000.0;

    if !embedding_dim.is_multiple_of(2) {
        candle::bail!("embedding_dim {} must be even", embedding_dim);
    }

    let device = timesteps.device();
    let timestep_dtype = timesteps.dtype();
    let half_dim = embedding_dim / 2;

    // Scale timesteps by 1000 (as in diffusion models)
    let timesteps = (timesteps * SCALE)?;

    // Compute frequencies: exp(-log(max_period) * i / half_dim) for i in 0..half_dim
    let exponent: Vec<f32> = (0..half_dim)
        .map(|i| (-(MAX_PERIOD.ln()) * i as f64 / half_dim as f64).exp() as f32)
        .collect();
    let freqs = Tensor::from_vec(exponent, (1, half_dim), device)?.to_dtype(timestep_dtype)?;

    // args = timesteps[:, None] * freqs[None, :]
    let args = timesteps.unsqueeze(1)?.broadcast_mul(&freqs)?;

    // Concatenate [cos(args), sin(args)] - matches flip_sin_to_cos=True in Python
    let emb = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?;

    emb.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_creation() -> Result<()> {
        let device = Device::Cpu;
        let rope = QwenEmbedRope::new(10000, vec![16, 56, 56], true, &device, DType::F32)?;

        // Check that frequencies were created with correct shape
        // Total dim/2 = (16 + 56 + 56) / 2 = 64
        assert_eq!(rope.pos_freqs.dims(), &[4096, 64, 2]);
        assert_eq!(rope.neg_freqs.dims(), &[4096, 64, 2]);
        Ok(())
    }

    #[test]
    fn test_video_freqs() -> Result<()> {
        let device = Device::Cpu;
        let rope = QwenEmbedRope::new(10000, vec![16, 56, 56], true, &device, DType::F32)?;

        // Test with a small latent size: 1 frame, 8x8 spatial, region index 0
        let vid_freqs = rope.compute_video_freqs(1, 8, 8, 0)?;
        assert_eq!(vid_freqs.dims(), &[64, 64, 2]); // 1*8*8 = 64 tokens
        Ok(())
    }

    #[test]
    fn test_apply_rotary() -> Result<()> {
        let device = Device::Cpu;

        // Create dummy input: [batch=2, seq=4, heads=24, head_dim=128]
        let x = Tensor::randn(0f32, 1f32, (2, 4, 24, 128), &device)?;

        // Create dummy frequencies: [seq=4, head_dim/2=64, 2]
        let freqs = Tensor::randn(0f32, 1f32, (4, 64, 2), &device)?;

        let out = apply_rotary_emb_qwen(&x, &freqs)?;
        assert_eq!(out.dims(), x.dims());
        Ok(())
    }

    #[test]
    fn test_timestep_embedding() -> Result<()> {
        let device = Device::Cpu;
        let timesteps = Tensor::from_vec(vec![0.5f32, 0.7, 0.9], 3, &device)?;
        let emb = timestep_embedding(&timesteps, 256, DType::F32)?;
        assert_eq!(emb.dims(), &[3, 256]);
        Ok(())
    }
}
