//! Qwen2.5-VL Text Decoder with Multimodal RoPE (M-RoPE).
//!
//! This module implements the Qwen2.5 language model decoder with:
//! - Grouped Query Attention (GQA)
//! - RMSNorm (no bias)
//! - SwiGLU MLP
//! - M-RoPE for 3D position encoding (temporal, height, width)
//!
//! Key difference from Qwen3-VL: No QK-normalization in attention.

use std::sync::Arc;

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{embedding, linear_b, rms_norm, Embedding, Linear, Module, RmsNorm, VarBuilder};

use super::config::Config;

// ============================================================================
// Flash Attention Support
// ============================================================================

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

#[cfg(feature = "flash-attn")]
fn flash_attn_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    window_size: usize,
) -> Result<Tensor> {
    // For causal sliding window: window_size_left = window_size, window_size_right = 0
    candle_flash_attn::flash_attn_windowed(q, k, v, softmax_scale, Some(window_size), Some(0))
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn_windowed(
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: f32,
    _: usize,
) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

// ============================================================================
// Multimodal Rotary Position Embedding (M-RoPE)
// ============================================================================

/// Multimodal Rotary Position Embedding (M-RoPE).
///
/// Unlike standard 1D RoPE, M-RoPE supports 3D position IDs for vision tokens:
/// - Temporal position (for video frames, always 0 for images)
/// - Height position (row in the image grid)
/// - Width position (column in the image grid)
///
/// Text tokens use the same position for all 3 dimensions (equivalent to 1D RoPE).
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    /// Precomputed cos values: [max_seq_len, head_dim/2]
    cos: Tensor,
    /// Precomputed sin values: [max_seq_len, head_dim/2]
    sin: Tensor,
    /// M-RoPE section sizes: [temporal, height, width]
    mrope_section: Vec<usize>,
    head_dim: usize,
}

impl RotaryEmbedding {
    pub fn new(cfg: &Config, device: &Device, dtype: DType) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;

        // Compute inverse frequencies
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / (cfg.rope_theta as f32).powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;

        // Compute cos/sin for all positions
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;

        Ok(Self {
            cos,
            sin,
            mrope_section: cfg.mrope_section(),
            head_dim: dim,
        })
    }

    /// Apply Multimodal RoPE with 3D position IDs.
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, heads, seq_len, head_dim]
    /// * `k` - Key tensor [batch, kv_heads, seq_len, head_dim]
    /// * `position_ids` - 3D position IDs [3, batch, seq_len]
    pub fn apply_multimodal_rotary_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (three, _batch, _seq_len) = position_ids.dims3()?;
        assert_eq!(three, 3, "position_ids must have 3 dimensions");

        // Compute cos/sin for each position dimension
        let (cos_3d, sin_3d) = self.compute_3d_rope_embeddings(position_ids)?;

        // Apply mrope_section to select appropriate bands from each dimension
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
}

// ============================================================================
// Position ID Computation
// ============================================================================

/// Image grid specification for M-RoPE position computation.
#[derive(Debug, Clone)]
pub struct ImageGrid {
    /// Grid height (patches after spatial merge)
    pub grid_h: usize,
    /// Grid width (patches after spatial merge)
    pub grid_w: usize,
}

/// Result of M-RoPE position computation, including delta for generation.
#[derive(Debug, Clone)]
pub struct MRopePositionIds {
    /// Position IDs tensor of shape [3, batch, seq_len]
    pub position_ids: Tensor,
    /// Delta for generation: next_position - seq_len per batch item.
    /// During decode, new token positions = seq_len + delta
    pub mrope_position_delta: i64,
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

/// Compute M-RoPE position IDs with delta for generation.
///
/// Returns both position IDs and the delta needed for autoregressive decoding.
/// During generation, new token position = seq_len + delta.
pub fn compute_mrope_position_ids_with_delta(
    input_ids: &Tensor,
    image_token_id: u32,
    image_grids: &[ImageGrid],
    device: &Device,
) -> Result<MRopePositionIds> {
    let (batch, seq_len) = input_ids.dims2()?;
    let input_ids_vec: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;

    let mut pos_t = vec![0i64; batch * seq_len];
    let mut pos_h = vec![0i64; batch * seq_len];
    let mut pos_w = vec![0i64; batch * seq_len];
    let mut max_pos = 0i64;

    for b in 0..batch {
        let batch_start = b * seq_len;
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

        if image_ranges.len() != image_grids.len() {
            return Err(candle::Error::Msg(format!(
                "Mismatch: found {} image ranges but {} grids provided",
                image_ranges.len(),
                image_grids.len()
            )));
        }

        let mut current_pos = 0i64;
        let mut range_idx = 0usize;

        for s in 0..seq_len {
            let idx = batch_start + s;

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

                let offset = current_pos;
                for vision_idx in 0..num_vision_tokens {
                    let token_idx = batch_start + img_start + vision_idx;
                    let t_pos = 0i64;
                    let h_pos = (vision_idx / grid.grid_w) as i64;
                    let w_pos = (vision_idx % grid.grid_w) as i64;

                    pos_t[token_idx] = t_pos + offset;
                    pos_h[token_idx] = h_pos + offset;
                    pos_w[token_idx] = w_pos + offset;
                }

                let max_h = (grid.grid_h - 1) as i64;
                let max_w = (grid.grid_w - 1) as i64;
                current_pos = offset + max_h.max(max_w) + 1;

                range_idx += 1;
                continue;
            }

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

            pos_t[idx] = current_pos;
            pos_h[idx] = current_pos;
            pos_w[idx] = current_pos;
            current_pos += 1;
        }

        max_pos = max_pos.max(current_pos);
    }

    let pos_t = Tensor::from_vec(pos_t, (batch, seq_len), device)?;
    let pos_h = Tensor::from_vec(pos_h, (batch, seq_len), device)?;
    let pos_w = Tensor::from_vec(pos_w, (batch, seq_len), device)?;

    let position_ids = Tensor::stack(&[pos_t, pos_h, pos_w], 0)?;

    // Delta: next_position - seq_len
    // This is how much to add to raw position during generation
    let mrope_position_delta = max_pos - seq_len as i64;

    Ok(MRopePositionIds {
        position_ids,
        mrope_position_delta,
    })
}

/// Video grid specification.
#[derive(Debug, Clone)]
pub struct VideoGrid {
    pub grid_t: usize,
    pub grid_h: usize,
    pub grid_w: usize,
}

/// Compute 3D M-RoPE position IDs for video input.
///
/// Uses temporal scaling: t_pos = frame_index * second_per_grid_t * tokens_per_second
/// where tokens_per_second comes from vision config (default: 4).
pub fn compute_mrope_position_ids_video(
    input_ids: &Tensor,
    video_token_id: u32,
    video_grid: &VideoGrid,
    second_per_grid_t: f32,
    tokens_per_second: usize,
    device: &Device,
) -> Result<Tensor> {
    let (batch, seq_len) = input_ids.dims2()?;
    let input_ids_vec: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;

    let grid_t = video_grid.grid_t;
    let grid_h = video_grid.grid_h;
    let grid_w = video_grid.grid_w;
    let num_vision_tokens = grid_t * grid_h * grid_w;

    let mut pos_t = vec![0i64; batch * seq_len];
    let mut pos_h = vec![0i64; batch * seq_len];
    let mut pos_w = vec![0i64; batch * seq_len];

    for b in 0..batch {
        let batch_start = b * seq_len;

        // Find video token range
        let mut video_start = None;
        let mut video_end = None;
        let mut in_video = false;

        for s in 0..seq_len {
            let token_id = input_ids_vec[batch_start + s];
            if token_id == video_token_id {
                if !in_video {
                    in_video = true;
                    video_start = Some(s);
                }
            } else if in_video {
                video_end = Some(s);
                break;
            }
        }
        if in_video && video_end.is_none() {
            video_end = Some(seq_len);
        }

        // Verify token count
        if let (Some(start), Some(end)) = (video_start, video_end) {
            let actual = end - start;
            if actual != num_vision_tokens {
                return Err(candle::Error::Msg(format!(
                    "Video has {} tokens but grid {}x{}x{} = {} expected",
                    actual, grid_t, grid_h, grid_w, num_vision_tokens
                )));
            }
        }

        // Compute positions
        let mut current_pos = 0i64;
        let video_range = video_start.zip(video_end);

        for s in 0..seq_len {
            let idx = batch_start + s;

            if let Some((v_start, v_end)) = video_range {
                if s == v_start {
                    let offset = current_pos;

                    for vision_idx in 0..num_vision_tokens {
                        let token_idx = batch_start + v_start + vision_idx;

                        // Qwen2.5-VL temporal scaling: frame_index * second_per_grid_t * tokens_per_second
                        let frame_index = vision_idx / (grid_h * grid_w);
                        let t_pos =
                            (frame_index as f32 * second_per_grid_t * tokens_per_second as f32)
                                as i64;
                        let spatial_idx = vision_idx % (grid_h * grid_w);
                        let h_pos = (spatial_idx / grid_w) as i64;
                        let w_pos = (spatial_idx % grid_w) as i64;

                        pos_t[token_idx] = t_pos + offset;
                        pos_h[token_idx] = h_pos + offset;
                        pos_w[token_idx] = w_pos + offset;
                    }

                    let max_t = ((grid_t - 1) as f32 * second_per_grid_t * tokens_per_second as f32)
                        as i64;
                    let max_h = (grid_h - 1) as i64;
                    let max_w = (grid_w - 1) as i64;
                    current_pos = offset + max_t.max(max_h).max(max_w) + 1;

                    continue;
                }

                if s > v_start && s < v_end {
                    continue;
                }
            }

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

/// Compute M-RoPE position IDs for video with delta for generation.
pub fn compute_mrope_position_ids_video_with_delta(
    input_ids: &Tensor,
    video_token_id: u32,
    video_grid: &VideoGrid,
    second_per_grid_t: f32,
    tokens_per_second: usize,
    device: &Device,
) -> Result<MRopePositionIds> {
    let (batch, seq_len) = input_ids.dims2()?;
    let input_ids_vec: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;

    let grid_t = video_grid.grid_t;
    let grid_h = video_grid.grid_h;
    let grid_w = video_grid.grid_w;
    let num_vision_tokens = grid_t * grid_h * grid_w;

    let mut pos_t = vec![0i64; batch * seq_len];
    let mut pos_h = vec![0i64; batch * seq_len];
    let mut pos_w = vec![0i64; batch * seq_len];
    let mut max_pos = 0i64;

    for b in 0..batch {
        let batch_start = b * seq_len;

        let mut video_start = None;
        let mut video_end = None;
        let mut in_video = false;

        for s in 0..seq_len {
            let token_id = input_ids_vec[batch_start + s];
            if token_id == video_token_id {
                if !in_video {
                    in_video = true;
                    video_start = Some(s);
                }
            } else if in_video {
                video_end = Some(s);
                break;
            }
        }
        if in_video && video_end.is_none() {
            video_end = Some(seq_len);
        }

        if let (Some(start), Some(end)) = (video_start, video_end) {
            let actual = end - start;
            if actual != num_vision_tokens {
                return Err(candle::Error::Msg(format!(
                    "Video has {} tokens but grid {}x{}x{} = {} expected",
                    actual, grid_t, grid_h, grid_w, num_vision_tokens
                )));
            }
        }

        let mut current_pos = 0i64;
        let video_range = video_start.zip(video_end);

        for s in 0..seq_len {
            let idx = batch_start + s;

            if let Some((v_start, v_end)) = video_range {
                if s == v_start {
                    let offset = current_pos;

                    for vision_idx in 0..num_vision_tokens {
                        let token_idx = batch_start + v_start + vision_idx;
                        let frame_index = vision_idx / (grid_h * grid_w);
                        let t_pos =
                            (frame_index as f32 * second_per_grid_t * tokens_per_second as f32)
                                as i64;
                        let spatial_idx = vision_idx % (grid_h * grid_w);
                        let h_pos = (spatial_idx / grid_w) as i64;
                        let w_pos = (spatial_idx % grid_w) as i64;

                        pos_t[token_idx] = t_pos + offset;
                        pos_h[token_idx] = h_pos + offset;
                        pos_w[token_idx] = w_pos + offset;
                    }

                    let max_t = ((grid_t - 1) as f32 * second_per_grid_t * tokens_per_second as f32)
                        as i64;
                    let max_h = (grid_h - 1) as i64;
                    let max_w = (grid_w - 1) as i64;
                    current_pos = offset + max_t.max(max_h).max(max_w) + 1;

                    continue;
                }

                if s > v_start && s < v_end {
                    continue;
                }
            }

            pos_t[idx] = current_pos;
            pos_h[idx] = current_pos;
            pos_w[idx] = current_pos;
            current_pos += 1;
        }

        max_pos = max_pos.max(current_pos);
    }

    let pos_t = Tensor::from_vec(pos_t, (batch, seq_len), device)?;
    let pos_h = Tensor::from_vec(pos_h, (batch, seq_len), device)?;
    let pos_w = Tensor::from_vec(pos_w, (batch, seq_len), device)?;

    let position_ids = Tensor::stack(&[pos_t, pos_h, pos_w], 0)?;
    let mrope_position_delta = max_pos - seq_len as i64;

    Ok(MRopePositionIds {
        position_ids,
        mrope_position_delta,
    })
}

// ============================================================================
// MLP
// ============================================================================

/// SwiGLU MLP block.
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: candle_nn::Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        // Qwen2.5-VL uses no bias in projections
        let gate_proj = linear_b(hidden_sz, intermediate_sz, false, vb.pp("gate_proj"))?;
        let up_proj = linear_b(hidden_sz, intermediate_sz, false, vb.pp("up_proj"))?;
        let down_proj = linear_b(intermediate_sz, hidden_sz, false, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate_proj.forward(xs)?.apply(&self.act_fn)?;
        let rhs = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

// ============================================================================
// Attention
// ============================================================================

/// Create a sliding window causal attention mask.
///
/// For a window size of W, each token can only attend to the previous W tokens
/// (including itself) while still respecting causality.
fn create_sliding_window_causal_mask(
    tgt_len: usize,
    kv_len: usize,
    window_size: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mask: Vec<f32> = (0..tgt_len)
        .flat_map(|i| {
            // Query position in the full sequence is at the end of kv_len for decode
            let query_pos = kv_len - tgt_len + i;
            (0..kv_len).map(move |j| {
                // Mask out: future positions (causal) OR positions outside sliding window
                if j > query_pos || query_pos - j >= window_size {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (tgt_len, kv_len), device)?.to_dtype(dtype)
}

/// GQA attention without QK-normalization (unlike Qwen3-VL).
///
/// Supports:
/// - Flash Attention 2 (when compiled with flash-attn feature)
/// - Sliding Window Attention (per-layer, based on layer_idx >= max_window_layers)
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
    softmax_scale: f64,
    use_flash_attn: bool,
    sliding_window: Option<usize>,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let num_kv_groups = num_heads / num_kv_heads;

        // Qwen2.5-VL has bias in Q, K, V projections but not O projection
        let q_proj = linear_b(hidden_sz, num_heads * head_dim, true, vb.pp("q_proj"))?;
        let k_proj = linear_b(hidden_sz, num_kv_heads * head_dim, true, vb.pp("k_proj"))?;
        let v_proj = linear_b(hidden_sz, num_kv_heads * head_dim, true, vb.pp("v_proj"))?;
        let o_proj = linear_b(num_heads * head_dim, hidden_sz, false, vb.pp("o_proj"))?;

        // Determine sliding window for this layer
        let sliding_window = cfg.get_sliding_window(layer_idx);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rotary_emb,
            kv_cache: None,
            softmax_scale: 1.0 / (head_dim as f64).sqrt(),
            use_flash_attn: cfg.use_flash_attn,
            sliding_window,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply M-RoPE (no QK-norm in Qwen2.5-VL, unlike Qwen3-VL)
        let (query_states, key_states) = self
            .rotary_emb
            .apply_multimodal_rotary_emb(&query_states, &key_states, position_ids)?;

        // KV cache
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        // Compute attention using either Flash Attention or standard path
        let attn_output = if self.use_flash_attn {
            // Flash Attention path
            // Flash Attention expects (batch, seq_len, num_heads, head_dim)
            let q = query_states.transpose(1, 2)?;
            let k = key_states.transpose(1, 2)?;
            let v = value_states.transpose(1, 2)?;
            let softmax_scale = self.softmax_scale as f32;

            let attn_out = match self.sliding_window {
                Some(window_size) => {
                    // Sliding window attention with flash-attn
                    flash_attn_windowed(&q, &k, &v, softmax_scale, window_size)?
                }
                None => {
                    // Full causal attention with flash-attn
                    flash_attn(&q, &k, &v, softmax_scale, q_len > 1)?
                }
            };
            attn_out.transpose(1, 2)?
        } else {
            // Standard attention path
            // Repeat KV for GQA
            let key_states =
                crate::utils::repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
            let value_states =
                crate::utils::repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

            let attn_weights =
                (query_states.matmul(&key_states.transpose(2, 3)?)? * self.softmax_scale)?;

            // Apply attention mask (either provided causal mask or sliding window mask)
            let kv_len = key_states.dim(2)?;
            let attn_weights = match (attention_mask, self.sliding_window) {
                (_, Some(window_size)) => {
                    // Create sliding window causal mask for this layer
                    let sw_mask = create_sliding_window_causal_mask(
                        q_len,
                        kv_len,
                        window_size,
                        attn_weights.device(),
                        attn_weights.dtype(),
                    )?;
                    // Expand for batch and heads: [1, 1, q_len, kv_len]
                    let sw_mask = sw_mask.unsqueeze(0)?.unsqueeze(0)?;
                    attn_weights.broadcast_add(&sw_mask)?
                }
                (Some(mask), None) => attn_weights.broadcast_add(mask)?,
                (None, None) => attn_weights,
            };

            // Softmax in F32 for stability
            let original_dtype = attn_weights.dtype();
            let attn_weights = if original_dtype != DType::F32 {
                let attn_weights = attn_weights.to_dtype(DType::F32)?;
                let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
                attn_weights.to_dtype(original_dtype)?
            } else {
                candle_nn::ops::softmax_last_dim(&attn_weights)?
            };

            attn_weights.matmul(&value_states)?
        };

        attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_sz, q_len, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

// ============================================================================
// Decoder Layer
// ============================================================================

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, layer_idx, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, position_ids)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ============================================================================
// Text Model
// ============================================================================

/// Qwen2.5-VL Text Model.
pub struct Qwen25VLTextModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    pub dtype: DType,
    pub hidden_size: usize,
    device: Device,
}

impl Qwen25VLTextModel {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        // Weight path: model.* (not model.language_model.*)
        let vb_m = vb.pp("model");

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let rotary_emb = Arc::new(RotaryEmbedding::new(cfg, vb.device(), vb.dtype())?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, layer_idx, vb_l.pp(layer_idx))?;
            layers.push(layer);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_b(cfg.hidden_size, cfg.vocab_size, false, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
            device: vb.device().clone(),
        })
    }

    /// Get token embeddings.
    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    /// Prepare causal attention mask.
    fn prepare_causal_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<f32> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    /// Forward pass with M-RoPE position IDs.
    pub fn forward_with_mrope(&mut self, xs: Tensor, position_ids: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;

        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_causal_attention_mask(b_sz, seq_len, 0)?)
        };

        let mut hidden = xs;
        for layer in self.layers.iter_mut() {
            hidden = layer.forward(&hidden, attention_mask.as_ref(), position_ids)?;
        }

        hidden = hidden.apply(&self.norm)?;

        // Logits for last token only
        self.lm_head
            .forward(&hidden)?
            .i((.., seq_len - 1, ..))?
            .contiguous()
    }

    /// Forward pass with M-RoPE using chunked prefill for reduced memory.
    ///
    /// Processes the input in chunks to reduce peak memory usage, which is essential
    /// for long sequences on memory-constrained devices (e.g., Metal GPUs).
    ///
    /// # Arguments
    /// * `xs` - Input embeddings [batch, seq_len, hidden_dim]
    /// * `position_ids` - M-RoPE position IDs [3, batch, seq_len]
    /// * `chunk_size` - Number of tokens to process per chunk
    ///
    /// # Memory Savings
    /// For seq_len=4808 with chunk_size=2048:
    /// - Without chunking: attention matrix = 4808² elements
    /// - With chunking: peak = 2048×4096 elements (64% reduction)
    pub fn forward_with_mrope_chunked(
        &mut self,
        xs: Tensor,
        position_ids: &Tensor,
        chunk_size: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;

        // For small sequences, use standard path (no benefit from chunking)
        if seq_len <= chunk_size {
            return self.forward_with_mrope(xs, position_ids);
        }

        // Clear KV cache to start fresh prefill
        self.clear_kv_cache();

        let num_chunks = seq_len.div_ceil(chunk_size);

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * chunk_size;
            let end = ((chunk_idx + 1) * chunk_size).min(seq_len);
            let chunk_len = end - start;

            // Extract chunk of embeddings: [batch, chunk_len, hidden_dim]
            let chunk_embeds = xs.narrow(1, start, chunk_len)?;

            // Extract chunk of position IDs: [3, batch, chunk_len]
            // M-RoPE uses absolute positions, so slicing gives correct positions
            let chunk_positions = position_ids.narrow(2, start, chunk_len)?;

            // Create attention mask for this chunk:
            // - Can attend to all cached tokens (positions 0..start)
            // - Causal attention within the chunk
            let attention_mask = if chunk_len <= 1 && start == 0 {
                None
            } else {
                Some(self.prepare_causal_attention_mask(b_sz, chunk_len, start)?)
            };

            // Forward through all layers (KV cache accumulates automatically)
            let mut hidden = chunk_embeds;
            for layer in self.layers.iter_mut() {
                hidden = layer.forward(&hidden, attention_mask.as_ref(), &chunk_positions)?;
            }

            // On the final chunk, apply norm and compute logits
            if chunk_idx == num_chunks - 1 {
                hidden = hidden.apply(&self.norm)?;
                return self.lm_head
                    .forward(&hidden)?
                    .i((.., chunk_len - 1, ..))?
                    .contiguous();
            }
        }

        unreachable!("Should have returned in the last chunk iteration")
    }

    /// Clear all KV caches.
    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }

    /// Debug helper: Check tensor for NaN/Inf and print statistics.
    fn check_tensor_health(name: &str, tensor: &Tensor) -> Result<bool> {
        let t_f32 = tensor.to_dtype(DType::F32)?;
        let vec: Vec<f32> = t_f32.flatten_all()?.to_vec1()?;
        let nan_count = vec.iter().filter(|x| x.is_nan()).count();
        let inf_count = vec.iter().filter(|x| x.is_infinite()).count();

        if nan_count > 0 || inf_count > 0 {
            eprintln!(
                "[DEBUG] ❌ {}: shape={:?}, {} NaN, {} Inf (out of {})",
                name,
                tensor.dims(),
                nan_count,
                inf_count,
                vec.len()
            );
            return Ok(false);
        }

        let mean = vec.iter().sum::<f32>() / vec.len() as f32;
        let variance =
            vec.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / vec.len() as f32;
        let std = variance.sqrt();
        let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!(
            "[DEBUG] ✓ {}: shape={:?}, mean={:.4}, std={:.4}, min={:.4}, max={:.4}",
            name,
            tensor.dims(),
            mean,
            std,
            min,
            max
        );
        Ok(true)
    }

    /// Forward pass with vision embeddings, returning hidden states for diffusion.
    ///
    /// This is used by Qwen-Image Edit and Layered pipelines for encoding prompts
    /// with both text and image understanding. The vision embeddings are inserted
    /// at image placeholder token positions.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs with image placeholders, shape (batch, seq_len)
    /// * `vision_embeds` - Vision embeddings from the vision encoder, shape (total_vision_tokens, hidden_size)
    /// * `image_grid_thw` - Grid dimensions for each image, shape (num_images, 3) containing [temporal, height, width]
    /// * `attention_mask` - Optional attention mask of shape (batch, seq_len)
    /// * `spatial_merge_size` - Spatial merge factor from vision config (typically 2)
    /// * `image_token_id` - Token ID for image placeholders (typically 151655)
    ///
    /// # Returns
    /// Hidden states tensor of shape (batch, seq_len, hidden_size) for use as diffusion conditioning
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
        // Clear KV cache to ensure fresh forward pass (no stale state from previous calls)
        self.clear_kv_cache();

        let (batch_size, seq_len) = input_ids.dims2()?;
        let hidden_dim = self.hidden_size;

        // 1. Get base token embeddings
        let mut input_embeds = self.embed_tokens(input_ids)?;

        // 2. Replace image placeholder tokens with vision embeddings
        let input_ids_flat: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
        let vision_embeds = vision_embeds.to_dtype(self.dtype)?;
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
            compute_mrope_position_ids_multi(input_ids, image_token_id, &image_grids, &self.device)?;

        // 4. Create causal attention mask if sequence length > 1
        let causal_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_causal_attention_mask(batch_size, seq_len, 0)?)
        };

        // Combine with provided attention mask if any
        let attention_mask = match (causal_mask, attention_mask) {
            (Some(causal), Some(mask)) => {
                let mask = mask.unsqueeze(1)?.unsqueeze(1)?;
                // IMPORTANT: Cannot use (1-mask)*-inf because 0*-inf = NaN in IEEE 754!
                // Use a large finite negative instead (Metal doesn't support where_cond)
                let mask = mask.to_dtype(DType::F32)?;
                let mask = ((1.0 - mask)? * -1e9)?.to_dtype(causal.dtype())?;
                let mask = mask.broadcast_as(causal.dims())?;
                Some((&causal + &mask)?)
            }
            (Some(causal), None) => Some(causal),
            (None, Some(mask)) => {
                let mask = mask.unsqueeze(1)?.unsqueeze(1)?;
                // IMPORTANT: Cannot use (1-mask)*-inf because 0*-inf = NaN in IEEE 754!
                // Use a large finite negative instead (Metal doesn't support where_cond)
                let mask = mask.to_dtype(DType::F32)?;
                Some(((1.0 - mask)? * -1e9)?.to_dtype(self.dtype)?)
            }
            (None, None) => None,
        };

        // 5. Forward through all transformer layers
        let mut hidden_states = input_embeds;
        for layer in self.layers.iter_mut() {
            hidden_states = layer.forward(&hidden_states, attention_mask.as_ref(), &position_ids)?;
        }

        // 6. Apply final layer norm (but NOT lm_head - we want hidden states)
        hidden_states.apply(&self.norm)
    }

    /// Forward pass for text-only input, returning hidden states.
    ///
    /// This is used by Qwen-Image for text encoding, where we need the raw hidden
    /// states from the transformer (not logits from lm_head).
    ///
    /// Unlike `forward_with_mrope`, this method:
    /// - Uses simple sequential position IDs (no vision token handling)
    /// - Returns all hidden states, not just the last token's logits
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
        // Clear KV cache to ensure fresh forward pass (no stale state from previous calls)
        self.clear_kv_cache();

        let debug_mode = std::env::var("QWEN_DEBUG").is_ok();
        let (b_sz, seq_len) = input_ids.dims2()?;

        // Get token embeddings
        let mut hidden_states = self.embed_tokens(input_ids)?;

        // Debug: Check embeddings for NaN
        if debug_mode {
            Self::check_tensor_health("embed_tokens output", &hidden_states)?;
        }

        // Create simple sequential position IDs: [3, batch, seq_len]
        // For text-only, all three M-RoPE dimensions use the same sequential positions
        let positions: Vec<i64> = (0..seq_len as i64).collect();
        let pos_tensor = Tensor::from_vec(positions.clone(), seq_len, &self.device)?;
        let pos_tensor = pos_tensor.unsqueeze(0)?.expand((b_sz, seq_len))?;
        let position_ids = Tensor::stack(&[&pos_tensor, &pos_tensor, &pos_tensor], 0)?;

        // Create causal attention mask if sequence length > 1
        let causal_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_causal_attention_mask(b_sz, seq_len, 0)?)
        };

        // Combine with provided attention mask if any
        let attention_mask = match (causal_mask, attention_mask) {
            (Some(causal), Some(mask)) => {
                // Debug: Check input mask before conversion
                if debug_mode {
                    Self::check_tensor_health("input_attention_mask (before conversion)", mask)?;
                }

                // Expand mask to match causal mask shape: [batch, 1, 1, seq_len]
                let mask = mask.unsqueeze(1)?.unsqueeze(1)?;
                // Convert 0/1 mask to large_negative/0 additive mask
                // IMPORTANT: Cannot use (1-mask)*-inf because 0*-inf = NaN in IEEE 754!
                // Use a large finite negative instead (Metal doesn't support where_cond)
                let mask = mask.to_dtype(DType::F32)?;
                let mask = ((1.0 - mask)? * -1e9)?;
                let mask = mask.to_dtype(causal.dtype())?;

                // Debug: Check mask after conversion
                if debug_mode {
                    Self::check_tensor_health("attention_mask (after -1e9 conversion)", &mask)?;
                }

                let mask = mask.broadcast_as(causal.dims())?;
                let combined = (&causal + &mask)?;

                // Debug: Check final combined mask
                if debug_mode {
                    Self::check_tensor_health("attention_mask (combined)", &combined)?;
                }

                Some(combined)
            }
            (Some(causal), None) => {
                if debug_mode {
                    Self::check_tensor_health("causal_mask", &causal)?;
                }
                Some(causal)
            }
            (None, Some(mask)) => {
                let mask = mask.unsqueeze(1)?.unsqueeze(1)?;
                // IMPORTANT: Cannot use (1-mask)*-inf because 0*-inf = NaN in IEEE 754!
                // Use a large finite negative instead (Metal doesn't support where_cond)
                let mask = mask.to_dtype(DType::F32)?;
                let converted = ((1.0 - mask)? * -1e9)?.to_dtype(self.dtype)?;
                if debug_mode {
                    Self::check_tensor_health("attention_mask (no causal)", &converted)?;
                }
                Some(converted)
            }
            (None, None) => None,
        };

        // Forward through all transformer layers
        let num_layers = self.layers.len();
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, attention_mask.as_ref(), &position_ids)?;

            // Debug: Check first 3 layers, then every 10th layer, and last layer
            if debug_mode && (layer_idx < 3 || layer_idx % 10 == 0 || layer_idx == num_layers - 1) {
                let healthy =
                    Self::check_tensor_health(&format!("layer {} output", layer_idx), &hidden_states)?;
                if !healthy {
                    eprintln!("[DEBUG] ⚠️  NaN first detected at layer {}!", layer_idx);
                    // Continue to see the pattern, but we found the culprit
                }
            }
        }

        // Apply final layer norm - HuggingFace applies norm BEFORE adding to all_hidden_states,
        // so hidden_states[-1] includes the final layer norm (see modeling_qwen2_5_vl.py:946-950)
        let final_hidden_states = hidden_states.apply(&self.norm)?;

        if debug_mode {
            Self::check_tensor_health("final_norm output", &final_hidden_states)?;
        }

        Ok(final_hidden_states)
    }
}
