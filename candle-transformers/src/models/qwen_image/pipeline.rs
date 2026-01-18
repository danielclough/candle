//! Pipeline utilities for Qwen-Image inference.
//!
//! This module provides reusable building blocks for Qwen-Image pipelines:
//! - Image preprocessing for VAE and vision encoder
//! - True CFG with norm rescaling
//! - Token expansion for vision prompts
//!
//! These utilities are used by the example pipelines (text-to-image, edit, inpaint, etc.)
//! and can be composed to build custom pipelines.
//!
//! Note: Text encoding functions that require the `tokenizers` crate are implemented
//! in the examples, as `candle-transformers` intentionally avoids heavy dependencies.

use candle::{DType, Device, Result, Tensor, D};

use crate::models::qwen2_5_vl::{
    get_image_grid_thw, normalize_image, patchify_image, smart_resize, DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
};

// ============================================================================
// Vision Encoder Constants
// ============================================================================

/// Patch size for vision encoder (14x14 pixels).
pub const VISION_PATCH_SIZE: usize = 14;

/// Temporal patch size for video frames.
pub const VISION_TEMPORAL_PATCH_SIZE: usize = 2;

/// Spatial merge size for patch merging (2x2).
pub const VISION_MERGE_SIZE: usize = 2;

/// Token ID for image placeholder in Qwen2.5-VL tokenizer.
pub const IMAGE_TOKEN_ID: u32 = 151655;

// ============================================================================
// Image Preprocessing
// ============================================================================

/// Prepare raw RGB image bytes for VAE encoding.
///
/// Converts RGB byte data to a tensor normalized to [-1, 1] range,
/// in the format expected by the Qwen-Image VAE encoder.
///
/// # Arguments
/// * `rgb_data` - Raw RGB bytes in row-major order (H×W×3)
/// * `height` - Image height in pixels
/// * `width` - Image width in pixels
/// * `device` - Compute device
/// * `dtype` - Data type for output tensor
///
/// # Returns
/// Tensor of shape `[1, 3, 1, height, width]` with values in [-1, 1]
pub fn prepare_image_for_vae(
    rgb_data: &[u8],
    height: usize,
    width: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut data = Vec::with_capacity(3 * height * width);

    // Convert to channels-first and normalize to [-1, 1]
    for c in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3 + c;
                let value = (rgb_data[idx] as f32 / 127.5) - 1.0;
                data.push(value);
            }
        }
    }

    Tensor::from_vec(data, (1, 3, 1, height, width), device)?.to_dtype(dtype)
}

/// Preprocess an image for the Qwen2.5-VL vision encoder.
///
/// Performs smart resizing, normalization, and patchification to prepare
/// an image for the vision encoder.
///
/// # Arguments
/// * `rgb_data` - Raw RGB bytes in row-major order (H×W×3)
/// * `orig_height` - Original image height
/// * `orig_width` - Original image width
/// * `device` - Compute device
/// * `dtype` - Data type for output tensor
///
/// # Returns
/// Tuple of:
/// - `pixel_values`: Patchified image tensor for vision encoder
/// - `grid_thw`: Grid dimensions tensor `[[T, H, W]]`
/// - `resized_height`: Height after smart resize
/// - `resized_width`: Width after smart resize
pub fn prepare_image_for_vision(
    rgb_data: &[u8],
    orig_height: usize,
    orig_width: usize,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor, usize, usize)> {
    // Compute smart resize dimensions
    let factor = VISION_PATCH_SIZE * VISION_MERGE_SIZE;
    let (resized_height, resized_width) = smart_resize(
        orig_height,
        orig_width,
        factor,
        DEFAULT_MIN_PIXELS,
        DEFAULT_MAX_PIXELS,
    );

    // The caller should resize the image before calling this function
    // We assume rgb_data is already at resized dimensions
    // Normalize to channels-first float
    let normalized = normalize_image(rgb_data, resized_height, resized_width);

    // Patchify for vision encoder
    let patches = patchify_image(
        &normalized,
        resized_height,
        resized_width,
        VISION_PATCH_SIZE,
        VISION_TEMPORAL_PATCH_SIZE,
        VISION_MERGE_SIZE,
    );

    // Compute grid dimensions
    let (grid_t, grid_h, grid_w) = get_image_grid_thw(
        resized_height,
        resized_width,
        None,
        None,
        Some(VISION_PATCH_SIZE),
        Some(VISION_MERGE_SIZE),
    );

    // Create tensors
    let num_patches = grid_t * grid_h * grid_w;
    let patch_elements = 3 * VISION_TEMPORAL_PATCH_SIZE * VISION_PATCH_SIZE * VISION_PATCH_SIZE;
    let pixel_values =
        Tensor::from_vec(patches, (num_patches, patch_elements), device)?.to_dtype(dtype)?;

    let grid_thw = Tensor::new(&[[grid_t as u32, grid_h as u32, grid_w as u32]], device)?;

    Ok((pixel_values, grid_thw, resized_height, resized_width))
}

/// Compute smart resize dimensions for vision encoder.
///
/// This is a convenience wrapper around `smart_resize` with Qwen-Image defaults.
pub fn compute_vision_size(orig_height: usize, orig_width: usize) -> (usize, usize) {
    let factor = VISION_PATCH_SIZE * VISION_MERGE_SIZE;
    smart_resize(
        orig_height,
        orig_width,
        factor,
        DEFAULT_MIN_PIXELS,
        DEFAULT_MAX_PIXELS,
    )
}

// ============================================================================
// Token Processing
// ============================================================================

/// Expand image placeholder tokens in a token sequence.
///
/// Finds the first occurrence of `IMAGE_TOKEN_ID` and replaces it with
/// `num_image_tokens` copies of the token.
///
/// # Arguments
/// * `tokens` - Input token sequence
/// * `num_image_tokens` - Number of image tokens to insert
///
/// # Returns
/// Expanded token sequence
pub fn expand_image_tokens(tokens: &[u32], num_image_tokens: usize) -> Vec<u32> {
    if let Some(pos) = tokens.iter().position(|&t| t == IMAGE_TOKEN_ID) {
        tokens[..pos]
            .iter()
            .chain(std::iter::repeat_n(&IMAGE_TOKEN_ID, num_image_tokens))
            .chain(tokens[pos + 1..].iter())
            .copied()
            .collect()
    } else {
        tokens.to_vec()
    }
}

// ============================================================================
// Classifier-Free Guidance
// ============================================================================

/// Apply True CFG with norm rescaling.
///
/// Combines positive and negative predictions using classifier-free guidance,
/// then rescales the result to prevent magnitude explosion.
///
/// The formula is:
/// ```text
/// comb = neg + scale × (pos - neg)
/// output = comb × (norm(pos) / norm(comb))
/// ```
///
/// # Arguments
/// * `pos_pred` - Positive prediction
/// * `neg_pred` - Negative prediction
/// * `cfg_scale` - Guidance scale (typically 4.0)
///
/// # Returns
/// Guided prediction with normalized magnitude
pub fn apply_true_cfg(pos_pred: &Tensor, neg_pred: &Tensor, cfg_scale: f64) -> Result<Tensor> {
    // Step 1: Combine predictions
    let comb_pred = (neg_pred + ((pos_pred - neg_pred)? * cfg_scale)?)?;

    // Step 2: Compute norms and rescale to prevent magnitude explosion
    let pos_norm = pos_pred.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let comb_norm = comb_pred.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;

    // Add small epsilon to avoid division by zero
    let scale_factor = (&pos_norm / (&comb_norm + 1e-8)?)?;
    // Explicitly broadcast scale_factor to match comb_pred shape
    // (Candle doesn't auto-broadcast like PyTorch)
    let scale_factor = scale_factor.broadcast_as(comb_pred.shape())?;
    &comb_pred * &scale_factor
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepare_image_for_vae() -> Result<()> {
        let device = Device::Cpu;

        // Create a simple 4x4 RGB image (48 bytes)
        let rgb_data: Vec<u8> = (0..48).collect();

        let tensor = prepare_image_for_vae(&rgb_data, 4, 4, &device, DType::F32)?;

        assert_eq!(tensor.dims(), &[1, 3, 1, 4, 4]);

        // Check normalization: 0 -> -1.0, 255 -> 1.0
        let data = tensor.flatten_all()?.to_vec1::<f32>()?;
        assert!((data[0] - (-1.0)).abs() < 0.01); // 0 -> -1.0

        Ok(())
    }

    #[test]
    fn test_apply_true_cfg() -> Result<()> {
        let device = Device::Cpu;

        let pos = Tensor::new(&[[1.0f32, 2.0, 3.0]], &device)?;
        let neg = Tensor::new(&[[0.0f32, 0.0, 0.0]], &device)?;

        let result = apply_true_cfg(&pos, &neg, 2.0)?;

        // With scale=2.0: comb = 0 + 2*(pos - 0) = 2*pos
        // Then rescaled to have same norm as pos
        // So result should have same direction as pos but magnitude of pos
        assert_eq!(result.dims(), pos.dims());

        // The result should have the same norm as pos (due to rescaling)
        let pos_norm = pos.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let result_norm = result.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        assert!((pos_norm - result_norm).abs() < 0.01);

        Ok(())
    }

    #[test]
    fn test_compute_vision_size() {
        // Test that smart resize produces dimensions divisible by factor (28)
        let (h, w) = compute_vision_size(1024, 768);
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }
}
