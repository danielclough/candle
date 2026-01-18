//! Image and video processing utilities for Qwen2.5-VL.
//!
//! This module provides utilities for computing image/video grid dimensions
//! and token counts, matching the HuggingFace implementation.
//!
//! # Normalization
//!
//! Qwen2.5-VL uses CLIP-style normalization with:
//! - Mean: [0.48145466, 0.4578275, 0.40821073]
//! - Std: [0.26862954, 0.26130258, 0.27577711]
//!
//! Formula: `(pixel / 255.0 - mean) / std`

/// Default values for image processing.
pub const DEFAULT_MIN_PIXELS: usize = 56 * 56; // 3136
pub const DEFAULT_MAX_PIXELS: usize = 28 * 28 * 1280; // 1003520
pub const DEFAULT_PATCH_SIZE: usize = 14;
pub const DEFAULT_MERGE_SIZE: usize = 2;
pub const DEFAULT_TEMPORAL_PATCH_SIZE: usize = 2;

/// CLIP-style normalization mean (RGB).
pub const IMAGE_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];

/// CLIP-style normalization std (RGB).
pub const IMAGE_STD: [f32; 3] = [0.26862954, 0.261_302_6, 0.275_777_1];

/// Smart resize that maintains aspect ratio while ensuring:
/// 1. Both dimensions are divisible by `factor`
/// 2. Total pixels are within `[min_pixels, max_pixels]`
/// 3. Aspect ratio is preserved as closely as possible
///
/// # Arguments
/// * `height` - Original image height
/// * `width` - Original image width
/// * `factor` - Dimensions must be divisible by this (typically patch_size * merge_size = 28)
/// * `min_pixels` - Minimum total pixels (default: 56*56 = 3136)
/// * `max_pixels` - Maximum total pixels (default: 28*28*1280 = 1003520)
///
/// # Returns
/// Tuple of (resized_height, resized_width)
///
/// # Panics
/// Panics if aspect ratio exceeds 200:1
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    let max_dim = height.max(width) as f64;
    let min_dim = height.min(width) as f64;

    if max_dim / min_dim > 200.0 {
        panic!(
            "Aspect ratio must be smaller than 200, got {}",
            max_dim / min_dim
        );
    }

    let h = height as f64;
    let w = width as f64;
    let factor_f = factor as f64;

    // Round to nearest factor multiple
    let mut h_bar = ((h / factor_f).round() * factor_f) as usize;
    let mut w_bar = ((w / factor_f).round() * factor_f) as usize;

    if h_bar * w_bar > max_pixels {
        // Scale down to fit max_pixels
        let beta = ((h * w) / max_pixels as f64).sqrt();
        h_bar = factor.max(((h / beta / factor_f).floor() * factor_f) as usize);
        w_bar = factor.max(((w / beta / factor_f).floor() * factor_f) as usize);
    } else if h_bar * w_bar < min_pixels {
        // Scale up to meet min_pixels
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = ((h * beta / factor_f).ceil() * factor_f) as usize;
        w_bar = ((w * beta / factor_f).ceil() * factor_f) as usize;
    }

    (h_bar, w_bar)
}

/// Compute the number of image patches for a given image size.
///
/// This accounts for smart resizing and spatial merging.
///
/// # Arguments
/// * `height` - Original image height
/// * `width` - Original image width
/// * `min_pixels` - Minimum total pixels (default: 3136)
/// * `max_pixels` - Maximum total pixels (default: 1003520)
/// * `patch_size` - Vision encoder patch size (default: 14)
/// * `merge_size` - Spatial merge size (default: 2)
///
/// # Returns
/// Number of patches (before merge) that the image will produce
pub fn get_number_of_image_patches(
    height: usize,
    width: usize,
    min_pixels: Option<usize>,
    max_pixels: Option<usize>,
    patch_size: Option<usize>,
    merge_size: Option<usize>,
) -> usize {
    let min_pixels = min_pixels.unwrap_or(DEFAULT_MIN_PIXELS);
    let max_pixels = max_pixels.unwrap_or(DEFAULT_MAX_PIXELS);
    let patch_size = patch_size.unwrap_or(DEFAULT_PATCH_SIZE);
    let merge_size = merge_size.unwrap_or(DEFAULT_MERGE_SIZE);

    let factor = patch_size * merge_size;
    let (resized_height, resized_width) =
        smart_resize(height, width, factor, min_pixels, max_pixels);

    let grid_h = resized_height / patch_size;
    let grid_w = resized_width / patch_size;

    grid_h * grid_w
}

/// Compute the number of image tokens (after spatial merge).
///
/// # Arguments
/// * `height` - Original image height
/// * `width` - Original image width
/// * `min_pixels` - Minimum total pixels
/// * `max_pixels` - Maximum total pixels
/// * `patch_size` - Vision encoder patch size
/// * `merge_size` - Spatial merge size
///
/// # Returns
/// Number of tokens that the image will produce in the LLM
pub fn get_number_of_image_tokens(
    height: usize,
    width: usize,
    min_pixels: Option<usize>,
    max_pixels: Option<usize>,
    patch_size: Option<usize>,
    merge_size: Option<usize>,
) -> usize {
    let merge_size = merge_size.unwrap_or(DEFAULT_MERGE_SIZE);
    let num_patches = get_number_of_image_patches(
        height,
        width,
        min_pixels,
        max_pixels,
        patch_size,
        Some(merge_size),
    );
    num_patches / (merge_size * merge_size)
}

/// Compute the number of video patches for a given video size.
///
/// # Arguments
/// * `num_frames` - Number of frames in the video
/// * `height` - Frame height
/// * `width` - Frame width
/// * `min_pixels` - Minimum total pixels per frame
/// * `max_pixels` - Maximum total pixels per frame
/// * `patch_size` - Vision encoder patch size
/// * `merge_size` - Spatial merge size
/// * `temporal_patch_size` - Temporal patch size
///
/// # Returns
/// Number of patches (before merge) that the video will produce
#[allow(clippy::too_many_arguments)]
pub fn get_number_of_video_patches(
    num_frames: usize,
    height: usize,
    width: usize,
    min_pixels: Option<usize>,
    max_pixels: Option<usize>,
    patch_size: Option<usize>,
    merge_size: Option<usize>,
    temporal_patch_size: Option<usize>,
) -> usize {
    let min_pixels = min_pixels.unwrap_or(DEFAULT_MIN_PIXELS);
    let max_pixels = max_pixels.unwrap_or(DEFAULT_MAX_PIXELS);
    let patch_size = patch_size.unwrap_or(DEFAULT_PATCH_SIZE);
    let merge_size = merge_size.unwrap_or(DEFAULT_MERGE_SIZE);
    let temporal_patch_size = temporal_patch_size.unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE);

    let factor = patch_size * merge_size;
    let (resized_height, resized_width) =
        smart_resize(height, width, factor, min_pixels, max_pixels);

    let grid_h = resized_height / patch_size;
    let grid_w = resized_width / patch_size;
    let grid_t = num_frames / temporal_patch_size;

    grid_t * grid_h * grid_w
}

/// Compute the number of video tokens (after spatial merge).
///
/// # Arguments
/// * `num_frames` - Number of frames in the video
/// * `height` - Frame height
/// * `width` - Frame width
/// * `min_pixels` - Minimum total pixels per frame
/// * `max_pixels` - Maximum total pixels per frame
/// * `patch_size` - Vision encoder patch size
/// * `merge_size` - Spatial merge size
/// * `temporal_patch_size` - Temporal patch size
///
/// # Returns
/// Number of tokens that the video will produce in the LLM
#[allow(clippy::too_many_arguments)]
pub fn get_number_of_video_tokens(
    num_frames: usize,
    height: usize,
    width: usize,
    min_pixels: Option<usize>,
    max_pixels: Option<usize>,
    patch_size: Option<usize>,
    merge_size: Option<usize>,
    temporal_patch_size: Option<usize>,
) -> usize {
    let merge_size = merge_size.unwrap_or(DEFAULT_MERGE_SIZE);
    let num_patches = get_number_of_video_patches(
        num_frames,
        height,
        width,
        min_pixels,
        max_pixels,
        patch_size,
        Some(merge_size),
        temporal_patch_size,
    );
    num_patches / (merge_size * merge_size)
}

/// Compute the grid dimensions (T, H, W) for an image.
///
/// # Arguments
/// * `height` - Original image height
/// * `width` - Original image width
/// * `min_pixels` - Minimum total pixels
/// * `max_pixels` - Maximum total pixels
/// * `patch_size` - Vision encoder patch size
/// * `merge_size` - Spatial merge size
///
/// # Returns
/// Tuple of (grid_t, grid_h, grid_w) where grid_t=1 for images
pub fn get_image_grid_thw(
    height: usize,
    width: usize,
    min_pixels: Option<usize>,
    max_pixels: Option<usize>,
    patch_size: Option<usize>,
    merge_size: Option<usize>,
) -> (usize, usize, usize) {
    let min_pixels = min_pixels.unwrap_or(DEFAULT_MIN_PIXELS);
    let max_pixels = max_pixels.unwrap_or(DEFAULT_MAX_PIXELS);
    let patch_size = patch_size.unwrap_or(DEFAULT_PATCH_SIZE);
    let merge_size = merge_size.unwrap_or(DEFAULT_MERGE_SIZE);

    let factor = patch_size * merge_size;
    let (resized_height, resized_width) =
        smart_resize(height, width, factor, min_pixels, max_pixels);

    let grid_h = resized_height / patch_size;
    let grid_w = resized_width / patch_size;

    (1, grid_h, grid_w) // T=1 for images
}

/// Compute the grid dimensions (T, H, W) for a video.
///
/// # Arguments
/// * `num_frames` - Number of frames in the video
/// * `height` - Frame height
/// * `width` - Frame width
/// * `min_pixels` - Minimum total pixels per frame
/// * `max_pixels` - Maximum total pixels per frame
/// * `patch_size` - Vision encoder patch size
/// * `merge_size` - Spatial merge size
/// * `temporal_patch_size` - Temporal patch size
///
/// # Returns
/// Tuple of (grid_t, grid_h, grid_w)
#[allow(clippy::too_many_arguments)]
pub fn get_video_grid_thw(
    num_frames: usize,
    height: usize,
    width: usize,
    min_pixels: Option<usize>,
    max_pixels: Option<usize>,
    patch_size: Option<usize>,
    merge_size: Option<usize>,
    temporal_patch_size: Option<usize>,
) -> (usize, usize, usize) {
    let min_pixels = min_pixels.unwrap_or(DEFAULT_MIN_PIXELS);
    let max_pixels = max_pixels.unwrap_or(DEFAULT_MAX_PIXELS);
    let patch_size = patch_size.unwrap_or(DEFAULT_PATCH_SIZE);
    let merge_size = merge_size.unwrap_or(DEFAULT_MERGE_SIZE);
    let temporal_patch_size = temporal_patch_size.unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE);

    let factor = patch_size * merge_size;
    let (resized_height, resized_width) =
        smart_resize(height, width, factor, min_pixels, max_pixels);

    let grid_t = num_frames / temporal_patch_size;
    let grid_h = resized_height / patch_size;
    let grid_w = resized_width / patch_size;

    (grid_t, grid_h, grid_w)
}

// ============================================================================
// Normalization Utilities
// ============================================================================

/// Normalize a single pixel value using CLIP-style normalization.
///
/// Formula: `(pixel / 255.0 - mean) / std`
#[inline]
pub fn normalize_pixel(value: u8, channel: usize) -> f32 {
    (value as f32 / 255.0 - IMAGE_MEAN[channel]) / IMAGE_STD[channel]
}

/// Normalize an RGB image to a channels-first f32 tensor.
///
/// # Arguments
/// * `rgb_data` - Raw RGB bytes in row-major order (H * W * 3)
/// * `height` - Image height
/// * `width` - Image width
///
/// # Returns
/// Normalized data in channels-first format (C, H, W) as a flat Vec<f32>
pub fn normalize_image(rgb_data: &[u8], height: usize, width: usize) -> Vec<f32> {
    let mut normalized = vec![0f32; 3 * height * width];

    for c in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let src_idx = (y * width + x) * 3 + c;
                let dst_idx = c * height * width + y * width + x;
                normalized[dst_idx] = normalize_pixel(rgb_data[src_idx], c);
            }
        }
    }

    normalized
}

// ============================================================================
// Patchification
// ============================================================================

/// Patchify an image following HuggingFace Qwen2.5-VL preprocessing.
///
/// The key operation is a 9D reshape + transpose:
/// 1. Start with image as (T, C, H, W) where T=temporal (padded to 2 for single images)
/// 2. Reshape to (grid_t, temporal_patch, C, grid_h/merge, merge, patch, grid_w/merge, merge, patch)
/// 3. Transpose to (grid_t, grid_h/merge, grid_w/merge, merge, merge, C, temporal_patch, patch, patch)
/// 4. Flatten to (num_patches, C * temporal_patch * patch * patch)
///
/// # Arguments
/// * `image_data` - Normalized image data in channels-first format (C, H, W)
/// * `height` - Image height (must be divisible by patch_size)
/// * `width` - Image width (must be divisible by patch_size)
/// * `patch_size` - Patch size (default: 14)
/// * `temporal_patch_size` - Temporal patch size (default: 2)
/// * `merge_size` - Spatial merge size (default: 2)
///
/// # Returns
/// Flattened patches as Vec<f32> with shape (num_patches, patch_elements)
pub fn patchify_image(
    image_data: &[f32],
    height: usize,
    width: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
) -> Vec<f32> {
    let channels = 3;
    let grid_h = height / patch_size;
    let grid_w = width / patch_size;
    let grid_t = 1; // Single image = 1 temporal frame

    let merged_h = grid_h / merge_size;
    let merged_w = grid_w / merge_size;
    let num_patches = grid_t * grid_h * grid_w;
    let patch_elements = channels * temporal_patch_size * patch_size * patch_size;

    let mut patches = vec![0f32; num_patches * patch_elements];

    // Iterate in the transposed order to fill patches correctly
    let mut patch_idx = 0;
    for _t in 0..grid_t {
        for mh in 0..merged_h {
            for mw in 0..merged_w {
                for sh in 0..merge_size {
                    for sw in 0..merge_size {
                        // This is one patch at position (mh*merge+sh, mw*merge+sw)
                        let patch_h = mh * merge_size + sh;
                        let patch_w = mw * merge_size + sw;
                        let h_start = patch_h * patch_size;
                        let w_start = patch_w * patch_size;

                        // Fill patch data: (C, temporal, patch_h, patch_w)
                        let mut elem_idx = 0;
                        for c in 0..channels {
                            for _t_sub in 0..temporal_patch_size {
                                // For single image, both temporal frames are the same
                                for ph in 0..patch_size {
                                    for pw in 0..patch_size {
                                        let y = h_start + ph;
                                        let x = w_start + pw;
                                        let src_idx = c * height * width + y * width + x;
                                        patches[patch_idx * patch_elements + elem_idx] =
                                            image_data[src_idx];
                                        elem_idx += 1;
                                    }
                                }
                            }
                        }
                        patch_idx += 1;
                    }
                }
            }
        }
    }

    patches
}

/// Patchify video frames following HuggingFace Qwen2.5-VL preprocessing.
///
/// Similar to image patchification but handles multiple temporal frames.
///
/// # Arguments
/// * `frames_data` - Vec of normalized frame data, each in channels-first format (C, H, W)
/// * `height` - Frame height (must be divisible by patch_size)
/// * `width` - Frame width (must be divisible by patch_size)
/// * `patch_size` - Patch size (default: 14)
/// * `temporal_patch_size` - Temporal patch size (default: 2)
/// * `merge_size` - Spatial merge size (default: 2)
///
/// # Returns
/// Flattened patches as Vec<f32> with shape (num_patches, patch_elements)
pub fn patchify_video(
    frames_data: &[Vec<f32>],
    height: usize,
    width: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
) -> Vec<f32> {
    let channels = 3;
    let grid_h = height / patch_size;
    let grid_w = width / patch_size;
    let num_frames = frames_data.len();

    // Pad frames to multiple of temporal_patch_size
    let grid_t = num_frames.div_ceil(temporal_patch_size);

    let merged_h = grid_h / merge_size;
    let merged_w = grid_w / merge_size;
    let num_patches = grid_t * grid_h * grid_w;
    let patch_elements = channels * temporal_patch_size * patch_size * patch_size;

    let mut patches = vec![0f32; num_patches * patch_elements];

    let mut patch_idx = 0;
    for t in 0..grid_t {
        for mh in 0..merged_h {
            for mw in 0..merged_w {
                for sh in 0..merge_size {
                    for sw in 0..merge_size {
                        let patch_h = mh * merge_size + sh;
                        let patch_w = mw * merge_size + sw;
                        let h_start = patch_h * patch_size;
                        let w_start = patch_w * patch_size;

                        let mut elem_idx = 0;
                        for c in 0..channels {
                            for t_sub in 0..temporal_patch_size {
                                // Get frame index, repeat last frame if padded
                                let frame_idx =
                                    (t * temporal_patch_size + t_sub).min(num_frames - 1);
                                let frame_data = &frames_data[frame_idx];

                                for ph in 0..patch_size {
                                    for pw in 0..patch_size {
                                        let y = h_start + ph;
                                        let x = w_start + pw;
                                        let src_idx = c * height * width + y * width + x;
                                        patches[patch_idx * patch_elements + elem_idx] =
                                            frame_data[src_idx];
                                        elem_idx += 1;
                                    }
                                }
                            }
                        }
                        patch_idx += 1;
                    }
                }
            }
        }
    }

    patches
}

// ============================================================================
// Image Processor Configuration
// ============================================================================

/// Configuration for Qwen2.5-VL image/video processing.
#[derive(Debug, Clone)]
pub struct ProcessorConfig {
    /// Minimum total pixels for an image.
    pub min_pixels: usize,
    /// Maximum total pixels for an image.
    pub max_pixels: usize,
    /// Vision encoder patch size.
    pub patch_size: usize,
    /// Spatial merge size for the patch merger.
    pub merge_size: usize,
    /// Temporal patch size for video frames.
    pub temporal_patch_size: usize,
    /// Normalization mean (RGB).
    pub image_mean: [f32; 3],
    /// Normalization std (RGB).
    pub image_std: [f32; 3],
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            min_pixels: DEFAULT_MIN_PIXELS,
            max_pixels: DEFAULT_MAX_PIXELS,
            patch_size: DEFAULT_PATCH_SIZE,
            merge_size: DEFAULT_MERGE_SIZE,
            temporal_patch_size: DEFAULT_TEMPORAL_PATCH_SIZE,
            image_mean: IMAGE_MEAN,
            image_std: IMAGE_STD,
        }
    }
}

impl ProcessorConfig {
    /// Create a new processor config with custom pixel bounds.
    pub fn with_pixel_bounds(min_pixels: usize, max_pixels: usize) -> Self {
        Self {
            min_pixels,
            max_pixels,
            ..Default::default()
        }
    }

    /// The factor that dimensions must be divisible by.
    pub fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }

    /// Compute target dimensions for an image.
    pub fn compute_size(&self, height: usize, width: usize) -> (usize, usize) {
        smart_resize(
            height,
            width,
            self.factor(),
            self.min_pixels,
            self.max_pixels,
        )
    }

    /// Compute the grid dimensions (T, H, W) for an image.
    pub fn compute_image_grid(&self, height: usize, width: usize) -> (usize, usize, usize) {
        let (h, w) = self.compute_size(height, width);
        (1, h / self.patch_size, w / self.patch_size)
    }

    /// Compute the grid dimensions (T, H, W) for a video.
    pub fn compute_video_grid(
        &self,
        num_frames: usize,
        height: usize,
        width: usize,
    ) -> (usize, usize, usize) {
        let (h, w) = self.compute_size(height, width);
        let grid_t = num_frames.div_ceil(self.temporal_patch_size);
        (grid_t, h / self.patch_size, w / self.patch_size)
    }

    /// Compute the number of LLM tokens for an image.
    pub fn compute_image_tokens(&self, height: usize, width: usize) -> usize {
        let (_, grid_h, grid_w) = self.compute_image_grid(height, width);
        (grid_h / self.merge_size) * (grid_w / self.merge_size)
    }

    /// Compute the number of LLM tokens for a video.
    pub fn compute_video_tokens(&self, num_frames: usize, height: usize, width: usize) -> usize {
        let (grid_t, grid_h, grid_w) = self.compute_video_grid(num_frames, height, width);
        grid_t * (grid_h / self.merge_size) * (grid_w / self.merge_size)
    }

    /// Size of each patch in elements.
    pub fn patch_elements(&self) -> usize {
        3 * self.temporal_patch_size * self.patch_size * self.patch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smart_resize_basic() {
        // 224x224 with factor 28 should stay approximately the same
        let (h, w) = smart_resize(224, 224, 28, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS);
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }

    #[test]
    fn test_smart_resize_scale_up() {
        // Very small image should be scaled up
        let (h, w) = smart_resize(28, 28, 28, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS);
        assert!(h * w >= DEFAULT_MIN_PIXELS);
    }

    #[test]
    fn test_smart_resize_scale_down() {
        // Very large image should be scaled down
        let (h, w) = smart_resize(4000, 4000, 28, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS);
        assert!(h * w <= DEFAULT_MAX_PIXELS);
    }

    #[test]
    fn test_image_patches() {
        // 224x224 image with patch_size=14, merge_size=2
        // factor = 28, so 224 stays 224
        // grid_h = 224/14 = 16, grid_w = 16
        // patches = 16 * 16 = 256
        let patches = get_number_of_image_patches(224, 224, None, None, None, None);
        assert_eq!(patches, 256);
    }

    #[test]
    fn test_image_tokens() {
        // 256 patches / (2*2) = 64 tokens
        let tokens = get_number_of_image_tokens(224, 224, None, None, None, None);
        assert_eq!(tokens, 64);
    }

    #[test]
    fn test_video_patches() {
        // 4 frames, 224x224
        // grid_t = 4/2 = 2
        // grid_h = 16, grid_w = 16
        // patches = 2 * 16 * 16 = 512
        let patches = get_number_of_video_patches(4, 224, 224, None, None, None, None, None);
        assert_eq!(patches, 512);
    }

    #[test]
    fn test_video_tokens() {
        // 512 patches / 4 = 128 tokens
        let tokens = get_number_of_video_tokens(4, 224, 224, None, None, None, None, None);
        assert_eq!(tokens, 128);
    }

    #[test]
    fn test_image_grid_thw() {
        let (t, h, w) = get_image_grid_thw(224, 224, None, None, None, None);
        assert_eq!(t, 1);
        assert_eq!(h, 16);
        assert_eq!(w, 16);
    }

    #[test]
    fn test_video_grid_thw() {
        let (t, h, w) = get_video_grid_thw(8, 224, 224, None, None, None, None, None);
        assert_eq!(t, 4); // 8 frames / 2 temporal_patch_size
        assert_eq!(h, 16);
        assert_eq!(w, 16);
    }
}
