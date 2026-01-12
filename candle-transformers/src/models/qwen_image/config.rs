//! Configuration structures for Qwen-Image model components.
//!
//! This module contains configuration structs for the transformer, VAE, and scheduler
//! components of the Qwen-Image text-to-image generation model.

use serde::Deserialize;

/// Configuration for the Qwen-Image Transformer (MMDiT architecture).
///
/// The transformer uses a dual-stream architecture where image and text streams
/// are processed jointly through attention, then separately through MLPs.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    /// Patch size for patchifying latents (default: 2).
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,

    /// Number of input channels after packing (default: 64 = 16 z_dim × 4 from 2×2 packing).
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,

    /// Number of output channels matching VAE z_dim (default: 16).
    #[serde(default = "default_out_channels")]
    pub out_channels: usize,

    /// Number of transformer blocks (default: 60).
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,

    /// Dimension per attention head (default: 128).
    #[serde(default = "default_attention_head_dim")]
    pub attention_head_dim: usize,

    /// Number of attention heads (default: 24).
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,

    /// Dimension of text encoder embeddings from Qwen2.5-VL (default: 3584).
    #[serde(default = "default_joint_attention_dim")]
    pub joint_attention_dim: usize,

    /// RoPE dimensions for [frame, height, width] axes (default: [16, 56, 56]).
    #[serde(default = "default_axes_dims_rope")]
    pub axes_dims_rope: (usize, usize, usize),

    /// RoPE theta for frequency computation (default: 10000).
    #[serde(default = "default_theta")]
    pub theta: usize,

    /// Whether to use zero conditioning for timestep (for CFG) (default: false).
    #[serde(default)]
    pub zero_cond_t: bool,

    /// Whether to use additional timestep conditioning (default: false).
    #[serde(default)]
    pub use_additional_t_cond: bool,

    /// Whether to use layer3d rope for layered generation (default: false).
    #[serde(default)]
    pub use_layer3d_rope: bool,
}

fn default_patch_size() -> usize {
    2
}
fn default_in_channels() -> usize {
    64
}
fn default_out_channels() -> usize {
    16
}
fn default_num_layers() -> usize {
    60
}
fn default_attention_head_dim() -> usize {
    128
}
fn default_num_attention_heads() -> usize {
    24
}
fn default_joint_attention_dim() -> usize {
    3584
}
fn default_axes_dims_rope() -> (usize, usize, usize) {
    (16, 56, 56)
}
fn default_theta() -> usize {
    10000
}

impl Default for Config {
    fn default() -> Self {
        Self::qwen_image()
    }
}

impl Config {
    /// Default configuration for Qwen-Image base model (20B parameters).
    pub fn qwen_image() -> Self {
        Self {
            patch_size: 2,
            in_channels: 64,
            out_channels: 16,
            num_layers: 60,
            attention_head_dim: 128,
            num_attention_heads: 24,
            joint_attention_dim: 3584,
            axes_dims_rope: (16, 56, 56),
            theta: 10000,
            zero_cond_t: false,
            use_additional_t_cond: false,
            use_layer3d_rope: false,
        }
    }

    /// Configuration for Qwen-Image Edit model (legacy).
    ///
    /// The Edit model uses the same architecture as the base model but is
    /// fine-tuned for image editing tasks with vision-language conditioning.
    /// Key difference: `zero_cond_t: true` enables per-token modulation where:
    /// - Noise latents (first sequence) use timestep-based modulation
    /// - Reference image latents (subsequent sequences) use zero-timestep modulation
    pub fn qwen_image_edit() -> Self {
        Self {
            zero_cond_t: true, // Enables per-token modulation for edit mode
            ..Self::qwen_image()
        }
    }

    /// Configuration for Qwen-Image Edit Plus model (2509/2511+).
    ///
    /// Enhanced edit model with dual-stream image conditioning:
    /// - Low-res (384px) condition image for vision encoder understanding
    /// - High-res (1024px) VAE image for detail preservation
    ///
    /// Uses `zero_cond_t: true` for per-token modulation (same as Edit).
    pub fn qwen_image_edit_plus() -> Self {
        Self {
            zero_cond_t: true,
            ..Self::qwen_image()
        }
    }

    /// Configuration for Qwen-Image Layered model.
    ///
    /// The Layered model decomposes images into transparent layers with:
    /// - `use_layer3d_rope: true` - Enables 3D RoPE for layer dimension
    /// - `use_additional_t_cond: true` - Enables is_rgb conditioning
    pub fn qwen_image_layered() -> Self {
        Self {
            patch_size: 2,
            in_channels: 64,
            out_channels: 16,
            num_layers: 60,
            attention_head_dim: 128,
            num_attention_heads: 24,
            joint_attention_dim: 3584,
            axes_dims_rope: (16, 56, 56),
            theta: 10000,
            zero_cond_t: false,
            use_additional_t_cond: true, // Enables is_rgb conditioning for RGBA output
            use_layer3d_rope: true,      // Enables 3D RoPE for layer dimension
        }
    }

    /// Inner dimension = num_heads × head_dim = 24 × 128 = 3072.
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// Total RoPE dimension = sum of axes dimensions = 16 + 56 + 56 = 128.
    pub fn rope_dim(&self) -> usize {
        self.axes_dims_rope.0 + self.axes_dims_rope.1 + self.axes_dims_rope.2
    }

    /// MLP hidden dimension (4x expansion like standard transformers).
    pub fn mlp_dim(&self) -> usize {
        self.inner_dim() * 4
    }
}

/// Configuration for the Qwen-Image 3D Causal VAE.
///
/// This VAE is derived from Wan Video VAE and uses causal 3D convolutions
/// for temporal consistency. It features iterative encoding/decoding with
/// feature caching for memory efficiency.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct VaeConfig {
    /// Base channel dimension (default: 96).
    #[serde(default = "default_base_dim")]
    pub base_dim: usize,

    /// Latent space dimension (default: 16).
    #[serde(default = "default_z_dim")]
    pub z_dim: usize,

    /// Channel multipliers for each block (default: [1, 2, 4, 4]).
    #[serde(default = "default_dim_mult")]
    pub dim_mult: Vec<usize>,

    /// Number of residual blocks per stage (default: 2).
    #[serde(default = "default_num_res_blocks")]
    pub num_res_blocks: usize,

    /// Which blocks perform temporal downsampling (default: [false, true, true]).
    #[serde(default = "default_temporal_downsample")]
    pub temporal_downsample: Vec<bool>,

    /// Number of input/output image channels (default: 3 for RGB).
    #[serde(default = "default_input_channels")]
    pub input_channels: usize,

    /// Dropout rate (default: 0.0).
    #[serde(default)]
    pub dropout: f32,

    /// Latent normalization mean values (16 channels).
    #[serde(default = "default_latents_mean")]
    pub latents_mean: Vec<f32>,

    /// Latent normalization std values (16 channels).
    #[serde(default = "default_latents_std")]
    pub latents_std: Vec<f32>,
}

fn default_base_dim() -> usize {
    96
}
fn default_z_dim() -> usize {
    16
}
fn default_dim_mult() -> Vec<usize> {
    vec![1, 2, 4, 4]
}
fn default_num_res_blocks() -> usize {
    2
}
fn default_temporal_downsample() -> Vec<bool> {
    vec![false, true, true]
}
fn default_input_channels() -> usize {
    3
}
fn default_latents_mean() -> Vec<f32> {
    vec![
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715,
        0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    ]
}
fn default_latents_std() -> Vec<f32> {
    vec![
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652,
        1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
    ]
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self::qwen_image()
    }
}

impl VaeConfig {
    /// Default VAE configuration for Qwen-Image.
    pub fn qwen_image() -> Self {
        Self {
            base_dim: 96,
            z_dim: 16,
            dim_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            temporal_downsample: vec![false, true, true],
            input_channels: 3,
            dropout: 0.0,
            latents_mean: default_latents_mean(),
            latents_std: default_latents_std(),
        }
    }

    /// Spatial compression ratio (2^num_downsample_blocks = 8).
    pub fn spatial_compression_ratio(&self) -> usize {
        1 << self.temporal_downsample.len()
    }

    /// Temporal upsample flags (reverse of downsample).
    pub fn temporal_upsample(&self) -> Vec<bool> {
        self.temporal_downsample.iter().rev().copied().collect()
    }
}

/// Configuration for the FlowMatch Euler Discrete Scheduler.
///
/// This scheduler implements the flow matching formulation where the model
/// learns a velocity field, and inference uses Euler integration.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SchedulerConfig {
    /// Number of training timesteps (default: 1000).
    #[serde(default = "default_num_train_timesteps")]
    pub num_train_timesteps: usize,

    /// Whether to use dynamic shifting based on image size (default: true).
    #[serde(default = "default_use_dynamic_shifting")]
    pub use_dynamic_shifting: bool,

    /// Base shift value for dynamic shifting (default: 0.5).
    #[serde(default = "default_base_shift")]
    pub base_shift: f64,

    /// Maximum shift value for dynamic shifting (default: 1.15).
    #[serde(default = "default_max_shift")]
    pub max_shift: f64,

    /// Base image sequence length for shift calculation (default: 256).
    #[serde(default = "default_base_image_seq_len")]
    pub base_image_seq_len: usize,

    /// Maximum image sequence length for shift calculation (default: 4096).
    #[serde(default = "default_max_image_seq_len")]
    pub max_image_seq_len: usize,
}

fn default_num_train_timesteps() -> usize {
    1000
}
fn default_use_dynamic_shifting() -> bool {
    true
}
fn default_base_shift() -> f64 {
    0.5
}
fn default_max_shift() -> f64 {
    1.15
}
fn default_base_image_seq_len() -> usize {
    256
}
fn default_max_image_seq_len() -> usize {
    4096
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self::qwen_image()
    }
}

impl SchedulerConfig {
    /// Default scheduler configuration for Qwen-Image.
    pub fn qwen_image() -> Self {
        Self {
            num_train_timesteps: 1000,
            use_dynamic_shifting: true,
            base_shift: 0.5,
            max_shift: 1.15,
            base_image_seq_len: 256,
            max_image_seq_len: 4096,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = Config::qwen_image();
        assert_eq!(config.inner_dim(), 3072); // 24 * 128
        assert_eq!(config.rope_dim(), 128); // 16 + 56 + 56
        assert_eq!(config.mlp_dim(), 12288); // 3072 * 4
    }

    #[test]
    fn test_vae_config_defaults() {
        let config = VaeConfig::qwen_image();
        assert_eq!(config.spatial_compression_ratio(), 8); // 2^3
        assert_eq!(config.temporal_upsample(), vec![true, true, false]);
    }
}
