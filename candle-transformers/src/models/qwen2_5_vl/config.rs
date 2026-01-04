//! Configuration for Qwen2.5-VL Vision-Language Model.
//!
//! Qwen2.5-VL is a multimodal model that combines a vision encoder (ViT with 2D RoPE)
//! with a Qwen2.5 text decoder using Multimodal RoPE (M-RoPE) for position encoding.
//!
//! Available model sizes: 3B, 7B, 72B (no 2B variant exists).

use candle_nn::Activation;
use serde::Deserialize;

// ============================================================================
// Vision Configuration
// ============================================================================

fn default_depth() -> usize {
    32
}
fn default_vision_hidden_size() -> usize {
    1280
}
fn default_vision_intermediate_size() -> usize {
    3420
}
fn default_num_heads() -> usize {
    16
}
fn default_in_channels() -> usize {
    3
}
fn default_out_hidden_size() -> usize {
    3584 // 7B default, overridden by config
}
fn default_patch_size() -> usize {
    14
}
fn default_spatial_merge_size() -> usize {
    2
}
fn default_temporal_patch_size() -> usize {
    2
}
fn default_fullatt_block_indexes() -> Vec<usize> {
    vec![7, 15, 23, 31]
}
fn default_window_size() -> usize {
    112
}
fn default_tokens_per_second() -> usize {
    4
}
fn default_vision_hidden_act() -> Activation {
    Activation::Silu
}

/// Vision encoder configuration for Qwen2.5-VL.
///
/// These values are mostly consistent across all model sizes (3B, 7B, 72B),
/// except for `out_hidden_size` which must match the text model's hidden_size.
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    /// Number of transformer blocks in the vision encoder.
    #[serde(default = "default_depth")]
    pub depth: usize,

    /// Hidden dimension of the vision encoder.
    #[serde(default = "default_vision_hidden_size")]
    pub hidden_size: usize,

    /// Intermediate dimension in the MLP.
    #[serde(default = "default_vision_intermediate_size")]
    pub intermediate_size: usize,

    /// Number of attention heads.
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,

    /// Number of input image channels.
    #[serde(default = "default_in_channels")]
    pub in_chans: usize,

    /// Output hidden size - must match text model's hidden_size.
    /// 2048 (3B) / 3584 (7B) / 8192 (72B)
    #[serde(default = "default_out_hidden_size")]
    pub out_hidden_size: usize,

    /// Patch size for the vision encoder.
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,

    /// Spatial merge size (2x2 pooling in the merger).
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,

    /// Temporal patch size for video frames.
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,

    /// Block indices that use full attention (others use window attention).
    #[serde(default = "default_fullatt_block_indexes")]
    pub fullatt_block_indexes: Vec<usize>,

    /// Window size for windowed attention layers.
    #[serde(default = "default_window_size")]
    pub window_size: usize,

    /// Tokens per second for video temporal position encoding.
    #[serde(default = "default_tokens_per_second")]
    pub tokens_per_second: usize,

    /// Activation function (SiLU for Qwen2.5-VL).
    #[serde(default = "default_vision_hidden_act")]
    pub hidden_act: Activation,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            depth: default_depth(),
            hidden_size: default_vision_hidden_size(),
            intermediate_size: default_vision_intermediate_size(),
            num_heads: default_num_heads(),
            in_chans: default_in_channels(),
            out_hidden_size: default_out_hidden_size(),
            patch_size: default_patch_size(),
            spatial_merge_size: default_spatial_merge_size(),
            temporal_patch_size: default_temporal_patch_size(),
            fullatt_block_indexes: default_fullatt_block_indexes(),
            window_size: default_window_size(),
            tokens_per_second: default_tokens_per_second(),
            hidden_act: default_vision_hidden_act(),
        }
    }
}

impl VisionConfig {
    /// Head dimension for attention.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

// ============================================================================
// RoPE Scaling Configuration
// ============================================================================

fn default_mrope_section() -> Vec<usize> {
    vec![16, 24, 24]
}

/// RoPE scaling configuration for multimodal position embeddings.
///
/// M-RoPE (Multimodal RoPE) splits the head_dim into sections for
/// temporal, height, and width position encoding.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    /// Sections for multimodal RoPE: [temporal, height, width].
    /// Total must equal head_dim/2 = 64 for head_dim=128.
    /// Default: [16, 24, 24]
    #[serde(default = "default_mrope_section")]
    pub mrope_section: Vec<usize>,

    /// RoPE type identifier.
    #[serde(default)]
    pub rope_type: Option<String>,
}

impl Default for RopeScaling {
    fn default() -> Self {
        Self {
            mrope_section: default_mrope_section(),
            rope_type: Some("mrope".to_string()),
        }
    }
}

// ============================================================================
// Main Configuration
// ============================================================================

fn default_vocab_size() -> usize {
    152064
}
fn default_hidden_size() -> usize {
    3584 // 7B default
}
fn default_intermediate_size() -> usize {
    18944 // 7B default
}
fn default_num_hidden_layers() -> usize {
    28 // 7B default
}
fn default_num_attention_heads() -> usize {
    28 // 7B default
}
fn default_num_key_value_heads() -> usize {
    4 // 7B default
}
fn default_head_dim() -> usize {
    128
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    1_000_000.0
}
fn default_max_position_embeddings() -> usize {
    128000 // 128K context
}
fn default_hidden_act() -> Activation {
    Activation::Silu
}
fn default_image_token_id() -> u32 {
    151655
}
fn default_video_token_id() -> u32 {
    151656
}
fn default_vision_start_token_id() -> u32 {
    151652
}
fn default_vision_end_token_id() -> u32 {
    151653
}
fn default_sliding_window() -> usize {
    4096
}
fn default_max_window_layers() -> usize {
    80
}

/// Combined configuration for Qwen2.5-VL model.
///
/// The text model parameters are at the top level (not nested in `text_config`),
/// following the HuggingFace format.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Vision encoder configuration.
    #[serde(default)]
    pub vision_config: VisionConfig,

    /// Vocabulary size for the text model.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    /// Hidden size of the text model.
    /// 2048 (3B) / 3584 (7B) / 8192 (72B)
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    /// Intermediate size in the MLP.
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,

    /// Number of transformer layers.
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,

    /// Number of attention heads.
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,

    /// Number of key-value heads for GQA.
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,

    /// Head dimension for attention.
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,

    /// RMS normalization epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,

    /// RoPE base frequency.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    /// Maximum position embeddings (128K context).
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    /// Activation function.
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,

    /// Whether to tie word embeddings with lm_head.
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// Image placeholder token ID.
    #[serde(default = "default_image_token_id")]
    pub image_token_id: u32,

    /// Video placeholder token ID.
    #[serde(default = "default_video_token_id")]
    pub video_token_id: u32,

    /// Vision start token ID.
    #[serde(default = "default_vision_start_token_id")]
    pub vision_start_token_id: u32,

    /// Vision end token ID.
    #[serde(default = "default_vision_end_token_id")]
    pub vision_end_token_id: u32,

    /// RoPE scaling configuration with mrope_section.
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,

    /// Whether to use sliding window attention.
    #[serde(default)]
    pub use_sliding_window: bool,

    /// Sliding window size (default: 4096).
    /// Only used when use_sliding_window is true.
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,

    /// Layers >= this index use sliding window attention (default: 80).
    /// For 7B model with 28 layers, this means no layers use sliding window by default.
    #[serde(default = "default_max_window_layers")]
    pub max_window_layers: usize,

    /// Whether to use flash attention (requires flash-attn feature and CUDA).
    #[serde(default)]
    pub use_flash_attn: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vision_config: VisionConfig::default(),
            vocab_size: default_vocab_size(),
            hidden_size: default_hidden_size(),
            intermediate_size: default_intermediate_size(),
            num_hidden_layers: default_num_hidden_layers(),
            num_attention_heads: default_num_attention_heads(),
            num_key_value_heads: default_num_key_value_heads(),
            head_dim: default_head_dim(),
            rms_norm_eps: default_rms_norm_eps(),
            rope_theta: default_rope_theta(),
            max_position_embeddings: default_max_position_embeddings(),
            hidden_act: default_hidden_act(),
            tie_word_embeddings: false,
            image_token_id: default_image_token_id(),
            video_token_id: default_video_token_id(),
            vision_start_token_id: default_vision_start_token_id(),
            vision_end_token_id: default_vision_end_token_id(),
            rope_scaling: Some(RopeScaling::default()),
            use_sliding_window: false,
            sliding_window: default_sliding_window(),
            max_window_layers: default_max_window_layers(),
            use_flash_attn: false,
        }
    }
}

impl Config {
    /// Get the mrope_section from rope_scaling, or use default.
    pub fn mrope_section(&self) -> Vec<usize> {
        self.rope_scaling
            .as_ref()
            .map(|rs| rs.mrope_section.clone())
            .unwrap_or_else(default_mrope_section)
    }

    /// Number of KV groups for grouped-query attention.
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Check if a layer uses sliding window attention.
    ///
    /// Returns true if sliding window is enabled AND the layer index >= max_window_layers.
    pub fn uses_sliding_window(&self, layer_idx: usize) -> bool {
        self.use_sliding_window && layer_idx >= self.max_window_layers
    }

    /// Get the sliding window size for a layer, or None if full attention.
    pub fn get_sliding_window(&self, layer_idx: usize) -> Option<usize> {
        if self.uses_sliding_window(layer_idx) {
            Some(self.sliding_window)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = Config::default();
        assert_eq!(cfg.vocab_size, 152064);
        assert_eq!(cfg.hidden_size, 3584);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.max_position_embeddings, 128000);
        assert_eq!(cfg.image_token_id, 151655);
    }

    #[test]
    fn test_mrope_section() {
        let cfg = Config::default();
        let section = cfg.mrope_section();
        assert_eq!(section, vec![16, 24, 24]);
        assert_eq!(section.iter().sum::<usize>(), 64); // head_dim / 2
    }

    #[test]
    fn test_vision_config() {
        let cfg = VisionConfig::default();
        assert_eq!(cfg.depth, 32);
        assert_eq!(cfg.hidden_size, 1280);
        assert_eq!(cfg.head_dim(), 80); // 1280 / 16
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.spatial_merge_size, 2);
    }

    #[test]
    fn test_sliding_window_defaults() {
        let cfg = Config::default();
        assert!(!cfg.use_sliding_window);
        assert_eq!(cfg.sliding_window, 4096);
        assert_eq!(cfg.max_window_layers, 80);
        assert!(!cfg.use_flash_attn);
    }

    #[test]
    fn test_sliding_window_layer_detection() {
        let cfg = Config {
            use_sliding_window: true,
            max_window_layers: 20,
            num_hidden_layers: 28,
            ..Default::default()
        };

        // Layers 0-19 should use full attention
        assert!(!cfg.uses_sliding_window(0));
        assert!(!cfg.uses_sliding_window(10));
        assert!(!cfg.uses_sliding_window(19));

        // Layers 20-27 should use sliding window
        assert!(cfg.uses_sliding_window(20));
        assert!(cfg.uses_sliding_window(25));
        assert!(cfg.uses_sliding_window(27));

        // Test get_sliding_window helper
        assert_eq!(cfg.get_sliding_window(10), None);
        assert_eq!(cfg.get_sliding_window(20), Some(4096));
    }

    #[test]
    fn test_sliding_window_disabled() {
        let cfg = Config {
            use_sliding_window: false,
            max_window_layers: 0, // Would enable all layers if use_sliding_window were true
            ..Default::default()
        };

        // All layers should use full attention when sliding window is disabled
        for layer_idx in 0..cfg.num_hidden_layers {
            assert!(!cfg.uses_sliding_window(layer_idx));
            assert_eq!(cfg.get_sliding_window(layer_idx), None);
        }
    }
}
