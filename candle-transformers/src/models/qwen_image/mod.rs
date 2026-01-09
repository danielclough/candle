//! Qwen-Image: A 20B parameter Multimodal Diffusion Transformer for text-to-image generation.
//!
//! This module implements the Qwen-Image model architecture in Candle, featuring:
//! - **Dual-stream MMDiT**: Text and image streams processed jointly through attention
//! - **3D RoPE**: Rotary position embeddings for spatial (height, width) and temporal (frame) dimensions
//! - **3D Causal VAE**: Video-compatible VAE with 8× spatial compression and feature caching
//! - **FlowMatch Scheduler**: Euler discrete scheduler with dynamic shift for varying resolutions
//!
//! # Architecture Overview
//!
//! ```text
//! Text Prompt → Qwen2.5-VL Encoder → Text Embeddings (3584-dim)
//!                                           ↓
//!                            ┌──────────────┴──────────────┐
//!                            │   Dual-Stream Transformer   │
//!                            │     (60 blocks, 3072-dim)   │
//!                            │                             │
//!                            │  Image Stream ←→ Text Stream│
//!                            │   (Joint Attention + MLP)   │
//!                            └──────────────┬──────────────┘
//!                                           ↓
//!                                    Output Latents
//!                                           ↓
//!                            ┌──────────────┴──────────────┐
//!                            │      3D Causal VAE          │
//!                            │   (16 channels, 8× spatial) │
//!                            └──────────────┬──────────────┘
//!                                           ↓
//!                                     RGB Image
//! ```
//!
//! # Key Differences from Flux
//!
//! | Aspect | Flux | Qwen-Image |
//! |--------|------|------------|
//! | **RoPE** | Real-valued (cos/sin) | Complex multiplication |
//! | **Position Encoding** | Standard positive | Center-aligned with negatives |
//! | **Single-stream Phase** | Yes | No (dual-stream only) |
//! | **Context Dim** | 4096 (T5/CLIP) | 3584 (Qwen2.5-VL) |
//! | **VAE** | 2D AutoencoderKL | 3D Causal VAE |
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --example qwen_image -- \
//!     --prompt "A serene mountain landscape at sunset" \
//!     --height 1024 --width 1024 \
//!     --output landscape.png
//! ```
//!
//! # References
//!
//! - Model: <https://huggingface.co/Qwen/Qwen-Image>
//! - Paper: Qwen-Image Technical Report

pub mod config;
pub mod controlnet;
pub mod debug;
pub mod edit_helpers;
pub mod pipeline;
pub mod rope;
pub mod scheduler;
pub mod vae;

mod blocks;
mod model;

// Re-export debug types for block overrides
pub use debug::{BlockOverrides, load_block_overrides, is_block_override_enabled, set_cfg_pass};

pub use config::{Config, SchedulerConfig, VaeConfig};
pub use controlnet::{ControlNetConfig, ControlNetOutput, QwenImageControlNetModel};
pub use edit_helpers::{
    calculate_dimensions, calculate_dimensions_with_resolution, extract_and_pad_embeddings,
    pack_layered_latents, unpack_layered_latents, PromptMode, CAPTION_PROMPT_CN, CAPTION_PROMPT_EN,
    EDIT_DROP_TOKENS, EDIT_PROMPT_TEMPLATE, LAYERED_DROP_TOKENS, LAYERED_PROMPT_TEMPLATE,
    TEXT_ONLY_DROP_TOKENS, TEXT_ONLY_PROMPT_TEMPLATE,
};
pub use model::{pack_latents, unpack_latents, QwenImageTransformer2DModel, QwenTimestepProjEmbeddings};
pub use rope::{apply_rotary_emb_qwen, QwenEmbedRope};
pub use scheduler::{calculate_shift, FlowMatchEulerDiscreteScheduler};
pub use pipeline::{
    apply_true_cfg, compute_vision_size, expand_image_tokens, prepare_image_for_vae,
    prepare_image_for_vision, IMAGE_TOKEN_ID, VISION_MERGE_SIZE, VISION_PATCH_SIZE,
    VISION_TEMPORAL_PATCH_SIZE,
};
pub use vae::{AutoencoderKLQwenImage, TiledDecodeConfig};
