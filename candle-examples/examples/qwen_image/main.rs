//! Qwen-Image: Text-to-image generation with Qwen diffusion models.
//!
//! This example demonstrates text-to-image generation using the Qwen-Image model,
//! with support for both full-precision and quantized (GGUF) inference.
//!
//! # Usage
//!
//! ```bash
//! # Basic text-to-image with quantized model (low VRAM)
//! cargo run --release --features cuda --example qwen_image -- \
//!     --gguf-transformer --gguf-text-encoder \
//!     generate --prompt "A serene mountain landscape at sunset" \
//!     --output landscape.png
//!
//! # With streaming for very low VRAM (~4GB)
//! cargo run --release --features cuda --example qwen_image -- \
//!     --gguf-transformer --gguf-text-encoder --streaming \
//!     generate --prompt "A cat sitting on a windowsill" \
//!     --output cat.png
//!
//! # Full precision (requires more VRAM)
//! cargo run --release --features cuda --example qwen_image -- \
//!     generate --prompt "An astronaut riding a horse on Mars" \
//!     --height 1024 --width 1024 \
//!     --output astronaut.png
//! ```

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

mod common;
mod generate;
mod mt_box_muller_rng;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "qwen-image",
    about = "Qwen-Image: Text-to-image generation with diffusion models",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Run on CPU instead of GPU.
    #[arg(long, global = true)]
    cpu: bool,

    /// Use F32 dtype instead of mixed precision (BF16/F16).
    #[arg(long, global = true)]
    use_f32: bool,

    /// Use F16 dtype instead of BF16 for lower memory usage.
    #[arg(long, global = true)]
    use_f16: bool,

    /// Enable Chrome tracing profiler.
    #[arg(long, global = true)]
    tracing: bool,

    /// Random seed for reproducibility.
    #[arg(long, global = true)]
    seed: Option<u64>,

    /// Local path to transformer weights.
    #[arg(long, global = true)]
    transformer_path: Option<String>,

    /// Local path to VAE weights.
    #[arg(long, global = true)]
    vae_path: Option<String>,

    /// Local path to text encoder weights.
    #[arg(long, global = true)]
    text_encoder_path: Option<String>,

    /// Local path to tokenizer.
    #[arg(long, global = true)]
    tokenizer_path: Option<String>,

    /// Quantized GGUF diffusion transformer. Accepts:
    /// - --gguf-transformer → uses default from HuggingFace
    /// - --gguf-transformer=/path/to/model.gguf → local file
    /// - --gguf-transformer=owner/repo/file.gguf → downloads from HuggingFace
    #[arg(long, global = true, num_args = 0..=1, default_missing_value = "auto", require_equals = true)]
    gguf_transformer: Option<String>,

    /// Quantized GGUF text encoder. Same format as --gguf-transformer.
    #[arg(long, global = true, num_args = 0..=1, default_missing_value = "auto", require_equals = true)]
    gguf_text_encoder: Option<String>,

    // =========================================================================
    // Memory optimization flags
    // =========================================================================
    /// Enable VAE slicing to reduce memory usage for batch processing.
    #[arg(long, global = true)]
    enable_vae_slicing: bool,

    /// Enable VAE tiling to process large images with less memory.
    #[arg(long, global = true)]
    enable_vae_tiling: bool,

    /// Tile size for VAE tiling (pixels).
    #[arg(long, global = true, default_value_t = 256)]
    vae_tile_size: usize,

    /// Tile stride for VAE tiling (pixels).
    #[arg(long, global = true, default_value_t = 192)]
    vae_tile_stride: usize,

    /// Enable text Q/K/V caching for CFG optimization.
    #[arg(long, global = true)]
    enable_text_cache: bool,

    /// Enable streaming mode for GGUF transformer to reduce GPU memory.
    /// Loads transformer blocks on-demand during inference.
    #[arg(long, global = true)]
    streaming: bool,

    /// Upcast attention to F32 for numerical stability.
    #[arg(long, global = true)]
    upcast_attention: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Text-to-image generation (with optional img2img).
    Generate {
        /// The prompt to generate an image from.
        #[arg(long, default_value = "A serene mountain landscape at sunset")]
        prompt: String,

        /// Negative prompt describing what to avoid.
        #[arg(long, default_value = "")]
        negative_prompt: String,

        /// Height of the generated image (must be divisible by 16).
        #[arg(long, default_value_t = 512)]
        height: usize,

        /// Width of the generated image (must be divisible by 16).
        #[arg(long, default_value_t = 512)]
        width: usize,

        /// Number of denoising steps.
        #[arg(long, default_value_t = 20)]
        steps: usize,

        /// True CFG guidance scale.
        #[arg(long, default_value_t = 4.0)]
        true_cfg_scale: f64,

        /// Input image for img2img mode (optional).
        #[arg(long)]
        init_image: Option<String>,

        /// Strength for img2img (0.0 = no change, 1.0 = full generation).
        #[arg(long, default_value_t = 0.75)]
        strength: f64,

        /// Output filename.
        #[arg(long, default_value = "qwen_image_output.png")]
        output: String,

        /// HuggingFace model ID for the transformer.
        #[arg(long, default_value = "Qwen/Qwen-Image-2512")]
        model_id: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Setup tracing
    let _guard = common::setup_tracing(cli.tracing);

    // Setup device and dtype
    let (device, dtype) =
        common::setup_device_and_dtype(cli.cpu, cli.use_f32, cli.use_f16, cli.seed)?;

    match cli.command {
        Command::Generate {
            prompt,
            negative_prompt,
            height,
            width,
            steps,
            true_cfg_scale,
            init_image,
            strength,
            output,
            model_id,
        } => {
            let args = generate::GenerateArgs {
                prompt,
                negative_prompt,
                height,
                width,
                steps,
                true_cfg_scale,
                init_image,
                strength,
                output,
                seed: cli.seed,
                model_id,
                enable_vae_slicing: cli.enable_vae_slicing,
                enable_vae_tiling: cli.enable_vae_tiling,
                vae_tile_size: cli.vae_tile_size,
                vae_tile_stride: cli.vae_tile_stride,
                enable_text_cache: cli.enable_text_cache,
                streaming: cli.streaming,
                upcast_attention: cli.upcast_attention,
            };

            let paths = generate::ModelPaths {
                transformer_path: cli.transformer_path,
                gguf_transformer_path: cli.gguf_transformer,
                vae_path: cli.vae_path,
                text_encoder_path: cli.text_encoder_path,
                gguf_text_encoder_path: cli.gguf_text_encoder,
                tokenizer_path: cli.tokenizer_path,
            };

            generate::run(args, paths, &device, dtype)
        }
    }
}
