//! Qwen-Image: Unified CLI for all Qwen-Image generation pipelines.
//!
//! This example provides a single entry point for:
//! - **generate**: Text-to-image generation (with optional img2img)
//! - **edit**: Image editing with text instructions
//! - **inpaint**: Fill in masked regions of an image
//! - **layered**: Decompose an image into transparent layers
//! - **controlnet**: ControlNet-guided generation
//!
//! # Usage
//!
//! ```bash
//! # Text-to-image
//! cargo run --release --example qwen_image -- generate \
//!     --prompt "A serene mountain landscape" \
//!     --output landscape.png
//!
//! # Image editing
//! cargo run --release --example qwen_image -- edit \
//!     --input-image photo.jpg \
//!     --prompt "Make the sky purple" \
//!     --output edited.png
//!
//! # Inpainting
//! cargo run --release --example qwen_image -- inpaint \
//!     --input-image photo.jpg \
//!     --mask mask.png \
//!     --prompt "A beautiful sunset" \
//!     --output inpainted.png
//!
//! # Layer decomposition
//! cargo run --release --example qwen_image -- layered \
//!     --input-image composite.png \
//!     --layers 4 \
//!     --output-dir ./layers/
//!
//! # ControlNet
//! cargo run --release --example qwen_image -- controlnet \
//!     --prompt "A beautiful landscape" \
//!     --control-image edges.png \
//!     --controlnet-path /path/to/controlnet \
//!     --output controlled.png
//! ```

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

mod common;
mod controlnet;
mod edit;
mod generate;
mod inpaint;
mod layered;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "qwen-image",
    about = "Qwen-Image: Text-to-image and image editing with diffusion models",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Run on CPU instead of GPU.
    #[arg(long, global = true)]
    cpu: bool,

    /// Use F32 dtype instead of BF16.
    #[arg(long, global = true)]
    use_f32: bool,

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
        #[arg(long, default_value_t = 28)]
        num_inference_steps: usize,

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
    },

    /// Image editing with text instructions.
    Edit {
        /// Input image to edit.
        #[arg(long)]
        input_image: String,

        /// Editing instruction prompt.
        #[arg(long, default_value = "Make the colors more vibrant")]
        prompt: String,

        /// Negative prompt.
        #[arg(long, default_value = "")]
        negative_prompt: String,

        /// Number of denoising steps.
        #[arg(long, default_value_t = 50)]
        num_inference_steps: usize,

        /// True CFG guidance scale.
        #[arg(long, default_value_t = 4.0)]
        true_cfg_scale: f64,

        /// HuggingFace model ID for the transformer.
        #[arg(long, default_value = "Qwen/Qwen-Image-Edit-2511")]
        model_id: String,

        /// HuggingFace model ID for the VAE.
        #[arg(long, default_value = "Qwen/Qwen-Image")]
        vae_model_id: String,

        /// Maximum output resolution (longest side). Use 0 to preserve input size.
        #[arg(long, default_value_t = 1024)]
        max_resolution: usize,

        /// Use tiled VAE decoding for memory efficiency.
        #[arg(long)]
        tiled_decode: Option<bool>,

        /// Tile size for tiled decoding.
        #[arg(long, default_value_t = 256)]
        tile_size: usize,

        /// Local path to vision encoder weights.
        #[arg(long)]
        vision_encoder_path: Option<String>,

        /// Output filename.
        #[arg(long, default_value = "edited_output.png")]
        output: String,
    },

    /// Inpainting: fill in masked regions.
    Inpaint {
        /// Input image to inpaint.
        #[arg(long)]
        input_image: String,

        /// Mask image (white = inpaint, black = keep).
        #[arg(long)]
        mask: String,

        /// Prompt describing what to generate in the masked region.
        #[arg(long, default_value = "A beautiful landscape")]
        prompt: String,

        /// Negative prompt.
        #[arg(long, default_value = "")]
        negative_prompt: String,

        /// Number of denoising steps.
        #[arg(long, default_value_t = 50)]
        num_inference_steps: usize,

        /// True CFG guidance scale.
        #[arg(long, default_value_t = 4.0)]
        true_cfg_scale: f64,

        /// Output filename.
        #[arg(long, default_value = "inpainted_output.png")]
        output: String,
    },

    /// Layer decomposition: split image into transparent layers.
    Layered {
        /// Input image to decompose.
        #[arg(long)]
        input_image: String,

        /// Description of the image content.
        #[arg(long, default_value = "A scene with multiple objects")]
        prompt: String,

        /// Negative prompt.
        #[arg(long, default_value = "")]
        negative_prompt: String,

        /// Number of output layers.
        #[arg(long, default_value_t = 4)]
        layers: usize,

        /// Resolution bucket (640 or 1024).
        #[arg(long, default_value_t = 640)]
        resolution: usize,

        /// Number of denoising steps.
        #[arg(long, default_value_t = 50)]
        num_inference_steps: usize,

        /// True CFG guidance scale.
        #[arg(long, default_value_t = 4.0)]
        true_cfg_scale: f64,

        /// Output directory for layer images.
        #[arg(long, default_value = "./layers")]
        output_dir: String,
    },

    /// ControlNet-guided generation.
    Controlnet {
        /// The prompt to generate an image from.
        #[arg(long, default_value = "A beautiful landscape")]
        prompt: String,

        /// Negative prompt.
        #[arg(long, default_value = "")]
        negative_prompt: String,

        /// Control image (edge map, depth map, etc.).
        #[arg(long)]
        control_image: String,

        /// Height of the generated image (must be divisible by 16).
        #[arg(long, default_value_t = 1024)]
        height: usize,

        /// Width of the generated image (must be divisible by 16).
        #[arg(long, default_value_t = 1024)]
        width: usize,

        /// Number of denoising steps.
        #[arg(long, default_value_t = 50)]
        num_inference_steps: usize,

        /// True CFG guidance scale.
        #[arg(long, default_value_t = 4.0)]
        true_cfg_scale: f64,

        /// ControlNet conditioning scale (0.0 = no control, 1.0 = full control).
        #[arg(long, default_value_t = 0.8)]
        controlnet_scale: f64,

        /// Local path to ControlNet weights.
        #[arg(long)]
        controlnet_path: Option<String>,

        /// Output filename.
        #[arg(long, default_value = "controlnet_output.png")]
        output: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Setup tracing
    let _guard = common::setup_tracing(cli.tracing);

    // Setup device and dtype
    let (device, dtype) = common::setup_device_and_dtype(cli.cpu, cli.use_f32, cli.seed)?;

    // Dispatch to the appropriate pipeline
    match cli.command {
        Command::Generate {
            prompt,
            negative_prompt,
            height,
            width,
            num_inference_steps,
            true_cfg_scale,
            init_image,
            strength,
            output,
        } => {
            let args = generate::GenerateArgs {
                prompt,
                negative_prompt,
                height,
                width,
                num_inference_steps,
                true_cfg_scale,
                init_image,
                strength,
                output,
            };
            let paths = generate::ModelPaths {
                transformer_path: cli.transformer_path,
                vae_path: cli.vae_path,
                text_encoder_path: cli.text_encoder_path,
                tokenizer_path: cli.tokenizer_path,
            };
            generate::run(args, paths, &device, dtype)
        }

        Command::Edit {
            input_image,
            prompt,
            negative_prompt,
            num_inference_steps,
            true_cfg_scale,
            model_id,
            vae_model_id,
            max_resolution,
            tiled_decode,
            tile_size,
            vision_encoder_path,
            output,
        } => {
            let args = edit::EditArgs {
                input_image,
                prompt,
                negative_prompt,
                num_inference_steps,
                true_cfg_scale,
                model_id,
                vae_model_id,
                max_resolution,
                tiled_decode,
                tile_size,
                output,
            };
            let paths = edit::EditModelPaths {
                transformer_path: cli.transformer_path,
                vae_path: cli.vae_path,
                text_encoder_path: cli.text_encoder_path,
                vision_encoder_path,
                tokenizer_path: cli.tokenizer_path,
            };
            edit::run(args, paths, &device, dtype)
        }

        Command::Inpaint {
            input_image,
            mask,
            prompt,
            negative_prompt,
            num_inference_steps,
            true_cfg_scale,
            output,
        } => {
            let args = inpaint::InpaintArgs {
                input_image,
                mask,
                prompt,
                negative_prompt,
                num_inference_steps,
                true_cfg_scale,
                output,
            };
            let paths = inpaint::InpaintModelPaths {
                transformer_path: cli.transformer_path,
                vae_path: cli.vae_path,
                text_encoder_path: cli.text_encoder_path,
                tokenizer_path: cli.tokenizer_path,
            };
            inpaint::run(args, paths, &device, dtype)
        }

        Command::Layered {
            input_image,
            prompt,
            negative_prompt,
            layers,
            resolution,
            num_inference_steps,
            true_cfg_scale,
            output_dir,
        } => {
            let args = layered::LayeredArgs {
                input_image,
                prompt,
                negative_prompt,
                layers,
                resolution,
                num_inference_steps,
                true_cfg_scale,
                output_dir,
            };
            let paths = layered::LayeredModelPaths {
                transformer_path: cli.transformer_path,
                vae_path: cli.vae_path,
                text_encoder_path: cli.text_encoder_path,
                tokenizer_path: cli.tokenizer_path,
            };
            layered::run(args, paths, &device, dtype)
        }

        Command::Controlnet {
            prompt,
            negative_prompt,
            control_image,
            height,
            width,
            num_inference_steps,
            true_cfg_scale,
            controlnet_scale,
            controlnet_path,
            output,
        } => {
            let args = controlnet::ControlnetArgs {
                prompt,
                negative_prompt,
                control_image,
                height,
                width,
                num_inference_steps,
                true_cfg_scale,
                controlnet_scale,
                output,
            };
            let paths = controlnet::ControlnetModelPaths {
                transformer_path: cli.transformer_path,
                controlnet_path,
                vae_path: cli.vae_path,
                text_encoder_path: cli.text_encoder_path,
                tokenizer_path: cli.tokenizer_path,
            };
            controlnet::run(args, paths, &device, dtype)
        }
    }
}
