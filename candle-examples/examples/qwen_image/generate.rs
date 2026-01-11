//! Text-to-image generation pipeline.
//!
//! Generates images from text prompts using the Qwen-Image diffusion model.
//! Optionally supports img2img mode with an initial image.

use anyhow::Result;
use candle::{DType, Device, Tensor};
use candle_transformers::models::qwen_image::{
    apply_true_cfg, pack_latents, unpack_latents, Config, PromptMode,
};

use crate::common;
use crate::mt_box_muller_rng::MtBoxMullerRng;

/// Arguments specific to the generate pipeline.
pub struct GenerateArgs {
    pub prompt: String,
    pub negative_prompt: String,
    pub height: usize,
    pub width: usize,
    pub num_inference_steps: usize,
    pub true_cfg_scale: f64,
    pub init_image: Option<String>,
    pub strength: f64,
    pub output: String,
    pub seed: Option<u64>,
}

/// Shared model path arguments.
pub struct ModelPaths {
    pub transformer_path: Option<String>,
    pub gguf_transformer_path: Option<String>,
    pub vae_path: Option<String>,
    pub text_encoder_path: Option<String>,
    pub gguf_text_encoder_path: Option<String>,
    pub tokenizer_path: Option<String>,
}

/// Run the generate pipeline.
pub fn run(
    args: GenerateArgs,
    paths: ModelPaths,
    device: &Device,
    dtype: DType,
) -> Result<()> {
    common::validate_dimensions(args.height, args.width)?;

    println!("Qwen-Image Text-to-Image");
    println!("Device: {:?}, DType: {:?}", device, dtype);
    println!("Generating {}x{} image", args.width, args.height);
    println!("Prompt: {}", args.prompt);
    if !args.negative_prompt.is_empty() {
        println!("Negative prompt: {}", args.negative_prompt);
    }

    let api = hf_hub::api::sync::Api::new()?;
    let dims = common::calculate_latent_dims(args.height, args.width);

    // =========================================================================
    // Stage 1: Load tokenizer and text encoder
    // =========================================================================
    println!("\n[1/5] Loading text encoder...");

    let tokenizer = common::load_tokenizer(paths.tokenizer_path.as_deref(), &api)?;
    let mut text_model = common::load_text_encoder_variant(
        paths.text_encoder_path.as_deref(),
        paths.gguf_text_encoder_path.as_deref(),
        &api,
        device,
    )?;
    let encoder_type = if text_model.is_quantized() { "quantized GGUF" } else { "FP16" };
    println!("  Text encoder loaded ({})", encoder_type);

    // =========================================================================
    // Stage 2: Encode prompts
    // =========================================================================
    println!("\n[2/5] Encoding prompts...");

    let (pos_embeds, pos_mask) = common::encode_text_prompt_variant(
        &tokenizer,
        &mut text_model,
        &args.prompt,
        PromptMode::TextOnly,
        device,
        dtype,
    )?;
    println!("  Positive embeddings: {:?}", pos_embeds.dims());

    // Only encode negative prompt if provided - True CFG requires an explicit negative prompt
    let do_true_cfg = args.true_cfg_scale > 1.0 && !args.negative_prompt.is_empty();
    let neg_data = if do_true_cfg {
        let (neg_embeds, neg_mask) = common::encode_text_prompt_variant(
            &tokenizer,
            &mut text_model,
            &args.negative_prompt,
            PromptMode::TextOnly,
            device,
            dtype,
        )?;
        println!("  Negative embeddings: {:?}", neg_embeds.dims());
        Some((neg_embeds, neg_mask))
    } else {
        if args.true_cfg_scale > 1.0 {
            println!("  Note: true_cfg_scale={} but no negative prompt provided, CFG disabled", args.true_cfg_scale);
        }
        None
    };

    drop(text_model);
    println!("  Text encoder freed");

    // =========================================================================
    // Stage 3: Setup scheduler and initial latents
    // =========================================================================
    println!("\n[3/5] Setting up scheduler and latents...");

    let vae = common::load_vae(paths.vae_path.as_deref(), &api, device, dtype)?;
    println!("  VAE loaded");

    let mut scheduler = common::create_scheduler(args.num_inference_steps, dims.image_seq_len);

    // Calculate start step for img2img
    let (num_actual_steps, start_step) = if args.init_image.is_some() {
        let t_start = ((1.0 - args.strength) * args.num_inference_steps as f64).round() as usize;
        let actual_steps = args.num_inference_steps - t_start;
        println!(
            "  img2img mode: strength={:.2}, skipping first {} steps",
            args.strength, t_start
        );
        scheduler.set_begin_index(t_start);
        (actual_steps, t_start)
    } else {
        (args.num_inference_steps, 0)
    };

    // Create initial latents
    let latents = if let Some(ref init_path) = args.init_image {
        create_img2img_latents(
            init_path,
            &vae,
            &scheduler,
            &dims,
            start_step,
            args.height,
            args.width,
            device,
            dtype,
        )?
    } else {
        // Always use PyTorch-compatible MT19937 + Box-Muller RNG for consistent noise distribution
        // Keep latents in F32 to avoid BF16 quantization error accumulating across steps
        // Use [B, T, C, H, W] format to match PyTorch diffusers
        let seed = args.seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
        });
        println!("  Using seed: {}", seed);
        let mut rng = MtBoxMullerRng::new(seed);
        rng.randn(
            &[1, 1, 16, dims.latent_height, dims.latent_width],
            device,
            DType::F32,
        )?
    };
    println!("  Initial latents shape: {:?}", latents.dims());

    // =========================================================================
    // Stage 4: Load transformer and run denoising loop
    // =========================================================================
    println!("\n[4/5] Loading transformer and denoising...");

    let config = Config::qwen_image();
    let transformer = common::load_transformer_variant(
        paths.transformer_path.as_deref(),
        paths.gguf_transformer_path.as_deref(),
        common::DEFAULT_TRANSFORMER_ID,
        &config,
        &api,
        device,
        dtype,
    )?;
    let model_type = if transformer.is_quantized() { "quantized GGUF" } else { "FP16" };
    println!("  Transformer loaded ({} layers, {})", config.num_layers, model_type);

    let img_shapes = vec![(1, dims.packed_height, dims.packed_width)];
    let txt_seq_lens = vec![pos_embeds.dim(1)?];
    let timesteps = scheduler.timesteps().to_vec();

    // Pack latents immediately - PyTorch keeps latents packed throughout the loop
    // latents is [B, T, C, H, W], pack to [B, seq, C*4]
    let mut latents = pack_latents(&latents, dims.latent_height, dims.latent_width)?;

    println!("  Running {} denoising steps...", num_actual_steps);
    for (step_idx, &timestep) in timesteps
        .iter()
        .skip(start_step)
        .take(num_actual_steps)
        .enumerate()
    {
        let step = step_idx + start_step;
        if step_idx % 10 == 0 || step_idx == num_actual_steps - 1 {
            println!(
                "    Step {}/{}, timestep: {:.2}",
                step + 1,
                args.num_inference_steps,
                timestep
            );
        }

        let t = Tensor::new(&[timestep as f32 / 1000.0], device)?.to_dtype(dtype)?;

        // Convert F32 latents to BF16 for transformer (weights are BF16)
        let latents_bf16 = latents.to_dtype(dtype)?;

        let noise_pred =
            transformer.forward(&latents_bf16, &pos_embeds, &pos_mask, &t, &img_shapes, &txt_seq_lens, dtype)?;

        // Apply True CFG only if negative prompt was provided
        let noise_pred = if let Some((ref neg_embeds, ref neg_mask)) = neg_data {
            let neg_pred =
                transformer.forward(&latents_bf16, neg_embeds, neg_mask, &t, &img_shapes, &txt_seq_lens, dtype)?;
            apply_true_cfg(&noise_pred, &neg_pred, args.true_cfg_scale)?
        } else {
            noise_pred
        };

        // Convert noise_pred to F32 for scheduler arithmetic (matching PyTorch behavior)
        let noise_pred = noise_pred.to_dtype(DType::F32)?;

        // Scheduler operates on PACKED latents (matching PyTorch diffusers)
        latents = scheduler.step(&noise_pred, &latents)?;
    }

    drop(transformer);
    println!("  Transformer freed");

    // =========================================================================
    // Stage 5: VAE decode and save
    // =========================================================================
    println!("\n[5/5] Decoding latents...");

    // Unpack latents for VAE: [B, seq, C*4] -> [B, C, T, H, W]
    let latents = unpack_latents(&latents, dims.latent_height, dims.latent_width, 16)?;
    let denormalized = vae.denormalize_latents(&latents)?;

    // Use decode_image for single-frame output (matches PyTorch: vae.decode(z)[:, :, 0])
    // Note: Both latents and VAE are F32 for numerical precision
    let image = vae.decode_image(&denormalized)?;

    common::postprocess_and_save_4d(&image, &args.output)?;
    println!("\nImage saved to: {}", args.output);

    Ok(())
}

/// Create initial latents for img2img by encoding the init image and adding noise.
#[allow(clippy::too_many_arguments)]
fn create_img2img_latents(
    init_path: &str,
    vae: &candle_transformers::models::qwen_image::AutoencoderKLQwenImage,
    scheduler: &candle_transformers::models::qwen_image::FlowMatchEulerDiscreteScheduler,
    dims: &common::LatentDims,
    start_step: usize,
    height: usize,
    width: usize,
    device: &Device,
    _dtype: DType,
) -> Result<Tensor> {
    println!("  Loading init image: {}", init_path);

    let init_image = common::load_image_for_vae(init_path, height, width, device)?;

    let dist = vae.encode(&init_image)?;
    // VAE outputs [B, C, T, H, W] format
    let encoded_latents = vae.normalize_latents(&dist.mode().clone())?;
    // Transpose to [B, T, C, H, W] format for packing
    let encoded_latents = encoded_latents.permute([0, 2, 1, 3, 4])?;

    // Keep noise in F32 to avoid BF16 quantization error
    // Use [B, T, C, H, W] format to match transposed VAE output
    // Use PyTorch-compatible RNG for consistent noise distribution
    use std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
    let mut rng = MtBoxMullerRng::new(seed);
    let noise = rng.randn(
        &[1, 1, 16, dims.latent_height, dims.latent_width],
        device,
        DType::F32,
    )?;

    let sigmas = scheduler.sigmas();
    let start_sigma = sigmas[start_step];
    println!("  Starting sigma: {:.4}", start_sigma);

    Ok(scheduler.scale_noise(&encoded_latents, &noise, start_sigma)?)
}
