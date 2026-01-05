//! Text-to-image generation pipeline.
//!
//! Generates images from text prompts using the Qwen-Image diffusion model.
//! Optionally supports img2img mode with an initial image.

use anyhow::Result;
use candle::{DType, Device, Tensor};
use candle_transformers::models::qwen_image::{
    apply_true_cfg, pack_latents, unpack_latents, Config, TEXT_ONLY_DROP_TOKENS,
    TEXT_ONLY_PROMPT_TEMPLATE,
};

use crate::common;

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
}

/// Shared model path arguments.
pub struct ModelPaths {
    pub transformer_path: Option<String>,
    pub vae_path: Option<String>,
    pub text_encoder_path: Option<String>,
    pub tokenizer_path: Option<String>,
}

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
    let mut text_model = common::load_text_encoder(
        paths.text_encoder_path.as_deref(),
        &api,
        device,
        dtype,
    )?;
    println!("  Text encoder loaded");

    // =========================================================================
    // Stage 2: Encode prompts
    // =========================================================================
    println!("\n[2/5] Encoding prompts...");

    let (pos_embeds, pos_mask) = common::encode_text_prompt(
        &tokenizer,
        &mut text_model,
        &args.prompt,
        TEXT_ONLY_PROMPT_TEMPLATE,
        TEXT_ONLY_DROP_TOKENS,
        device,
    )?;
    println!("  Positive embeddings: {:?}", pos_embeds.dims());

    // Only encode negative prompt if provided - True CFG requires an explicit negative prompt
    let do_true_cfg = args.true_cfg_scale > 1.0 && !args.negative_prompt.is_empty();
    let neg_data = if do_true_cfg {
        let (neg_embeds, neg_mask) = common::encode_text_prompt(
            &tokenizer,
            &mut text_model,
            &args.negative_prompt,
            TEXT_ONLY_PROMPT_TEMPLATE,
            TEXT_ONLY_DROP_TOKENS,
            device,
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
        Tensor::randn(
            0f32,
            1f32,
            (1, 16, 1, dims.latent_height, dims.latent_width),
            device,
        )?
        .to_dtype(dtype)?
    };
    println!("  Initial latents shape: {:?}", latents.dims());

    // =========================================================================
    // Stage 4: Load transformer and run denoising loop
    // =========================================================================
    println!("\n[4/5] Loading transformer and denoising...");

    let config = Config::qwen_image();
    let transformer = common::load_transformer(
        paths.transformer_path.as_deref(),
        common::DEFAULT_TRANSFORMER_ID,
        &config,
        &api,
        device,
        dtype,
    )?;
    println!("  Transformer loaded ({} layers)", config.num_layers);

    let img_shapes = vec![(1, dims.packed_height, dims.packed_width)];
    let txt_seq_lens = vec![pos_embeds.dim(1)?];
    let timesteps = scheduler.timesteps().to_vec();
    let mut latents = latents;

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

        let packed = pack_latents(&latents, dims.latent_height, dims.latent_width)?;
        let t = Tensor::new(&[timestep as f32 / 1000.0], device)?.to_dtype(dtype)?;

        let noise_pred =
            transformer.forward(&packed, &pos_embeds, &pos_mask, &t, &img_shapes, &txt_seq_lens)?;

        // Apply True CFG only if negative prompt was provided
        let noise_pred = if let Some((ref neg_embeds, ref neg_mask)) = neg_data {
            let neg_pred =
                transformer.forward(&packed, neg_embeds, neg_mask, &t, &img_shapes, &txt_seq_lens)?;
            apply_true_cfg(&noise_pred, &neg_pred, args.true_cfg_scale)?
        } else {
            noise_pred
        };

        let unpacked = unpack_latents(&noise_pred, dims.latent_height, dims.latent_width, 16)?;

        latents = scheduler.step(&unpacked, &latents)?;
    }

    drop(transformer);
    println!("  Transformer freed");

    // =========================================================================
    // Stage 5: VAE decode and save
    // =========================================================================
    println!("\n[5/5] Decoding latents...");

    let latents = vae.denormalize_latents(&latents)?;
    let image = vae.decode(&latents)?;

    common::postprocess_and_save(&image, &args.output)?;
    println!("\nImage saved to: {}", args.output);

    Ok(())
}

/// Create initial latents for img2img by encoding the init image and adding noise.
fn create_img2img_latents(
    init_path: &str,
    vae: &candle_transformers::models::qwen_image::AutoencoderKLQwenImage,
    scheduler: &candle_transformers::models::qwen_image::FlowMatchEulerDiscreteScheduler,
    dims: &common::LatentDims,
    start_step: usize,
    height: usize,
    width: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    println!("  Loading init image: {}", init_path);

    let init_image = common::load_image_for_vae(init_path, height, width, device, dtype)?;

    let dist = vae.encode(&init_image)?;
    let encoded_latents = vae.normalize_latents(&dist.mode().clone())?;

    let noise = Tensor::randn(
        0f32,
        1f32,
        (1, 16, 1, dims.latent_height, dims.latent_width),
        device,
    )?
    .to_dtype(dtype)?;

    let sigmas = scheduler.sigmas();
    let start_sigma = sigmas[start_step];
    println!("  Starting sigma: {:.4}", start_sigma);

    Ok(scheduler.scale_noise(&encoded_latents, &noise, start_sigma)?)
}
