//! Inpainting pipeline: fill in masked regions of an image.
//!
//! White regions in the mask are regenerated, black regions are preserved.
//! During denoising, the latents are blended with the original image based
//! on the mask and current noise level.

use anyhow::Result;
use candle::{DType, Device, Tensor};
use candle_transformers::models::qwen_image::{
    apply_true_cfg, pack_latents, unpack_latents, Config, PromptMode,
};

use crate::common;

/// Arguments specific to the inpaint pipeline.
pub struct InpaintArgs {
    pub input_image: String,
    pub mask: String,
    pub prompt: String,
    pub negative_prompt: String,
    pub num_inference_steps: usize,
    pub true_cfg_scale: f64,
    pub output: String,
}

/// Model paths for the inpaint pipeline.
pub struct InpaintModelPaths {
    pub transformer_path: Option<String>,
    pub vae_path: Option<String>,
    pub text_encoder_path: Option<String>,
    pub tokenizer_path: Option<String>,
}

pub fn run(
    args: InpaintArgs,
    paths: InpaintModelPaths,
    device: &Device,
    dtype: DType,
) -> Result<()> {
    println!("Qwen-Image Inpainting");
    println!("Device: {:?}, DType: {:?}", device, dtype);
    println!("Input image: {}", args.input_image);
    println!("Mask: {}", args.mask);
    println!("Prompt: {}", args.prompt);

    let api = hf_hub::api::sync::Api::new()?;

    // =========================================================================
    // Stage 1: Load VAE and encode input image
    // =========================================================================
    println!("\n[1/5] Loading VAE and encoding input image...");

    let vae = common::load_vae(paths.vae_path.as_deref(), &api, device, dtype)?;
    println!("  VAE loaded");

    // Load input image and get dimensions
    let input_img = image::ImageReader::open(&args.input_image)?
        .decode()
        .map_err(|e| anyhow::anyhow!("Failed to decode image: {}", e))?;
    let (orig_width, orig_height) = (input_img.width() as usize, input_img.height() as usize);

    // Round to nearest 16 for compatibility
    let (target_height, target_width) = common::round_to_16(orig_height, orig_width);
    println!(
        "  Original size: {}x{}, Target size: {}x{}",
        orig_width, orig_height, target_width, target_height
    );

    let dims = common::calculate_latent_dims(target_height, target_width);
    println!("  Latent size: {}x{}", dims.latent_height, dims.latent_width);

    // Encode input image
    let input_image =
        common::load_image_for_vae(&args.input_image, target_height, target_width, device, dtype)?;
    let dist = vae.encode(&input_image)?;
    let original_latents = vae.normalize_latents(&dist.mode().clone())?;
    println!("  Encoded latents shape: {:?}", original_latents.dims());

    // Load mask
    let mask = common::load_mask_for_latents(
        &args.mask,
        dims.latent_height,
        dims.latent_width,
        device,
        dtype,
    )?;
    let mask_coverage = mask.mean_all()?.to_scalar::<f32>()?;
    println!("  Mask coverage: {:.1}% to inpaint", mask_coverage * 100.0);

    // =========================================================================
    // Stage 2: Load tokenizer and text encoder
    // =========================================================================
    println!("\n[2/5] Loading text encoder...");

    let tokenizer = common::load_tokenizer(paths.tokenizer_path.as_deref(), &api)?;
    let mut text_model =
        common::load_text_encoder(paths.text_encoder_path.as_deref(), &api, device)?;
    println!("  Text encoder loaded");

    // Encode prompts
    let (pos_embeds, pos_mask) = common::encode_text_prompt(
        &tokenizer,
        &mut text_model,
        &args.prompt,
        PromptMode::TextOnly,
        device,
        dtype,
    )?;

    let neg_prompt = if args.negative_prompt.is_empty() {
        ""
    } else {
        &args.negative_prompt
    };
    let (neg_embeds, neg_mask) = common::encode_text_prompt(
        &tokenizer,
        &mut text_model,
        neg_prompt,
        PromptMode::TextOnly,
        device,
        dtype,
    )?;

    drop(text_model);
    println!("  Text encoder freed");

    // =========================================================================
    // Stage 3: Setup scheduler
    // =========================================================================
    println!("\n[3/5] Setting up scheduler...");

    let mut scheduler = common::create_scheduler(args.num_inference_steps, dims.image_seq_len);

    // Create initial noise (F32 to avoid BF16 quantization error)
    let noise = Tensor::randn(
        0f32,
        1f32,
        (1, 16, 1, dims.latent_height, dims.latent_width),
        device,
    )?;

    // Start with pure noise (we'll blend with original during denoising)
    let mut latents = noise.clone();

    // =========================================================================
    // Stage 4: Load transformer and run inpainting loop
    // =========================================================================
    println!("\n[4/5] Loading transformer and inpainting...");

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
    let sigmas = scheduler.sigmas().to_vec();

    println!(
        "  Running {} denoising steps...",
        args.num_inference_steps
    );
    for (step, &timestep) in timesteps
        .iter()
        .take(args.num_inference_steps)
        .enumerate()
    {
        if step % 10 == 0 || step == args.num_inference_steps - 1 {
            println!(
                "    Step {}/{}, timestep: {:.2}",
                step + 1,
                args.num_inference_steps,
                timestep
            );
        }

        // Get current sigma
        let sigma = sigmas[step];

        // Blend: in masked areas use denoising latents, in unmasked use noised original
        latents = blend_latents(&latents, &original_latents, &noise, &mask, sigma)?;

        // Pack latents for transformer
        let packed = pack_latents(&latents, dims.latent_height, dims.latent_width)?;
        // Convert F32 latents to BF16 for transformer (weights are BF16)
        let packed = packed.to_dtype(dtype)?;
        let t = Tensor::new(&[timestep as f32 / 1000.0], device)?.to_dtype(dtype)?;

        let pos_pred =
            transformer.forward(&packed, &pos_embeds, &pos_mask, &t, &img_shapes, &txt_seq_lens)?;
        let neg_pred =
            transformer.forward(&packed, &neg_embeds, &neg_mask, &t, &img_shapes, &txt_seq_lens)?;

        let guided_pred = apply_true_cfg(&pos_pred, &neg_pred, args.true_cfg_scale)?;
        let unpacked = unpack_latents(&guided_pred, dims.latent_height, dims.latent_width, 16)?;
        // Convert back to F32 for scheduler arithmetic
        let unpacked = unpacked.to_dtype(DType::F32)?;

        latents = scheduler.step(&unpacked, &latents)?;
    }

    // Final blend: ensure unmasked regions are exactly the original
    let inv_mask = (Tensor::ones_like(&mask)? - &mask)?;
    latents = (mask.broadcast_mul(&latents)? + inv_mask.broadcast_mul(&original_latents)?)?;

    drop(transformer);
    println!("  Transformer freed");

    // =========================================================================
    // Stage 5: VAE decode and save
    // =========================================================================
    println!("\n[5/5] Decoding latents...");

    // Note: Both latents and VAE are F32 for numerical precision
    let latents = vae.denormalize_latents(&latents)?;
    let image = vae.decode(&latents)?;

    common::postprocess_and_save(&image, &args.output)?;
    println!("\nInpainted image saved to: {}", args.output);

    Ok(())
}

/// Blend noisy latents with the original latents based on the mask.
///
/// In masked regions (mask=1), use the noisy latents (being denoised).
/// In unmasked regions (mask=0), use the original latents with added noise at current sigma.
fn blend_latents(
    noisy: &Tensor,
    original: &Tensor,
    noise: &Tensor,
    mask: &Tensor,
    sigma: f64,
) -> Result<Tensor> {
    // Add noise to original at current sigma level
    let noised_original = ((original * (1.0 - sigma))? + (noise * sigma)?)?;

    // Blend based on mask: mask * noisy + (1 - mask) * noised_original
    let inv_mask = (Tensor::ones_like(mask)? - mask)?;
    let result = (mask.broadcast_mul(noisy)? + inv_mask.broadcast_mul(&noised_original)?)?;

    Ok(result)
}
