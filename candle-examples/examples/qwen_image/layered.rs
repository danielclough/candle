//! Layered image decomposition pipeline.
//!
//! Decomposes an input image into multiple transparent PNG layers.
//! The first frame is the combined/reference image, subsequent frames
//! are individual decomposed layers.

use anyhow::{anyhow, Result};
use candle::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::qwen_image::{
    apply_true_cfg, calculate_dimensions_with_resolution, pack_layered_latents,
    unpack_layered_latents, Config, PromptMode,
};

use crate::common;

/// Arguments specific to the layered pipeline.
pub struct LayeredArgs {
    pub input_image: String,
    pub prompt: String,
    pub negative_prompt: String,
    pub layers: usize,
    pub resolution: usize,
    pub num_inference_steps: usize,
    pub true_cfg_scale: f64,
    pub output_dir: String,
}

/// Model paths for the layered pipeline.
pub struct LayeredModelPaths {
    pub transformer_path: Option<String>,
    pub vae_path: Option<String>,
    pub text_encoder_path: Option<String>,
    pub tokenizer_path: Option<String>,
}

pub fn run(
    args: LayeredArgs,
    paths: LayeredModelPaths,
    device: &Device,
    dtype: DType,
) -> Result<()> {
    // Validate resolution
    if args.resolution != 640 && args.resolution != 1024 {
        return Err(anyhow!(
            "Resolution must be 640 or 1024, got {}",
            args.resolution
        ));
    }

    println!("Qwen-Image Layered");
    println!("Device: {:?}, DType: {:?}", device, dtype);
    println!("Input image: {}", args.input_image);
    println!("Layers: {}", args.layers);
    println!("Resolution bucket: {}", args.resolution);

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;

    let api = hf_hub::api::sync::Api::new()?;

    // =========================================================================
    // Stage 1: Calculate output dimensions
    // =========================================================================
    println!("\n[1/5] Calculating dimensions...");

    let input_img = image::ImageReader::open(&args.input_image)?
        .decode()
        .map_err(|e| anyhow!("Failed to decode image: {}", e))?;
    let (orig_width, orig_height) = (input_img.width() as usize, input_img.height() as usize);
    let aspect_ratio = orig_width as f64 / orig_height as f64;
    println!("  Original size: {}x{}", orig_width, orig_height);

    let (target_width, target_height) =
        calculate_dimensions_with_resolution(args.resolution, aspect_ratio);
    let target_width = (target_width / 16) * 16;
    let target_height = (target_height / 16) * 16;
    println!("  Output size: {}x{}", target_width, target_height);

    let dims = common::calculate_latent_dims(target_height, target_width);

    // =========================================================================
    // Stage 2: Load VAE and encode input image
    // =========================================================================
    println!("\n[2/5] Loading VAE and encoding input image...");

    let vae = common::load_vae(paths.vae_path.as_deref(), &api, device, dtype)?;
    println!("  VAE loaded");

    let vae_input =
        common::load_image_for_vae(&args.input_image, target_height, target_width, device)?;
    let dist = vae.encode(&vae_input)?;
    let image_latents = vae.normalize_latents(&dist.mode().clone())?;
    println!("  Image latents shape: {:?}", image_latents.dims());

    // =========================================================================
    // Stage 3: Load text encoder and encode prompts
    // =========================================================================
    println!("\n[3/5] Loading text encoder and encoding prompts...");

    let tokenizer = common::load_tokenizer(paths.tokenizer_path.as_deref(), &api)?;
    let mut text_model =
        common::load_text_encoder(paths.text_encoder_path.as_deref(), &api, device)?;
    println!("  Text encoder loaded");

    let (pos_embeds, pos_mask) = common::encode_text_prompt(
        &tokenizer,
        &mut text_model,
        &args.prompt,
        PromptMode::Layered,
        device,
        dtype,
    )?;
    println!("  Positive prompt embeddings: {:?}", pos_embeds.dims());

    let neg_prompt = if args.negative_prompt.is_empty() {
        ""
    } else {
        &args.negative_prompt
    };
    let (neg_embeds, neg_mask) = common::encode_text_prompt(
        &tokenizer,
        &mut text_model,
        neg_prompt,
        PromptMode::Layered,
        device,
        dtype,
    )?;

    drop(text_model);
    println!("  Text encoder freed");

    // =========================================================================
    // Stage 4: Load transformer and setup scheduler
    // =========================================================================
    println!("\n[4/5] Loading transformer and setting up denoising...");

    println!(
        "  Frames: {} (1 combined + {} layers)",
        args.layers + 1,
        args.layers
    );

    let mut scheduler = common::create_scheduler(args.num_inference_steps, dims.image_seq_len);

    // Create initial noise for all layers: [batch, layers+1, channels, height, width]
    // Keep in F32 to avoid BF16 quantization error accumulating across steps
    // Use PyTorch-compatible RNG for consistent noise distribution
    use std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
    let mut rng = crate::mt_box_muller_rng::MtBoxMullerRng::new(seed);
    let noise_latents = rng.randn(
        &[1, args.layers + 1, 16, dims.latent_height, dims.latent_width],
        device,
        DType::F32,
    )?;

    // Pack layered latents
    let packed_noise =
        pack_layered_latents(&noise_latents, dims.latent_height, dims.latent_width, args.layers)?;
    println!("  Packed noise shape: {:?}", packed_noise.dims());

    // Pack image latents (single frame) for conditioning, convert to BF16 for transformer
    let image_latents_frame = image_latents.squeeze(2)?; // [1, 16, h, w]
    let image_latents_frame = image_latents_frame.unsqueeze(1)?; // [1, 1, 16, h, w]
    let packed_image =
        pack_layered_latents(&image_latents_frame, dims.latent_height, dims.latent_width, 0)?
            .to_dtype(dtype)?;

    // Load transformer with layered config
    let config = Config::qwen_image_layered();
    let transformer = common::load_transformer(
        paths.transformer_path.as_deref(),
        common::DEFAULT_TRANSFORMER_ID,
        &config,
        &api,
        device,
        dtype,
    )?;
    println!("  Transformer loaded ({} layers)", config.num_layers);

    // =========================================================================
    // Stage 5: Denoising loop
    // =========================================================================
    println!("\n[5/5] Running denoising loop...");

    // Image shapes for RoPE: one entry per frame, plus one for the image condition
    let mut img_shapes = Vec::with_capacity(args.layers + 2);
    for _ in 0..=args.layers {
        img_shapes.push((1, dims.packed_height, dims.packed_width));
    }
    img_shapes.push((1, dims.packed_height, dims.packed_width)); // For image condition

    let txt_seq_lens = vec![pos_embeds.dim(1)?];
    let timesteps = scheduler.timesteps().to_vec();
    let mut latents = packed_noise;

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

        // Convert F32 latents to BF16 for transformer (weights are BF16)
        let latents_bf16 = latents.to_dtype(dtype)?;
        // Concatenate layered noise with image condition (both now BF16)
        let latent_model_input = Tensor::cat(&[&latents_bf16, &packed_image], 1)?;
        let t = Tensor::new(&[timestep as f32 / 1000.0], device)?.to_dtype(dtype)?;

        let pos_pred = transformer.forward(
            &latent_model_input,
            &pos_embeds,
            &pos_mask,
            &t,
            &img_shapes,
            &txt_seq_lens,
        )?;
        let neg_pred = transformer.forward(
            &latent_model_input,
            &neg_embeds,
            &neg_mask,
            &t,
            &img_shapes,
            &txt_seq_lens,
        )?;

        // Extract only the layered noise prediction (not the image condition part)
        let noise_seq_len = latents.dim(1)?;
        let pos_pred = pos_pred.narrow(1, 0, noise_seq_len)?;
        let neg_pred = neg_pred.narrow(1, 0, noise_seq_len)?;

        let guided_pred = apply_true_cfg(&pos_pred, &neg_pred, args.true_cfg_scale)?;

        // Unpack, step, and repack
        let unpacked = unpack_layered_latents(
            &guided_pred,
            dims.latent_height,
            dims.latent_width,
            args.layers,
            16,
        )?;
        // Convert back to F32 for scheduler arithmetic
        let unpacked = unpacked.to_dtype(DType::F32)?;
        let unpacked_latents = unpack_layered_latents(
            &latents,
            dims.latent_height,
            dims.latent_width,
            args.layers,
            16,
        )?;

        let stepped = scheduler.step(&unpacked, &unpacked_latents)?;

        // stepped is [batch, channels, layers+1, height, width], convert to [batch, layers+1, channels, h, w]
        let stepped = stepped.permute([0, 2, 1, 3, 4])?;
        latents =
            pack_layered_latents(&stepped, dims.latent_height, dims.latent_width, args.layers)?;
    }

    // Unpack final latents: [batch, channels, layers+1, height, width]
    let final_latents = unpack_layered_latents(
        &latents,
        dims.latent_height,
        dims.latent_width,
        args.layers,
        16,
    )?;

    drop(transformer);
    println!("  Transformer freed");

    // =========================================================================
    // Decode and save layers
    // =========================================================================
    println!("\nDecoding and saving layers...");

    for layer_idx in 0..=args.layers {
        // Extract single layer: [batch, channels, 1, height, width]
        let layer_latents = final_latents.narrow(2, layer_idx, 1)?;

        // Denormalize and decode (both latents and VAE are F32 for precision)
        let layer_latents = vae.denormalize_latents(&layer_latents)?;
        let decoded = vae.decode(&layer_latents)?;

        // Post-process: [1, 3, T, H, W] -> [3, H, W]
        let decoded = decoded.squeeze(0)?;
        let decoded = decoded.i((.., 0, .., ..))?;
        let decoded = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?;
        let decoded = decoded.to_dtype(DType::U8)?;

        let output_path = format!("{}/layer_{}.png", args.output_dir, layer_idx);
        candle_examples::save_image(&decoded, &output_path)?;
        println!("  Saved layer {} to: {}", layer_idx, output_path);
    }

    println!("\nLayer decomposition complete!");
    println!("Output directory: {}", args.output_dir);

    Ok(())
}
