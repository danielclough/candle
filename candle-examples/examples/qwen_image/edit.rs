//! Image editing pipeline with text instructions.
//!
//! Performs image editing by conditioning on both the input image latents
//! and an editing instruction. The noise latents are concatenated with
//! the encoded input image latents during denoising.

use anyhow::{anyhow, Result};
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;

use crate::mt_box_muller_rng::MtBoxMullerRng;
use candle_transformers::models::{
    qwen2_5_vl::{
        get_image_grid_thw, normalize_image, patchify_image, smart_resize, Qwen25VLTextModel,
        Qwen25VLVisionModel, VisionConfig, DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS,
    },
    qwen_image::{
        apply_true_cfg, pack_latents, set_cfg_pass, unpack_latents, Config, TiledDecodeConfig,
        EDIT_DROP_TOKENS, EDIT_PROMPT_TEMPLATE,
    },
};
use tokenizers::Tokenizer;

use crate::common;
use crate::debug::{self, DebugContext};

/// Vision encoder configuration constants.
const VISION_PATCH_SIZE: usize = 14;
const VISION_TEMPORAL_PATCH_SIZE: usize = 2;
const VISION_MERGE_SIZE: usize = 2;
const IMAGE_TOKEN_ID: u32 = 151655;

/// Arguments specific to the edit pipeline.
pub struct EditArgs {
    pub input_image: String,
    pub prompt: String,
    pub negative_prompt: String,
    pub num_inference_steps: usize,
    pub true_cfg_scale: f64,
    pub model_id: String,
    pub vae_model_id: String,
    pub height: Option<usize>,
    pub width: Option<usize>,
    pub max_resolution: usize,
    pub tiled_decode: Option<bool>,
    pub tile_size: usize,
    pub output: String,
    pub seed: Option<u64>,
}

/// Model paths for the edit pipeline.
pub struct EditModelPaths {
    pub transformer_path: Option<String>,
    pub vae_path: Option<String>,
    pub text_encoder_path: Option<String>,
    pub vision_encoder_path: Option<String>,
    pub tokenizer_path: Option<String>,
}

pub fn run(
    args: EditArgs,
    paths: EditModelPaths,
    device: &Device,
    dtype: DType,
    mut debug_ctx: Option<&mut DebugContext>,
) -> Result<()> {
    println!("Qwen-Image Edit");
    println!("Device: {:?}, DType: {:?}", device, dtype);
    println!("Model: {}", args.model_id);
    println!("Input image: {}", args.input_image);
    println!("Prompt: {}", args.prompt);
    println!("Negative prompt: {}", if args.negative_prompt.is_empty() { "(none)" } else { &args.negative_prompt });
    println!("CFG scale: {}", args.true_cfg_scale);
    println!("Steps: {}", args.num_inference_steps);

    // Warn if CFG is specified but no negative prompt (matching Python behavior)
    if args.negative_prompt.is_empty() && args.true_cfg_scale != 1.0 {
        println!(
            "Warning: true_cfg_scale is passed as {}, but classifier-free guidance is not enabled since no negative_prompt is provided.",
            args.true_cfg_scale
        );
    }

    let api = hf_hub::api::sync::Api::new()?;

    // =========================================================================
    // Stage 1: Load input image and determine output dimensions
    // =========================================================================
    println!("\n[1/5] Loading input image...");

    let input_img = image::ImageReader::open(&args.input_image)?
        .decode()
        .map_err(|e| anyhow!("Failed to decode image: {}", e))?;
    let (orig_width, orig_height) = (input_img.width() as usize, input_img.height() as usize);
    println!("  Original size: {}x{}", orig_width, orig_height);

    let (target_width, target_height) = if let (Some(h), Some(w)) = (args.height, args.width) {
        // Use explicit dimensions if both specified
        println!("  Using explicit dimensions: {}x{}", w, h);
        (w, h)
    } else {
        common::calculate_output_dims(orig_width, orig_height, args.max_resolution)
    };
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
    let vae_input = debug::checkpoint(&mut debug_ctx, "vae_input", vae_input)?;

    let dist = vae.encode(&vae_input)?;
    let image_latents_raw = dist.mode().clone();
    let image_latents_raw = debug::checkpoint(&mut debug_ctx, "image_latents_raw", image_latents_raw)?;

    let image_latents = vae.normalize_latents(&image_latents_raw)?;
    let image_latents = debug::checkpoint(&mut debug_ctx, "image_latents_normalized", image_latents)?;
    // Permute from VAE format [B, C, T, H, W] to pack format [B, T, C, H, W]
    let image_latents = image_latents.permute([0, 2, 1, 3, 4])?;
    println!("  Image latents shape: {:?}", image_latents.dims());

    // =========================================================================
    // Stage 3: Load vision encoder, text encoder, and encode prompts
    // =========================================================================
    println!("\n[3/5] Loading encoders and encoding prompts...");

    let tokenizer = common::load_tokenizer(paths.tokenizer_path.as_deref(), &api)?;

    // Load vision encoder
    let vision_config = VisionConfig::default();
    let vision_model_files = match &paths.vision_encoder_path {
        Some(path) => {
            candle_examples::hub_load_local_safetensors(path, "model.safetensors.index.json")?
        }
        None => {
            let repo = api.repo(hf_hub::Repo::model(common::DEFAULT_TEXT_ENCODER_ID.to_string()));
            candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?
        }
    };

    // Load vision encoder in F32 for numerical precision
    let vb_vision =
        unsafe { VarBuilder::from_mmaped_safetensors(&vision_model_files, DType::F32, device)? };
    let vision_model = Qwen25VLVisionModel::new(&vision_config, vb_vision.pp("visual"))?;
    println!("  Vision encoder loaded (F32)");

    // Preprocess input image for vision encoder (F32)
    // Constrain to target dimensions to match Python behavior
    let vision_max_pixels = target_width * target_height;
    let (pixel_values, image_grid_thw, _) =
        preprocess_image_for_vision(&input_img, device, DType::F32, Some(vision_max_pixels))?;
    let pixel_values = debug::checkpoint(&mut debug_ctx, "vision_pixel_values", pixel_values)?;
    println!(
        "  Image preprocessed for vision: {} patches",
        pixel_values.dim(0)?
    );

    // Compute vision embeddings (convert to F32 for text encoder which uses F32)
    let vision_embeds = vision_model.forward(&pixel_values, &image_grid_thw)?
        .to_dtype(DType::F32)?;
    let vision_embeds = debug::checkpoint(&mut debug_ctx, "vision_embeds", vision_embeds)?;
    let num_image_tokens = vision_embeds.dim(0)?;
    println!("  Vision embeddings: {} tokens", num_image_tokens);

    // Free vision encoder from memory before loading text encoder
    drop(vision_model);

    // Load text encoder (always F32 for numerical precision)
    let mut text_model =
        common::load_text_encoder(paths.text_encoder_path.as_deref(), &api, device)?;
    println!("  Text encoder loaded");

    // Encode prompts with vision (text encoder outputs F32, convert to transformer dtype)
    let (pos_embeds, pos_mask) = encode_prompt_with_vision(
        &tokenizer,
        &mut text_model,
        &args.prompt,
        &vision_embeds,
        &image_grid_thw,
        num_image_tokens,
        device,
    )?;
    let pos_embeds = debug::checkpoint(&mut debug_ctx, "prompt_embeds", pos_embeds)?;
    let pos_mask = debug::checkpoint(&mut debug_ctx, "prompt_mask", pos_mask)?;
    let pos_embeds = pos_embeds.to_dtype(dtype)?;
    println!("  Positive prompt embeddings: {:?}", pos_embeds.dims());

    // Only encode negative prompt if one is provided (matching Python behavior)
    let (neg_embeds, neg_mask) = if !args.negative_prompt.is_empty() {
        let (neg_embeds, neg_mask) = encode_prompt_with_vision(
            &tokenizer,
            &mut text_model,
            &args.negative_prompt,
            &vision_embeds,
            &image_grid_thw,
            num_image_tokens,
            device,
        )?;
        let neg_embeds = debug::checkpoint(&mut debug_ctx, "negative_prompt_embeds", neg_embeds)?;
        let neg_mask = debug::checkpoint(&mut debug_ctx, "negative_prompt_mask", neg_mask)?;
        let neg_embeds = neg_embeds.to_dtype(dtype)?;
        println!("  Negative prompt embeddings: {:?}", neg_embeds.dims());
        (Some(neg_embeds), Some(neg_mask))
    } else {
        // No negative prompt → skip encoding entirely (Python skips CFG in this case)
        (None, None)
    };

    // Free text encoder from memory before loading transformer
    drop(text_model);
    println!("  Text encoder freed");

    // =========================================================================
    // Stage 4: Load transformer and setup scheduler
    // =========================================================================
    println!("\n[4/5] Loading transformer and setting up denoising...");

    let mut scheduler = common::create_scheduler(args.num_inference_steps, dims.image_seq_len);

    // Create initial noise (F32 to avoid BF16 quantization error)
    // Use [B, T, C, H, W] format to match PyTorch diffusers
    // Always use PyTorch-compatible MT19937 + Box-Muller RNG for consistent noise distribution
    let seed = args.seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
    });
    println!("  Using seed: {}", seed);
    let mut rng = MtBoxMullerRng::new(seed);
    let noise_latents = rng.randn(
        &[1, 1, 16, dims.latent_height, dims.latent_width],
        &Device::Cpu,
        DType::F32,
    )?.to_device(device)?;
    let noise_latents = debug::checkpoint(&mut debug_ctx, "noise_latents_unpacked", noise_latents)?;

    let packed_noise = pack_latents(&noise_latents, dims.latent_height, dims.latent_width)?;
    let packed_noise = debug::checkpoint(&mut debug_ctx, "packed_noise_latents", packed_noise)?;

    // Pack image latents and convert to BF16 for transformer input
    let packed_image = pack_latents(&image_latents, dims.latent_height, dims.latent_width)?;
    let packed_image = debug::checkpoint(&mut debug_ctx, "packed_image_latents", packed_image)?;
    let packed_image = packed_image.to_dtype(dtype)?;

    // Load transformer
    let config = Config::qwen_image_edit();
    let transformer = common::load_transformer(
        paths.transformer_path.as_deref(),
        &args.model_id,
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

    // Image shapes for RoPE: noise + image regions (both same size)
    let img_shapes = vec![
        (1, dims.packed_height, dims.packed_width),  // Noise latents
        (1, dims.packed_height, dims.packed_width),  // Image latents
    ];
    // Text sequence lengths for RoPE - must match actual embedding lengths!
    // Positive and negative prompts may have different token counts.
    let pos_txt_seq_lens = vec![pos_embeds.dim(1)?];
    let neg_txt_seq_lens = neg_embeds.as_ref().map(|e| vec![e.dim(1).unwrap()]);

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
        // Concatenate noise and image latents (both BF16)
        let latent_model_input = Tensor::cat(&[&latents_bf16, &packed_image], 1)?;
        let t = Tensor::new(&[timestep as f32 / 1000.0], device)?.to_dtype(dtype)?;

        // Debug: capture transformer inputs at step 0
        if step == 0 {
            debug::checkpoint(&mut debug_ctx, "transformer_input_hidden_states", latent_model_input.clone())?;
            debug::checkpoint(&mut debug_ctx, "transformer_input_timestep", t.clone())?;
        }

        // Set CFG pass for block override loading (positive = 0)
        set_cfg_pass(0);
        // Note: Timestep doubling for zero_cond_t is handled internally by the transformer
        let pos_pred = transformer.forward(
            &latent_model_input,
            &pos_embeds,
            &pos_mask,
            &t,
            &img_shapes,
            &pos_txt_seq_lens,
        )?;

        // Debug: capture full transformer output at step 0 (before extraction)
        if step == 0 {
            debug::checkpoint(&mut debug_ctx, "transformer_noise_pred_full", pos_pred.clone())?;
        }

        // Extract only the noise prediction part
        let noise_seq_len = latents.dim(1)?;
        let pos_pred = pos_pred.narrow(1, 0, noise_seq_len)?;

        // Debug: capture positive noise predictions at each step
        let pos_name = format!("transformer_noise_pred_pos_step{}", step);
        let pos_pred = debug::checkpoint(&mut debug_ctx, &pos_name, pos_pred)?;

        // Only apply CFG if we have a negative prompt (matching Python behavior)
        let model_pred = if let (Some(neg_embeds), Some(neg_mask), Some(neg_seq_lens)) = (&neg_embeds, &neg_mask, &neg_txt_seq_lens) {
            // Set CFG pass for block override loading (negative = 1)
            set_cfg_pass(1);
            let neg_pred = transformer.forward(
                &latent_model_input,
                neg_embeds,
                neg_mask,
                &t,
                &img_shapes,
                neg_seq_lens,
            )?;

            let neg_pred = neg_pred.narrow(1, 0, noise_seq_len)?;

            // Debug: capture negative noise predictions
            let neg_name = format!("transformer_noise_pred_neg_step{}", step);
            let neg_pred = debug::checkpoint(&mut debug_ctx, &neg_name, neg_pred)?;

            let guided_pred = apply_true_cfg(&pos_pred, &neg_pred, args.true_cfg_scale)?;

            // Debug: capture guided prediction at each step
            let guided_name = format!("guided_pred_step{}", step);
            debug::checkpoint(&mut debug_ctx, &guided_name, guided_pred.clone())?;

            if step == 0 {
                debug::checkpoint(&mut debug_ctx, "transformer_guided_pred_step0", guided_pred.clone())?;
            }
            guided_pred
        } else {
            // No negative prompt → skip CFG, use positive prediction directly
            // (CFG requires both pos and neg predictions to compute the guidance direction)
            pos_pred
        };

        // Convert model_pred to F32 for scheduler arithmetic (matching PyTorch behavior)
        let model_pred = model_pred.to_dtype(DType::F32)?;

        // Scheduler operates on PACKED latents (matching PyTorch diffusers)
        latents = scheduler.step(&model_pred, &latents)?;

        // Debug: capture latents after each step
        let step_name = format!("latents_after_step{}", step);
        debug::checkpoint(&mut debug_ctx, &step_name, latents.clone())?;

        // Keep backwards-compatible names for step 0
        if step == 0 {
            debug::checkpoint(&mut debug_ctx, "packed_latents_after_step0", latents.clone())?;
            let latents_unpacked = unpack_latents(&latents, dims.latent_height, dims.latent_width, 16)?;
            debug::checkpoint(&mut debug_ctx, "latents_after_step0", latents_unpacked)?;
        }
    }

    let final_latents = unpack_latents(&latents, dims.latent_height, dims.latent_width, 16)?;
    let final_latents = debug::checkpoint(&mut debug_ctx, "final_latents", final_latents)?;
    debug::checkpoint(&mut debug_ctx, "final_latents_packed", latents)?;

    drop(transformer);
    println!("  Transformer freed");

    // =========================================================================
    // Decode and save
    // =========================================================================
    let use_tiled = args
        .tiled_decode
        .unwrap_or_else(|| target_height * target_width > 512 * 512);

    // Note: Both latents and VAE are F32 for numerical precision
    let latents = vae.denormalize_latents(&final_latents)?;
    let latents = debug::checkpoint(&mut debug_ctx, "denormalized_latents", latents)?;

    // Decode latents to image
    // PyTorch: image = vae.decode(latents)[0][:, :, 0]
    // The [:, :, 0] extracts first temporal frame: [B, C, T, H, W] -> [B, C, H, W]
    let image = if use_tiled {
        let tile_stride = (args.tile_size * 3) / 4;
        let config = TiledDecodeConfig {
            tile_sample_min_size: args.tile_size,
            tile_sample_stride: tile_stride,
        };
        println!(
            "\nDecoding with tiled VAE (tile: {}px, stride: {}px)...",
            config.tile_sample_min_size, config.tile_sample_stride
        );
        let decoded = vae.tiled_decode(&latents, &config)?;
        // Extract first frame: [B, C, T, H, W] -> [B, C, H, W]
        let (b, c, _t, h, w) = decoded.dims5()?;
        decoded.i((.., .., 0, .., ..))?.reshape((b, c, h, w))?
    } else {
        println!("\nDecoding latents...");
        vae.decode_image(&latents)?
    };

    debug::checkpoint(&mut debug_ctx, "decoded_image", image.clone())?;

    common::postprocess_and_save_4d(&image, &args.output)?;
    println!("Edited image saved to: {}", args.output);

    Ok(())
}

/// Preprocess an image for the vision encoder.
///
/// If `max_pixels` is provided, it constrains the resize to that maximum.
/// This is used to match the output dimensions (e.g., 256x256 = 65536 pixels).
fn preprocess_image_for_vision(
    img: &image::DynamicImage,
    device: &Device,
    dtype: DType,
    max_pixels: Option<usize>,
) -> Result<(Tensor, Tensor, usize)> {
    let (orig_width, orig_height) = (img.width() as usize, img.height() as usize);

    let factor = VISION_PATCH_SIZE * VISION_MERGE_SIZE;
    let max_px = max_pixels.unwrap_or(DEFAULT_MAX_PIXELS);
    let (resized_height, resized_width) = smart_resize(
        orig_height,
        orig_width,
        factor,
        DEFAULT_MIN_PIXELS,
        max_px,
    );

    // Use CatmullRom (bicubic) to match PyTorch's default resample method
    let resized = img.resize_exact(
        resized_width as u32,
        resized_height as u32,
        image::imageops::FilterType::CatmullRom,
    );

    let rgb = resized.to_rgb8();
    let rgb_data: Vec<u8> = rgb.as_raw().to_vec();

    let normalized = normalize_image(&rgb_data, resized_height, resized_width);
    let patches = patchify_image(
        &normalized,
        resized_height,
        resized_width,
        VISION_PATCH_SIZE,
        VISION_TEMPORAL_PATCH_SIZE,
        VISION_MERGE_SIZE,
    );

    let (grid_t, grid_h, grid_w) = get_image_grid_thw(
        resized_height,
        resized_width,
        None,
        None,
        Some(VISION_PATCH_SIZE),
        Some(VISION_MERGE_SIZE),
    );

    let num_patches = grid_t * grid_h * grid_w;
    let patch_elements = 3 * VISION_TEMPORAL_PATCH_SIZE * VISION_PATCH_SIZE * VISION_PATCH_SIZE;
    let pixel_values =
        Tensor::from_vec(patches, (num_patches, patch_elements), device)?.to_dtype(dtype)?;

    let grid_thw = Tensor::new(&[[grid_t as u32, grid_h as u32, grid_w as u32]], device)?;

    Ok((pixel_values, grid_thw, num_patches))
}

/// Encode a prompt with vision embeddings.
fn encode_prompt_with_vision(
    tokenizer: &Tokenizer,
    text_model: &mut Qwen25VLTextModel,
    prompt: &str,
    vision_embeds: &Tensor,
    image_grid_thw: &Tensor,
    num_image_tokens: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let templated = EDIT_PROMPT_TEMPLATE.replace("{}", prompt);

    let encoding = tokenizer
        .encode(templated, false)
        .map_err(|e| anyhow!("Tokenizer error: {}", e))?;

    let mut tokens: Vec<u32> = encoding.get_ids().to_vec();

    // DEBUG: Print original tokens
    eprintln!("[ENCODE_PROMPT] original tokens len: {}", tokens.len());
    eprintln!("[ENCODE_PROMPT] IMAGE_TOKEN_ID we're looking for: {}", IMAGE_TOKEN_ID);
    let original_image_pos = tokens.iter().position(|&t| t == IMAGE_TOKEN_ID);
    eprintln!("[ENCODE_PROMPT] IMAGE_TOKEN_ID found at position: {:?}", original_image_pos);

    // Expand image_pad token to the correct number
    if let Some(pos) = tokens.iter().position(|&t| t == IMAGE_TOKEN_ID) {
        eprintln!("[ENCODE_PROMPT] Expanding single IMAGE_TOKEN_ID at pos {} to {} copies", pos, num_image_tokens);
        let expanded: Vec<u32> = tokens[..pos]
            .iter()
            .chain(std::iter::repeat(&IMAGE_TOKEN_ID).take(num_image_tokens))
            .chain(tokens[pos + 1..].iter())
            .copied()
            .collect();
        tokens = expanded;
    } else {
        eprintln!("[ENCODE_PROMPT] WARNING: No IMAGE_TOKEN_ID found in tokens!");
    }

    let seq_len = tokens.len();
    eprintln!("[ENCODE_PROMPT] final tokens len: {}", seq_len);

    // Verify expansion
    let image_count = tokens.iter().filter(|&&t| t == IMAGE_TOKEN_ID).count();
    eprintln!("[ENCODE_PROMPT] IMAGE_TOKEN_ID count in final tokens: {}", image_count);

    // Print tokens around the image region
    if let Some(first_img_pos) = tokens.iter().position(|&t| t == IMAGE_TOKEN_ID) {
        let start = first_img_pos.saturating_sub(3);
        let end = (first_img_pos + 10).min(tokens.len());
        eprintln!("[ENCODE_PROMPT] tokens[{}..{}] around image start: {:?}", start, end, &tokens[start..end]);
    }
    let input_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
    let attention_mask = Tensor::ones((1, seq_len), DType::F32, device)?;

    let hidden_states = text_model.forward_with_vision(
        &input_ids,
        vision_embeds,
        image_grid_thw,
        Some(&attention_mask),
        VISION_MERGE_SIZE,
        IMAGE_TOKEN_ID,
    )?;

    let out_seq_len = hidden_states.dim(1)?;
    if out_seq_len <= EDIT_DROP_TOKENS {
        return Err(anyhow!(
            "Prompt too short: {} tokens, need > {}",
            out_seq_len,
            EDIT_DROP_TOKENS
        ));
    }

    let embeddings = hidden_states.narrow(1, EDIT_DROP_TOKENS, out_seq_len - EDIT_DROP_TOKENS)?;
    let mask = attention_mask.narrow(1, EDIT_DROP_TOKENS, out_seq_len - EDIT_DROP_TOKENS)?;

    Ok((embeddings, mask))
}
