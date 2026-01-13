//! Image editing pipeline with text instructions.
//!
//! Performs image editing by conditioning on both the input image latents
//! and an editing instruction. The noise latents are concatenated with
//! the encoded input image latents during denoising.

use anyhow::{anyhow, Result};
use candle::{DType, Device, IndexOp, Tensor};

use crate::mt_box_muller_rng::MtBoxMullerRng;
use candle_transformers::models::{
    qwen2_5_vl::{
        get_image_grid_thw, normalize_image, patchify_image, smart_resize, DEFAULT_MAX_PIXELS,
        DEFAULT_MIN_PIXELS,
    },
    qwen_image::{
        apply_true_cfg, pack_latents, unpack_latents, Config, InferenceConfig, TiledDecodeConfig,
        EDIT_DROP_TOKENS, EDIT_PROMPT_TEMPLATE,
    },
};
use tokenizers::Tokenizer;

use crate::common;

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
    pub steps: usize,
    pub true_cfg_scale: f64,
    pub model_id: String,
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
    pub gguf_transformer_path: Option<String>,
    pub vae_path: Option<String>,
    pub text_encoder_path: Option<String>,
    pub gguf_text_encoder_path: Option<String>,
    pub vision_encoder_path: Option<String>,
    pub gguf_vision_encoder_path: Option<String>,
    pub tokenizer_path: Option<String>,
}

pub fn run(args: EditArgs, paths: EditModelPaths, device: &Device, dtype: DType) -> Result<()> {
    // Auto-detect pipeline type based on model ID
    let use_plus = args.model_id.contains("2509")
        || args.model_id.contains("2511")
        || args.model_id.contains("Plus");
    let pipeline_name = if use_plus {
        "QwenImageEditPlusPipeline"
    } else {
        "QwenImageEditPipeline"
    };

    println!("Qwen-Image Edit");
    println!("Pipeline: {}", pipeline_name);
    println!("Device: {:?}, DType: {:?}", device, dtype);
    println!("Model: {}", args.model_id);
    println!("Input image: {}", args.input_image);
    println!("Prompt: {}", args.prompt);
    println!(
        "Negative prompt: {}",
        if args.negative_prompt.is_empty() {
            "(none)"
        } else {
            &args.negative_prompt
        }
    );
    println!("True CFG scale: {}", args.true_cfg_scale);
    println!("Steps: {}", args.steps);

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

    let dims = common::OutputDims::new(target_height, target_width);

    // =========================================================================
    // Stage 2: Load VAE and encode input image
    // =========================================================================
    println!("\n[2/5] Loading VAE and encoding input image...");

    let vae = common::load_vae(
        paths.vae_path.as_deref(),
        &api,
        device,
        dtype,
        &args.model_id,
    )?;
    println!("  VAE loaded");

    let vae_input =
        common::load_image_for_vae(&args.input_image, target_height, target_width, device)?;

    let image_latents = common::encode_and_normalize_image(&vae, &vae_input)?;
    // Permute from VAE format [B, C, T, H, W] to pack format [B, T, C, H, W]
    let image_latents = image_latents.permute([0, 2, 1, 3, 4])?;
    println!("  Image latents shape: {:?}", image_latents.dims());

    // =========================================================================
    // Stage 3: Load vision encoder, text encoder, and encode prompts
    // =========================================================================
    println!("\n[3/5] Loading encoders and encoding prompts...");

    let tokenizer = common::load_tokenizer(paths.tokenizer_path.as_deref(), &api)?;

    // Load vision encoder (F32 for numerical precision, supports GGUF mmproj or safetensors)
    let vision_model = common::load_vision_encoder_variant(
        paths.vision_encoder_path.as_deref(),
        paths.gguf_vision_encoder_path.as_deref(),
        &api,
        device,
        DType::F32,
    )?;
    println!("  Vision encoder loaded ({:?})", dtype);

    // Preprocess input image for vision encoder (F32)
    // Constrain to target dimensions to match Python behavior
    let vision_max_pixels = target_width * target_height;
    let (pixel_values, image_grid_thw, _) =
        preprocess_image_for_vision(&input_img, device, DType::F32, Some(vision_max_pixels))?;
    println!(
        "  Image preprocessed for vision: {} patches",
        pixel_values.dim(0)?
    );

    // Compute vision embeddings (convert to F32 for text encoder which uses F32)
    let vision_embeds = vision_model
        .forward(&pixel_values, &image_grid_thw)?
        .to_dtype(DType::F32)?;
    let num_image_tokens = vision_embeds.dim(0)?;
    println!("  Vision embeddings: {} tokens", num_image_tokens);

    // Free vision encoder from memory before loading text encoder
    drop(vision_model);

    // Load text encoder (FP16 or quantized GGUF)
    let mut text_model = common::load_text_encoder_variant(
        paths.text_encoder_path.as_deref(),
        paths.gguf_text_encoder_path.as_deref(),
        &api,
        device,
    )?;
    common::log_text_encoder_loaded(text_model.is_quantized(), dtype);

    // Encode prompts with vision (text encoder outputs F32, convert to transformer dtype)
    let (pos_embeds, _) = encode_prompt_with_vision(
        &tokenizer,
        &mut text_model,
        &args.prompt,
        &vision_embeds,
        &image_grid_thw,
        num_image_tokens,
        device,
    )?;
    let pos_embeds = pos_embeds.to_dtype(dtype)?;
    println!("  Positive prompt embeddings: {:?}", pos_embeds.dims());

    // Only encode negative prompt if one is provided (matching Python behavior)
    let neg_embeds = if !args.negative_prompt.is_empty() {
        let (neg_embeds, _) = encode_prompt_with_vision(
            &tokenizer,
            &mut text_model,
            &args.negative_prompt,
            &vision_embeds,
            &image_grid_thw,
            num_image_tokens,
            device,
        )?;
        let neg_embeds = neg_embeds.to_dtype(dtype)?;
        println!("  Negative prompt embeddings: {:?}", neg_embeds.dims());
        Some(neg_embeds)
    } else {
        // No negative prompt → skip encoding entirely (Python skips CFG in this case)
        None
    };

    // Free text encoder from memory before loading transformer
    drop(text_model);
    println!("  Text encoder freed");

    // =========================================================================
    // Stage 4: Load transformer and setup scheduler
    // =========================================================================
    println!("\n[4/5] Loading transformer and setting up denoising...");

    let mut scheduler = common::create_scheduler(args.steps, dims.image_seq_len);

    // Create initial noise (F32 to avoid BF16 quantization error)
    // Use [B, T, C, H, W] format to match PyTorch diffusers
    // Always use PyTorch-compatible MT19937 + Box-Muller RNG for consistent noise distribution
    let seed = common::get_seed_or_current_time(args.seed);
    println!("  Using seed: {}", seed);
    let mut rng = MtBoxMullerRng::new(seed);
    let noise_latents = rng
        .randn(
            &[1, 1, 16, dims.latent_height, dims.latent_width],
            &Device::Cpu,
            DType::F32,
        )?
        .to_device(device)?;

    let packed_noise = pack_latents(&noise_latents, dims.latent_height, dims.latent_width)?;

    // Pack image latents and convert to BF16 for transformer input
    let packed_image = pack_latents(&image_latents, dims.latent_height, dims.latent_width)?;
    let packed_image = packed_image.to_dtype(dtype)?;

    // Load transformer - select config based on pipeline type
    let config = if use_plus {
        Config::qwen_image_edit_plus()
    } else {
        Config::qwen_image_edit()
    };
    let inference_config = InferenceConfig::default();
    let transformer = common::load_transformer_variant(
        paths.transformer_path.as_deref(),
        paths.gguf_transformer_path.as_deref(),
        &args.model_id,
        &config,
        &api,
        device,
        dtype,
        &inference_config,
    )?;
    common::log_transformer_loaded(
        config.num_layers,
        transformer.is_quantized(),
        dtype,
        &inference_config,
    );

    // =========================================================================
    // Stage 5: Denoising loop
    // =========================================================================
    println!("\n[5/5] Running denoising loop...");

    // Image shapes for RoPE: noise + image regions (both same size)
    let img_shapes = vec![
        (1, dims.packed_height, dims.packed_width), // Noise latents
        (1, dims.packed_height, dims.packed_width), // Image latents
    ];
    let timesteps = scheduler.timesteps().to_vec();
    let mut latents = packed_noise;

    for (step, &timestep) in timesteps.iter().take(args.steps).enumerate() {
        common::log_denoising_step(step, args.steps, timestep);

        // Convert F32 latents to BF16 for transformer (weights are BF16)
        let latents_bf16 = latents.to_dtype(dtype)?;
        // Concatenate noise and image latents (both BF16)
        let latent_model_input = Tensor::cat(&[&latents_bf16, &packed_image], 1)?;
        let t = Tensor::new(&[timestep as f32 / 1000.0], device)?.to_dtype(dtype)?;

        // Note: Timestep doubling for zero_cond_t is handled internally by the transformer
        let pos_pred = transformer.forward(&latent_model_input, &pos_embeds, &t, &img_shapes)?;

        // Extract only the noise prediction part
        let noise_seq_len = latents.dim(1)?;
        let pos_pred = pos_pred.narrow(1, 0, noise_seq_len)?;

        // Only apply CFG if we have a negative prompt (matching Python behavior)
        let model_pred = if let Some(ref neg_emb) = &neg_embeds {
            let neg_pred = transformer.forward(&latent_model_input, neg_emb, &t, &img_shapes)?;

            let neg_pred = neg_pred.narrow(1, 0, noise_seq_len)?;

            apply_true_cfg(&pos_pred, &neg_pred, args.true_cfg_scale)?
        } else {
            // No negative prompt → skip CFG, use positive prediction directly
            pos_pred
        };

        // Convert model_pred to F32 for scheduler arithmetic (matching PyTorch behavior)
        let model_pred = model_pred.to_dtype(DType::F32)?;

        // Scheduler operates on PACKED latents (matching PyTorch diffusers)
        latents = scheduler.step(&model_pred, &latents)?;
    }

    let final_latents = unpack_latents(&latents, dims.latent_height, dims.latent_width, 16)?;

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

    // Decode latents to image
    // PyTorch: image = vae.decode(latents)[0][:, :, 0]
    // The [:, :, 0] extracts first temporal frame: [B, C, T, H, W] -> [B, C, H, W]
    let image = if use_tiled {
        let tile_stride = (args.tile_size * 3) / 4;
        let config = TiledDecodeConfig::uniform(args.tile_size, tile_stride);
        println!(
            "\nDecoding with tiled VAE (tile: {}px, stride: {}px)...",
            args.tile_size, tile_stride
        );
        let decoded = vae.tiled_decode(&latents, &config)?;
        // Extract first frame: [B, C, T, H, W] -> [B, C, H, W]
        let (b, c, _t, h, w) = decoded.dims5()?;
        decoded.i((.., .., 0, .., ..))?.reshape((b, c, h, w))?
    } else {
        println!("\nDecoding latents...");
        vae.decode_image(&latents)?
    };

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
    let (resized_height, resized_width) =
        smart_resize(orig_height, orig_width, factor, DEFAULT_MIN_PIXELS, max_px);

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
    text_model: &mut common::TextEncoderVariant,
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

    // Expand image_pad token to the correct number
    if let Some(pos) = tokens.iter().position(|&t| t == IMAGE_TOKEN_ID) {
        let expanded: Vec<u32> = tokens[..pos]
            .iter()
            .chain(std::iter::repeat_n(&IMAGE_TOKEN_ID, num_image_tokens))
            .chain(tokens[pos + 1..].iter())
            .copied()
            .collect();
        tokens = expanded;
    }

    let seq_len = tokens.len();
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
