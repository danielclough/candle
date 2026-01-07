//! Image editing pipeline with text instructions.
//!
//! Performs image editing by conditioning on both the input image latents
//! and an editing instruction. The noise latents are concatenated with
//! the encoded input image latents during denoising.

use anyhow::{anyhow, Result};
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{
    qwen2_5_vl::{
        get_image_grid_thw, normalize_image, patchify_image, smart_resize, Qwen25VLTextModel,
        Qwen25VLVisionModel, VisionConfig, DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS,
    },
    qwen_image::{
        apply_true_cfg, pack_latents, unpack_latents, Config, TiledDecodeConfig,
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
    pub max_resolution: usize,
    pub tiled_decode: Option<bool>,
    pub tile_size: usize,
    pub output: String,
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
    println!("Steps: {}", args.num_inference_steps);

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

    let (target_width, target_height) =
        common::calculate_output_dims(orig_width, orig_height, args.max_resolution);
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
    // Transpose from VAE format [B, C, T, H, W] to pack format [B, T, C, H, W]
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

    let vb_vision =
        unsafe { VarBuilder::from_mmaped_safetensors(&vision_model_files, dtype, device)? };
    let vision_model = Qwen25VLVisionModel::new(&vision_config, vb_vision.pp("visual"))?;
    println!("  Vision encoder loaded");

    // Preprocess input image for vision encoder
    let (pixel_values, image_grid_thw, _) =
        preprocess_image_for_vision(&input_img, device, dtype)?;
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

    let neg_prompt = if args.negative_prompt.is_empty() {
        ""
    } else {
        &args.negative_prompt
    };
    let (neg_embeds, neg_mask) = encode_prompt_with_vision(
        &tokenizer,
        &mut text_model,
        neg_prompt,
        &vision_embeds,
        &image_grid_thw,
        num_image_tokens,
        device,
    )?;
    let neg_embeds = debug::checkpoint(&mut debug_ctx, "negative_prompt_embeds", neg_embeds)?;
    let neg_mask = debug::checkpoint(&mut debug_ctx, "negative_prompt_mask", neg_mask)?;
    let neg_embeds = neg_embeds.to_dtype(dtype)?;

    drop(text_model);
    drop(vision_model);
    println!("  Encoders freed");

    // =========================================================================
    // Stage 4: Load transformer and setup scheduler
    // =========================================================================
    println!("\n[4/5] Loading transformer and setting up denoising...");

    let mut scheduler = common::create_scheduler(args.num_inference_steps, dims.image_seq_len);

    // Create initial noise and pack latents (F32 to avoid BF16 quantization error)
    // Use [B, C, T, H, W] format to match PyTorch convention (edit pipeline)
    let noise_latents = Tensor::randn(
        0f32,
        1f32,
        (1, 16, 1, dims.latent_height, dims.latent_width),
        device,
    )?;
    let noise_latents = debug::checkpoint(&mut debug_ctx, "noise_latents_unpacked", noise_latents)?;
    // Transpose from [B, C, T, H, W] to [B, T, C, H, W] for pack_latents (edit pipeline)
    let noise_latents = noise_latents.permute([0, 2, 1, 3, 4])?;

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
        // Concatenate noise and image latents (both BF16)
        let latent_model_input = Tensor::cat(&[&latents_bf16, &packed_image], 1)?;
        let t = Tensor::new(&[timestep as f32 / 1000.0], device)?.to_dtype(dtype)?;

        // Debug: capture transformer inputs at step 0
        if step == 0 {
            debug::checkpoint(&mut debug_ctx, "transformer_input_hidden_states", latent_model_input.clone())?;
            debug::checkpoint(&mut debug_ctx, "transformer_input_timestep", t.clone())?;
        }

        let pos_pred = transformer.forward(
            &latent_model_input,
            &pos_embeds,
            &pos_mask,
            &t,
            &img_shapes,
            &txt_seq_lens,
        )?;

        // Debug: capture full transformer output at step 0 (before extraction)
        if step == 0 {
            debug::checkpoint(&mut debug_ctx, "transformer_noise_pred_full", pos_pred.clone())?;
        }

        let neg_pred = transformer.forward(
            &latent_model_input,
            &neg_embeds,
            &neg_mask,
            &t,
            &img_shapes,
            &txt_seq_lens,
        )?;

        // Extract only the noise prediction part
        let noise_seq_len = latents.dim(1)?;
        let pos_pred = pos_pred.narrow(1, 0, noise_seq_len)?;
        let neg_pred = neg_pred.narrow(1, 0, noise_seq_len)?;

        // Debug: capture extracted noise predictions at step 0
        if step == 0 {
            debug::checkpoint(&mut debug_ctx, "noise_pred_pos_step0", pos_pred.clone())?;
            debug::checkpoint(&mut debug_ctx, "noise_pred_neg_step0", neg_pred.clone())?;
        }

        let guided_pred = apply_true_cfg(&pos_pred, &neg_pred, args.true_cfg_scale)?;

        // Debug: capture guided prediction at step 0
        if step == 0 {
            debug::checkpoint(&mut debug_ctx, "guided_pred_step0", guided_pred.clone())?;
        }

        // Unpack and step
        // unpack_latents outputs [B, C, T, H, W] to match PyTorch checkpoints
        // NOTE: Noise predictions use output/target dimensions
        let unpacked = unpack_latents(&guided_pred, dims.latent_height, dims.latent_width, 16)?;
        // Convert back to F32 for scheduler arithmetic
        let unpacked = unpacked.to_dtype(DType::F32)?;
        let unpacked_latents =
            unpack_latents(&latents, dims.latent_height, dims.latent_width, 16)?;
        let stepped = scheduler.step(&unpacked, &unpacked_latents)?;
        // Transpose from [B, C, T, H, W] to [B, T, C, H, W] for pack_latents
        let stepped = stepped.permute([0, 2, 1, 3, 4])?;
        latents = pack_latents(&stepped, dims.latent_height, dims.latent_width)?;

        // Debug: capture latents after step 0
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
fn preprocess_image_for_vision(
    img: &image::DynamicImage,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor, usize)> {
    let (orig_width, orig_height) = (img.width() as usize, img.height() as usize);

    let factor = VISION_PATCH_SIZE * VISION_MERGE_SIZE;
    let (resized_height, resized_width) = smart_resize(
        orig_height,
        orig_width,
        factor,
        DEFAULT_MIN_PIXELS,
        DEFAULT_MAX_PIXELS,
    );

    let resized = img.resize_exact(
        resized_width as u32,
        resized_height as u32,
        image::imageops::FilterType::Lanczos3,
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

    // Expand image_pad token to the correct number
    if let Some(pos) = tokens.iter().position(|&t| t == IMAGE_TOKEN_ID) {
        let expanded: Vec<u32> = tokens[..pos]
            .iter()
            .chain(std::iter::repeat(&IMAGE_TOKEN_ID).take(num_image_tokens))
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
