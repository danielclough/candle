//! ControlNet-guided image generation pipeline.
//!
//! Uses a control image (edges, depth, pose, etc.) to guide the spatial
//! structure of the generated image.

use anyhow::{anyhow, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen_image::{
    apply_true_cfg, pack_latents, unpack_latents, Config, ControlNetConfig,
    QwenImageControlNetModel, TEXT_ONLY_DROP_TOKENS, TEXT_ONLY_PROMPT_TEMPLATE,
};

use crate::common;

/// Arguments specific to the controlnet pipeline.
pub struct ControlnetArgs {
    pub prompt: String,
    pub negative_prompt: String,
    pub control_image: String,
    pub height: usize,
    pub width: usize,
    pub num_inference_steps: usize,
    pub true_cfg_scale: f64,
    pub controlnet_scale: f64,
    pub output: String,
}

/// Model paths for the controlnet pipeline.
pub struct ControlnetModelPaths {
    pub transformer_path: Option<String>,
    pub controlnet_path: Option<String>,
    pub vae_path: Option<String>,
    pub text_encoder_path: Option<String>,
    pub tokenizer_path: Option<String>,
}

pub fn run(
    args: ControlnetArgs,
    paths: ControlnetModelPaths,
    device: &Device,
    dtype: DType,
) -> Result<()> {
    common::validate_dimensions(args.height, args.width)?;

    println!("Qwen-Image with ControlNet");
    println!("Device: {:?}, DType: {:?}", device, dtype);
    println!("Generating {}x{} image", args.width, args.height);
    println!("Prompt: {}", args.prompt);
    println!("Control image: {}", args.control_image);
    println!("ControlNet scale: {:.2}", args.controlnet_scale);

    let api = hf_hub::api::sync::Api::new()?;
    let dims = common::calculate_latent_dims(args.height, args.width);

    // =========================================================================
    // Stage 1: Load VAE and encode control image
    // =========================================================================
    println!("\n[1/6] Loading VAE and encoding control image...");

    let vae = common::load_vae(paths.vae_path.as_deref(), &api, device, dtype)?;
    println!("  VAE loaded");

    // Load and encode control image
    let control_image =
        common::load_image_for_vae(&args.control_image, args.height, args.width, device, dtype)?;

    let control_condition =
        encode_control_condition(&control_image, &vae, dims.latent_height, dims.latent_width)?;
    println!("  Control condition shape: {:?}", control_condition.dims());

    // =========================================================================
    // Stage 2: Load tokenizer and text encoder
    // =========================================================================
    println!("\n[2/6] Loading text encoder...");

    let tokenizer = common::load_tokenizer(paths.tokenizer_path.as_deref(), &api)?;
    let mut text_model =
        common::load_text_encoder(paths.text_encoder_path.as_deref(), &api, device, dtype)?;
    println!("  Text encoder loaded");

    // Encode prompts
    let (pos_embeds, pos_mask) = common::encode_text_prompt(
        &tokenizer,
        &mut text_model,
        &args.prompt,
        TEXT_ONLY_PROMPT_TEMPLATE,
        TEXT_ONLY_DROP_TOKENS,
        device,
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
        TEXT_ONLY_PROMPT_TEMPLATE,
        TEXT_ONLY_DROP_TOKENS,
        device,
    )?;

    drop(text_model);
    println!("  Text encoder freed");

    // =========================================================================
    // Stage 3: Setup scheduler and initial latents
    // =========================================================================
    println!("\n[3/6] Setting up scheduler and latents...");

    let mut scheduler = common::create_scheduler(args.num_inference_steps, dims.image_seq_len);

    let latents = Tensor::randn(
        0f32,
        1f32,
        (1, 16, 1, dims.latent_height, dims.latent_width),
        device,
    )?
    .to_dtype(dtype)?;

    println!("  Latent size: {}x{}", dims.latent_height, dims.latent_width);

    // =========================================================================
    // Stage 4: Load ControlNet
    // =========================================================================
    println!("\n[4/6] Loading ControlNet...");

    let controlnet_config = ControlNetConfig::default_5_layers();
    let controlnet_files = match &paths.controlnet_path {
        Some(path) => {
            println!("  Loading ControlNet from local path: {}", path);
            candle_examples::hub_load_local_safetensors(
                path,
                "diffusion_pytorch_model.safetensors.index.json",
            )?
        }
        None => {
            return Err(anyhow!(
                "ControlNet weights not found. Please provide --controlnet-path."
            ));
        }
    };

    let vb_controlnet =
        unsafe { VarBuilder::from_mmaped_safetensors(&controlnet_files, dtype, device)? };
    let controlnet = QwenImageControlNetModel::new(&controlnet_config, vb_controlnet)?;
    println!("  ControlNet loaded ({} layers)", controlnet.num_layers());

    // =========================================================================
    // Stage 5: Load transformer and run denoising loop
    // =========================================================================
    println!("\n[5/6] Loading transformer and denoising...");

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

    println!(
        "  Running {} denoising steps with ControlNet...",
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

        let packed = pack_latents(&latents, dims.latent_height, dims.latent_width)?;
        let t = Tensor::new(&[timestep as f32 / 1000.0], device)?.to_dtype(dtype)?;

        // Get ControlNet residuals for positive prediction
        let controlnet_output = controlnet.forward(
            &packed,
            &control_condition,
            args.controlnet_scale,
            &pos_embeds,
            &t,
            &img_shapes,
            &txt_seq_lens,
        )?;

        // Positive prediction with ControlNet residuals
        let pos_pred = transformer.forward_with_controlnet(
            &packed,
            &pos_embeds,
            &pos_mask,
            &t,
            &img_shapes,
            &txt_seq_lens,
            Some(&controlnet_output.block_residuals),
        )?;

        // Negative prediction (without ControlNet for True CFG)
        let neg_pred =
            transformer.forward(&packed, &neg_embeds, &neg_mask, &t, &img_shapes, &txt_seq_lens)?;

        let guided_pred = apply_true_cfg(&pos_pred, &neg_pred, args.true_cfg_scale)?;
        let unpacked = unpack_latents(&guided_pred, dims.latent_height, dims.latent_width, 16)?;

        latents = scheduler.step(&unpacked, &latents)?;
    }

    drop(transformer);
    drop(controlnet);
    println!("  Models freed");

    // =========================================================================
    // Stage 6: VAE decode and save
    // =========================================================================
    println!("\n[6/6] Decoding latents...");

    let latents = vae.denormalize_latents(&latents)?;
    let image = vae.decode(&latents)?;

    common::postprocess_and_save(&image, &args.output)?;
    println!("\nControlNet image saved to: {}", args.output);

    Ok(())
}

/// Encode a control image to latent space for ControlNet conditioning.
fn encode_control_condition(
    control_image: &Tensor,
    vae: &candle_transformers::models::qwen_image::AutoencoderKLQwenImage,
    latent_height: usize,
    latent_width: usize,
) -> Result<Tensor> {
    let dist = vae.encode(control_image)?;
    let latents = vae.normalize_latents(&dist.mode().clone())?;
    Ok(pack_latents(&latents, latent_height, latent_width)?)
}
