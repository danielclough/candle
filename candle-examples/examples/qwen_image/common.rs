//! Common utilities for Qwen-Image pipelines.
//!
//! This module provides shared functionality across all Qwen-Image generation modes:
//! - Device and tracing setup
//! - Image loading and saving with VAE normalization
//! - Model loading (VAE, tokenizer, text encoder, transformer)
//! - Prompt encoding utilities
//! - Scheduler setup

use anyhow::{anyhow, Result};
use candle::{DType, Device, IndexOp, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::{
    qwen2_5_vl::{Config as TextConfig, Qwen25VLTextModel},
    qwen_image::{
        calculate_shift, AutoencoderKLQwenImage, Config as TransformerConfig,
        FlowMatchEulerDiscreteScheduler, PromptMode, QwenImageTransformer2DModel, SchedulerConfig,
        VaeConfig,
    },
};
use tokenizers::Tokenizer;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

/// VAE spatial compression factor (8x).
pub const VAE_SCALE_FACTOR: usize = 8;

/// Patch size for latent packing (2x2 patches).
pub const PATCH_SIZE: usize = 2;

/// Default HuggingFace model IDs.
pub const DEFAULT_TEXT_ENCODER_ID: &str = "Qwen/Qwen2.5-VL-7B-Instruct";
pub const DEFAULT_VAE_MODEL_ID: &str = "Qwen/Qwen-Image";
pub const DEFAULT_TRANSFORMER_ID: &str = "Qwen/Qwen-Image";

// ============================================================================
// Setup utilities
// ============================================================================

/// Initialize Chrome tracing if enabled.
///
/// Returns a guard that must be kept alive for the duration of tracing.
pub fn setup_tracing(enabled: bool) -> Option<tracing_chrome::FlushGuard> {
    if enabled {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    }
}

/// Setup compute device and dtype based on arguments.
///
/// Note: Device seeding is only supported on Metal/CUDA. For CPU, the seed is
/// handled by `MtBoxMullerRng` in the generate pipeline for reproducible latents.
pub fn setup_device_and_dtype(cpu: bool, use_f32: bool, seed: Option<u64>) -> Result<(Device, DType)> {
    let device = candle_examples::device(cpu)?;
    if let Some(seed) = seed {
        // Only seed non-CPU devices (CPU doesn't support set_seed, but we use
        // MtBoxMullerRng for reproducible latent generation anyway)
        if !matches!(device, Device::Cpu) {
            device.set_seed(seed)?;
        }
    }
    let dtype = if use_f32 {
        DType::F32
    } else {
        device.bf16_default_to_f32()
    };
    Ok((device, dtype))
}

// ============================================================================
// Image loading utilities
// ============================================================================

/// Load an image and prepare it for VAE encoding.
///
/// Returns a tensor of shape [1, 3, 1, height, width] normalized to [-1, 1].
/// Uses (B, C, T, H, W) convention which VAE expects internally.
/// Always returns F32 since the VAE uses F32 for numerical precision.
pub fn load_image_for_vae(
    path: &str,
    target_height: usize,
    target_width: usize,
    device: &Device,
) -> Result<Tensor> {
    let img = image::ImageReader::open(path)
        .map_err(|e| anyhow!("Failed to open image '{}': {}", path, e))?
        .decode()
        .map_err(|e| anyhow!("Failed to decode image '{}': {}", path, e))?;

    let img = img.resize_exact(
        target_width as u32,
        target_height as u32,
        image::imageops::FilterType::Lanczos3,
    );

    let rgb = img.to_rgb8();
    let mut data = Vec::with_capacity(3 * target_height * target_width);

    // Convert to CHW format and normalize to [-1, 1]
    for c in 0..3 {
        for y in 0..target_height {
            for x in 0..target_width {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                let value = (pixel[c] as f32 / 127.5) - 1.0;
                data.push(value);
            }
        }
    }

    // Shape: [1, 3, 1, height, width] for 3D VAE - (B, C, T, H, W) convention
    Ok(Tensor::from_vec(data, (1, 3, 1, target_height, target_width), device)?)
}

/// Load a grayscale mask image and resize to latent dimensions.
///
/// Returns a tensor of shape [1, 1, 1, latent_height, latent_width] (B, T, C, H, W) with values in [0, 1].
/// White (1.0) = region to process, Black (0.0) = region to preserve.
pub fn load_mask_for_latents(
    path: &str,
    latent_height: usize,
    latent_width: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let img = image::ImageReader::open(path)
        .map_err(|e| anyhow!("Failed to open mask '{}': {}", path, e))?
        .decode()
        .map_err(|e| anyhow!("Failed to decode mask '{}': {}", path, e))?;

    // Resize mask directly to latent dimensions
    let img = img.resize_exact(
        latent_width as u32,
        latent_height as u32,
        image::imageops::FilterType::Lanczos3,
    );

    let gray = img.to_luma8();
    let mut data = Vec::with_capacity(latent_height * latent_width);

    for y in 0..latent_height {
        for x in 0..latent_width {
            let pixel = gray.get_pixel(x as u32, y as u32);
            let value = pixel[0] as f32 / 255.0;
            data.push(value);
        }
    }

    Ok(Tensor::from_vec(data, (1, 1, 1, latent_height, latent_width), device)?.to_dtype(dtype)?)
}

/// Post-process VAE output and save to file.
///
/// Handles the 3D VAE output format [1, 3, T, H, W], takes the first temporal frame,
/// clamps to [-1, 1], converts to [0, 255] uint8, and saves.
pub fn postprocess_and_save(image: &Tensor, output_path: &str) -> Result<()> {
    // Remove batch: [1, 3, T, H, W] -> [3, T, H, W]
    let image = image.squeeze(0)?;
    // Take first temporal frame: [3, T, H, W] -> [3, H, W]
    let image = image.i((.., 0, .., ..))?;
    // Clamp and convert to [0, 255]
    let image = ((image.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?;
    let image = image.to_dtype(DType::U8)?;

    candle_examples::save_image(&image, output_path)?;
    Ok(())
}

/// Save a 4D image tensor [B, C, H, W] (already frame-selected).
pub fn postprocess_and_save_4d(image: &Tensor, output_path: &str) -> Result<()> {
    // Remove batch: [1, 3, H, W] -> [3, H, W]
    let image = image.squeeze(0)?;
    // Clamp and convert to [0, 255]
    let image = ((image.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?;
    let image = image.to_dtype(DType::U8)?;

    candle_examples::save_image(&image, output_path)?;
    Ok(())
}

// ============================================================================
// Model loading utilities
// ============================================================================

/// Load the Qwen-Image VAE.
///
/// NOTE: The VAE always runs in F32 for numerical precision, regardless of the
/// `dtype` parameter. BF16 causes significant quality degradation in the 3D
/// convolutions and upsampling layers. Input tensors are converted to F32
/// before VAE operations.
pub fn load_vae(
    vae_path: Option<&str>,
    api: &hf_hub::api::sync::Api,
    device: &Device,
    _dtype: DType, // Ignored - VAE always uses F32 for precision
) -> Result<AutoencoderKLQwenImage> {
    let vae_config = VaeConfig::qwen_image();
    let vae_file = match vae_path {
        Some(path) => std::path::PathBuf::from(path).join("diffusion_pytorch_model.safetensors"),
        None => {
            let repo = api.repo(hf_hub::Repo::model(DEFAULT_VAE_MODEL_ID.to_string()));
            repo.get("vae/diffusion_pytorch_model.safetensors")?
        }
    };

    // Always use F32 for the VAE to maintain numerical precision
    // through the 3D convolution and upsampling layers.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[vae_file], DType::F32, device)? };
    Ok(AutoencoderKLQwenImage::new(&vae_config, vb)?)
}

/// Load the tokenizer.
pub fn load_tokenizer(
    tokenizer_path: Option<&str>,
    api: &hf_hub::api::sync::Api,
) -> Result<Tokenizer> {
    let path = match tokenizer_path {
        Some(p) => std::path::PathBuf::from(p),
        None => {
            let repo = api.repo(hf_hub::Repo::model(DEFAULT_TEXT_ENCODER_ID.to_string()));
            repo.get("tokenizer.json")?
        }
    };
    Tokenizer::from_file(&path).map_err(|e| anyhow!("Failed to load tokenizer: {}", e))
}

/// Load the Qwen2.5-VL text encoder.
///
/// NOTE: The text encoder always runs in F32 for numerical precision. The embeddings are converted
/// to the target dtype after encoding in `encode_text_prompt`. This mixed-precision approach provides
/// optimal quality without sacrificing performance in the transformer/VAE.
pub fn load_text_encoder(
    text_encoder_path: Option<&str>,
    api: &hf_hub::api::sync::Api,
    device: &Device,
) -> Result<Qwen25VLTextModel> {
    let text_config = TextConfig::default();
    let model_files = match text_encoder_path {
        Some(path) => {
            candle_examples::hub_load_local_safetensors(path, "model.safetensors.index.json")?
        }
        None => {
            let repo = api.repo(hf_hub::Repo::model(DEFAULT_TEXT_ENCODER_ID.to_string()));
            candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?
        }
    };

    // Always use F32 for the text encoder to maintain numerical precision
    // through the 28 transformer layers. BF16 accumulates too much error.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, DType::F32, device)? };
    Ok(Qwen25VLTextModel::new(&text_config, vb)?)
}

/// Load the Qwen-Image transformer with the specified configuration.
pub fn load_transformer(
    transformer_path: Option<&str>,
    model_id: &str,
    config: &TransformerConfig,
    api: &hf_hub::api::sync::Api,
    device: &Device,
    dtype: DType,
) -> Result<QwenImageTransformer2DModel> {
    let model_files = match transformer_path {
        Some(path) => {
            candle_examples::hub_load_local_safetensors(
                path,
                "diffusion_pytorch_model.safetensors.index.json",
            )?
        }
        None => {
            let repo = api.repo(hf_hub::Repo::model(model_id.to_string()));
            candle_examples::hub_load_safetensors(
                &repo,
                "transformer/diffusion_pytorch_model.safetensors.index.json",
            )?
        }
    };

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, dtype, device)? };
    Ok(QwenImageTransformer2DModel::new(config, vb)?)
}

// ============================================================================
// Prompt encoding utilities
// ============================================================================

/// Encode a text-only prompt using the specified mode.
///
/// This handles the standard text-only encoding flow:
/// 1. Apply the prompt template for the given mode
/// 2. Tokenize
/// 3. Run through text encoder (always in F32 for precision)
/// 4. Drop the system prefix tokens
/// 5. Convert embeddings to target dtype (for downstream BF16 transformer)
///
/// Returns (embeddings, attention_mask).
pub fn encode_text_prompt(
    tokenizer: &Tokenizer,
    text_model: &mut Qwen25VLTextModel,
    prompt: &str,
    mode: PromptMode,
    device: &Device,
    target_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let templated = mode.template().replace("{}", prompt);
    let drop_tokens = mode.drop_tokens();

    let encoding = tokenizer
        .encode(templated, false)
        .map_err(|e| anyhow!("Tokenizer error: {}", e))?;

    let tokens = encoding.get_ids();
    let attention_mask = encoding.get_attention_mask();

    let input_ids = Tensor::new(tokens, device)?.unsqueeze(0)?;
    let attention_mask_tensor = Tensor::new(attention_mask, device)?
        .to_dtype(DType::F32)?
        .unsqueeze(0)?;

    // Debug: Check input tensors before forward pass
    if std::env::var("QWEN_DEBUG").is_ok() {
        // Check attention mask values
        let mask_f32 = attention_mask_tensor.to_dtype(DType::F32)?;
        let mask_vec: Vec<f32> = mask_f32.flatten_all()?.to_vec1()?;
        let mask_min = mask_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let mask_max = mask_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mask_sum: f32 = mask_vec.iter().sum();
        let mask_len = attention_mask_tensor.dim(1)?;
        println!("[DEBUG] attention_mask: shape={:?}, min={}, max={}, sum={} (expect sum={})",
            attention_mask_tensor.dims(), mask_min, mask_max, mask_sum, mask_len);

        // Warn if mask values are outside expected range [0, 1]
        if mask_min < 0.0 || mask_max > 1.0 {
            println!("[DEBUG] ⚠️  WARNING: attention_mask values outside [0, 1] range!");
        }

        // Check input_ids range
        let ids_vec: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
        let ids_min = *ids_vec.iter().min().unwrap_or(&0);
        let ids_max = *ids_vec.iter().max().unwrap_or(&0);
        println!("[DEBUG] input_ids: shape={:?}, min={}, max={}, num_tokens={}",
            input_ids.dims(), ids_min, ids_max, ids_vec.len());

        // Print first and last few tokens for context
        if ids_vec.len() > 10 {
            println!("[DEBUG] input_ids first 5: {:?}", &ids_vec[..5]);
            println!("[DEBUG] input_ids last 5: {:?}", &ids_vec[ids_vec.len()-5..]);
        } else {
            println!("[DEBUG] input_ids all: {:?}", ids_vec);
        }
    }

    let hidden_states = text_model.forward_text_only(&input_ids, Some(&attention_mask_tensor))?;

    // Debug: Check output hidden states for NaN
    if std::env::var("QWEN_DEBUG").is_ok() {
        let hs_f32 = hidden_states.to_dtype(DType::F32)?;
        let hs_vec: Vec<f32> = hs_f32.flatten_all()?.to_vec1()?;
        let nan_count = hs_vec.iter().filter(|x| x.is_nan()).count();
        let inf_count = hs_vec.iter().filter(|x| x.is_infinite()).count();

        if nan_count > 0 || inf_count > 0 {
            println!("[DEBUG] ❌ hidden_states contains {} NaN, {} Inf values out of {}",
                nan_count, inf_count, hs_vec.len());
        } else {
            let mean = hs_vec.iter().sum::<f32>() / hs_vec.len() as f32;
            let variance = hs_vec.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hs_vec.len() as f32;
            let std = variance.sqrt();
            let min = hs_vec.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = hs_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("[DEBUG] ✓ hidden_states: shape={:?}, mean={:.4}, std={:.4}, min={:.4}, max={:.4}",
                hidden_states.dims(), mean, std, min, max);
        }
    }

    let seq_len = hidden_states.dim(1)?;
    if seq_len <= drop_tokens {
        return Err(anyhow!(
            "Prompt too short: {} tokens after encoding, need more than {}",
            seq_len,
            drop_tokens
        ));
    }

    let embeddings = hidden_states.narrow(1, drop_tokens, seq_len - drop_tokens)?;
    let mask = attention_mask_tensor.narrow(1, drop_tokens, seq_len - drop_tokens)?;

    // Convert embeddings from F32 to target dtype (typically BF16) for the downstream transformer
    let embeddings = embeddings.to_dtype(target_dtype)?;

    Ok((embeddings, mask))
}

// ============================================================================
// Scheduler utilities
// ============================================================================

/// Create and configure the FlowMatch Euler scheduler.
///
/// Automatically calculates the dynamic shift (mu) based on image sequence length.
pub fn create_scheduler(
    num_inference_steps: usize,
    image_seq_len: usize,
) -> FlowMatchEulerDiscreteScheduler {
    let config = SchedulerConfig::default();
    let mu = calculate_shift(
        image_seq_len,
        config.base_image_seq_len,
        config.max_image_seq_len,
        config.base_shift,
        config.max_shift,
    );

    let mut scheduler = FlowMatchEulerDiscreteScheduler::new(&config);
    scheduler.set_timesteps(num_inference_steps, None, Some(mu));
    scheduler
}

/// Calculate latent dimensions from image dimensions.
pub fn calculate_latent_dims(height: usize, width: usize) -> LatentDims {
    let latent_height = height / VAE_SCALE_FACTOR;
    let latent_width = width / VAE_SCALE_FACTOR;
    let packed_height = latent_height / PATCH_SIZE;
    let packed_width = latent_width / PATCH_SIZE;
    let image_seq_len = packed_height * packed_width;

    LatentDims {
        latent_height,
        latent_width,
        packed_height,
        packed_width,
        image_seq_len,
    }
}

/// Latent space dimensions.
#[derive(Debug, Clone, Copy)]
pub struct LatentDims {
    pub latent_height: usize,
    pub latent_width: usize,
    pub packed_height: usize,
    pub packed_width: usize,
    pub image_seq_len: usize,
}

// ============================================================================
// Dimension validation
// ============================================================================

/// Validate that dimensions are divisible by 16.
pub fn validate_dimensions(height: usize, width: usize) -> Result<()> {
    if height % 16 != 0 || width % 16 != 0 {
        return Err(anyhow!(
            "Height ({}) and width ({}) must be divisible by 16",
            height,
            width
        ));
    }
    Ok(())
}

/// Round dimensions to nearest multiple of 16.
pub fn round_to_16(height: usize, width: usize) -> (usize, usize) {
    let h = ((height + 15) / 16) * 16;
    let w = ((width + 15) / 16) * 16;
    (h, w)
}

/// Calculate output dimensions based on max resolution while preserving aspect ratio.
pub fn calculate_output_dims(
    orig_width: usize,
    orig_height: usize,
    max_resolution: usize,
) -> (usize, usize) {
    if max_resolution == 0 {
        // Preserve input size, just round to 16
        round_to_16(orig_height, orig_width)
    } else {
        let aspect_ratio = orig_width as f64 / orig_height as f64;
        let (w, h) = if orig_width >= orig_height {
            let w = max_resolution;
            let h = (w as f64 / aspect_ratio).round() as usize;
            (w, h)
        } else {
            let h = max_resolution;
            let w = (h as f64 * aspect_ratio).round() as usize;
            (w, h)
        };
        round_to_16(h, w)
    }
}

/// Calculate source image encoding dimensions for edit pipeline.
///
/// PyTorch's QwenImageEditPipeline encodes the source image at a HIGHER resolution
/// (targeting 1M pixels = 1024×1024) to preserve detail for editing, while the
/// output is generated at the target resolution.
///
/// This function replicates PyTorch's `calculate_dimensions(1024*1024, ratio)`:
/// ```python
/// width = math.sqrt(target_area * ratio)
/// height = width / ratio
/// width = round(width / 32) * 32
/// height = round(height / 32) * 32
/// ```
pub fn calculate_source_image_dims(orig_width: usize, orig_height: usize) -> (usize, usize) {
    const TARGET_AREA: f64 = 1024.0 * 1024.0; // 1M pixels, matching PyTorch

    let ratio = orig_width as f64 / orig_height as f64;
    let width = (TARGET_AREA * ratio).sqrt();
    let height = width / ratio;

    // Round to nearest 32 (matching PyTorch's rounding)
    let width = ((width / 32.0).round() * 32.0) as usize;
    let height = ((height / 32.0).round() * 32.0) as usize;

    (width, height)
}
