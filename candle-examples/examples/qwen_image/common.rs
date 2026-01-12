//! Common utilities for Qwen-Image pipelines.
//!
//! This module provides shared functionality across all Qwen-Image generation modes:
//! - Device and tracing setup
//! - Image loading and saving with VAE normalization
//! - Model loading (VAE, tokenizer, text encoder, transformer)
//! - Prompt encoding utilities
//! - Scheduler setup

use anyhow::{anyhow, Result};
use candle::quantized::gguf_file;
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{
    qwen2_5_vl::{Config as TextConfig, Qwen25VLTextModel},
    qwen_image::{
        calculate_shift, AutoencoderKLQwenImage, Config as TransformerConfig,
        FlowMatchEulerDiscreteScheduler, PromptMode, QwenImageTransformer2DModel,
        QwenImageTransformer2DModelQuantized, SchedulerConfig, VaeConfig,
    },
};
use tokenizers::Tokenizer;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

/// VAE spatial compression factor (8x).
pub const VAE_SCALE_FACTOR: usize = 8;

/// Patch size for latent packing (2x2 patches).
pub const PATCH_SIZE: usize = 2;

/// Default HuggingFace model IDs (FP16 safetensors).
pub const DEFAULT_TEXT_ENCODER_ID: &str = "Qwen/Qwen2.5-VL-7B-Instruct";

/// Default GGUF model paths (HuggingFace format: owner/repo/filename).
/// Text encoder and vision encoder use Qwen2.5-VL GGUF files from unsloth.
pub const DEFAULT_GGUF_TEXT_ENCODER: &str =
    "unsloth/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf";
/// Vision encoder (mmproj) files at different precision levels - matched to working dtype.
pub const DEFAULT_GGUF_VISION_ENCODER_F32: &str =
    "unsloth/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-F32.gguf";
pub const DEFAULT_GGUF_VISION_ENCODER_F16: &str =
    "unsloth/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-F16.gguf";
pub const DEFAULT_GGUF_VISION_ENCODER_BF16: &str =
    "unsloth/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-BF16.gguf";
/// Default GGUF transformer from city96/Qwen-Image-gguf.
/// Available quants: Q2_K (7GB), Q3_K_M (9.7GB), Q4_K_M (13GB), Q5_K_M (15GB), Q8_0 (22GB), BF16 (41GB)
pub const DEFAULT_GGUF_TRANSFORMER: &str = "city96/Qwen-Image-gguf/qwen-image-Q4_K_M.gguf";

// ============================================================================
// GGUF Path Resolution
// ============================================================================

/// Resolve a GGUF path to a local file, downloading from HuggingFace if needed.
///
/// Accepts:
/// - "auto" → uses the provided default
/// - Local path → returns as-is if exists
/// - "owner/repo/filename.gguf" → downloads from HuggingFace
pub fn resolve_gguf_path(
    value: &str,
    default: &str,
    api: &hf_hub::api::sync::Api,
) -> Result<std::path::PathBuf> {
    let path_str = if value == "auto" { default } else { value };

    // Check if it's a local file
    let local_path = std::path::Path::new(path_str);
    if local_path.exists() {
        return Ok(local_path.to_path_buf());
    }

    // Parse as HuggingFace: owner/repo/filename
    let parts: Vec<&str> = path_str.splitn(3, '/').collect();
    if parts.len() == 3 {
        let repo_id = format!("{}/{}", parts[0], parts[1]);
        let filename = parts[2];
        println!("Downloading {} from {}...", filename, repo_id);
        let repo = api.repo(hf_hub::Repo::model(repo_id));
        let path = repo.get(filename)?;
        return Ok(path);
    }

    // If we get here, it's neither a local file nor valid HF format
    Err(anyhow!(
        "GGUF path '{}' not found locally and not in owner/repo/file format",
        path_str
    ))
}

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

pub fn setup_device_and_dtype(
    cpu: bool,
    use_f32: bool,
    use_f16: bool,
    seed: Option<u64>,
) -> Result<(Device, DType)> {
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
    } else if use_f16 {
        DType::F16
    } else {
        // Default: BF16 with F32 fallback for devices that don't support BF16
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
    Ok(Tensor::from_vec(
        data,
        (1, 3, 1, target_height, target_width),
        device,
    )?)
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
    model_id: &str,
) -> Result<AutoencoderKLQwenImage> {
    let vae_config = VaeConfig::qwen_image();
    let vae_file = match vae_path {
        Some(path) => std::path::PathBuf::from(path).join("diffusion_pytorch_model.safetensors"),
        None => {
            let repo = api.repo(hf_hub::Repo::model(model_id.to_owned()));
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
        Some(path) => candle_examples::hub_load_local_safetensors(
            path,
            "diffusion_pytorch_model.safetensors.index.json",
        )?,
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

/// Load the quantized transformer model from GGUF file.
///
/// # Arguments
/// * `gguf_path` - Path to the GGUF transformer file
/// * `device` - Device to load on
/// * `dtype` - Working dtype for biases and normalization layers
///
/// # Returns
/// Quantized transformer model
pub fn load_transformer_quantized(
    gguf_path: &str,
    device: &Device,
    dtype: DType,
) -> Result<QwenImageTransformer2DModelQuantized> {
    println!("Loading quantized transformer from {}...", gguf_path);
    let mut file = std::fs::File::open(gguf_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    Ok(QwenImageTransformer2DModelQuantized::from_gguf(
        content, &mut file, device, dtype,
    )?)
}

// ============================================================================
// TransformerVariant - Unified interface for FP16 and quantized transformers
// ============================================================================

/// Unified transformer variant that abstracts over FP16 and quantized models.
///
/// This enum provides a common interface for both model types, handling the
/// differences in their forward signatures internally.
#[allow(clippy::large_enum_variant)]
pub enum TransformerVariant {
    FP16(QwenImageTransformer2DModel),
    Quantized(QwenImageTransformer2DModelQuantized),
}

impl TransformerVariant {
    /// Unified forward pass.
    pub fn forward(
        &self,
        img: &Tensor,
        txt: &Tensor,
        timestep: &Tensor,
        img_shapes: &[(usize, usize, usize)],
    ) -> candle::Result<Tensor> {
        match self {
            Self::FP16(model) => model.forward(img, txt, timestep, img_shapes),
            Self::Quantized(model) => model.forward(img, txt, timestep, img_shapes),
        }
    }

    /// Returns true if this is a quantized model.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized(_))
    }
}

/// Load the transformer model, choosing FP16 or quantized based on provided paths.
///
/// If `gguf_path` is provided, loads a quantized GGUF model.
/// Otherwise, loads the FP16 safetensors model.
pub fn load_transformer_variant(
    transformer_path: Option<&str>,
    gguf_path: Option<&str>,
    model_id: &str,
    config: &TransformerConfig,
    api: &hf_hub::api::sync::Api,
    device: &Device,
    dtype: DType,
) -> Result<TransformerVariant> {
    if let Some(gguf_value) = gguf_path {
        // Resolve GGUF path (handles "auto", local paths, and HF paths)
        let resolved_path = resolve_gguf_path(gguf_value, DEFAULT_GGUF_TRANSFORMER, api)?;
        println!("Loading quantized transformer from {:?}...", resolved_path);
        let model = load_transformer_quantized(resolved_path.to_str().unwrap(), device, dtype)?;
        Ok(TransformerVariant::Quantized(model))
    } else {
        let model = load_transformer(transformer_path, model_id, config, api, device, dtype)?;
        Ok(TransformerVariant::FP16(model))
    }
}

// ============================================================================
// TextEncoderVariant - Unified interface for FP16 and quantized text encoders
// ============================================================================

use candle_transformers::models::quantized_qwen2_5_vl::{
    load_vision_from_mmproj, ModelWeights as QuantizedTextModel,
};
use candle_transformers::models::qwen2_5_vl::Qwen25VLVisionModel;

/// Unified text encoder variant that abstracts over FP16 and quantized models.
pub enum TextEncoderVariant {
    FP16(Qwen25VLTextModel),
    Quantized(QuantizedTextModel),
}

impl TextEncoderVariant {
    /// Forward pass for text-only input, returning hidden states.
    pub fn forward_text_only(
        &mut self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> candle::Result<Tensor> {
        match self {
            Self::FP16(model) => model.forward_text_only(input_ids, attention_mask),
            Self::Quantized(model) => model.forward_text_only(input_ids, attention_mask),
        }
    }

    /// Forward pass with vision embeddings, returning hidden states.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_vision(
        &mut self,
        input_ids: &Tensor,
        vision_embeds: &Tensor,
        image_grid_thw: &Tensor,
        attention_mask: Option<&Tensor>,
        spatial_merge_size: usize,
        image_token_id: u32,
    ) -> candle::Result<Tensor> {
        match self {
            Self::FP16(model) => model.forward_with_vision(
                input_ids,
                vision_embeds,
                image_grid_thw,
                attention_mask,
                spatial_merge_size,
                image_token_id,
            ),
            Self::Quantized(model) => model.forward_with_vision(
                input_ids,
                vision_embeds,
                image_grid_thw,
                attention_mask,
                spatial_merge_size,
                image_token_id,
            ),
        }
    }

    /// Returns true if this is a quantized model.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized(_))
    }
}

/// Load the text encoder, choosing FP16 or quantized based on provided paths.
///
/// If `gguf_path` is provided (and not None), loads a quantized GGUF model.
/// Otherwise, loads the FP16 safetensors model.
pub fn load_text_encoder_variant(
    text_encoder_path: Option<&str>,
    gguf_path: Option<&str>,
    api: &hf_hub::api::sync::Api,
    device: &Device,
) -> Result<TextEncoderVariant> {
    if let Some(gguf_value) = gguf_path {
        // Resolve GGUF path (handles "auto", local paths, and HF paths)
        let resolved_path = resolve_gguf_path(gguf_value, DEFAULT_GGUF_TEXT_ENCODER, api)?;
        println!("Loading quantized text encoder from {:?}...", resolved_path);

        let mut file = std::fs::File::open(&resolved_path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let model = QuantizedTextModel::from_gguf(content, &mut file, device)?;
        println!("  Quantized text encoder loaded");
        Ok(TextEncoderVariant::Quantized(model))
    } else {
        let model = load_text_encoder(text_encoder_path, api, device)?;
        Ok(TextEncoderVariant::FP16(model))
    }
}

/// Get the default mmproj GGUF path for the given dtype.
///
/// Matches the mmproj precision to the working dtype for optimal compatibility.
fn default_mmproj_for_dtype(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => DEFAULT_GGUF_VISION_ENCODER_F32,
        DType::F16 => DEFAULT_GGUF_VISION_ENCODER_F16,
        DType::BF16 => DEFAULT_GGUF_VISION_ENCODER_BF16,
        // For other dtypes (quantized, etc.), default to F16 as a good balance
        _ => DEFAULT_GGUF_VISION_ENCODER_F16,
    }
}

/// Load the vision encoder from mmproj GGUF or safetensors.
///
/// If `gguf_path` is provided, loads from mmproj GGUF file.
/// When `gguf_path` is "auto", selects the mmproj precision to match the working dtype.
/// Otherwise, loads from safetensors.
pub fn load_vision_encoder_variant(
    vision_encoder_path: Option<&str>,
    gguf_path: Option<&str>,
    api: &hf_hub::api::sync::Api,
    device: &Device,
    dtype: DType,
) -> Result<Qwen25VLVisionModel> {
    if let Some(gguf_value) = gguf_path {
        // Select default mmproj based on dtype
        let default_mmproj = default_mmproj_for_dtype(dtype);
        // Resolve GGUF path (handles "auto", local paths, and HF paths)
        let resolved_path = resolve_gguf_path(gguf_value, default_mmproj, api)?;
        println!(
            "Loading vision encoder from mmproj GGUF: {:?}...",
            resolved_path
        );

        let mut file = std::fs::File::open(&resolved_path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let (vision_model, _config) = load_vision_from_mmproj(&content, &mut file, device, dtype)?;
        println!("  Vision encoder loaded from mmproj");
        Ok(vision_model)
    } else {
        // Load from safetensors
        load_vision_encoder(vision_encoder_path, api, device, dtype)
    }
}

/// Load vision encoder from safetensors.
pub fn load_vision_encoder(
    vision_encoder_path: Option<&str>,
    api: &hf_hub::api::sync::Api,
    device: &Device,
    dtype: DType,
) -> Result<Qwen25VLVisionModel> {
    use candle_transformers::models::qwen2_5_vl::VisionConfig;

    let vision_config = VisionConfig::default();
    let model_files = match vision_encoder_path {
        Some(path) => {
            candle_examples::hub_load_local_safetensors(path, "model.safetensors.index.json")?
        }
        None => {
            let repo = api.repo(hf_hub::Repo::model(DEFAULT_TEXT_ENCODER_ID.to_string()));
            candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?
        }
    };

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, dtype, device)? };
    let vb_vision = vb.pp("visual");
    Ok(Qwen25VLVisionModel::new(&vision_config, vb_vision)?)
}

// ============================================================================
// Prompt encoding utilities
// ============================================================================

/// Encode a text-only prompt using the specified mode (variant-aware version).
///
/// This handles the standard text-only encoding flow:
/// 1. Apply the prompt template for the given mode
/// 2. Tokenize
/// 3. Run through text encoder (FP16 or quantized)
/// 4. Drop the system prefix tokens
/// 5. Convert embeddings to target dtype (for downstream BF16 transformer)
///
/// Returns (embeddings, attention_mask).
pub fn encode_text_prompt_variant(
    tokenizer: &Tokenizer,
    text_model: &mut TextEncoderVariant,
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

    let hidden_states = text_model.forward_text_only(&input_ids, Some(&attention_mask_tensor))?;

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

    // Convert embeddings to target dtype (typically BF16) for the downstream transformer
    let embeddings = embeddings.to_dtype(target_dtype)?;

    Ok((embeddings, mask))
}

/// Encode a text-only prompt using the specified mode (FP16-only version).
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

    let hidden_states = text_model.forward_text_only(&input_ids, Some(&attention_mask_tensor))?;

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
pub fn create_scheduler(steps: usize, image_seq_len: usize) -> FlowMatchEulerDiscreteScheduler {
    let config = SchedulerConfig::default();
    let mu = calculate_shift(
        image_seq_len,
        config.base_image_seq_len,
        config.max_image_seq_len,
        config.base_shift,
        config.max_shift,
    );

    let mut scheduler = FlowMatchEulerDiscreteScheduler::new(&config);
    scheduler.set_timesteps(steps, None, Some(mu));
    scheduler
}

/// Output image dimensions and all derived sizes for the generation pipeline.
///
/// This struct encapsulates the relationship between:
/// - User-specified output image size (`image_height`, `image_width`)
/// - VAE latent space size (`latent_height`, `latent_width`) = image / 8
/// - Transformer patch size (`packed_height`, `packed_width`) = latent / 2
/// - Sequence length for transformer (`seq_len`)
#[derive(Debug, Clone, Copy)]
pub struct OutputDims {
    /// Output image height in pixels
    pub image_height: usize,
    /// Output image width in pixels
    pub image_width: usize,
    /// VAE latent height (image_height / 8)
    pub latent_height: usize,
    /// VAE latent width (image_width / 8)
    pub latent_width: usize,
    /// Transformer packed height (latent_height / 2)
    pub packed_height: usize,
    /// Transformer packed width (latent_width / 2)
    pub packed_width: usize,
    /// Transformer image sequence length (packed_height * packed_width)
    pub image_seq_len: usize,
}

impl OutputDims {
    /// Create output dimensions from the desired image size.
    ///
    /// Computes all derived dimensions (latent, packed, sequence length).
    pub fn new(height: usize, width: usize) -> Self {
        let latent_height = height / VAE_SCALE_FACTOR;
        let latent_width = width / VAE_SCALE_FACTOR;
        let packed_height = latent_height / PATCH_SIZE;
        let packed_width = latent_width / PATCH_SIZE;
        let image_seq_len = packed_height * packed_width;

        Self {
            image_height: height,
            image_width: width,
            latent_height,
            latent_width,
            packed_height,
            packed_width,
            image_seq_len,
        }
    }
}


// ============================================================================
// Dimension validation
// ============================================================================

/// Validate that dimensions are divisible by 16.
pub fn validate_dimensions(height: usize, width: usize) -> Result<()> {
    if !height.is_multiple_of(16) || !width.is_multiple_of(16) {
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
    let h = height.div_ceil(16) * 16;
    let w = width.div_ceil(16) * 16;
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
pub fn _calculate_source_image_dims(orig_width: usize, orig_height: usize) -> (usize, usize) {
    const TARGET_AREA: f64 = 1024.0 * 1024.0; // 1M pixels, matching PyTorch

    let ratio = orig_width as f64 / orig_height as f64;
    let width = (TARGET_AREA * ratio).sqrt();
    let height = width / ratio;

    // Round to nearest 32 (matching PyTorch's rounding)
    let width = ((width / 32.0).round() * 32.0) as usize;
    let height = ((height / 32.0).round() * 32.0) as usize;

    (width, height)
}
