//! Qwen2.5-VL: Vision-Language Model for Image Understanding.
//!
//! Qwen2.5-VL is a multimodal model that combines a vision encoder with the Qwen2.5
//! language model for understanding images and answering questions about them.
//!
//! Available model sizes: 3B, 7B, 72B (no 2B variant exists).
//!
//! ```bash
//! # Basic image Q&A (uses 7B model by default)
//! cargo run --example qwen2_5_vl --release -- \
//!     --image photo.jpg \
//!     --prompt "What is in this image?"
//!
//! # With 3B model (smaller, faster)
//! cargo run --example qwen2_5_vl --release -- \
//!     --model "Qwen/Qwen2.5-VL-3B-Instruct" \
//!     --image document.png \
//!     --prompt "Describe this image"
//!
//! # Multi-image with placeholders
//! cargo run --example qwen2_5_vl --release -- \
//!     --image before.jpg --image after.jpg \
//!     --prompt "Compare {image1} and {image2}. What changed?"
//!
//! # Multi-image without placeholders (images placed before prompt)
//! cargo run --example qwen2_5_vl --release -- \
//!     --image page1.png --image page2.png \
//!     --prompt "Compare these two pages"
//!
//! # Video understanding
//! cargo run --example qwen2_5_vl --release --features video -- \
//!     --video clip.mp4 \
//!     --prompt "Describe what happens in this video"
//!
//! # With sampling (temperature, top-p, top-k)
//! cargo run --example qwen2_5_vl --release -- \
//!     --image photo.jpg --prompt "Write a creative story about this image" \
//!     --temperature 0.8 --top-p 0.9
//!
//! # With streaming output (tokens printed as generated)
//! cargo run --example qwen2_5_vl --release -- \
//!     --image photo.jpg --prompt "Describe this image in detail" \
//!     --stream --temperature 0.7
//!
//! # With repeat penalty to reduce repetition
//! cargo run --example qwen2_5_vl --release -- \
//!     --image photo.jpg --prompt "Describe this image" \
//!     --repeat-penalty 1.1 --repeat-last-n 64
//!
//! # With Flash Attention (faster on CUDA, requires flash-attn feature)
//! cargo run --example qwen2_5_vl --release --features flash-attn -- \
//!     --flash-attn --image photo.jpg \
//!     --prompt "Describe this image"
//!
//! # With Sliding Window Attention (for very long sequences)
//! cargo run --example qwen2_5_vl --release -- \
//!     --sliding-window --sliding-window-size 4096 \
//!     --image photo.jpg --prompt "Describe this image"
//! ```

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::qwen2_5_vl::{
    patchify_image, patchify_video, smart_resize, Config, ImageGrid, Qwen25VLModel,
    DEFAULT_MAX_PIXELS, DEFAULT_MERGE_SIZE, DEFAULT_MIN_PIXELS, DEFAULT_PATCH_SIZE,
    DEFAULT_TEMPORAL_PATCH_SIZE, IMAGE_MEAN, IMAGE_STD,
};
use candle_transformers::utils::apply_repeat_penalty;
use clap::Parser;
use std::io::Write;
use tokenizers::Tokenizer;

const DEFAULT_MODEL_ID: &str = "Qwen/Qwen2.5-VL-7B-Instruct";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to image file(s). Can specify multiple times for multi-image.
    #[arg(long, num_args = 1..)]
    image: Vec<String>,

    /// Path to video file (requires ffmpeg)
    #[arg(long)]
    video: Option<String>,

    /// FPS for video frame extraction (default: 2.0)
    #[arg(long, default_value = "2.0")]
    video_fps: f32,

    /// Maximum frames to extract from video (default: 32)
    #[arg(long, default_value = "32")]
    max_frames: usize,

    /// Prompt/question about the image
    #[arg(long, default_value = "Describe this image in detail.")]
    prompt: String,

    /// Model repository or path
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    model_id: String,

    /// Model revision
    #[arg(long, default_value = "main")]
    revision: String,

    /// Run on CPU rather than GPU
    #[arg(long)]
    cpu: bool,

    /// Maximum generation length
    #[arg(long, default_value = "512")]
    max_length: usize,

    /// Use bfloat16 precision
    #[arg(long)]
    bf16: bool,

    /// Enable Flash Attention 2 (requires CUDA and --features flash-attn)
    #[arg(long)]
    flash_attn: bool,

    /// Enable sliding window attention for text layers
    #[arg(long)]
    sliding_window: bool,

    /// Sliding window size (default: 4096)
    #[arg(long, default_value_t = 4096)]
    sliding_window_size: usize,

    /// Layers >= this index use sliding window (default: 0 = all layers when enabled)
    #[arg(long, default_value_t = 0)]
    max_window_layers: usize,

    // === Sampling parameters ===
    /// Sampling temperature (0.0 = greedy/argmax, higher = more random)
    #[arg(long)]
    temperature: Option<f64>,

    /// Top-p (nucleus) sampling threshold (0.0-1.0)
    #[arg(long)]
    top_p: Option<f64>,

    /// Top-k sampling: only consider top k tokens
    #[arg(long)]
    top_k: Option<usize>,

    /// Repetition penalty (1.0 = no penalty, >1.0 = discourage repetition)
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    /// Number of recent tokens to apply repeat penalty to (0 = all generated tokens)
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Random seed for sampling
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Enable streaming output (print tokens as they're generated)
    #[arg(long)]
    stream: bool,
}

/// Load and preprocess image for Qwen2.5-VL.
///
/// Returns pre-patchified pixel values ready for the vision encoder.
fn load_image(path: &str, device: &Device, dtype: DType) -> Result<(Tensor, Tensor)> {
    let img = image::ImageReader::open(path)?
        .decode()
        .map_err(|e| E::msg(format!("Failed to decode image: {}", e)))?;

    let img = img.to_rgb8();
    let (width, height) = (img.width() as usize, img.height() as usize);

    let factor = DEFAULT_PATCH_SIZE * DEFAULT_MERGE_SIZE; // 28
    let (new_height, new_width) =
        smart_resize(height, width, factor, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS);

    let resized = image::imageops::resize(
        &img,
        new_width as u32,
        new_height as u32,
        image::imageops::FilterType::CatmullRom,
    );

    // Normalize using CLIP mean/std, channels-first format (C, H, W)
    let mut normalized = vec![0f32; 3 * new_height * new_width];
    for c in 0..3 {
        for y in 0..new_height {
            for x in 0..new_width {
                let pixel = resized.get_pixel(x as u32, y as u32);
                let idx = c * new_height * new_width + y * new_width + x;
                normalized[idx] = (pixel[c] as f32 / 255.0 - IMAGE_MEAN[c]) / IMAGE_STD[c];
            }
        }
    }

    // Patchify using library function
    let patches = patchify_image(
        &normalized,
        new_height,
        new_width,
        DEFAULT_PATCH_SIZE,
        DEFAULT_TEMPORAL_PATCH_SIZE,
        DEFAULT_MERGE_SIZE,
    );

    let h_patches = new_height / DEFAULT_PATCH_SIZE;
    let w_patches = new_width / DEFAULT_PATCH_SIZE;
    let num_patches = h_patches * w_patches;
    let patch_elements = 3 * DEFAULT_TEMPORAL_PATCH_SIZE * DEFAULT_PATCH_SIZE * DEFAULT_PATCH_SIZE;

    let pixel_values =
        Tensor::from_vec(patches, (num_patches, patch_elements), device)?.to_dtype(dtype)?;

    let grid_thw = Tensor::new(&[[1u32, h_patches as u32, w_patches as u32]], device)?;

    println!(
        "Image: {}x{} -> {}x{} ({} x {} patches = {} total)",
        width, height, new_width, new_height, h_patches, w_patches, num_patches
    );

    Ok((pixel_values, grid_thw))
}

/// Extract frames from video using ffmpeg.
///
/// Returns paths to extracted frame images in a temporary directory.
fn extract_frames_ffmpeg(
    video_path: &str,
    fps: f32,
    max_frames: usize,
) -> Result<(std::path::PathBuf, Vec<std::path::PathBuf>)> {
    use std::process::Command;

    // Create temp directory for frames
    let temp_dir = std::env::temp_dir().join(format!("qwen_vl_frames_{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir)?;

    // Run ffmpeg to extract frames
    let output = Command::new("ffmpeg")
        .args([
            "-i",
            video_path,
            "-vf",
            &format!("fps={}", fps),
            "-frames:v",
            &max_frames.to_string(),
            "-y", // Overwrite output files
            temp_dir.join("frame_%04d.png").to_str().unwrap(),
        ])
        .output()
        .map_err(|e| E::msg(format!("Failed to run ffmpeg: {}. Is ffmpeg installed?", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(E::msg(format!("ffmpeg failed: {}", stderr)));
    }

    // Collect frame paths
    let mut frame_paths: Vec<std::path::PathBuf> = std::fs::read_dir(&temp_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "png"))
        .collect();

    frame_paths.sort();

    if frame_paths.is_empty() {
        return Err(E::msg("No frames extracted from video"));
    }

    Ok((temp_dir, frame_paths))
}

/// Load and preprocess video for Qwen2.5-VL.
///
/// Returns (pixel_values, grid_thw, second_per_grid_t) for video processing.
fn load_video(
    path: &str,
    fps: f32,
    max_frames: usize,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor, f32)> {
    println!("Extracting frames from video at {} fps...", fps);

    let (temp_dir, frame_paths) = extract_frames_ffmpeg(path, fps, max_frames)?;
    let num_frames = frame_paths.len();
    println!("Extracted {} frames", num_frames);

    if num_frames == 0 {
        std::fs::remove_dir_all(&temp_dir).ok();
        return Err(E::msg("No frames extracted from video"));
    }

    let factor = DEFAULT_PATCH_SIZE * DEFAULT_MERGE_SIZE;

    // Load first frame to get dimensions and compute target size
    let first_img = image::ImageReader::open(&frame_paths[0])?
        .decode()
        .map_err(|e| E::msg(format!("Failed to decode frame: {}", e)))?
        .to_rgb8();
    let (width, height) = (first_img.width() as usize, first_img.height() as usize);
    let (new_height, new_width) =
        smart_resize(height, width, factor, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS);

    // Load and normalize all frames
    let mut frames_data: Vec<Vec<f32>> = Vec::with_capacity(num_frames);
    for frame_path in &frame_paths {
        let img = image::ImageReader::open(frame_path)?
            .decode()
            .map_err(|e| E::msg(format!("Failed to decode frame: {}", e)))?
            .to_rgb8();

        let resized = image::imageops::resize(
            &img,
            new_width as u32,
            new_height as u32,
            image::imageops::FilterType::CatmullRom,
        );

        // Normalize using CLIP mean/std, channels-first format (C, H, W)
        let mut normalized = vec![0f32; 3 * new_height * new_width];
        for c in 0..3 {
            for y in 0..new_height {
                for x in 0..new_width {
                    let pixel = resized.get_pixel(x as u32, y as u32);
                    let idx = c * new_height * new_width + y * new_width + x;
                    normalized[idx] = (pixel[c] as f32 / 255.0 - IMAGE_MEAN[c]) / IMAGE_STD[c];
                }
            }
        }
        frames_data.push(normalized);
    }

    // Clean up temp directory
    std::fs::remove_dir_all(&temp_dir).ok();

    // Patchify video frames using library function
    let patches = patchify_video(
        &frames_data,
        new_height,
        new_width,
        DEFAULT_PATCH_SIZE,
        DEFAULT_TEMPORAL_PATCH_SIZE,
        DEFAULT_MERGE_SIZE,
    );

    let h_patches = new_height / DEFAULT_PATCH_SIZE;
    let w_patches = new_width / DEFAULT_PATCH_SIZE;

    // Compute temporal grid (after padding to temporal_patch_size)
    let grid_t = num_frames.div_ceil(DEFAULT_TEMPORAL_PATCH_SIZE);

    let num_patches = grid_t * h_patches * w_patches;
    let patch_elements =
        3 * DEFAULT_TEMPORAL_PATCH_SIZE * DEFAULT_PATCH_SIZE * DEFAULT_PATCH_SIZE;

    let pixel_values =
        Tensor::from_vec(patches, (num_patches, patch_elements), device)?.to_dtype(dtype)?;

    let grid_thw = Tensor::new(&[[grid_t as u32, h_patches as u32, w_patches as u32]], device)?;

    // second_per_grid_t = temporal_patch_size / fps
    let second_per_grid_t = DEFAULT_TEMPORAL_PATCH_SIZE as f32 / fps;

    println!(
        "Video: {} frames -> {}x{} at {} fps ({} x {} x {} patches = {} total, second_per_grid_t={:.3})",
        num_frames, new_width, new_height, fps, grid_t, h_patches, w_patches, num_patches, second_per_grid_t
    );

    Ok((pixel_values, grid_thw, second_per_grid_t))
}

/// Build input tokens for video with Qwen2.5-VL chat template.
fn build_video_input_tokens(
    tokenizer: &Tokenizer,
    prompt: &str,
    num_video_tokens: usize,
    video_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
    device: &Device,
) -> Result<Tensor> {
    let system_msg = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
    let user_prefix = "<|im_start|>user\n";
    let user_suffix = "<|im_end|>\n<|im_start|>assistant\n";

    let system_enc = tokenizer
        .encode(system_msg, false)
        .map_err(|e| E::msg(format!("Tokenization error: {}", e)))?;
    let user_prefix_enc = tokenizer
        .encode(user_prefix, false)
        .map_err(|e| E::msg(format!("Tokenization error: {}", e)))?;
    let prompt_enc = tokenizer
        .encode(prompt, false)
        .map_err(|e| E::msg(format!("Tokenization error: {}", e)))?;
    let user_suffix_enc = tokenizer
        .encode(user_suffix, false)
        .map_err(|e| E::msg(format!("Tokenization error: {}", e)))?;

    let mut input_ids: Vec<u32> = Vec::new();
    input_ids.extend(system_enc.get_ids());
    input_ids.extend(user_prefix_enc.get_ids());
    input_ids.push(vision_start_token_id);
    input_ids.extend(vec![video_token_id; num_video_tokens]);
    input_ids.push(vision_end_token_id);
    input_ids.extend(prompt_enc.get_ids());
    input_ids.extend(user_suffix_enc.get_ids());

    let tensor = Tensor::new(input_ids.as_slice(), device)?.unsqueeze(0)?;
    Ok(tensor)
}

/// Part of a prompt - either text or an image placeholder.
enum PromptPart {
    Text(String),
    Image(usize), // Index into image_grids (0-indexed)
}

/// Parse prompt for {image1}, {image2}, etc. placeholders.
///
/// Returns a list of parts in order. If no placeholders found, returns None
/// so caller can fall back to all-images-first behavior.
fn parse_prompt_placeholders(prompt: &str) -> Option<Vec<PromptPart>> {
    // Simple parser for {imageN} patterns (1-indexed in prompt, converted to 0-indexed)
    let mut parts = Vec::new();
    let mut last_end = 0;
    let mut found_any = false;

    let chars: Vec<char> = prompt.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '{' {
            // Look for "image" followed by digits and "}"
            let start = i;
            i += 1;

            // Check for "image"
            let image_str = "image";
            let mut matched = true;
            for (j, c) in image_str.chars().enumerate() {
                if i + j >= chars.len() || chars[i + j] != c {
                    matched = false;
                    break;
                }
            }

            if matched {
                i += image_str.len();

                // Parse digits
                let digit_start = i;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }

                if i > digit_start && i < chars.len() && chars[i] == '}' {
                    // Valid placeholder found
                    let num_str: String = chars[digit_start..i].iter().collect();
                    if let Ok(num) = num_str.parse::<usize>() {
                        found_any = true;

                        // Add text before this placeholder
                        if start > last_end {
                            let text: String = chars[last_end..start].iter().collect();
                            parts.push(PromptPart::Text(text));
                        }

                        // Add image placeholder (convert 1-indexed to 0-indexed)
                        parts.push(PromptPart::Image(num.saturating_sub(1)));

                        last_end = i + 1; // Skip the closing '}'
                    }
                    i += 1;
                    continue;
                }
            }
            // Not a valid placeholder, continue from after '{'
            i = start + 1;
        } else {
            i += 1;
        }
    }

    if !found_any {
        return None;
    }

    // Add remaining text
    if last_end < chars.len() {
        let text: String = chars[last_end..].iter().collect();
        parts.push(PromptPart::Text(text));
    }

    Some(parts)
}

/// Build input tokens with Qwen2.5-VL chat template.
///
/// Supports {image1}, {image2}, etc. placeholders in the prompt for flexible image positioning.
/// If no placeholders found, all images are placed before the prompt text.
///
/// Format:
/// <|im_start|>system
/// You are a helpful assistant.<|im_end|>
/// <|im_start|>user
/// {images and text interleaved}<|im_end|>
/// <|im_start|>assistant
fn build_input_tokens(
    tokenizer: &Tokenizer,
    prompt: &str,
    image_grids: &[ImageGrid],
    image_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
    device: &Device,
) -> Result<Tensor> {
    let system_msg = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
    let user_prefix = "<|im_start|>user\n";
    let user_suffix = "<|im_end|>\n<|im_start|>assistant\n";

    let system_enc = tokenizer
        .encode(system_msg, false)
        .map_err(|e| E::msg(format!("Tokenization error: {}", e)))?;
    let user_prefix_enc = tokenizer
        .encode(user_prefix, false)
        .map_err(|e| E::msg(format!("Tokenization error: {}", e)))?;
    let user_suffix_enc = tokenizer
        .encode(user_suffix, false)
        .map_err(|e| E::msg(format!("Tokenization error: {}", e)))?;

    let mut input_ids: Vec<u32> = Vec::new();
    input_ids.extend(system_enc.get_ids());
    input_ids.extend(user_prefix_enc.get_ids());

    // Helper to add image tokens for a given image index
    let add_image_tokens = |ids: &mut Vec<u32>, img_idx: usize| {
        if img_idx < image_grids.len() {
            let grid = &image_grids[img_idx];
            let num_tokens = grid.grid_h * grid.grid_w;
            ids.push(vision_start_token_id);
            ids.extend(vec![image_token_id; num_tokens]);
            ids.push(vision_end_token_id);
        }
    };

    // Check for placeholders in prompt
    if let Some(parts) = parse_prompt_placeholders(prompt) {
        // Build with placeholders
        for part in parts {
            match part {
                PromptPart::Text(text) => {
                    if !text.is_empty() {
                        let enc = tokenizer
                            .encode(text.as_str(), false)
                            .map_err(|e| E::msg(format!("Tokenization error: {}", e)))?;
                        input_ids.extend(enc.get_ids());
                    }
                }
                PromptPart::Image(idx) => {
                    add_image_tokens(&mut input_ids, idx);
                }
            }
        }
    } else {
        // No placeholders - put all images first, then the prompt
        for img_idx in 0..image_grids.len() {
            add_image_tokens(&mut input_ids, img_idx);
        }
        let prompt_enc = tokenizer
            .encode(prompt, false)
            .map_err(|e| E::msg(format!("Tokenization error: {}", e)))?;
        input_ids.extend(prompt_enc.get_ids());
    }

    input_ids.extend(user_suffix_enc.get_ids());

    let tensor = Tensor::new(input_ids.as_slice(), device)?.unsqueeze(0)?;
    Ok(tensor)
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Validate input: must have either image(s) or video
    if args.image.is_empty() && args.video.is_none() {
        return Err(E::msg("At least one --image or --video must be specified"));
    }
    if !args.image.is_empty() && args.video.is_some() {
        return Err(E::msg("Cannot specify both --image and --video"));
    }

    let device = candle_examples::device(args.cpu)?;
    let dtype = if args.bf16 { DType::BF16 } else { DType::F32 };
    println!("Using device: {:?}, dtype: {:?}", device, dtype);

    // Load model from HuggingFace
    println!("Loading model from {}...", args.model_id);
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        args.model_id.clone(),
        hf_hub::RepoType::Model,
        args.revision.clone(),
    ));

    // Load config
    let config_file = repo.get("config.json")?;
    let mut config: Config = serde_json::from_str(&std::fs::read_to_string(&config_file)?)?;

    // Apply CLI overrides for attention optimization
    config.use_flash_attn = args.flash_attn;
    if args.sliding_window {
        config.use_sliding_window = true;
        config.sliding_window = args.sliding_window_size;
        config.max_window_layers = args.max_window_layers;
    }

    // Validate flash attention settings
    if config.use_flash_attn {
        if args.cpu {
            eprintln!("Warning: --flash-attn has no effect on CPU, using standard attention");
            config.use_flash_attn = false;
        }
        #[cfg(not(feature = "flash-attn"))]
        {
            eprintln!("Warning: flash-attn feature not enabled, compile with --features flash-attn");
            config.use_flash_attn = false;
        }
    }

    println!(
        "Vision: {}L {}H, Text: {}L {}H (GQA: {}KV)",
        config.vision_config.depth,
        config.vision_config.num_heads,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
    );
    if config.use_flash_attn {
        println!("Using Flash Attention 2");
    }
    if config.use_sliding_window {
        println!(
            "Using Sliding Window Attention (size: {}, layers >= {})",
            config.sliding_window, config.max_window_layers
        );
    }

    // Load tokenizer
    let tokenizer_file = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_file).map_err(E::msg)?;

    // Load model weights (handle sharded safetensors)
    let model_files: Vec<_> = [
        "model.safetensors",
        "model-00001-of-00005.safetensors",
        "model-00001-of-00004.safetensors",
        "model-00001-of-00003.safetensors",
        "model-00001-of-00002.safetensors",
    ]
    .iter()
    .filter_map(|f| repo.get(f).ok())
    .collect();

    if model_files.is_empty() {
        return Err(E::msg("Could not find model weights"));
    }

    // If first file is sharded, get all shards
    let model_files = if model_files[0]
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .contains("-00001-of-")
    {
        // Count how many shards
        let name = model_files[0]
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let parts: Vec<&str> = name.split("-of-").collect();
        if parts.len() == 2 {
            let total: usize = parts[1]
                .trim_end_matches(".safetensors")
                .parse()
                .unwrap_or(1);
            let mut files = Vec::new();
            for i in 1..=total {
                let shard_name = format!(
                    "model-{:05}-of-{:05}.safetensors",
                    i, total
                );
                files.push(repo.get(&shard_name)?);
            }
            files
        } else {
            model_files
        }
    } else {
        model_files
    };

    println!("Loading {} weight file(s)...", model_files.len());
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, dtype, &device)? };

    let mut model = Qwen25VLModel::new(&config, vb)?;
    println!("Model loaded successfully!");

    // Get EOS token
    let eos_token_id = tokenizer
        .token_to_id("<|im_end|>")
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        .unwrap_or(151643);

    let spatial_merge = config.vision_config.spatial_merge_size;

    // Set up sampling strategy based on CLI args
    let sampling = match (args.temperature, args.top_k, args.top_p) {
        // No temperature or very low temperature = greedy
        (None, None, None) => Sampling::ArgMax,
        (Some(t), _, _) if t < 1e-7 => Sampling::ArgMax,
        // All three specified: top-k then top-p
        (Some(temp), Some(k), Some(p)) => {
            Sampling::TopKThenTopP { k, p, temperature: temp }
        }
        // Temperature + top-k
        (Some(temp), Some(k), None) => Sampling::TopK { k, temperature: temp },
        // Temperature + top-p (nucleus)
        (Some(temp), None, Some(p)) => Sampling::TopP { p, temperature: temp },
        // Temperature only (sample from full distribution)
        (Some(temp), None, None) => Sampling::All { temperature: temp },
        // Top-k without explicit temperature (use default 1.0)
        (None, Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature: 1.0 },
        (None, Some(k), None) => Sampling::TopK { k, temperature: 1.0 },
        // Top-p without explicit temperature (use default 1.0)
        (None, None, Some(p)) => Sampling::TopP { p, temperature: 1.0 },
    };

    let mut logits_processor = LogitsProcessor::from_sampling(args.seed, sampling.clone());
    let repeat_penalty = args.repeat_penalty;
    let repeat_last_n = args.repeat_last_n;

    // Print sampling configuration
    match &sampling {
        Sampling::ArgMax => println!("Sampling: greedy (argmax)"),
        Sampling::All { temperature } => println!("Sampling: temperature={:.2}", temperature),
        Sampling::TopK { k, temperature } => {
            println!("Sampling: top-k={}, temperature={:.2}", k, temperature)
        }
        Sampling::TopP { p, temperature } => {
            println!("Sampling: top-p={:.2}, temperature={:.2}", p, temperature)
        }
        Sampling::TopKThenTopP { k, p, temperature } => {
            println!(
                "Sampling: top-k={}, top-p={:.2}, temperature={:.2}",
                k, p, temperature
            )
        }
        Sampling::GumbelSoftmax { temperature } => {
            println!("Sampling: gumbel-softmax, temperature={:.2}", temperature)
        }
    }
    if repeat_penalty != 1.0 {
        println!(
            "Repeat penalty: {:.2} (last {} tokens)",
            repeat_penalty, repeat_last_n
        );
    }

    // Generate based on input type (image or video)
    let start = std::time::Instant::now();
    let generated_tokens = if let Some(video_path) = &args.video {
        // Video mode
        println!("Loading: {}", video_path);
        let (pixel_values, grid_thw, second_per_grid_t) =
            load_video(video_path, args.video_fps, args.max_frames, &device, dtype)?;

        // Calculate video tokens
        let grid_vec = grid_thw.to_vec2::<u32>()?;
        let g = &grid_vec[0];
        let grid_t = g[0] as usize;
        let h_patches = g[1] as usize;
        let w_patches = g[2] as usize;
        let num_video_tokens = grid_t * (h_patches / spatial_merge) * (w_patches / spatial_merge);

        println!(
            "Total video tokens: {} (after {}x{} merge)",
            num_video_tokens, spatial_merge, spatial_merge
        );

        // Build input tokens for video
        let input_ids = build_video_input_tokens(
            &tokenizer,
            &args.prompt,
            num_video_tokens,
            config.video_token_id,
            config.vision_start_token_id,
            config.vision_end_token_id,
            &device,
        )?;
        println!("Input sequence length: {}", input_ids.dim(1)?);

        // Generate using video forward with sampling
        if args.stream {
            println!("\nGenerating (streaming, max {} tokens)...\n", args.max_length);
            model.generate_video_streaming(
                &input_ids,
                &pixel_values,
                &grid_thw,
                second_per_grid_t,
                args.max_length,
                eos_token_id,
                |logits, generated| {
                    // Apply repeat penalty if configured
                    let logits = if repeat_penalty != 1.0 && !generated.is_empty() {
                        let start = generated.len().saturating_sub(repeat_last_n);
                        apply_repeat_penalty(logits, repeat_penalty, &generated[start..])?
                    } else {
                        logits.clone()
                    };
                    logits_processor.sample(&logits)
                },
                |token, _is_eos| {
                    // Stream tokens to stdout
                    if let Ok(text) = tokenizer.decode(&[token], true) {
                        print!("{}", text);
                        std::io::stdout().flush().ok();
                    }
                },
            )?
        } else {
            println!("\nGenerating (max {} tokens)...", args.max_length);
            model.generate_video_with_sampler(
                &input_ids,
                &pixel_values,
                &grid_thw,
                second_per_grid_t,
                args.max_length,
                eos_token_id,
                |logits, generated| {
                    // Apply repeat penalty if configured
                    let logits = if repeat_penalty != 1.0 && !generated.is_empty() {
                        let start = generated.len().saturating_sub(repeat_last_n);
                        apply_repeat_penalty(logits, repeat_penalty, &generated[start..])?
                    } else {
                        logits.clone()
                    };
                    logits_processor.sample(&logits)
                },
            )?
        }
    } else {
        // Image mode
        let mut all_pixel_values = Vec::new();
        let mut all_grid_thw = Vec::new();
        let mut image_grids: Vec<ImageGrid> = Vec::new();

        for image_path in &args.image {
            println!("Loading: {}", image_path);
            let (pixel_values, grid_thw) = load_image(image_path, &device, dtype)?;

            let grid_vec = grid_thw.to_vec2::<u32>()?;
            let g = &grid_vec[0];
            let h_patches = g[1] as usize;
            let w_patches = g[2] as usize;

            image_grids.push(ImageGrid {
                grid_h: h_patches / spatial_merge,
                grid_w: w_patches / spatial_merge,
            });

            all_pixel_values.push(pixel_values);
            all_grid_thw.push(grid_thw);
        }

        // Concatenate pixel values and grid_thw
        let pixel_values = Tensor::cat(&all_pixel_values, 0)?;
        let grid_thw = Tensor::cat(&all_grid_thw, 0)?;

        // Calculate total image tokens for display
        let num_image_tokens: usize = image_grids.iter().map(|g| g.grid_h * g.grid_w).sum();

        println!(
            "Total image tokens: {} (after {}x{} merge)",
            num_image_tokens, spatial_merge, spatial_merge
        );

        // Build input tokens (supports {image1}, {image2} placeholders in prompt)
        let input_ids = build_input_tokens(
            &tokenizer,
            &args.prompt,
            &image_grids,
            config.image_token_id,
            config.vision_start_token_id,
            config.vision_end_token_id,
            &device,
        )?;
        println!("Input sequence length: {}", input_ids.dim(1)?);

        // Generate using image forward with sampling
        if args.stream {
            println!("\nGenerating (streaming, max {} tokens)...\n", args.max_length);
            model.generate_streaming(
                &input_ids,
                &pixel_values,
                &grid_thw,
                args.max_length,
                eos_token_id,
                |logits, generated| {
                    // Apply repeat penalty if configured
                    let logits = if repeat_penalty != 1.0 && !generated.is_empty() {
                        let start = generated.len().saturating_sub(repeat_last_n);
                        apply_repeat_penalty(logits, repeat_penalty, &generated[start..])?
                    } else {
                        logits.clone()
                    };
                    logits_processor.sample(&logits)
                },
                |token, _is_eos| {
                    // Stream tokens to stdout
                    if let Ok(text) = tokenizer.decode(&[token], true) {
                        print!("{}", text);
                        std::io::stdout().flush().ok();
                    }
                },
            )?
        } else {
            println!("\nGenerating (max {} tokens)...", args.max_length);
            model.generate_with_sampler(
                &input_ids,
                &pixel_values,
                &grid_thw,
                args.max_length,
                eos_token_id,
                |logits, generated| {
                    // Apply repeat penalty if configured
                    let logits = if repeat_penalty != 1.0 && !generated.is_empty() {
                        let start = generated.len().saturating_sub(repeat_last_n);
                        apply_repeat_penalty(logits, repeat_penalty, &generated[start..])?
                    } else {
                        logits.clone()
                    };
                    logits_processor.sample(&logits)
                },
            )?
        }
    };

    let elapsed = start.elapsed();

    // Display output (skip if streaming, since tokens were already printed)
    if args.stream {
        // Just print stats after streamed output
        println!("\n\n{:=<60}", "");
        println!(
            "Generated {} tokens in {:.2}s ({:.1} tokens/sec)",
            generated_tokens.len(),
            elapsed.as_secs_f32(),
            generated_tokens.len() as f32 / elapsed.as_secs_f32()
        );
    } else {
        // Decode and display full output
        let output_text = tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| E::msg(format!("Decoding error: {}", e)))?;

        println!("\n{:=<60}", "");
        println!("Prompt: {}", args.prompt);
        println!("{:=<60}", "");
        println!("{}", output_text.trim());
        println!("{:=<60}", "");
        println!(
            "Generated {} tokens in {:.2}s ({:.1} tokens/sec)",
            generated_tokens.len(),
            elapsed.as_secs_f32(),
            generated_tokens.len() as f32 / elapsed.as_secs_f32()
        );
    }

    Ok(())
}
