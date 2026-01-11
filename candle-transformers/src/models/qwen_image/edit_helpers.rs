//! Helper utilities for Qwen-Image Edit and Layered pipelines.
//!
//! This module provides prompt templates, latent packing operations, and
//! dimension calculations needed for vision-language editing and layer decomposition.

use candle::{IndexOp, Result, Tensor};

// ============================================================================
// Prompt Modes
// ============================================================================

/// Prompt encoding mode that bundles template and drop_tokens together.
///
/// This ensures template/drop_tokens are always correctly paired and makes
/// the API cleaner by avoiding separate constant arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptMode {
    /// Text-only pipelines (text-to-image, inpaint, controlnet, img2img).
    /// No vision tokens, drops 34 tokens from system prefix.
    TextOnly,
    /// Edit mode with vision tokens for image editing.
    /// Drops 64 tokens from system prefix.
    Edit,
    /// Layered mode for layer decomposition (same as TextOnly).
    Layered,
}

impl PromptMode {
    /// Get the prompt template for this mode.
    /// The `{}` placeholder is replaced with the user's prompt text.
    pub fn template(&self) -> &'static str {
        match self {
            Self::TextOnly | Self::Layered => TEXT_ONLY_PROMPT_TEMPLATE,
            Self::Edit => EDIT_PROMPT_TEMPLATE,
        }
    }

    /// Get the number of tokens to drop from the start of embeddings.
    /// This removes the system instruction prefix.
    pub fn drop_tokens(&self) -> usize {
        match self {
            Self::TextOnly | Self::Layered => 34,
            Self::Edit => 64,
        }
    }
}

// ============================================================================
// Prompt Templates (kept for direct access)
// ============================================================================

/// Prompt template for text-only pipelines (text-to-image, inpaint, controlnet, img2img).
///
/// This template is used for pipelines that don't include vision tokens.
/// Both positive AND negative prompts should use this same template.
/// The `{}` placeholder is replaced with the user's prompt text.
pub const TEXT_ONLY_PROMPT_TEMPLATE: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";

/// Number of tokens to drop from the start of text-only pipeline embeddings.
///
/// This removes the system instruction prefix, leaving only the prompt context.
pub const TEXT_ONLY_DROP_TOKENS: usize = 34;

/// Prompt template for Edit mode (vision pipeline with image tokens).
///
/// This template instructs the model to analyze the input image and apply
/// the user's editing instruction while maintaining consistency with the original.
pub const EDIT_PROMPT_TEMPLATE: &str = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n";

/// Number of tokens to drop from the start of Edit mode embeddings.
///
/// This removes the system instruction prefix, leaving only the image+prompt context.
pub const EDIT_DROP_TOKENS: usize = 64;

/// Prompt template for Layered mode (text-only, for diffusion conditioning).
///
/// This template is used when encoding the caption for layer decomposition.
/// Unlike Edit mode, it doesn't include vision tokens as the caption is either
/// provided by the user or auto-generated.
pub const LAYERED_PROMPT_TEMPLATE: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";

/// Number of tokens to drop from the start of Layered mode embeddings.
pub const LAYERED_DROP_TOKENS: usize = 34;

// ============================================================================
// Auto-Caption Prompts
// ============================================================================

/// English prompt for auto-captioning input images.
///
/// Used when no user prompt is provided to the Layered pipeline.
pub const CAPTION_PROMPT_EN: &str = "<|im_start|>system\n\
    You are a helpful assistant.<|im_end|>\n\
    <|im_start|>user\n\
    # Image Annotator\n\
    You are a professional image annotator. Please write an image caption based on the input image:\n\
    1. Write the caption using natural, descriptive language without structured formats or rich text.\n\
    2. Enrich caption details by including:\n\
       - Object attributes, such as quantity, color, shape, size, material, state, position, actions, and so on\n\
       - Vision Relations between objects, such as spatial relations, functional relations, possessive relations, \
         attachment relations, action relations, comparative relations, causal relations, and so on\n\
       - Environmental details, such as weather, lighting, colors, textures, atmosphere, and so on\n\
       - Identify the text clearly visible in the image, without translation or explanation, \
         and highlight it in the caption with quotation marks\n\
    3. Maintain authenticity and accuracy:\n\
       - Avoid generalizations\n\
       - Describe all visible information in the image, while do not add information not explicitly shown in the image\n\
    <|vision_start|><|image_pad|><|vision_end|><|im_end|>\n\
    <|im_start|>assistant\n";

/// Chinese prompt for auto-captioning input images.
///
/// Used when `use_en_prompt=false` in the Layered pipeline.
pub const CAPTION_PROMPT_CN: &str = "<|im_start|>system\n\
    You are a helpful assistant.<|im_end|>\n\
    <|im_start|>user\n\
    # 图像标注器\n\
    你是一个专业的图像标注器。请基于输入图像，撰写图注:\n\
    1. 使用自然、描述性的语言撰写图注，不要使用结构化形式或富文本形式。\n\
    2. 通过加入以下内容，丰富图注细节：\n\
       - 对象的属性：如数量、颜色、形状、大小、位置、材质、状态、动作等\n\
       - 对象间的视觉关系：如空间关系、功能关系、动作关系、从属关系、比较关系、因果关系等\n\
       - 环境细节：例如天气、光照、颜色、纹理、气氛等\n\
       - 文字内容：识别图像中清晰可见的文字，不做翻译和解释，用引号在图注中强调\n\
    3. 保持真实性与准确性：\n\
       - 不要使用笼统的描述\n\
       - 描述图像中所有可见的信息，但不要加入没有在图像中出现的内容\n\
    <|vision_start|><|image_pad|><|vision_end|><|im_end|>\n\
    <|im_start|>assistant\n";

// ============================================================================
// Latent Packing for Layered Mode
// ============================================================================

/// Pack layered latents for the transformer.
///
/// Converts VAE latents with a layer dimension into the packed format expected
/// by the Qwen-Image transformer. Each layer's latents are packed with 2x2 spatial
/// patches flattened into the channel dimension.
///
/// # Arguments
/// * `latents` - Tensor of shape [batch, layers+1, channels, height, width]
/// * `height` - Latent height (before packing)
/// * `width` - Latent width (before packing)
/// * `layers` - Number of output layers (not including the combined first frame)
///
/// # Returns
/// Packed tensor of shape [batch, (layers+1) * (H/2) * (W/2), channels * 4]
pub fn pack_layered_latents(
    latents: &Tensor,
    height: usize,
    width: usize,
    layers: usize,
) -> Result<Tensor> {
    let (batch, num_frames, channels, h, w) = latents.dims5()?;
    assert_eq!(h, height, "Height mismatch");
    assert_eq!(w, width, "Width mismatch");
    assert_eq!(num_frames, layers + 1, "Frame count mismatch");

    // Process each frame separately to avoid 7D tensor operations
    let half_h = height / 2;
    let half_w = width / 2;
    let mut packed_frames = Vec::with_capacity(num_frames);

    for frame_idx in 0..num_frames {
        // Extract single frame: [batch, channels, height, width]
        let frame = latents.i((.., frame_idx, .., .., ..))?;

        // Reshape to [batch, channels, height/2, 2, width/2, 2]
        let frame = frame.reshape((batch, channels, half_h, 2, half_w, 2))?;

        // Permute to [batch, height/2, width/2, channels, 2, 2]
        let frame = frame.permute([0, 2, 4, 1, 3, 5])?;

        // Reshape to [batch, (height/2) * (width/2), channels * 4]
        let frame = frame.reshape((batch, half_h * half_w, channels * 4))?;
        packed_frames.push(frame);
    }

    // Concatenate all frames along sequence dimension
    Tensor::cat(&packed_frames, 1)
}

/// Unpack layered latents from transformer output.
///
/// Reverses the packing operation to recover the original latent shape with
/// the layer dimension.
///
/// # Arguments
/// * `latents` - Packed tensor of shape [batch, (layers+1) * (H/2) * (W/2), channels * 4]
/// * `height` - Target latent height
/// * `width` - Target latent width
/// * `layers` - Number of output layers (not including the combined first frame)
/// * `out_channels` - Number of output channels (typically 16)
///
/// # Returns
/// Unpacked tensor of shape [batch, channels, layers+1, height, width]
pub fn unpack_layered_latents(
    latents: &Tensor,
    height: usize,
    width: usize,
    layers: usize,
    out_channels: usize,
) -> Result<Tensor> {
    let (batch, _num_patches, packed_channels) = latents.dims3()?;
    let channels = packed_channels / 4;
    assert_eq!(channels, out_channels, "Channel mismatch");

    let num_frames = layers + 1;
    let half_h = height / 2;
    let half_w = width / 2;
    let tokens_per_frame = half_h * half_w;

    // Process each frame separately to avoid 7D tensor operations
    let mut unpacked_frames = Vec::with_capacity(num_frames);

    for frame_idx in 0..num_frames {
        // Extract single frame: [batch, (height/2)*(width/2), channels*4]
        let start = frame_idx * tokens_per_frame;
        let frame = latents.narrow(1, start, tokens_per_frame)?;

        // Reshape to [batch, height/2, width/2, channels, 2, 2]
        let frame = frame.reshape((batch, half_h, half_w, channels, 2, 2))?;

        // Permute to [batch, channels, height/2, 2, width/2, 2]
        let frame = frame.permute([0, 3, 1, 4, 2, 5])?;

        // Reshape to [batch, channels, height, width]
        let frame = frame.reshape((batch, channels, height, width))?;

        // Add frame dimension: [batch, channels, 1, height, width]
        let frame = frame.unsqueeze(2)?;
        unpacked_frames.push(frame);
    }

    // Concatenate all frames along frame dimension: [batch, channels, layers+1, height, width]
    Tensor::cat(&unpacked_frames, 2)
}

// ============================================================================
// Dimension Calculations
// ============================================================================

/// Calculate output dimensions that preserve aspect ratio.
///
/// Given a target area (e.g., 1024*1024 = 1M pixels) and aspect ratio,
/// computes width and height that:
/// 1. Preserve the aspect ratio
/// 2. Have approximately the target area
/// 3. Are divisible by 32 (required for the VAE)
///
/// # Arguments
/// * `target_area` - Target pixel count (e.g., 1024 * 1024)
/// * `aspect_ratio` - Width / Height ratio
///
/// # Returns
/// (width, height) tuple
pub fn calculate_dimensions(target_area: usize, aspect_ratio: f64) -> (usize, usize) {
    // width = sqrt(area * ratio), height = width / ratio
    let width = (target_area as f64 * aspect_ratio).sqrt();
    let height = width / aspect_ratio;

    // Round to nearest multiple of 32
    let width = ((width / 32.0).round() * 32.0) as usize;
    let height = ((height / 32.0).round() * 32.0) as usize;

    (width, height)
}

/// Calculate output dimensions with specified resolution bucket.
///
/// For Layered mode which uses specific resolution buckets (640 or 1024).
///
/// # Arguments
/// * `resolution` - Target resolution (640 or 1024)
/// * `aspect_ratio` - Width / Height ratio from input image
///
/// # Returns
/// (width, height) tuple
pub fn calculate_dimensions_with_resolution(resolution: usize, aspect_ratio: f64) -> (usize, usize) {
    calculate_dimensions(resolution * resolution, aspect_ratio)
}

/// Extract hidden states with masking and offset.
///
/// Extracts valid hidden states based on attention mask, drops initial tokens,
/// and pads to a consistent sequence length for batching.
///
/// # Arguments
/// * `hidden_states` - Hidden states tensor [batch, seq_len, hidden_dim]
/// * `attention_mask` - Attention mask [batch, seq_len]
/// * `drop_first` - Number of initial tokens to drop (system message prefix)
///
/// # Returns
/// (padded_embeddings, attention_mask) tuple
pub fn extract_and_pad_embeddings(
    hidden_states: &Tensor,
    attention_mask: &Tensor,
    drop_first: usize,
) -> Result<(Tensor, Tensor)> {
    let (batch_size, seq_len, hidden_dim) = hidden_states.dims3()?;
    let device = hidden_states.device();
    let dtype = hidden_states.dtype();

    // Get attention mask as boolean for extraction
    let mask_vec: Vec<f32> = attention_mask.flatten_all()?.to_vec1()?;

    // Calculate valid lengths per batch item (after dropping first tokens)
    let mut valid_lengths = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        let batch_start = b * seq_len;
        let mut valid = 0usize;
        for s in drop_first..seq_len {
            if mask_vec[batch_start + s] > 0.5 {
                valid += 1;
            }
        }
        valid_lengths.push(valid);
    }

    // Find max sequence length after extraction
    let max_len = valid_lengths.iter().copied().max().unwrap_or(0);

    // Extract and pad
    let mut padded_embeds = Vec::new();
    let mut padded_masks = Vec::new();

    for (b, &valid_len) in valid_lengths.iter().enumerate() {
        let batch_hidden = hidden_states.i(b)?;

        // Extract valid embeddings (skip first `drop_first` tokens)
        let extracted = batch_hidden.narrow(0, drop_first, seq_len - drop_first)?;
        let extracted = extracted.narrow(0, 0, valid_len)?;

        // Pad to max_len
        if valid_len < max_len {
            let padding = Tensor::zeros((max_len - valid_len, hidden_dim), dtype, device)?;
            padded_embeds.push(Tensor::cat(&[&extracted, &padding], 0)?);
        } else {
            padded_embeds.push(extracted);
        }

        // Create attention mask
        let ones = Tensor::ones(valid_len, dtype, device)?;
        let zeros = Tensor::zeros(max_len - valid_len, dtype, device)?;
        padded_masks.push(Tensor::cat(&[&ones, &zeros], 0)?);
    }

    let padded_embeds = Tensor::stack(&padded_embeds, 0)?;
    let padded_masks = Tensor::stack(&padded_masks, 0)?;

    Ok((padded_embeds, padded_masks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn test_calculate_dimensions() {
        // Square aspect ratio
        let (w, h) = calculate_dimensions(1024 * 1024, 1.0);
        assert_eq!(w, 1024);
        assert_eq!(h, 1024);

        // 16:9 aspect ratio
        let (w, h) = calculate_dimensions(1024 * 1024, 16.0 / 9.0);
        assert!(w > h);
        assert_eq!(w % 32, 0);
        assert_eq!(h % 32, 0);
    }

    #[test]
    fn test_pack_unpack_layered_latents() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let layers = 2; // Use smaller values for faster testing
        let channels = 4;
        let height = 8;
        let width = 8;

        // Create random latents: [batch, layers+1, channels, height, width]
        let latents = Tensor::randn(
            0f32,
            1f32,
            (batch, layers + 1, channels, height, width),
            &device,
        )?;

        // Pack
        let packed = pack_layered_latents(&latents, height, width, layers)?;
        let expected_seq_len = (layers + 1) * (height / 2) * (width / 2);
        assert_eq!(packed.dims(), &[batch, expected_seq_len, channels * 4]);

        // Unpack
        let unpacked = unpack_layered_latents(&packed, height, width, layers, channels)?;
        // Note: unpacked is [batch, channels, layers+1, height, width]
        assert_eq!(unpacked.dims(), &[batch, channels, layers + 1, height, width]);

        Ok(())
    }
}
