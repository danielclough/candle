//! Qwen2.5-VL Vision-Language Model.
//!
//! Qwen2.5-VL is a multimodal model combining:
//! - Vision encoder: ViT with 2D RoPE and window attention
//! - Text decoder: Qwen2.5 with M-RoPE (Multimodal RoPE)
//!
//! Key features:
//! - Dynamic resolution via patch-based processing
//! - 3D position encoding (temporal, height, width)
//! - No DeepStack (simpler than Qwen3-VL)
//! - No QK-normalization in text attention
//!
//! Available model sizes: 3B, 7B, 72B (no 2B variant exists).
//!
//! # Example
//!
//! ```ignore
//! use candle_transformers::models::qwen2_5_vl::{Config, Qwen25VLModel};
//!
//! let model = Qwen25VLModel::new(&config, vb)?;
//! let logits = model.forward(&input_ids, &pixel_values, &image_grid_thw)?;
//! ```

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle::{Result, Tensor};
use candle_nn::VarBuilder;

pub mod config;
pub mod processing;
pub mod text;
pub mod vision;

pub use config::{Config, RopeScaling, VisionConfig};
pub use processing::{
    get_image_grid_thw, get_number_of_image_patches, get_number_of_image_tokens,
    get_number_of_video_patches, get_number_of_video_tokens, get_video_grid_thw, normalize_image,
    normalize_pixel, patchify_image, patchify_video, smart_resize, ProcessorConfig,
    DEFAULT_MAX_PIXELS, DEFAULT_MERGE_SIZE, DEFAULT_MIN_PIXELS, DEFAULT_PATCH_SIZE,
    DEFAULT_TEMPORAL_PATCH_SIZE, IMAGE_MEAN, IMAGE_STD,
};
pub use text::{
    compute_mrope_position_ids, compute_mrope_position_ids_multi, compute_mrope_position_ids_video,
    compute_mrope_position_ids_video_with_delta, compute_mrope_position_ids_with_delta, ImageGrid,
    MRopePositionIds, Qwen25VLTextModel, VideoGrid,
};
pub use vision::Qwen25VLVisionModel;

/// Qwen2.5-VL Vision-Language Model.
///
/// Combines a vision encoder (ViT with 2D RoPE) and text decoder (Qwen2.5 with M-RoPE)
/// for multimodal understanding.
pub struct Qwen25VLModel {
    vision: Qwen25VLVisionModel,
    text: Qwen25VLTextModel,
    image_token_id: u32,
    video_token_id: u32,
    spatial_merge_size: usize,
    tokens_per_second: usize,
    /// Chunk size for prefill to reduce peak memory (None = disabled).
    prefill_chunk_size: Option<usize>,
}

impl Qwen25VLModel {
    /// Create a new Qwen2.5-VL model.
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        // Vision encoder uses "visual.*" prefix (not "model.visual.*")
        let vision = Qwen25VLVisionModel::new(&cfg.vision_config, vb.pp("visual"))?;
        let text = Qwen25VLTextModel::new(cfg, vb.clone())?;

        Ok(Self {
            vision,
            text,
            image_token_id: cfg.image_token_id,
            video_token_id: cfg.video_token_id,
            spatial_merge_size: cfg.vision_config.spatial_merge_size,
            tokens_per_second: cfg.vision_config.tokens_per_second,
            prefill_chunk_size: cfg.prefill_chunk_size,
        })
    }

    /// Forward pass for image understanding.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs with image placeholders, shape (batch, seq_len)
    /// * `pixel_values` - Preprocessed image pixels, shape (num_patches, channels * temporal * patch * patch)
    /// * `image_grid_thw` - Grid dimensions for each image, shape (num_images, 3)
    ///
    /// # Returns
    /// Logits for the last token, shape (batch, vocab_size)
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        image_grid_thw: &Tensor,
    ) -> Result<Tensor> {
        let device = input_ids.device();

        // 1. Process images through vision encoder
        let vision_embeds = self.vision.forward(pixel_values, image_grid_thw)?;
        let vision_embeds = vision_embeds.to_dtype(self.text.dtype)?;

        // 2. Get text embeddings
        let mut input_embeds = self.text.embed_tokens(input_ids)?;
        let (_batch_size, seq_len, hidden_dim) = input_embeds.dims3()?;

        // 3. Compute M-RoPE position IDs
        let grid_thw_vec = image_grid_thw.to_vec2::<u32>()?;
        let image_grids: Vec<ImageGrid> = grid_thw_vec
            .iter()
            .map(|g| {
                let h = g[1] as usize;
                let w = g[2] as usize;
                ImageGrid {
                    grid_h: h / self.spatial_merge_size,
                    grid_w: w / self.spatial_merge_size,
                }
            })
            .collect();

        let position_ids =
            compute_mrope_position_ids_multi(input_ids, self.image_token_id, &image_grids, device)?;

        // 4. Find image placeholder positions and replace with vision embeddings
        let input_ids_flat: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
        let mut vision_offset = 0usize;
        let batch_size = input_ids.dim(0)?;

        for batch_idx in 0..batch_size {
            let batch_start = batch_idx * seq_len;
            let mut token_idx = 0usize;
            while token_idx < seq_len {
                if input_ids_flat[batch_start + token_idx] == self.image_token_id {
                    // Find contiguous image tokens
                    let start = token_idx;
                    while token_idx < seq_len
                        && input_ids_flat[batch_start + token_idx] == self.image_token_id
                    {
                        token_idx += 1;
                    }
                    let len = token_idx - start;

                    // Replace with vision embeddings
                    let vision_chunk = vision_embeds.narrow(0, vision_offset, len)?;
                    input_embeds = input_embeds.slice_assign(
                        &[batch_idx..batch_idx + 1, start..start + len, 0..hidden_dim],
                        &vision_chunk.unsqueeze(0)?,
                    )?;
                    vision_offset += len;
                } else {
                    token_idx += 1;
                }
            }
        }

        // 5. Forward through text model (use chunked prefill if configured)
        match self.prefill_chunk_size {
            Some(chunk_size) => {
                self.text
                    .forward_with_mrope_chunked(input_embeds, &position_ids, chunk_size)
            }
            None => self.text.forward_with_mrope(input_embeds, &position_ids),
        }
    }

    /// Forward pass for video understanding.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs with video placeholders
    /// * `pixel_values_video` - Preprocessed video frames
    /// * `video_grid_thw` - Grid dimensions (temporal, height, width)
    /// * `second_per_grid_t` - Temporal spacing = temporal_patch_size / fps
    pub fn forward_video(
        &mut self,
        input_ids: &Tensor,
        pixel_values_video: &Tensor,
        video_grid_thw: &Tensor,
        second_per_grid_t: f32,
    ) -> Result<Tensor> {
        let device = input_ids.device();

        // 1. Process video through vision encoder
        let vision_embeds = self.vision.forward(pixel_values_video, video_grid_thw)?;
        let vision_embeds = vision_embeds.to_dtype(self.text.dtype)?;

        // 2. Get text embeddings
        let mut input_embeds = self.text.embed_tokens(input_ids)?;
        let (_batch_size, seq_len, hidden_dim) = input_embeds.dims3()?;

        // 3. Compute M-RoPE position IDs for video
        let grid_thw_vec = video_grid_thw.to_vec2::<u32>()?;
        let g = &grid_thw_vec[0]; // Assume single video
        let video_grid = VideoGrid {
            grid_t: g[0] as usize,
            grid_h: g[1] as usize / self.spatial_merge_size,
            grid_w: g[2] as usize / self.spatial_merge_size,
        };

        let position_ids = compute_mrope_position_ids_video(
            input_ids,
            self.video_token_id,
            &video_grid,
            second_per_grid_t,
            self.tokens_per_second,
            device,
        )?;

        // 4. Replace video placeholders with vision embeddings
        let input_ids_flat: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
        let mut vision_offset = 0usize;
        let batch_size = input_ids.dim(0)?;

        for batch_idx in 0..batch_size {
            let batch_start = batch_idx * seq_len;
            let mut token_idx = 0usize;
            while token_idx < seq_len {
                if input_ids_flat[batch_start + token_idx] == self.video_token_id {
                    let start = token_idx;
                    while token_idx < seq_len
                        && input_ids_flat[batch_start + token_idx] == self.video_token_id
                    {
                        token_idx += 1;
                    }
                    let len = token_idx - start;

                    let vision_chunk = vision_embeds.narrow(0, vision_offset, len)?;
                    input_embeds = input_embeds.slice_assign(
                        &[batch_idx..batch_idx + 1, start..start + len, 0..hidden_dim],
                        &vision_chunk.unsqueeze(0)?,
                    )?;
                    vision_offset += len;
                } else {
                    token_idx += 1;
                }
            }
        }

        // 5. Forward through text model (use chunked prefill if configured)
        match self.prefill_chunk_size {
            Some(chunk_size) => {
                self.text
                    .forward_with_mrope_chunked(input_embeds, &position_ids, chunk_size)
            }
            None => self.text.forward_with_mrope(input_embeds, &position_ids),
        }
    }

    /// Generate tokens autoregressively with a custom sampler.
    ///
    /// This method handles the complex M-RoPE position tracking internally while
    /// delegating the sampling decision to the caller via a closure. This allows
    /// for flexible sampling strategies (temperature, top-k, top-p, repeat penalty)
    /// without coupling them to the model.
    ///
    /// # Arguments
    /// * `input_ids` - Initial token IDs with image placeholders
    /// * `pixel_values` - Preprocessed image pixels
    /// * `image_grid_thw` - Grid dimensions for each image
    /// * `max_length` - Maximum number of tokens to generate
    /// * `eos_token_id` - End of sequence token ID
    /// * `sampler` - Closure that takes (logits, generated_tokens) and returns sampled token ID.
    ///   The generated_tokens slice contains all tokens generated so far (useful for repeat penalty).
    ///
    /// # Returns
    /// Vector of generated token IDs (excluding input tokens)
    ///
    /// # Example
    /// ```ignore
    /// use candle_transformers::generation::LogitsProcessor;
    /// use candle_transformers::utils::apply_repeat_penalty;
    ///
    /// let mut processor = LogitsProcessor::new(seed, Some(0.8), Some(0.9));
    /// let tokens = model.generate_with_sampler(
    ///     &input_ids, &pixel_values, &grid_thw, 512, eos_id,
    ///     |logits, generated| {
    ///         let logits = apply_repeat_penalty(logits, 1.1, generated)?;
    ///         processor.sample(&logits)
    ///     },
    /// )?;
    /// ```
    pub fn generate_with_sampler<F>(
        &mut self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        image_grid_thw: &Tensor,
        max_length: usize,
        eos_token_id: u32,
        mut sampler: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(&Tensor, &[u32]) -> Result<u32>,
    {
        self.clear_kv_cache();
        let device = input_ids.device();
        let input_len = input_ids.dim(1)?;

        // Compute image grids and M-RoPE position IDs with delta
        let grid_thw_vec = image_grid_thw.to_vec2::<u32>()?;
        let image_grids: Vec<ImageGrid> = grid_thw_vec
            .iter()
            .map(|g| {
                let h = g[1] as usize;
                let w = g[2] as usize;
                ImageGrid {
                    grid_h: h / self.spatial_merge_size,
                    grid_w: w / self.spatial_merge_size,
                }
            })
            .collect();

        let mrope_result = compute_mrope_position_ids_with_delta(
            input_ids,
            self.image_token_id,
            &image_grids,
            device,
        )?;
        let mrope_delta = mrope_result.mrope_position_delta;

        // Prefill: process the full input with image
        // Squeeze batch dimension: [batch, vocab] -> [vocab] for sampler
        let logits = self
            .forward(input_ids, pixel_values, image_grid_thw)?
            .squeeze(0)?;
        let mut generated: Vec<u32> = Vec::new();
        let mut next_token = sampler(&logits, &generated)?;
        generated.push(next_token);

        // Decode loop
        for _ in 1..max_length {
            if next_token == eos_token_id {
                break;
            }

            // Create input for next token
            let next_input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;

            // Position for new token: seq_len - 1 + delta
            // This correctly accounts for vision tokens using 2D/3D spatial positions
            let seq_len = input_len + generated.len();
            let pos = (seq_len as i64 - 1) + mrope_delta;
            let position_ids = Tensor::new(&[[[pos]], [[pos]], [[pos]]], device)?;

            let next_embeds = self.text.embed_tokens(&next_input)?;
            let logits = self
                .text
                .forward_with_mrope(next_embeds, &position_ids)?
                .squeeze(0)?;

            next_token = sampler(&logits, &generated)?;
            generated.push(next_token);
        }

        Ok(generated)
    }

    /// Generate tokens autoregressively with streaming callback.
    ///
    /// Like `generate_with_sampler`, but calls `on_token` after each token is generated,
    /// allowing for real-time streaming output.
    ///
    /// # Arguments
    /// * `input_ids` - Initial token IDs with image placeholders
    /// * `pixel_values` - Preprocessed image pixels
    /// * `image_grid_thw` - Grid dimensions for each image
    /// * `max_length` - Maximum number of tokens to generate
    /// * `eos_token_id` - End of sequence token ID
    /// * `sampler` - Closure that takes (logits, generated_tokens) and returns sampled token ID
    /// * `on_token` - Callback invoked after each token is generated with (token_id, is_eos)
    ///
    /// # Returns
    /// Vector of generated token IDs (excluding input tokens)
    ///
    /// # Example
    /// ```ignore
    /// model.generate_streaming(
    ///     &input_ids, &pixel_values, &grid_thw, 512, eos_id,
    ///     |logits, _| processor.sample(logits),
    ///     |token, _is_eos| {
    ///         if let Some(text) = tokenizer.decode(&[token], true).ok() {
    ///             print!("{}", text);
    ///             std::io::stdout().flush().ok();
    ///         }
    ///     },
    /// )?;
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming<F, C>(
        &mut self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        image_grid_thw: &Tensor,
        max_length: usize,
        eos_token_id: u32,
        mut sampler: F,
        mut on_token: C,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(&Tensor, &[u32]) -> Result<u32>,
        C: FnMut(u32, bool),
    {
        self.clear_kv_cache();
        let device = input_ids.device();
        let input_len = input_ids.dim(1)?;

        // Compute image grids and M-RoPE position IDs with delta
        let grid_thw_vec = image_grid_thw.to_vec2::<u32>()?;
        let image_grids: Vec<ImageGrid> = grid_thw_vec
            .iter()
            .map(|g| {
                let h = g[1] as usize;
                let w = g[2] as usize;
                ImageGrid {
                    grid_h: h / self.spatial_merge_size,
                    grid_w: w / self.spatial_merge_size,
                }
            })
            .collect();

        let mrope_result = compute_mrope_position_ids_with_delta(
            input_ids,
            self.image_token_id,
            &image_grids,
            device,
        )?;
        let mrope_delta = mrope_result.mrope_position_delta;

        // Prefill: process the full input with image
        // Squeeze batch dimension: [batch, vocab] -> [vocab] for sampler
        let logits = self
            .forward(input_ids, pixel_values, image_grid_thw)?
            .squeeze(0)?;
        let mut generated: Vec<u32> = Vec::new();
        let mut next_token = sampler(&logits, &generated)?;
        generated.push(next_token);

        let is_eos = next_token == eos_token_id;
        on_token(next_token, is_eos);
        if is_eos {
            return Ok(generated);
        }

        // Decode loop
        for _ in 1..max_length {
            // Create input for next token
            let next_input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;

            // Position for new token: seq_len - 1 + delta
            let seq_len = input_len + generated.len();
            let pos = (seq_len as i64 - 1) + mrope_delta;
            let position_ids = Tensor::new(&[[[pos]], [[pos]], [[pos]]], device)?;

            let next_embeds = self.text.embed_tokens(&next_input)?;
            let logits = self
                .text
                .forward_with_mrope(next_embeds, &position_ids)?
                .squeeze(0)?;

            next_token = sampler(&logits, &generated)?;
            generated.push(next_token);

            let is_eos = next_token == eos_token_id;
            on_token(next_token, is_eos);
            if is_eos {
                break;
            }
        }

        Ok(generated)
    }

    /// Generate tokens autoregressively using greedy decoding (argmax).
    ///
    /// This is a convenience method that uses greedy decoding. For more control
    /// over sampling (temperature, top-k, top-p), use `generate_with_sampler`.
    ///
    /// # Arguments
    /// * `input_ids` - Initial token IDs with image placeholders
    /// * `pixel_values` - Preprocessed image pixels
    /// * `image_grid_thw` - Grid dimensions for each image
    /// * `max_length` - Maximum number of tokens to generate
    /// * `eos_token_id` - End of sequence token ID
    ///
    /// # Returns
    /// Vector of generated token IDs (excluding input tokens)
    pub fn generate(
        &mut self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        image_grid_thw: &Tensor,
        max_length: usize,
        eos_token_id: u32,
    ) -> Result<Vec<u32>> {
        self.generate_with_sampler(
            input_ids,
            pixel_values,
            image_grid_thw,
            max_length,
            eos_token_id,
            |logits, _generated| logits.argmax(D::Minus1)?.to_scalar::<u32>(),
        )
    }

    /// Generate tokens autoregressively from video input with a custom sampler.
    ///
    /// This method handles the complex M-RoPE position tracking (including temporal
    /// dimensions) internally while delegating sampling to the caller.
    ///
    /// # Arguments
    /// * `input_ids` - Initial token IDs with video placeholders
    /// * `pixel_values_video` - Preprocessed video pixels
    /// * `video_grid_thw` - Grid dimensions (temporal, height, width)
    /// * `second_per_grid_t` - Temporal spacing = temporal_patch_size / fps
    /// * `max_length` - Maximum number of tokens to generate
    /// * `eos_token_id` - End of sequence token ID
    /// * `sampler` - Closure that takes (logits, generated_tokens) and returns sampled token ID
    ///
    /// # Returns
    /// Vector of generated token IDs (excluding input tokens)
    #[allow(clippy::too_many_arguments)]
    pub fn generate_video_with_sampler<F>(
        &mut self,
        input_ids: &Tensor,
        pixel_values_video: &Tensor,
        video_grid_thw: &Tensor,
        second_per_grid_t: f32,
        max_length: usize,
        eos_token_id: u32,
        mut sampler: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(&Tensor, &[u32]) -> Result<u32>,
    {
        self.clear_kv_cache();
        let device = input_ids.device();
        let input_len = input_ids.dim(1)?;

        // Compute video grid and M-RoPE position IDs with delta
        let grid_thw_vec = video_grid_thw.to_vec2::<u32>()?;
        let g = &grid_thw_vec[0]; // Assume single video
        let video_grid = VideoGrid {
            grid_t: g[0] as usize,
            grid_h: g[1] as usize / self.spatial_merge_size,
            grid_w: g[2] as usize / self.spatial_merge_size,
        };

        let mrope_result = compute_mrope_position_ids_video_with_delta(
            input_ids,
            self.video_token_id,
            &video_grid,
            second_per_grid_t,
            self.tokens_per_second,
            device,
        )?;
        let mrope_delta = mrope_result.mrope_position_delta;

        // Prefill: process the full input with video
        // Squeeze batch dimension: [batch, vocab] -> [vocab] for sampler
        let logits = self
            .forward_video(
                input_ids,
                pixel_values_video,
                video_grid_thw,
                second_per_grid_t,
            )?
            .squeeze(0)?;
        let mut generated: Vec<u32> = Vec::new();
        let mut next_token = sampler(&logits, &generated)?;
        generated.push(next_token);

        // Decode loop
        for _ in 1..max_length {
            if next_token == eos_token_id {
                break;
            }

            // Create input for next token
            let next_input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;

            // Position for new token: seq_len - 1 + delta
            // This correctly accounts for video tokens using 3D spatial/temporal positions
            let seq_len = input_len + generated.len();
            let pos = (seq_len as i64 - 1) + mrope_delta;
            let position_ids = Tensor::new(&[[[pos]], [[pos]], [[pos]]], device)?;

            let next_embeds = self.text.embed_tokens(&next_input)?;
            let logits = self
                .text
                .forward_with_mrope(next_embeds, &position_ids)?
                .squeeze(0)?;

            next_token = sampler(&logits, &generated)?;
            generated.push(next_token);
        }

        Ok(generated)
    }

    /// Generate tokens autoregressively from video with streaming callback.
    ///
    /// Like `generate_video_with_sampler`, but calls `on_token` after each token is generated.
    ///
    /// # Arguments
    /// * `input_ids` - Initial token IDs with video placeholders
    /// * `pixel_values_video` - Preprocessed video pixels
    /// * `video_grid_thw` - Grid dimensions (temporal, height, width)
    /// * `second_per_grid_t` - Temporal spacing = temporal_patch_size / fps
    /// * `max_length` - Maximum number of tokens to generate
    /// * `eos_token_id` - End of sequence token ID
    /// * `sampler` - Closure that takes (logits, generated_tokens) and returns sampled token ID
    /// * `on_token` - Callback invoked after each token is generated with (token_id, is_eos)
    ///
    /// # Returns
    /// Vector of generated token IDs (excluding input tokens)
    #[allow(clippy::too_many_arguments)]
    pub fn generate_video_streaming<F, C>(
        &mut self,
        input_ids: &Tensor,
        pixel_values_video: &Tensor,
        video_grid_thw: &Tensor,
        second_per_grid_t: f32,
        max_length: usize,
        eos_token_id: u32,
        mut sampler: F,
        mut on_token: C,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(&Tensor, &[u32]) -> Result<u32>,
        C: FnMut(u32, bool),
    {
        self.clear_kv_cache();
        let device = input_ids.device();
        let input_len = input_ids.dim(1)?;

        // Compute video grid and M-RoPE position IDs with delta
        let grid_thw_vec = video_grid_thw.to_vec2::<u32>()?;
        let g = &grid_thw_vec[0]; // Assume single video
        let video_grid = VideoGrid {
            grid_t: g[0] as usize,
            grid_h: g[1] as usize / self.spatial_merge_size,
            grid_w: g[2] as usize / self.spatial_merge_size,
        };

        let mrope_result = compute_mrope_position_ids_video_with_delta(
            input_ids,
            self.video_token_id,
            &video_grid,
            second_per_grid_t,
            self.tokens_per_second,
            device,
        )?;
        let mrope_delta = mrope_result.mrope_position_delta;

        // Prefill: process the full input with video
        // Squeeze batch dimension: [batch, vocab] -> [vocab] for sampler
        let logits = self
            .forward_video(
                input_ids,
                pixel_values_video,
                video_grid_thw,
                second_per_grid_t,
            )?
            .squeeze(0)?;
        let mut generated: Vec<u32> = Vec::new();
        let mut next_token = sampler(&logits, &generated)?;
        generated.push(next_token);

        let is_eos = next_token == eos_token_id;
        on_token(next_token, is_eos);
        if is_eos {
            return Ok(generated);
        }

        // Decode loop
        for _ in 1..max_length {
            // Create input for next token
            let next_input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;

            // Position for new token: seq_len - 1 + delta
            let seq_len = input_len + generated.len();
            let pos = (seq_len as i64 - 1) + mrope_delta;
            let position_ids = Tensor::new(&[[[pos]], [[pos]], [[pos]]], device)?;

            let next_embeds = self.text.embed_tokens(&next_input)?;
            let logits = self
                .text
                .forward_with_mrope(next_embeds, &position_ids)?
                .squeeze(0)?;

            next_token = sampler(&logits, &generated)?;
            generated.push(next_token);

            let is_eos = next_token == eos_token_id;
            on_token(next_token, is_eos);
            if is_eos {
                break;
            }
        }

        Ok(generated)
    }

    /// Generate tokens autoregressively from video input using greedy decoding.
    ///
    /// This is a convenience method that uses greedy decoding. For more control
    /// over sampling, use `generate_video_with_sampler`.
    ///
    /// # Arguments
    /// * `input_ids` - Initial token IDs with video placeholders
    /// * `pixel_values_video` - Preprocessed video pixels
    /// * `video_grid_thw` - Grid dimensions (temporal, height, width)
    /// * `second_per_grid_t` - Temporal spacing = temporal_patch_size / fps
    /// * `max_length` - Maximum number of tokens to generate
    /// * `eos_token_id` - End of sequence token ID
    ///
    /// # Returns
    /// Vector of generated token IDs (excluding input tokens)
    pub fn generate_video(
        &mut self,
        input_ids: &Tensor,
        pixel_values_video: &Tensor,
        video_grid_thw: &Tensor,
        second_per_grid_t: f32,
        max_length: usize,
        eos_token_id: u32,
    ) -> Result<Vec<u32>> {
        self.generate_video_with_sampler(
            input_ids,
            pixel_values_video,
            video_grid_thw,
            second_per_grid_t,
            max_length,
            eos_token_id,
            |logits, _generated| logits.argmax(D::Minus1)?.to_scalar::<u32>(),
        )
    }

    /// Clear all KV caches for fresh generation.
    pub fn clear_kv_cache(&mut self) {
        self.text.clear_kv_cache();
    }
}

use candle::D;
