//! Streaming Qwen-Image Transformer Model.
//!
//! This module provides memory-efficient GGUF loading that streams transformer blocks
//! on-demand during inference instead of loading all 60 blocks into GPU memory at once.
//!
//! This reduces peak GPU memory from ~10-12GB (all blocks) to ~200-400MB (one block at a time),
//! at the cost of increased latency from disk I/O during each denoising step.
//!
//! # Usage
//!
//! ```ignore
//! use candle::quantized::gguf_file;
//!
//! let model = QwenImageTransformer2DModelStreaming::from_gguf_path(
//!     "qwen-image-q4_k.gguf",
//!     &device,
//!     dtype,
//!     &inference_config,
//!     false,
//! )?;
//!
//! // Forward pass streams blocks automatically
//! let output = model.forward(&latents, &text_embeds, &timestep, &img_shapes)?;
//! ```

use candle::{quantized::gguf_file, DType, Device, Result, Tensor};
use candle_nn::RmsNorm;
use std::io::{Read, Seek};
use std::path::PathBuf;
use std::sync::Arc;

use crate::models::with_tracing::QMatMul;

use super::blocks::{
    apply_modulation_with_index, layer_norm_no_affine, FeedForward, Modulation, QkNorm,
    QwenDoubleStreamAttention, QwenImageTransformerBlock,
};
use super::config::InferenceConfig;
use super::model::{AdaLayerNormContinuous, QwenTimestepProjEmbeddings};
use super::quantized::QLinear;
use super::rope::QwenEmbedRope;

// ============================================================================
// Streaming Block Loader
// ============================================================================

/// Metadata required to load a single transformer block from GGUF.
///
/// Stores the tensor names and offsets for lazy loading.
#[derive(Debug, Clone)]
pub struct BlockLoadInfo {
    /// Block index (0-59)
    pub idx: usize,
    /// Tensor prefix (e.g., "transformer_blocks.0")
    pub prefix: String,
}

/// Streaming loader for transformer blocks.
///
/// Holds GGUF metadata and file path, loads blocks on-demand.
pub struct StreamingBlockLoader {
    /// Path to GGUF file
    path: PathBuf,
    /// GGUF file content (metadata only, tensors loaded on-demand)
    content: gguf_file::Content,
    /// Device to load tensors onto
    device: Device,
    /// Working dtype for biases
    dtype: DType,
    /// Model dimensions
    inner_dim: usize,
    num_heads: usize,
    head_dim: usize,
    /// Whether to upcast attention to F32
    upcast_attention: bool,
}

impl StreamingBlockLoader {
    /// Create a new streaming loader from GGUF file path.
    pub fn new(
        path: PathBuf,
        device: Device,
        dtype: DType,
        inference_config: &InferenceConfig,
    ) -> Result<Self> {
        let mut file = std::fs::File::open(&path)?;
        let content = gguf_file::Content::read(&mut file)?;

        // Extract dimensions from metadata
        let get_u32 = |keys: &[&str]| -> Option<u32> {
            for key in keys {
                if let Some(v) = content.metadata.get(*key) {
                    return v.to_u32().ok();
                }
            }
            None
        };

        let inner_dim =
            get_u32(&["qwen_image.inner_dim", "transformer.inner_dim"]).unwrap_or(3072) as usize;
        let num_heads = get_u32(&[
            "qwen_image.num_attention_heads",
            "transformer.num_attention_heads",
        ])
        .unwrap_or(24) as usize;
        let head_dim = inner_dim / num_heads;

        Ok(Self {
            path,
            content,
            device,
            dtype,
            inner_dim,
            num_heads,
            head_dim,
            upcast_attention: inference_config.upcast_attention,
        })
    }

    /// Get number of transformer blocks.
    pub fn num_blocks(&self) -> usize {
        // Count blocks by checking tensor names
        let mut max_idx = 0;
        for name in self.content.tensor_infos.keys() {
            if name.starts_with("transformer_blocks.") {
                if let Some(idx_str) = name
                    .strip_prefix("transformer_blocks.")
                    .and_then(|s| s.split('.').next())
                {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        max_idx = max_idx.max(idx + 1);
                    }
                }
            }
        }
        max_idx
    }

    /// Load a single transformer block by index.
    ///
    /// Opens the GGUF file, seeks to the block's tensors, and loads them.
    pub fn load_block(
        &self,
        idx: usize,
    ) -> Result<QwenImageTransformerBlock<QLinear>> {
        let mut file = std::fs::File::open(&self.path)?;
        self.load_block_from_reader(idx, &mut file)
    }

    /// Load a transformer block using an existing reader.
    fn load_block_from_reader<R: Read + Seek>(
        &self,
        idx: usize,
        reader: &mut R,
    ) -> Result<QwenImageTransformerBlock<QLinear>> {
        let prefix = format!("transformer_blocks.{idx}");
        let bias_dtype = self.dtype;

        // Macro to load and dequantize tensor (for small params like norms)
        macro_rules! load_dequant {
            ($name:expr, $dtype:expr) => {{
                let name: &str = &$name;
                let qt = self.content.tensor(reader, name, &self.device)?;
                qt.dequantize(&self.device)?.to_dtype($dtype)?
            }};
        }

        // Macro to load QLinear with weight and optional bias
        macro_rules! load_qlinear {
            ($weight_name:expr, $bias_name:expr) => {{
                let weight_name: &str = &$weight_name;
                let bias_name: &str = &$bias_name;
                let qt = self.content.tensor(reader, weight_name, &self.device)?;
                let weight = QMatMul::from_weights_with_transposed_data(Arc::new(qt))?;
                let bias = if self.content.tensor_infos.contains_key(bias_name) {
                    let bt = self.content.tensor(reader, bias_name, &self.device)?;
                    Some(bt.dequantize(&self.device)?.to_dtype(bias_dtype)?)
                } else {
                    None
                };
                QLinear::new(weight, bias)
            }};
        }

        // Load modulation layers
        let img_mod = Modulation::from_linear(load_qlinear!(
            format!("{prefix}.img_mod.1.weight"),
            format!("{prefix}.img_mod.1.bias")
        ));
        let txt_mod = Modulation::from_linear(load_qlinear!(
            format!("{prefix}.txt_mod.1.weight"),
            format!("{prefix}.txt_mod.1.bias")
        ));

        // Load attention norms
        let img_norm = QkNorm::from_rms_norms(
            RmsNorm::new(
                load_dequant!(format!("{prefix}.attn.norm_q.weight"), DType::F32),
                1e-6,
            ),
            RmsNorm::new(
                load_dequant!(format!("{prefix}.attn.norm_k.weight"), DType::F32),
                1e-6,
            ),
        );
        let txt_norm = QkNorm::from_rms_norms(
            RmsNorm::new(
                load_dequant!(format!("{prefix}.attn.norm_added_q.weight"), DType::F32),
                1e-6,
            ),
            RmsNorm::new(
                load_dequant!(format!("{prefix}.attn.norm_added_k.weight"), DType::F32),
                1e-6,
            ),
        );

        let attn = QwenDoubleStreamAttention::from_parts(
            load_qlinear!(
                format!("{prefix}.attn.to_q.weight"),
                format!("{prefix}.attn.to_q.bias")
            ),
            load_qlinear!(
                format!("{prefix}.attn.to_k.weight"),
                format!("{prefix}.attn.to_k.bias")
            ),
            load_qlinear!(
                format!("{prefix}.attn.to_v.weight"),
                format!("{prefix}.attn.to_v.bias")
            ),
            load_qlinear!(
                format!("{prefix}.attn.to_out.0.weight"),
                format!("{prefix}.attn.to_out.0.bias")
            ),
            load_qlinear!(
                format!("{prefix}.attn.add_q_proj.weight"),
                format!("{prefix}.attn.add_q_proj.bias")
            ),
            load_qlinear!(
                format!("{prefix}.attn.add_k_proj.weight"),
                format!("{prefix}.attn.add_k_proj.bias")
            ),
            load_qlinear!(
                format!("{prefix}.attn.add_v_proj.weight"),
                format!("{prefix}.attn.add_v_proj.bias")
            ),
            load_qlinear!(
                format!("{prefix}.attn.to_add_out.weight"),
                format!("{prefix}.attn.to_add_out.bias")
            ),
            img_norm,
            txt_norm,
            self.num_heads,
            self.head_dim,
            self.upcast_attention,
        );

        // Load MLPs
        let img_mlp = FeedForward::from_linears(
            load_qlinear!(
                format!("{prefix}.img_mlp.net.0.proj.weight"),
                format!("{prefix}.img_mlp.net.0.proj.bias")
            ),
            load_qlinear!(
                format!("{prefix}.img_mlp.net.2.weight"),
                format!("{prefix}.img_mlp.net.2.bias")
            ),
        );
        let txt_mlp = FeedForward::from_linears(
            load_qlinear!(
                format!("{prefix}.txt_mlp.net.0.proj.weight"),
                format!("{prefix}.txt_mlp.net.0.proj.bias")
            ),
            load_qlinear!(
                format!("{prefix}.txt_mlp.net.2.weight"),
                format!("{prefix}.txt_mlp.net.2.bias")
            ),
        );

        // Layer norms (parameter-free)
        let img_norm1 = layer_norm_no_affine(self.inner_dim, 1e-6, &self.device, DType::F32)?;
        let txt_norm1 = layer_norm_no_affine(self.inner_dim, 1e-6, &self.device, DType::F32)?;

        Ok(QwenImageTransformerBlock {
            img_mod,
            img_norm1,
            img_mlp,
            txt_mod,
            txt_norm1,
            txt_mlp,
            attn,
        })
    }

    /// Accessor for content (needed for loading non-block components)
    pub fn content(&self) -> &gguf_file::Content {
        &self.content
    }

    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get dtype
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get inner dim
    pub fn inner_dim(&self) -> usize {
        self.inner_dim
    }
}

// ============================================================================
// Streaming Transformer Model
// ============================================================================

/// Streaming Qwen-Image Transformer that loads blocks on-demand.
///
/// Unlike `QwenImageTransformer2DModelQuantized` which loads all 60 blocks at once
/// (~10-12GB GPU memory), this variant streams blocks during the forward pass,
/// reducing peak memory to ~200-400MB (one block at a time).
///
/// Trade-off: Higher latency due to disk I/O during inference.
pub struct QwenImageTransformer2DModelStreaming {
    /// RoPE embeddings for 3D positioning
    pos_embed: QwenEmbedRope,
    /// Timestep embedding projection
    time_text_embed: QwenTimestepProjEmbeddings<QLinear>,
    /// Text input normalization
    txt_norm: RmsNorm,
    /// Image input projection
    img_in: QLinear,
    /// Text input projection
    txt_in: QLinear,
    /// Output normalization with AdaLN
    norm_out: AdaLayerNormContinuous<QLinear>,
    /// Output projection
    proj_out: QLinear,
    /// Whether to use zero conditioning for timestep (edit mode)
    zero_cond_t: bool,
    /// Block loader for streaming
    block_loader: StreamingBlockLoader,
    /// Number of transformer blocks
    num_blocks: usize,
}

impl QwenImageTransformer2DModelStreaming {
    /// Load streaming model from GGUF file path.
    ///
    /// Loads only the non-block components (small, ~100MB) into GPU memory.
    /// Transformer blocks are streamed on-demand during forward pass.
    pub fn from_gguf_path(
        path: &str,
        device: &Device,
        dtype: DType,
        inference_config: &InferenceConfig,
        zero_cond_t: bool,
    ) -> Result<Self> {
        println!("Loading streaming transformer from {}...", path);
        println!("  (Blocks will be streamed on-demand during inference)");

        let path = PathBuf::from(path);
        let block_loader = StreamingBlockLoader::new(path.clone(), device.clone(), dtype, inference_config)?;
        let num_blocks = block_loader.num_blocks();

        // Open file and load non-block components
        let mut file = std::fs::File::open(&path)?;
        let content = block_loader.content();
        let inner_dim = block_loader.inner_dim();
        let bias_dtype = dtype;

        // Macro to load and dequantize tensor
        macro_rules! load_dequant {
            ($name:expr, $dtype:expr) => {{
                let qt = content.tensor(&mut file, $name, device)?;
                qt.dequantize(device)?.to_dtype($dtype)?
            }};
        }

        // Macro to load QLinear with weight and optional bias
        macro_rules! load_qlinear {
            ($weight_name:expr, $bias_name:expr) => {{
                let qt = content.tensor(&mut file, $weight_name, device)?;
                let weight = QMatMul::from_weights_with_transposed_data(Arc::new(qt))?;
                let bias = if content.tensor_infos.contains_key($bias_name) {
                    let bt = content.tensor(&mut file, $bias_name, device)?;
                    Some(bt.dequantize(device)?.to_dtype(bias_dtype)?)
                } else {
                    None
                };
                QLinear::new(weight, bias)
            }};
        }

        // RoPE embeddings
        let theta = 10000usize;
        let axes_dims = (16usize, 56usize, 56usize);
        let pos_embed = QwenEmbedRope::new(
            theta,
            vec![axes_dims.0, axes_dims.1, axes_dims.2],
            true,
            device,
            DType::F32,
        )?;

        // Timestep embeddings
        let time_text_embed = QwenTimestepProjEmbeddings::from_linears(
            load_qlinear!(
                "time_text_embed.timestep_embedder.linear_1.weight",
                "time_text_embed.timestep_embedder.linear_1.bias"
            ),
            load_qlinear!(
                "time_text_embed.timestep_embedder.linear_2.weight",
                "time_text_embed.timestep_embedder.linear_2.bias"
            ),
        );

        // Text norm
        let txt_norm_weight = load_dequant!("txt_norm.weight", DType::F32);
        let txt_norm = RmsNorm::new(txt_norm_weight, 1e-6);

        // Input projections
        let img_in = load_qlinear!("img_in.weight", "img_in.bias");
        let txt_in = load_qlinear!("txt_in.weight", "txt_in.bias");

        // Output layers
        let norm_out = AdaLayerNormContinuous::from_parts(
            layer_norm_no_affine(inner_dim, 1e-6, device, DType::F32)?,
            load_qlinear!("norm_out.linear.weight", "norm_out.linear.bias"),
        );
        let proj_out = load_qlinear!("proj_out.weight", "proj_out.bias");

        println!("  Loaded {} non-block components", 7);
        println!("  {} transformer blocks will be streamed", num_blocks);

        Ok(Self {
            pos_embed,
            time_text_embed,
            txt_norm,
            img_in,
            txt_in,
            norm_out,
            proj_out,
            zero_cond_t,
            block_loader,
            num_blocks,
        })
    }

    /// Get number of transformer blocks.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Forward pass through the streaming transformer.
    ///
    /// Loads each transformer block on-demand, computes, then frees it.
    /// This dramatically reduces GPU memory at the cost of increased latency.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: &Tensor,
        img_shapes: &[(usize, usize, usize)],
    ) -> Result<Tensor> {
        let dtype = hidden_states.dtype();
        let device = hidden_states.device();

        // Project image latents
        let mut hidden_states = self.img_in.forward(hidden_states)?;

        // Normalize and project text
        let mut encoder_hidden_states = encoder_hidden_states.apply(&self.txt_norm)?;
        encoder_hidden_states = self.txt_in.forward(&encoder_hidden_states)?;

        // Handle zero_cond_t for edit mode
        let timestep = timestep.to_dtype(dtype)?;
        let (timestep, modulate_index) = if self.zero_cond_t {
            let zero_timestep = (&timestep * 0.0)?;
            let doubled_timestep = Tensor::cat(&[&timestep, &zero_timestep], 0)?;
            let modulate_idx = Self::create_modulate_index(img_shapes, device)?;
            (doubled_timestep, Some(modulate_idx))
        } else {
            (timestep, None)
        };

        // Compute timestep embedding
        let temb = self.time_text_embed.forward(&timestep, dtype)?;

        // Compute RoPE frequencies
        let txt_seq_len = encoder_hidden_states.dim(1)?;
        let txt_seq_lens = &[txt_seq_len];
        let image_rotary_emb = self.pos_embed.forward(img_shapes, txt_seq_lens)?;

        // Stream through transformer blocks
        for idx in 0..self.num_blocks {
            // Load block from GGUF
            let block = self.block_loader.load_block(idx)?;

            // Process through block
            let (enc_out, hid_out) = Self::forward_block(
                &block,
                &hidden_states,
                &encoder_hidden_states,
                &temb,
                Some(&image_rotary_emb),
                modulate_index.as_ref(),
            )?;

            encoder_hidden_states = enc_out;
            hidden_states = hid_out;

            // Block is dropped here, freeing GPU memory
        }

        // For zero_cond_t, use only the first half of temb
        let temb_for_norm = if self.zero_cond_t {
            let batch_size = temb.dim(0)? / 2;
            temb.narrow(0, 0, batch_size)?
        } else {
            temb
        };

        // Final normalization
        let hidden_states = self.norm_out.forward(&hidden_states, &temb_for_norm)?;

        // Project to output
        self.proj_out.forward(&hidden_states)
    }

    /// Forward pass through a single block.
    fn forward_block(
        block: &QwenImageTransformerBlock<QLinear>,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        temb: &Tensor,
        image_rotary_emb: Option<&(Tensor, Tensor)>,
        modulate_index: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        // Get modulation parameters
        let (img_mod1, img_mod2) = block.img_mod.forward(temb)?;

        // For text stream, use only first half of temb if modulate_index is provided (zero_cond_t mode)
        let batch_size = hidden_states.dim(0)?;
        let txt_temb = if modulate_index.is_some() {
            // In zero_cond_t mode, temb is doubled [2*batch, dim]
            // Text uses only actual timestep (first half)
            temb.narrow(0, 0, batch_size)?
        } else {
            temb.clone()
        };
        let (txt_mod1, txt_mod2) = block.txt_mod.forward(&txt_temb)?;

        // ===== Image stream: norm + modulate + attention =====
        let img_normed = hidden_states.apply(&block.img_norm1)?;
        let (img_modulated, img_gate_attn) = if let Some(mod_idx) = modulate_index {
            apply_modulation_with_index(&img_normed, &img_mod1, mod_idx)?
        } else {
            (img_mod1.scale_shift(&img_normed)?, img_mod1.gate.clone())
        };

        // ===== Text stream: norm + modulate =====
        let txt_normed = encoder_hidden_states.apply(&block.txt_norm1)?;
        let txt_modulated = txt_mod1.scale_shift(&txt_normed)?;

        // ===== Joint attention =====
        let (img_attn_out, txt_attn_out) =
            block.attn.forward(&img_modulated, &txt_modulated, image_rotary_emb)?;

        // ===== Image stream: gated residual + MLP =====
        let hidden_states = (hidden_states + &img_gate_attn.broadcast_mul(&img_attn_out)?)?;

        let img_normed2 = hidden_states.apply(&block.img_norm1)?;
        let (img_modulated2, img_gate_mlp) = if let Some(mod_idx) = modulate_index {
            apply_modulation_with_index(&img_normed2, &img_mod2, mod_idx)?
        } else {
            (img_mod2.scale_shift(&img_normed2)?, img_mod2.gate.clone())
        };
        let img_mlp_out = block.img_mlp.forward(&img_modulated2)?;
        let hidden_states = (&hidden_states + &img_gate_mlp.broadcast_mul(&img_mlp_out)?)?;

        // ===== Text stream: gated residual + MLP =====
        let encoder_hidden_states =
            (encoder_hidden_states + &txt_mod1.gate.broadcast_mul(&txt_attn_out)?)?;

        let txt_normed2 = encoder_hidden_states.apply(&block.txt_norm1)?;
        let txt_modulated2 = txt_mod2.scale_shift(&txt_normed2)?;
        let txt_mlp_out = block.txt_mlp.forward(&txt_modulated2)?;
        let encoder_hidden_states =
            (&encoder_hidden_states + &txt_mod2.gate.broadcast_mul(&txt_mlp_out)?)?;

        Ok((encoder_hidden_states, hidden_states))
    }

    /// Create modulate_index tensor for edit mode.
    fn create_modulate_index(
        img_shapes: &[(usize, usize, usize)],
        device: &Device,
    ) -> Result<Tensor> {
        if img_shapes.is_empty() {
            candle::bail!("img_shapes cannot be empty for modulate_index creation");
        }

        let mut indices: Vec<i64> = Vec::new();

        // First shape (noise latents) -> index 0
        let (f0, h0, w0) = img_shapes[0];
        let noise_seq_len = f0 * h0 * w0;
        indices.extend(std::iter::repeat_n(0i64, noise_seq_len));

        // Remaining shapes (reference images) -> index 1
        for &(f, h, w) in img_shapes.iter().skip(1) {
            let seq_len = f * h * w;
            indices.extend(std::iter::repeat_n(1i64, seq_len));
        }

        let total_len = indices.len();
        Tensor::from_vec(indices, (1, total_len), device)
    }
}
