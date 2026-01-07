//! Vision encoder for Qwen2.5-VL.
//!
//! This module implements a Vision Transformer (ViT) with:
//! - 3D patch embedding for temporal+spatial patches (NO bias)
//! - 2D rotary position embeddings (no learnable position embeddings)
//! - Window attention (with full attention at specific layers)
//! - Patch merger for 2x2 spatial merge + projection to text hidden size
//!
//! Key architectural notes:
//! - Uses SwiGLU MLP (gate_proj, up_proj, down_proj) instead of simple 2-layer MLP
//! - Uses RMSNorm (weight only) instead of LayerNorm
//! - No learnable position embeddings - relies entirely on 2D RoPE
//! - No DeepStack injection (unlike Qwen3-VL)

use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{rms_norm, Conv2d, Conv2dConfig, Linear, RmsNorm, VarBuilder};

use super::config::VisionConfig;
use crate::models::qwen_image::debug::debug_tensor;

// ============================================================================
// Conv3d for temporal patch size = 2
// ============================================================================

/// Configuration for 3D convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv3dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Default for Conv3dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

/// 3D convolution without bias, implemented as two 2D convolutions.
/// Assumes temporal patch size of 2.
pub struct Conv3dNoBias {
    conv2d_1: Conv2d,
    conv2d_2: Conv2d,
}

impl Conv3dNoBias {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_sizes: [usize; 3],
        cfg: Conv3dConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let ws = vb.get(
            (
                out_channels,
                in_channels / cfg.groups,
                kernel_sizes[0],
                kernel_sizes[1],
                kernel_sizes[2],
            ),
            "weight",
        )?;

        // Split on temporal dimension (assuming temporal_patch_size = 2)
        let w1 = ws.i((.., .., 0, .., ..))?;
        let w2 = ws.i((.., .., 1, .., ..))?;

        let cfg = Conv2dConfig {
            padding: cfg.padding,
            stride: cfg.stride,
            dilation: cfg.dilation,
            groups: cfg.groups,
            cudnn_fwd_algo: None,
        };

        Ok(Self {
            conv2d_1: Conv2d::new(w1.contiguous()?, None, cfg),
            conv2d_2: Conv2d::new(w2.contiguous()?, None, cfg),
        })
    }
}

impl Module for Conv3dNoBias {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs1 = xs.i((.., .., 0, .., ..))?;
        let xs2 = xs.i((.., .., 1, .., ..))?;
        (self.conv2d_1.forward(&xs1)? + self.conv2d_2.forward(&xs2)?)?.unsqueeze(2)
    }
}

// ============================================================================
// Patch Embedding
// ============================================================================

/// 3D Patch Embedding for temporal+spatial patches.
///
/// Expects pre-patchified input with shape:
/// `(num_patches, in_channels * temporal_patch_size * patch_size * patch_size)`
///
/// The patchification (including the 9D transpose) happens in image preprocessing,
/// following the HuggingFace Qwen2.5-VL implementation.
///
/// Note: Qwen2.5-VL does NOT have a bias in the patch embedding (unlike Qwen3-VL).
struct PatchEmbed {
    proj: Conv3dNoBias,
    in_channels: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    hidden_size: usize,
}

impl PatchEmbed {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let proj_vb = vb.pp("proj");
        let proj = Conv3dNoBias::new(
            cfg.in_chans,
            cfg.hidden_size,
            [cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size],
            Conv3dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            },
            proj_vb,
        )?;
        Ok(Self {
            proj,
            in_channels: cfg.in_chans,
            patch_size: cfg.patch_size,
            temporal_patch_size: cfg.temporal_patch_size,
            hidden_size: cfg.hidden_size,
        })
    }

    /// Forward pass for pre-patchified input.
    ///
    /// # Arguments
    /// * `xs` - Pre-patchified tensor of shape `(num_patches, C * T * patch * patch)`
    ///   where C=3, T=2, patch=14, so the last dim is 1176.
    ///
    /// # Returns
    /// Patch embeddings of shape `(num_patches, hidden_size)`
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Reshape from (num_patches, C*T*patch*patch) to (num_patches, C, T, patch, patch)
        let xs = xs.reshape((
            (),
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ))?;
        // Apply Conv3d projection: (num_patches, hidden, 1, 1, 1)
        let xs = self.proj.forward(&xs)?;
        // Flatten to (num_patches, hidden)
        xs.reshape(((), self.hidden_size))
    }
}

// ============================================================================
// Vision MLP (SwiGLU)
// ============================================================================

/// Vision MLP using SwiGLU activation.
/// Weight paths: mlp.gate_proj.*, mlp.up_proj.*, mlp.down_proj.*
struct VisionMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl VisionMlp {
    fn new(dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: candle_nn::linear(dim, hidden_dim, vb.pp("gate_proj"))?,
            up_proj: candle_nn::linear(dim, hidden_dim, vb.pp("up_proj"))?,
            down_proj: candle_nn::linear(hidden_dim, dim, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?.silu()?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ============================================================================
// Rotary Position Embedding (2D for Vision)
// ============================================================================

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

fn apply_rotary_pos_emb_vision(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let cos = cos.unsqueeze(D::Minus2)?;
    let sin = sin.unsqueeze(D::Minus2)?;

    let q_embed = (q.broadcast_mul(&cos)? + rotate_half(q)?.broadcast_mul(&sin)?)?;
    let k_embed = (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin)?)?;
    Ok((q_embed, k_embed))
}

struct VisionRotaryEmbedding {
    inv_freq: Tensor,
}

impl VisionRotaryEmbedding {
    const THETA: f32 = 10000.;

    fn new(dim: usize, device: &Device) -> Result<Self> {
        let inv_freq = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / Self::THETA.powf(i as f32 / dim as f32))
            .collect::<Vec<_>>();
        let inv_freq_len = inv_freq.len();
        Ok(Self {
            inv_freq: Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?,
        })
    }

    fn make_embeds(&self, seqlen: usize) -> Result<Tensor> {
        let seq =
            Tensor::arange(0f32, seqlen as f32, self.inv_freq.device())?.unsqueeze(D::Minus1)?;
        seq.broadcast_matmul(&self.inv_freq)
    }
}

// ============================================================================
// Vision Attention
// ============================================================================

struct VisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn new(dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            qkv: candle_nn::linear(dim, dim * 3, vb.pp("qkv"))?,
            proj: candle_nn::linear(dim, dim, vb.pp("proj"))?,
            num_heads,
            head_dim: dim / num_heads,
        })
    }

    /// Forward pass with chunked attention based on cu_seqlens.
    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let seq_len = xs.dim(0)?;
        let hidden_states = self.qkv.forward(xs)?;
        let qkv = hidden_states
            .reshape((seq_len, 3, self.num_heads, self.head_dim))?
            .permute((1, 0, 2, 3))?;
        let mut q = qkv.i(0)?.squeeze(0)?;
        let mut k = qkv.i(1)?.squeeze(0)?;
        let mut v = qkv.i(2)?.squeeze(0)?;

        // RoPE and attention in F32 for numerical precision
        let cos = cos.to_dtype(DType::F32)?;
        let sin = sin.to_dtype(DType::F32)?;
        q = q.to_dtype(DType::F32)?;
        k = k.to_dtype(DType::F32)?;
        v = v.to_dtype(DType::F32)?;
        (q, k) = apply_rotary_pos_emb_vision(&q, &k, &cos, &sin)?;

        let mut outputs = Vec::new();
        for window in cu_seqlens.windows(2) {
            let start = window[0];
            let end = window[1];
            if end <= start {
                continue;
            }
            let len = end - start;
            let q_chunk = q.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;
            let k_chunk = k.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;
            let v_chunk = v.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;

            let mut chunk_out = {
                let q = q_chunk.unsqueeze(0)?;
                let k = k_chunk.unsqueeze(0)?;
                let v = v_chunk.unsqueeze(0)?;

                let attn_weights =
                    (q.matmul(&k.transpose(2, 3)?)? / (self.head_dim as f64).sqrt())?;

                let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
                attn_weights.matmul(&v)?
            };
            chunk_out = chunk_out.squeeze(0)?.transpose(0, 1)?;

            chunk_out.device().synchronize()?;
            chunk_out = chunk_out.reshape((len, self.num_heads * self.head_dim))?;
            // Convert back to input dtype for output projection
            outputs.push(chunk_out.to_dtype(xs.dtype())?);
        }
        let attn_output = Tensor::cat(&outputs, 0)?;
        self.proj.forward(&attn_output)
    }
}

// ============================================================================
// Vision Block
// ============================================================================

/// Vision transformer block using RMSNorm (not LayerNorm).
struct VisionBlock {
    norm1: RmsNorm,
    norm2: RmsNorm,
    attn: VisionAttention,
    mlp: VisionMlp,
}

impl VisionBlock {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        // Qwen2.5-VL uses RMSNorm with eps=1e-6
        let norm1 = rms_norm(cfg.hidden_size, 1e-6, vb.pp("norm1"))?;
        let norm2 = rms_norm(cfg.hidden_size, 1e-6, vb.pp("norm2"))?;
        let attn = VisionAttention::new(cfg.hidden_size, cfg.num_heads, vb.pp("attn"))?;
        let mlp = VisionMlp::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"))?;
        Ok(Self {
            norm1,
            norm2,
            attn,
            mlp,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        // All operations in original dtype (BF16) to match weights
        // Only residual additions are done in F32 for numerical precision
        let normed = self.norm1.forward(xs)?;
        let attn_out = self.attn.forward(&normed, cu_seqlens, cos, sin)?;

        // Residual addition in F32 for precision
        let xs_att = xs
            .to_dtype(DType::F32)?
            .add(&attn_out.to_dtype(DType::F32)?)?
            .to_dtype(xs.dtype())?;

        let normed2 = self.norm2.forward(&xs_att)?;
        let mlp_out = self.mlp.forward(&normed2)?;

        // Residual addition in F32 for precision
        xs_att
            .to_dtype(DType::F32)?
            .add(&mlp_out.to_dtype(DType::F32)?)?
            .to_dtype(xs.dtype())
    }
}

// ============================================================================
// Patch Merger
// ============================================================================

/// Patch merger - 2×2 spatial merge + projection to text hidden dimension.
///
/// Weight structure:
/// - merger.ln_q.weight: RMSNorm on hidden_size (1280)
/// - merger.mlp.0.weight/bias: Linear from merged_size (5120) to merged_size (5120)
/// - merger.mlp.2.weight/bias: Linear from merged_size (5120) to out_hidden_size (2048)
struct PatchMerger {
    ln_q: RmsNorm,
    spatial_merge_unit: usize,
    merged_hidden_size: usize,
    mlp_0: Linear,
    mlp_2: Linear,
}

impl PatchMerger {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let merged_hidden_size = cfg.hidden_size * cfg.spatial_merge_size.pow(2);
        let mlp_vb = vb.pp("mlp");
        Ok(Self {
            ln_q: rms_norm(cfg.hidden_size, 1e-6, vb.pp("ln_q"))?,
            spatial_merge_unit: cfg.spatial_merge_size.pow(2),
            merged_hidden_size,
            mlp_0: candle_nn::linear(merged_hidden_size, merged_hidden_size, mlp_vb.pp("0"))?,
            mlp_2: candle_nn::linear(merged_hidden_size, cfg.out_hidden_size, mlp_vb.pp("2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let seq_len = xs.dim(0)?;
        if seq_len % self.spatial_merge_unit != 0 {
            candle::bail!(
                "Sequence length {} is not divisible by spatial merge unit {}",
                seq_len,
                self.spatial_merge_unit
            );
        }
        let grouped = seq_len / self.spatial_merge_unit;
        // Apply RMSNorm before spatial merge
        let normed = self.ln_q.forward(xs)?;
        // Spatially merge 4 adjacent patches into 1
        let reshaped = normed.reshape((grouped, self.merged_hidden_size))?;
        // Apply MLP: mlp.0 -> GELU -> mlp.2
        let xs = self.mlp_0.forward(&reshaped)?;
        let xs = xs.gelu()?;
        self.mlp_2.forward(&xs)
    }
}

// ============================================================================
// Main Vision Model
// ============================================================================

/// Qwen2.5-VL Vision Model.
///
/// A Vision Transformer with:
/// - 3D patch embedding (temporal + spatial, NO bias)
/// - 2D rotary position embeddings (NO learnable position embeddings)
/// - SwiGLU MLP and RMSNorm
/// - Patch merger for projection to text model dimension
///
/// Key differences from Qwen3-VL:
/// - No DeepStack injection
/// - No learnable position embeddings (relies entirely on 2D RoPE)
pub struct Qwen25VLVisionModel {
    patch_embed: PatchEmbed,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
    rotary_pos_emb: VisionRotaryEmbedding,
    spatial_merge_size: usize,
    spatial_merge_unit: usize,
    patch_size: usize,
    window_size: usize,
    fullatt_block_indexes: Vec<usize>,
}

impl Qwen25VLVisionModel {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embed = PatchEmbed::new(cfg, vb.pp("patch_embed"))?;

        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(VisionBlock::new(cfg, vb.pp(format!("blocks.{i}")))?);
        }

        let merger = PatchMerger::new(cfg, vb.pp("merger"))?;

        let head_dim = cfg.hidden_size / cfg.num_heads;
        let rotary_pos_emb = VisionRotaryEmbedding::new(head_dim / 2, vb.device())?;

        Ok(Self {
            patch_embed,
            blocks,
            merger,
            rotary_pos_emb,
            spatial_merge_size: cfg.spatial_merge_size,
            spatial_merge_unit: cfg.spatial_merge_size * cfg.spatial_merge_size,
            patch_size: cfg.patch_size,
            window_size: cfg.window_size,
            fullatt_block_indexes: cfg.fullatt_block_indexes.clone(),
        })
    }

    /// Compute rotary position embeddings based on grid.
    fn rot_pos_emb(&self, grid_thw: &Tensor, device: &Device) -> Result<Tensor> {
        let grid = grid_thw.to_vec2::<u32>()?;
        let max_hw = grid
            .iter()
            .flat_map(|v| v[1..3].iter())
            .copied()
            .max()
            .unwrap_or(0) as usize;
        let freq_table = self.rotary_pos_emb.make_embeds(max_hw)?;

        let mut coords: Vec<(i64, i64)> = Vec::new();
        for g in &grid {
            let h = g[1] as usize;
            let w = g[2] as usize;
            let merged_h = h / self.spatial_merge_size;
            let merged_w = w / self.spatial_merge_size;

            let mut base_coords: Vec<(i64, i64)> = Vec::with_capacity(h * w);
            for br in 0..merged_h {
                for bc in 0..merged_w {
                    for ir in 0..self.spatial_merge_size {
                        for ic in 0..self.spatial_merge_size {
                            base_coords.push((
                                (br * self.spatial_merge_size + ir) as i64,
                                (bc * self.spatial_merge_size + ic) as i64,
                            ));
                        }
                    }
                }
            }

            for _ in 0..(g[0] as usize) {
                coords.extend(base_coords.iter().cloned());
            }
        }

        let total_tokens = coords.len();
        let mut rows = Vec::with_capacity(total_tokens);
        let mut cols = Vec::with_capacity(total_tokens);
        for &(r, c) in &coords {
            rows.push(r);
            cols.push(c);
        }
        let rows = Tensor::from_vec(rows, (total_tokens,), device)?;
        let cols = Tensor::from_vec(cols, (total_tokens,), device)?;
        let row_embeds = freq_table.index_select(&rows, 0)?;
        let col_embeds = freq_table.index_select(&cols, 0)?;
        Tensor::stack(&[row_embeds, col_embeds], D::Minus2)?
            .reshape((total_tokens, freq_table.dim(D::Minus1)? * 2))
    }

    /// Build cumulative sequence lengths for chunked attention.
    fn build_cu_seqlens(&self, grid_thw: &Tensor) -> Result<Vec<usize>> {
        let grid = grid_thw.to_vec2::<u32>()?;
        let mut cu = Vec::with_capacity(grid.iter().map(|v| v[0] as usize).sum::<usize>() + 1);
        cu.push(0usize);
        let mut acc = 0usize;
        for g in &grid {
            let area = (g[1] * g[2]) as usize;
            for _ in 0..(g[0] as usize) {
                acc += area;
                cu.push(acc);
            }
        }
        Ok(cu)
    }

    /// Compute window indices for window attention.
    ///
    /// This reorders patches into spatial windows for efficient local attention.
    /// Returns (window_index, cu_window_seqlens) where:
    /// - window_index: indices to reorder patches into window groups
    /// - cu_window_seqlens: cumulative sequence lengths for each window
    ///
    /// The window size in terms of merged patches is:
    /// `vit_merger_window_size = window_size / spatial_merge_size / patch_size`
    /// For default config (112 / 2 / 14 = 4): each window is 4×4 merged patches.
    fn get_window_index(&self, grid_thw: &Tensor) -> Result<(Vec<usize>, Vec<usize>)> {
        let grid = grid_thw.to_vec2::<u32>()?;
        let vit_merger_window_size = self.window_size / self.spatial_merge_size / self.patch_size;

        let mut window_index: Vec<usize> = Vec::new();
        let mut cu_window_seqlens: Vec<usize> = vec![0];
        let mut window_index_id: usize = 0;

        for g in &grid {
            let grid_t = g[0] as usize;
            let grid_h = g[1] as usize;
            let grid_w = g[2] as usize;

            // LLM grid size (after spatial merge)
            let llm_grid_h = grid_h / self.spatial_merge_size;
            let llm_grid_w = grid_w / self.spatial_merge_size;

            // Compute padding needed to make grid divisible by window size
            let pad_h = (vit_merger_window_size - llm_grid_h % vit_merger_window_size)
                % vit_merger_window_size;
            let pad_w = (vit_merger_window_size - llm_grid_w % vit_merger_window_size)
                % vit_merger_window_size;
            let num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
            let num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;

            // For each temporal frame
            for _t in 0..grid_t {
                // Create index grid with padding (-1 indicates padding)
                let mut index_padded =
                    vec![vec![-1i64; llm_grid_w + pad_w]; llm_grid_h + pad_h];

                // Fill in actual indices
                for (h, row) in index_padded.iter_mut().enumerate().take(llm_grid_h) {
                    for (w, cell) in row.iter_mut().enumerate().take(llm_grid_w) {
                        *cell = (window_index_id + h * llm_grid_w + w) as i64;
                    }
                }

                // Reshape into windows and extract indices
                for wh in 0..num_windows_h {
                    for ww in 0..num_windows_w {
                        let mut window_indices: Vec<usize> = Vec::new();
                        for ih in 0..vit_merger_window_size {
                            for iw in 0..vit_merger_window_size {
                                let h = wh * vit_merger_window_size + ih;
                                let w = ww * vit_merger_window_size + iw;
                                let idx = index_padded[h][w];
                                if idx >= 0 {
                                    window_indices.push(idx as usize);
                                }
                            }
                        }
                        // Add window to output
                        if !window_indices.is_empty() {
                            let seqlen = window_indices.len() * self.spatial_merge_unit;
                            window_index.extend(window_indices);
                            cu_window_seqlens
                                .push(cu_window_seqlens.last().unwrap() + seqlen);
                        }
                    }
                }
                window_index_id += llm_grid_h * llm_grid_w;
            }
        }

        Ok((window_index, cu_window_seqlens))
    }

    /// Forward pass through the vision encoder.
    ///
    /// # Arguments
    /// * `xs` - Pixel values tensor, shape (num_patches, in_channels * temporal * patch * patch)
    /// * `grid_thw` - Grid dimensions tensor, shape (num_images, 3) with [temporal, height, width]
    ///
    /// # Returns
    /// Vision embeddings projected to text model dimension.
    pub fn forward(&self, xs: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        let device = xs.device();
        // Patch embedding (no position embedding - Qwen2.5-VL relies entirely on RoPE)
        let hidden_states = self.patch_embed.forward(xs)?;
        debug_tensor("vision_after_patch_embed", &hidden_states);

        // Compute 2D rotary position embeddings
        let rotary_pos_emb = self.rot_pos_emb(grid_thw, device)?;
        debug_tensor("vision_rotary_pos_emb", &rotary_pos_emb);

        // Get window index for reordering patches into spatial windows
        let (window_index, cu_window_seqlens) = self.get_window_index(grid_thw)?;

        // Reorder hidden states by window index (operate on spatial_merge_unit groups)
        let seq_len = hidden_states.dim(0)?;
        let hidden_dim = hidden_states.dim(1)?;
        let grouped = seq_len / self.spatial_merge_unit;

        // Reshape to (grouped, spatial_merge_unit, hidden_dim)
        let hidden_states = hidden_states.reshape((grouped, self.spatial_merge_unit, hidden_dim))?;

        // Reorder using window_index
        let window_index_u32: Vec<u32> = window_index.iter().map(|&x| x as u32).collect();
        let window_index_tensor =
            Tensor::from_vec(window_index_u32, (window_index.len(),), device)?;
        // NOTE: index_select may produce non-contiguous tensor, force contiguity for downstream ops
        let mut hidden_states = hidden_states.index_select(&window_index_tensor, 0)?.contiguous()?;

        // Reshape back to (seq_len, hidden_dim)
        hidden_states = hidden_states.reshape((seq_len, hidden_dim))?;
        debug_tensor("vision_after_window_reorder", &hidden_states);

        // Reorder rotary position embeddings similarly
        let rotary_pos_emb = rotary_pos_emb.reshape((seq_len, ()))?;
        let rotary_pos_emb = rotary_pos_emb.reshape((grouped, self.spatial_merge_unit, ()))?;
        let rotary_pos_emb = rotary_pos_emb.index_select(&window_index_tensor, 0)?.contiguous()?;
        let rotary_pos_emb = rotary_pos_emb.reshape((seq_len, ()))?;

        // Compute cos/sin for RoPE
        let emb = Tensor::cat(&[&rotary_pos_emb, &rotary_pos_emb], D::Minus1)?;
        let cos = emb.cos()?.to_dtype(DType::F32)?;
        let sin = emb.sin()?.to_dtype(DType::F32)?;
        debug_tensor("vision_cos", &cos);
        debug_tensor("vision_sin", &sin);

        // Build cu_seqlens for full attention (image-level chunking)
        let cu_seqlens = self.build_cu_seqlens(grid_thw)?;

        // Process through all blocks with layer-specific attention
        // Layers in fullatt_block_indexes use full attention (cu_seqlens)
        // Other layers use window attention (cu_window_seqlens)
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let cu_seqlens_now = if self.fullatt_block_indexes.contains(&layer_idx) {
                &cu_seqlens // Full attention for layers 7, 15, 23, 31
            } else {
                &cu_window_seqlens // Window attention for other 28 layers
            };
            hidden_states = block.forward(&hidden_states, cu_seqlens_now, &cos, &sin)?;
            // Debug after key blocks: 0 (first), 7 (first full-attn), 15, 23, 31 (last full-attn)
            if layer_idx == 0 || layer_idx == 7 || layer_idx == 15 || layer_idx == 23 || layer_idx == 31 {
                debug_tensor(&format!("vision_after_block_{}", layer_idx), &hidden_states);
            }
        }
        debug_tensor("vision_after_all_blocks", &hidden_states);

        // Apply merger to get final embeddings
        let hidden_states = self.merger.forward(&hidden_states)?;
        debug_tensor("vision_after_merger", &hidden_states);

        // Reverse the window reordering to restore original patch order
        // Compute argsort of window_index to get reverse indices
        let mut reverse_indices: Vec<u32> = (0..window_index.len() as u32).collect();
        reverse_indices.sort_by_key(|&i| window_index[i as usize]);
        let reverse_index_tensor =
            Tensor::from_vec(reverse_indices, (window_index.len(),), device)?;

        let output = hidden_states.index_select(&reverse_index_tensor, 0)?.contiguous()?;
        debug_tensor("vision_final_output", &output);
        Ok(output)
    }
}
