//! Debug utilities for Qwen-Image transformer.
//!
//! This module provides tensor debugging utilities that can be enabled
//! via the `QWEN_DEBUG` environment variable.
//!
//! # Usage
//!
//! Set `QWEN_DEBUG=1` to enable detailed tensor statistics at key checkpoints.

use candle::{DType, Result, Tensor};
use std::sync::atomic::{AtomicBool, Ordering};

/// Global flag for debug mode (checked once at startup).
static DEBUG_MODE: AtomicBool = AtomicBool::new(false);
static DEBUG_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Check if debug mode is enabled (via QWEN_DEBUG environment variable).
pub fn is_debug_mode() -> bool {
    if !DEBUG_INITIALIZED.load(Ordering::Relaxed) {
        let enabled = std::env::var("QWEN_DEBUG").is_ok();
        DEBUG_MODE.store(enabled, Ordering::Relaxed);
        DEBUG_INITIALIZED.store(true, Ordering::Relaxed);
    }
    DEBUG_MODE.load(Ordering::Relaxed)
}

/// Compute tensor statistics for debugging.
pub fn tensor_stats(t: &Tensor) -> Result<(f32, f32, f32, f32)> {
    let t_f32 = t.to_dtype(DType::F32)?.flatten_all()?;
    let mean = t_f32.mean_all()?.to_scalar::<f32>()?;
    let diff = t_f32.broadcast_sub(&t_f32.mean_all()?)?;
    let var = (&diff * &diff)?.mean_all()?.to_scalar::<f32>()?;
    let std = var.sqrt();
    let min = t_f32.min(0)?.to_scalar::<f32>()?;
    let max = t_f32.max(0)?.to_scalar::<f32>()?;
    Ok((mean, std, min, max))
}

/// Print tensor statistics for debugging.
///
/// Only prints if `QWEN_DEBUG=1` is set.
pub fn debug_tensor(name: &str, t: &Tensor) {
    if !is_debug_mode() {
        return;
    }

    match tensor_stats(t) {
        Ok((mean, std, min, max)) => {
            println!(
                "[TRANSFORMER] {}: shape={:?}, mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
                name,
                t.dims(),
                mean,
                std,
                min,
                max
            );
        }
        Err(e) => {
            println!(
                "[TRANSFORMER] {}: shape={:?}, stats error: {}",
                name,
                t.dims(),
                e
            );
        }
    }
}

/// Print a debug message (only if debug mode is enabled).
pub fn debug_print(msg: &str) {
    if is_debug_mode() {
        println!("[TRANSFORMER] {}", msg);
    }
}

/// Check if VAE debug mode is enabled (via VAE_DEBUG environment variable).
pub fn is_vae_debug() -> bool {
    std::env::var("VAE_DEBUG").is_ok()
}

/// Format tensor statistics as a string.
pub fn tensor_stats_string(t: &Tensor) -> Result<String> {
    let (mean, std, min, max) = tensor_stats(t)?;
    Ok(format!(
        "shape={:?}, mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
        t.dims(),
        mean,
        std,
        min,
        max
    ))
}

/// Check if VAE tensor saving is enabled (via VAE_SAVE environment variable).
pub fn is_vae_save() -> bool {
    std::env::var("VAE_SAVE").is_ok()
}

/// Print VAE tensor statistics (only if VAE_DEBUG is set).
/// Also saves tensors to debug_tensors/rust_edit/ if VAE_SAVE is set.
pub fn debug_vae_tensor(name: &str, t: &Tensor) {
    if !is_vae_debug() {
        return;
    }

    match tensor_stats_string(t) {
        Ok(stats) => {
            eprintln!("[VAE] {}: {}", name, stats);
        }
        Err(e) => {
            eprintln!("[VAE] {}: shape={:?}, stats error: {}", name, t.dims(), e);
        }
    }

    // Save tensor if VAE_SAVE is enabled
    if is_vae_save() {
        let output_dir = "debug_tensors/rust_edit";
        if let Err(e) = std::fs::create_dir_all(output_dir) {
            eprintln!("[VAE_SAVE] Failed to create dir: {}", e);
            return;
        }
        let path = format!("{}/vae_{}.npy", output_dir, name);
        if let Err(e) = t.write_npy(&path) {
            eprintln!("[VAE_SAVE] Failed to save {}: {}", name, e);
        } else {
            eprintln!("[VAE_SAVE] Saved {}", path);
        }
    }
}

/// Debug output for apply_modulation_with_index.
///
/// This traces the per-token modulation selection to help debug edit mode issues.
pub fn debug_modulation_with_index(
    xs: &Tensor,
    shift_all: &Tensor,
    scale_all: &Tensor,
    modulate_index: &Tensor,
    shift_0: &Tensor,
    shift_1: &Tensor,
    scale_0: &Tensor,
    scale_1: &Tensor,
    shift_result: &Tensor,
    scale_result: &Tensor,
    modulated: &Tensor,
) -> Result<()> {
    eprintln!("[MODULATE_INDEX] xs shape: {:?}, dtype: {:?}", xs.dims(), xs.dtype());
    eprintln!("[MODULATE_INDEX] shift_all shape: {:?}", shift_all.dims());
    eprintln!("[MODULATE_INDEX] modulate_index shape: {:?}", modulate_index.dims());
    eprintln!("[MODULATE_INDEX] batch_size from xs: {}", xs.dim(0)?);

    // Print some sample values from each half
    let shift_0_flat = shift_0.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let shift_1_flat = shift_1.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let scale_0_flat = scale_0.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let scale_1_flat = scale_1.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;

    eprintln!("[MODULATE_INDEX] shift_0[0..5]: {:?}", &shift_0_flat[0..5.min(shift_0_flat.len())]);
    eprintln!("[MODULATE_INDEX] shift_1[0..5]: {:?}", &shift_1_flat[0..5.min(shift_1_flat.len())]);
    eprintln!("[MODULATE_INDEX] scale_0[0..5]: {:?}", &scale_0_flat[0..5.min(scale_0_flat.len())]);
    eprintln!("[MODULATE_INDEX] scale_1[0..5]: {:?}", &scale_1_flat[0..5.min(scale_1_flat.len())]);

    // Print modulate_index values
    let idx_flat = modulate_index.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let num_zeros = idx_flat.iter().filter(|&&x| x == 0.0).count();
    let num_ones = idx_flat.iter().filter(|&&x| x == 1.0).count();
    eprintln!("[MODULATE_INDEX] index has {} zeros, {} ones, total {}", num_zeros, num_ones, idx_flat.len());
    eprintln!("[MODULATE_INDEX] index[0..10]: {:?}", &idx_flat[0..10.min(idx_flat.len())]);
    if idx_flat.len() > 100 {
        eprintln!("[MODULATE_INDEX] index[last 10]: {:?}", &idx_flat[idx_flat.len()-10..]);
    }

    // Check that selection worked correctly:
    // Token 0 should use shift_0, last token should use shift_1
    let shift_result_flat = shift_result.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let seq_len = modulate_index.dim(1)?;
    let dim = shift_0.dim(2)?;

    // First token's first few features (should match shift_0 since index=0)
    eprintln!("[MODULATE_INDEX] shift_result[token0, 0..5]: {:?}", &shift_result_flat[0..5.min(shift_result_flat.len())]);

    // Last token's first few features (should match shift_1 since index=1)
    let last_token_start = (seq_len - 1) * dim;
    if last_token_start + 5 <= shift_result_flat.len() {
        eprintln!("[MODULATE_INDEX] shift_result[last_token, 0..5]: {:?}", &shift_result_flat[last_token_start..last_token_start+5]);
    }

    eprintln!("[MODULATE_INDEX] shift_result shape: {:?}", shift_result.dims());
    eprintln!("[MODULATE_INDEX] scale_result shape: {:?}", scale_result.dims());

    let modulated_flat = modulated.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    eprintln!("[MODULATE_INDEX] modulated[0..5]: {:?}", &modulated_flat[0..5.min(modulated_flat.len())]);
    eprintln!("[MODULATE_INDEX] modulated shape: {:?}", modulated.dims());

    Ok(())
}

// ============================================================================
// Block Override Loading for Debugging
// ============================================================================

use candle::Device;
use std::path::Path;

/// Optional tensor overrides for debugging block internals.
///
/// Used to substitute intermediate tensors with PyTorch reference values
/// to isolate where divergence occurs in the transformer blocks.
///
/// # Supported Overrides
///
/// **Block-level:**
/// - `img_modulated`: Override the modulated image tensor before attention
/// - `img_gate2`: Override the MLP gate tensor
///
/// **Attention-level (Q/K/V projections):**
/// - `img_q`: Override image stream Q projection output
/// - `img_k`: Override image stream K projection output
/// - `img_v`: Override image stream V projection output
/// - `txt_q`: Override text stream Q projection output
/// - `txt_k`: Override text stream K projection output
/// - `txt_v`: Override text stream V projection output
///
/// **Attention internals:**
/// - `attn_weights`: Override Q@K.T/sqrt(d) scores [batch, heads, seq, seq]
/// - `attn_probs`: Override softmax output [batch, heads, seq, seq]
/// - `attn_output`: Override attention output before split [batch, seq, heads, head_dim]
#[derive(Debug, Clone, Default)]
pub struct BlockOverrides {
    /// Override img_modulated (input to attention after modulation)
    pub img_modulated: Option<Tensor>,
    /// Override img_gate2 (MLP gate for image stream)
    pub img_gate2: Option<Tensor>,
    /// Override image stream Q projection output [batch, seq, heads * head_dim]
    pub img_q: Option<Tensor>,
    /// Override image stream K projection output [batch, seq, heads * head_dim]
    pub img_k: Option<Tensor>,
    /// Override image stream V projection output [batch, seq, heads * head_dim]
    pub img_v: Option<Tensor>,
    /// Override text stream Q projection output [batch, seq, heads * head_dim]
    pub txt_q: Option<Tensor>,
    /// Override text stream K projection output [batch, seq, heads * head_dim]
    pub txt_k: Option<Tensor>,
    /// Override text stream V projection output [batch, seq, heads * head_dim]
    pub txt_v: Option<Tensor>,
    /// Override attention weights (Q@K.T/sqrt(d)) [batch, heads, seq, seq]
    pub attn_weights: Option<Tensor>,
    /// Override attention probs (softmax output) [batch, heads, seq, seq]
    pub attn_probs: Option<Tensor>,
    /// Override attention output before split [batch, seq, heads, head_dim]
    pub attn_output: Option<Tensor>,
}

impl BlockOverrides {
    /// Check if any Q/K/V overrides are set.
    pub fn has_qkv_overrides(&self) -> bool {
        self.img_q.is_some()
            || self.img_k.is_some()
            || self.img_v.is_some()
            || self.txt_q.is_some()
            || self.txt_k.is_some()
            || self.txt_v.is_some()
    }

    /// Check if any attention internal overrides are set.
    pub fn has_attn_overrides(&self) -> bool {
        self.attn_weights.is_some()
            || self.attn_probs.is_some()
            || self.attn_output.is_some()
    }

    /// Check if any override is set.
    pub fn is_empty(&self) -> bool {
        self.img_modulated.is_none()
            && self.img_gate2.is_none()
            && !self.has_qkv_overrides()
            && !self.has_attn_overrides()
    }
}

/// Environment variable to enable block override substitution.
/// Set to a comma-separated list of tensors to substitute, e.g.:
/// `QWEN_BLOCK_OVERRIDE=img_modulated,img_gate2`
/// `QWEN_BLOCK_OVERRIDE=img_q,img_k,img_v` (for Q/K/V debugging)
static BLOCK_OVERRIDE_VAR: &str = "QWEN_BLOCK_OVERRIDE";

/// Directory containing PyTorch reference tensors.
static PYTORCH_TENSOR_DIR: &str = "debug_tensors/pytorch_edit";

/// Thread-local CFG pass counter for loading appropriate block overrides.
/// 0 = positive prompt pass, 1 = negative prompt pass
thread_local! {
    static CFG_PASS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Set the current CFG pass for block override loading.
/// Call this before each transformer call with 0 for positive, 1 for negative.
pub fn set_cfg_pass(pass: usize) {
    CFG_PASS.with(|p| p.set(pass));
    eprintln!("[DEBUG] CFG pass set to {} ({})", pass, if pass == 0 { "positive" } else { "negative" });
}

/// Get the current CFG pass suffix for loading block overrides.
fn get_cfg_pass_suffix() -> &'static str {
    CFG_PASS.with(|p| {
        if p.get() == 0 { "_pos" } else { "_neg" }
    })
}

/// Check if block overrides are enabled via environment variable.
pub fn is_block_override_enabled() -> bool {
    std::env::var(BLOCK_OVERRIDE_VAR).is_ok()
}

/// Load a tensor from a NumPy .npy file.
fn load_npy_tensor(path: &Path, device: &Device, dtype: DType) -> Result<Tensor> {
    // read_npy takes a path directly
    let tensor = Tensor::read_npy(path)?;

    // Convert to target dtype and device
    tensor.to_dtype(dtype)?.to_device(device)
}

/// Load block overrides from PyTorch reference tensors if enabled.
///
/// Checks `QWEN_BLOCK_OVERRIDE` environment variable for which tensors to substitute:
/// - `img_modulated`: Substitute the modulated image tensor before attention
/// - `img_gate2`: Substitute the MLP gate tensor
/// - `img_q`, `img_k`, `img_v`: Substitute image stream Q/K/V projections
/// - `txt_q`, `txt_k`, `txt_v`: Substitute text stream Q/K/V projections
///
/// Example: `QWEN_BLOCK_OVERRIDE=img_modulated`
/// Example: `QWEN_BLOCK_OVERRIDE=img_q,img_k,img_v`
///
/// The `pass_suffix` parameter allows loading CFG-pass-specific tensors:
/// - `Some("_pos")` loads tensors from positive prompt pass (e.g., `block0_internal_attn_weights_pos.npy`)
/// - `Some("_neg")` loads tensors from negative prompt pass (e.g., `block0_internal_attn_weights_neg.npy`)
/// - `None` loads from unsuffixed files (backwards compatibility)
pub fn load_block_overrides(device: &Device, dtype: DType, pass_suffix: Option<&str>) -> Result<Option<BlockOverrides>> {
    let override_var = match std::env::var(BLOCK_OVERRIDE_VAR) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };

    let override_names: Vec<&str> = override_var.split(',').map(|s| s.trim()).collect();

    if override_names.is_empty() {
        return Ok(None);
    }

    // Use provided suffix, or fall back to thread-local CFG pass suffix
    let suffix = pass_suffix.unwrap_or_else(|| get_cfg_pass_suffix());
    eprintln!("[DEBUG] Loading block overrides{}: {:?}", suffix, override_names);

    let mut overrides = BlockOverrides::default();
    let tensor_dir = Path::new(PYTORCH_TENSOR_DIR);

    // Helper to construct filename with optional suffix
    let make_filename = |base: &str| -> String {
        format!("{}{}.npy", base, suffix)
    };

    for name in &override_names {
        match *name {
            "img_modulated" => {
                // Load from block0_internal_img_mod1_modulated{suffix}.npy
                let path = tensor_dir.join(make_filename("block0_internal_img_mod1_modulated"));
                if path.exists() {
                    let tensor = load_npy_tensor(&path, device, dtype)?;
                    eprintln!("[DEBUG] Loaded img_modulated{} override: shape={:?}", suffix, tensor.dims());
                    overrides.img_modulated = Some(tensor);
                } else {
                    eprintln!("[DEBUG] Warning: {} not found, skipping img_modulated override", path.display());
                }
            }
            "img_gate2" => {
                // Load from block0_internal_img_mod2_gate_result{suffix}.npy
                let path = tensor_dir.join(make_filename("block0_internal_img_mod2_gate_result"));
                if path.exists() {
                    let tensor = load_npy_tensor(&path, device, dtype)?;
                    eprintln!("[DEBUG] Loaded img_gate2{} override: shape={:?}", suffix, tensor.dims());
                    overrides.img_gate2 = Some(tensor);
                } else {
                    eprintln!("[DEBUG] Warning: {} not found, skipping img_gate2 override", path.display());
                }
            }
            // Q/K/V overrides for image stream
            "img_q" => {
                let path = tensor_dir.join(make_filename("block0_internal_img_q_proj"));
                if path.exists() {
                    let tensor = load_npy_tensor(&path, device, dtype)?;
                    eprintln!("[DEBUG] Loaded img_q{} override: shape={:?}", suffix, tensor.dims());
                    overrides.img_q = Some(tensor);
                } else {
                    eprintln!("[DEBUG] Warning: {} not found, skipping img_q override", path.display());
                }
            }
            "img_k" => {
                let path = tensor_dir.join(make_filename("block0_internal_img_k_proj"));
                if path.exists() {
                    let tensor = load_npy_tensor(&path, device, dtype)?;
                    eprintln!("[DEBUG] Loaded img_k{} override: shape={:?}", suffix, tensor.dims());
                    overrides.img_k = Some(tensor);
                } else {
                    eprintln!("[DEBUG] Warning: {} not found, skipping img_k override", path.display());
                }
            }
            "img_v" => {
                let path = tensor_dir.join(make_filename("block0_internal_img_v_proj"));
                if path.exists() {
                    let tensor = load_npy_tensor(&path, device, dtype)?;
                    eprintln!("[DEBUG] Loaded img_v{} override: shape={:?}", suffix, tensor.dims());
                    overrides.img_v = Some(tensor);
                } else {
                    eprintln!("[DEBUG] Warning: {} not found, skipping img_v override", path.display());
                }
            }
            // Q/K/V overrides for text stream
            "txt_q" => {
                let path = tensor_dir.join(make_filename("block0_internal_txt_q_proj"));
                if path.exists() {
                    let tensor = load_npy_tensor(&path, device, dtype)?;
                    eprintln!("[DEBUG] Loaded txt_q{} override: shape={:?}", suffix, tensor.dims());
                    overrides.txt_q = Some(tensor);
                } else {
                    eprintln!("[DEBUG] Warning: {} not found, skipping txt_q override", path.display());
                }
            }
            "txt_k" => {
                let path = tensor_dir.join(make_filename("block0_internal_txt_k_proj"));
                if path.exists() {
                    let tensor = load_npy_tensor(&path, device, dtype)?;
                    eprintln!("[DEBUG] Loaded txt_k{} override: shape={:?}", suffix, tensor.dims());
                    overrides.txt_k = Some(tensor);
                } else {
                    eprintln!("[DEBUG] Warning: {} not found, skipping txt_k override", path.display());
                }
            }
            "txt_v" => {
                let path = tensor_dir.join(make_filename("block0_internal_txt_v_proj"));
                if path.exists() {
                    let tensor = load_npy_tensor(&path, device, dtype)?;
                    eprintln!("[DEBUG] Loaded txt_v{} override: shape={:?}", suffix, tensor.dims());
                    overrides.txt_v = Some(tensor);
                } else {
                    eprintln!("[DEBUG] Warning: {} not found, skipping txt_v override", path.display());
                }
            }
            // Attention internal overrides
            "attn_weights" => {
                let path = tensor_dir.join(make_filename("block0_internal_attn_weights"));
                if path.exists() {
                    let tensor = load_npy_tensor(&path, device, dtype)?;
                    eprintln!("[DEBUG] Loaded attn_weights{} override: shape={:?}", suffix, tensor.dims());
                    overrides.attn_weights = Some(tensor);
                } else {
                    eprintln!("[DEBUG] Warning: {} not found, skipping attn_weights override", path.display());
                }
            }
            "attn_probs" => {
                let path = tensor_dir.join(make_filename("block0_internal_attn_probs"));
                if path.exists() {
                    let tensor = load_npy_tensor(&path, device, dtype)?;
                    eprintln!("[DEBUG] Loaded attn_probs{} override: shape={:?}", suffix, tensor.dims());
                    overrides.attn_probs = Some(tensor);
                } else {
                    eprintln!("[DEBUG] Warning: {} not found, skipping attn_probs override", path.display());
                }
            }
            "attn_output" => {
                let path = tensor_dir.join(make_filename("block0_internal_attn_output_pre_split"));
                if path.exists() {
                    let tensor = load_npy_tensor(&path, device, dtype)?;
                    eprintln!("[DEBUG] Loaded attn_output{} override: shape={:?}", suffix, tensor.dims());
                    overrides.attn_output = Some(tensor);
                } else {
                    eprintln!("[DEBUG] Warning: {} not found, skipping attn_output override", path.display());
                }
            }
            other => {
                eprintln!("[DEBUG] Warning: Unknown override '{}', ignoring", other);
            }
        }
    }

    // Return None if no overrides were actually loaded
    if overrides.is_empty() {
        return Ok(None);
    }

    Ok(Some(overrides))
}
