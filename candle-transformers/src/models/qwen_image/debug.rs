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

// ============================================================================
// Attention Debug Functions for CFG Bug Investigation
// ============================================================================

/// Check if attention debug mode is enabled (via QWEN_DEBUG_ATTENTION environment variable).
pub fn is_attention_debug() -> bool {
    std::env::var("QWEN_DEBUG_ATTENTION").is_ok()
}

/// Debug attention internals to isolate where divergence occurs.
///
/// This function captures and compares tensors at each step of the attention computation:
/// 1. Q@K.T (raw matmul output)
/// 2. attn_weights (after scaling by 1/sqrt(head_dim))
/// 3. attn_probs (after softmax)
///
/// By comparing with PyTorch reference at each step, we can identify exactly where
/// the Rust implementation diverges.
///
/// # Arguments
/// * `name` - Identifier for this attention (e.g., "block0")
/// * `attn_logits` - Q@K.T raw output before scaling
/// * `attn_weights` - Scaled attention weights before softmax
/// * `attn_probs` - Attention probabilities after softmax
/// * `pytorch_dir` - Optional path to directory containing PyTorch reference tensors
pub fn debug_attention_internals(
    name: &str,
    attn_logits: &Tensor,
    attn_weights: &Tensor,
    attn_probs: &Tensor,
    pytorch_dir: Option<&str>,
) -> Result<()> {
    if !is_attention_debug() {
        return Ok(());
    }

    let suffix = get_cfg_pass_suffix();
    let pass_name = if suffix == "_pos" { "POSITIVE" } else { "NEGATIVE" };

    eprintln!("\n[ATTN_DEBUG] {} {} pass attention internals:", name, pass_name);
    eprintln!("[ATTN_DEBUG] ─────────────────────────────────────────────");

    // Compute stats for each stage
    let logits_stats = tensor_stats(attn_logits)?;
    let weights_stats = tensor_stats(attn_weights)?;
    let probs_stats = tensor_stats(attn_probs)?;

    eprintln!("[ATTN_DEBUG] Q@K.T (raw):     mean={:10.4}, std={:10.4}, min={:10.4}, max={:10.4}",
        logits_stats.0, logits_stats.1, logits_stats.2, logits_stats.3);
    eprintln!("[ATTN_DEBUG] scaled weights:  mean={:10.4}, std={:10.4}, min={:10.4}, max={:10.4}",
        weights_stats.0, weights_stats.1, weights_stats.2, weights_stats.3);
    eprintln!("[ATTN_DEBUG] softmax probs:   mean={:10.4}, std={:10.4}, min={:10.4}, max={:10.4}",
        probs_stats.0, probs_stats.1, probs_stats.2, probs_stats.3);

    // Load and compare with PyTorch if available
    if let Some(dir) = pytorch_dir {
        let tensor_dir = Path::new(dir);

        // Try to load PyTorch reference tensors
        let logits_path = tensor_dir.join(format!("{}_attn_logits{}.npy", name, suffix));
        let weights_path = tensor_dir.join(format!("{}_attn_weights{}.npy", name, suffix));
        let probs_path = tensor_dir.join(format!("{}_attn_probs{}.npy", name, suffix));

        if logits_path.exists() {
            let py_logits = Tensor::read_npy(&logits_path)?;
            let diff = compare_tensors(attn_logits, &py_logits)?;
            eprintln!("[ATTN_DEBUG] Q@K.T vs PyTorch: max_diff={:.6}, mean_diff={:.6} {}",
                diff.0, diff.1, if diff.0 < 0.01 { "✅" } else { "⚠️" });
        }

        if weights_path.exists() {
            let py_weights = Tensor::read_npy(&weights_path)?;
            let diff = compare_tensors(attn_weights, &py_weights)?;
            eprintln!("[ATTN_DEBUG] scaled vs PyTorch: max_diff={:.6}, mean_diff={:.6} {}",
                diff.0, diff.1, if diff.0 < 0.01 { "✅" } else { "⚠️" });
        }

        if probs_path.exists() {
            let py_probs = Tensor::read_npy(&probs_path)?;
            let diff = compare_tensors(attn_probs, &py_probs)?;
            eprintln!("[ATTN_DEBUG] softmax vs PyTorch: max_diff={:.6}, mean_diff={:.6} {}",
                diff.0, diff.1, if diff.0 < 0.01 { "✅" } else { "❌ BUG HERE?" });
        }
    }

    eprintln!("[ATTN_DEBUG] ─────────────────────────────────────────────\n");
    Ok(())
}

/// Compare two tensors and return (max_diff, mean_diff).
fn compare_tensors(a: &Tensor, b: &Tensor) -> Result<(f32, f32)> {
    let a_f32 = a.to_dtype(DType::F32)?.flatten_all()?;
    let b_f32 = b.to_dtype(DType::F32)?.flatten_all()?;

    // Handle shape mismatch gracefully
    if a_f32.elem_count() != b_f32.elem_count() {
        return Ok((f32::INFINITY, f32::INFINITY));
    }

    let diff = (&a_f32 - &b_f32)?.abs()?;
    let max_diff = diff.max(0)?.to_scalar::<f32>()?;
    let mean_diff = diff.mean_all()?.to_scalar::<f32>()?;
    Ok((max_diff, mean_diff))
}

/// Analyze attention probabilities per head to identify which heads diverge most.
///
/// # Arguments
/// * `attn_probs` - Attention probabilities [batch, heads, seq, seq]
/// * `name` - Identifier for logging
pub fn debug_per_head_stats(
    attn_probs: &Tensor,
    name: &str,
) -> Result<()> {
    if !is_attention_debug() {
        return Ok(());
    }

    let suffix = get_cfg_pass_suffix();
    let pass_name = if suffix == "_pos" { "POSITIVE" } else { "NEGATIVE" };

    eprintln!("\n[HEAD_DEBUG] {} {} pass - Per-head attention analysis:", name, pass_name);
    eprintln!("[HEAD_DEBUG] Shape: {:?}", attn_probs.dims());

    let dims = attn_probs.dims();
    if dims.len() != 4 {
        eprintln!("[HEAD_DEBUG] Warning: Expected 4D tensor [batch, heads, seq, seq], got {:?}", dims);
        return Ok(());
    }

    let num_heads = dims[1];
    let probs_f32 = attn_probs.to_dtype(DType::F32)?;

    eprintln!("[HEAD_DEBUG] ┌────────┬───────────┬───────────┬───────────┬───────────┐");
    eprintln!("[HEAD_DEBUG] │  Head  │   Mean    │   Std     │   Min     │   Max     │");
    eprintln!("[HEAD_DEBUG] ├────────┼───────────┼───────────┼───────────┼───────────┤");

    let mut heads_with_max_1 = 0;
    let mut total_max = 0.0f32;

    for h in 0..num_heads {
        // Extract attention for this head: [batch, seq, seq]
        let head_attn = probs_f32.narrow(1, h, 1)?.squeeze(1)?;
        let stats = tensor_stats(&head_attn)?;

        // Check if any attention weight is close to 1.0 (focused attention)
        if stats.3 > 0.99 {
            heads_with_max_1 += 1;
        }
        total_max += stats.3;

        let max_indicator = if stats.3 > 0.99 { "★" } else if stats.3 > 0.9 { "●" } else { " " };
        eprintln!("[HEAD_DEBUG] │ {:>4}   │ {:>9.6} │ {:>9.6} │ {:>9.6} │ {:>9.6} │ {}",
            h, stats.0, stats.1, stats.2, stats.3, max_indicator);
    }

    eprintln!("[HEAD_DEBUG] └────────┴───────────┴───────────┴───────────┴───────────┘");
    eprintln!("[HEAD_DEBUG] Summary: {}/{} heads have max ≈ 1.0 (focused attention)",
        heads_with_max_1, num_heads);
    eprintln!("[HEAD_DEBUG] Average max across heads: {:.6}", total_max / num_heads as f32);
    eprintln!("[HEAD_DEBUG] (★ = max > 0.99, ● = max > 0.9)");

    Ok(())
}

/// Compare attention patterns between positive and negative CFG passes.
///
/// This helps understand if errors are correlated or anti-correlated between passes,
/// which affects how CFG amplifies the errors.
///
/// # Arguments
/// * `pos_attn` - Attention probs from positive prompt pass [batch, heads, seq, seq]
/// * `neg_attn` - Attention probs from negative prompt pass [batch, heads, seq, seq]
pub fn debug_cfg_attention_comparison(
    pos_attn: &Tensor,
    neg_attn: &Tensor,
) -> Result<()> {
    if !is_attention_debug() {
        return Ok(());
    }

    eprintln!("\n[CFG_DEBUG] Comparing positive vs negative attention patterns:");

    let pos_stats = tensor_stats(pos_attn)?;
    let neg_stats = tensor_stats(neg_attn)?;

    eprintln!("[CFG_DEBUG] Positive: mean={:.6}, std={:.6}, max={:.6}",
        pos_stats.0, pos_stats.1, pos_stats.3);
    eprintln!("[CFG_DEBUG] Negative: mean={:.6}, std={:.6}, max={:.6}",
        neg_stats.0, neg_stats.1, neg_stats.3);

    // Check if shapes match for detailed comparison
    if pos_attn.dims() != neg_attn.dims() {
        eprintln!("[CFG_DEBUG] Warning: Shape mismatch - pos: {:?}, neg: {:?}",
            pos_attn.dims(), neg_attn.dims());
        eprintln!("[CFG_DEBUG] (This is expected if prompts have different token counts)");
        return Ok(());
    }

    let diff = compare_tensors(pos_attn, neg_attn)?;
    eprintln!("[CFG_DEBUG] Attention diff: max={:.6}, mean={:.6}", diff.0, diff.1);

    // Compute correlation between errors
    // If errors are anti-correlated (opposite signs), CFG will amplify them
    let pos_f32 = pos_attn.to_dtype(DType::F32)?.flatten_all()?;
    let neg_f32 = neg_attn.to_dtype(DType::F32)?.flatten_all()?;

    // Compute element-wise product to check correlation direction
    let product = (&pos_f32 * &neg_f32)?;
    let product_mean = product.mean_all()?.to_scalar::<f32>()?;
    let pos_neg_product_sign = if product_mean > 0.0 { "positive (correlated)" } else { "negative (anti-correlated)" };

    eprintln!("[CFG_DEBUG] Error correlation: {} - mean(pos*neg) = {:.6}",
        pos_neg_product_sign, product_mean);

    if product_mean < 0.0 {
        eprintln!("[CFG_DEBUG] ⚠️ Anti-correlated errors will be AMPLIFIED by CFG!");
        eprintln!("[CFG_DEBUG]    CFG formula: guided = neg + scale*(pos - neg)");
        eprintln!("[CFG_DEBUG]    If pos ≈ -neg, then guided ≈ (1+2*scale)*pos");
    }

    Ok(())
}

/// Save attention tensors to disk for external analysis.
///
/// # Arguments
/// * `name` - Base name for the files
/// * `attn_logits` - Q@K.T raw output
/// * `attn_weights` - Scaled attention weights
/// * `attn_probs` - Attention probabilities after softmax
pub fn save_attention_tensors(
    name: &str,
    attn_logits: &Tensor,
    attn_weights: &Tensor,
    attn_probs: &Tensor,
) -> Result<()> {
    let suffix = get_cfg_pass_suffix();
    let output_dir = "debug_tensors/rust_edit";
    std::fs::create_dir_all(output_dir)?;

    let logits_path = format!("{}/{}_attn_logits{}.npy", output_dir, name, suffix);
    let weights_path = format!("{}/{}_attn_weights{}.npy", output_dir, name, suffix);
    let probs_path = format!("{}/{}_attn_probs{}.npy", output_dir, name, suffix);

    attn_logits.write_npy(&logits_path)?;
    attn_weights.write_npy(&weights_path)?;
    attn_probs.write_npy(&probs_path)?;

    eprintln!("[ATTN_DEBUG] Saved attention tensors to {}/*{}.npy", output_dir, suffix);
    Ok(())
}

// ============================================================================
// Q/K Pipeline Debug Functions for Stage-by-Stage Comparison
// ============================================================================

/// Check if Q/K pipeline saving is enabled (via QWEN_SAVE_QK environment variable).
pub fn is_qk_save_enabled() -> bool {
    std::env::var("QWEN_SAVE_QK").is_ok()
}

/// Save Q/K tensors at a specific pipeline stage.
///
/// This enables stage-by-stage comparison with PyTorch to identify where divergence occurs:
/// - Stage 1 (proj): After linear projection, before reshape
/// - Stage 2 (norm): After QkNorm (RMSNorm), before RoPE
/// - Stage 3 (rope): After RoPE application
///
/// # Arguments
/// * `block_name` - Block identifier (e.g., "block0")
/// * `stage` - Pipeline stage: "proj", "norm", or "rope"
/// * `img_q` - Image stream Q tensor
/// * `img_k` - Image stream K tensor
/// * `txt_q` - Text stream Q tensor
/// * `txt_k` - Text stream K tensor
pub fn save_qk_pipeline_tensors(
    block_name: &str,
    stage: &str,
    img_q: &Tensor,
    img_k: &Tensor,
    txt_q: &Tensor,
    txt_k: &Tensor,
) -> Result<()> {
    if !is_qk_save_enabled() {
        return Ok(());
    }

    let suffix = get_cfg_pass_suffix();
    let output_dir = "debug_tensors/rust_edit";
    std::fs::create_dir_all(output_dir)?;

    // Save each tensor with descriptive names matching PyTorch convention
    // Convert to F32 before saving (NumPy doesn't support BF16)
    let tensors = [
        (format!("{}_{}_img_q{}.npy", block_name, stage, suffix), img_q),
        (format!("{}_{}_img_k{}.npy", block_name, stage, suffix), img_k),
        (format!("{}_{}_txt_q{}.npy", block_name, stage, suffix), txt_q),
        (format!("{}_{}_txt_k{}.npy", block_name, stage, suffix), txt_k),
    ];

    for (filename, tensor) in &tensors {
        let path = format!("{}/{}", output_dir, filename);
        tensor.to_dtype(DType::F32)?.write_npy(&path)?;
    }

    // Print summary statistics for quick comparison
    let img_q_stats = tensor_stats(img_q)?;
    let img_k_stats = tensor_stats(img_k)?;
    let txt_q_stats = tensor_stats(txt_q)?;
    let txt_k_stats = tensor_stats(txt_k)?;

    eprintln!("\n[QK_SAVE] {} stage '{}' {}:", block_name, stage, suffix.trim_start_matches('_'));
    eprintln!("[QK_SAVE] img_q: shape={:?}, mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
        img_q.dims(), img_q_stats.0, img_q_stats.1, img_q_stats.2, img_q_stats.3);
    eprintln!("[QK_SAVE] img_k: shape={:?}, mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
        img_k.dims(), img_k_stats.0, img_k_stats.1, img_k_stats.2, img_k_stats.3);
    eprintln!("[QK_SAVE] txt_q: shape={:?}, mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
        txt_q.dims(), txt_q_stats.0, txt_q_stats.1, txt_q_stats.2, txt_q_stats.3);
    eprintln!("[QK_SAVE] txt_k: shape={:?}, mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
        txt_k.dims(), txt_k_stats.0, txt_k_stats.1, txt_k_stats.2, txt_k_stats.3);
    eprintln!("[QK_SAVE] Saved to {}/{}_{}_*{}.npy", output_dir, block_name, stage, suffix);

    Ok(())
}

/// Save RoPE frequency tensors for debugging.
///
/// # Arguments
/// * `name` - Identifier (e.g., "diffusion_rope")
/// * `img_freqs` - Image/video frequency tensor [seq, dim/2, 2]
/// * `txt_freqs` - Text frequency tensor [seq, dim/2, 2]
pub fn save_rope_freqs(
    name: &str,
    img_freqs: &Tensor,
    txt_freqs: &Tensor,
) -> Result<()> {
    if !is_qk_save_enabled() {
        return Ok(());
    }

    let output_dir = "debug_tensors/rust_edit";
    std::fs::create_dir_all(output_dir)?;

    let img_path = format!("{}/{}_img_freqs.npy", output_dir, name);
    let txt_path = format!("{}/{}_txt_freqs.npy", output_dir, name);

    // Convert to F32 before saving (NumPy doesn't support BF16)
    img_freqs.to_dtype(DType::F32)?.write_npy(&img_path)?;
    txt_freqs.to_dtype(DType::F32)?.write_npy(&txt_path)?;

    let img_stats = tensor_stats(img_freqs)?;
    let txt_stats = tensor_stats(txt_freqs)?;

    eprintln!("\n[ROPE_SAVE] {} frequencies:", name);
    eprintln!("[ROPE_SAVE] img_freqs: shape={:?}, mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
        img_freqs.dims(), img_stats.0, img_stats.1, img_stats.2, img_stats.3);
    eprintln!("[ROPE_SAVE] txt_freqs: shape={:?}, mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
        txt_freqs.dims(), txt_stats.0, txt_stats.1, txt_stats.2, txt_stats.3);
    eprintln!("[ROPE_SAVE] Saved to {}/{}_*.npy", output_dir, name);

    Ok(())
}
