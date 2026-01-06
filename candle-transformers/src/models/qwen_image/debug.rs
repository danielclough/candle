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

/// Print VAE tensor statistics (only if VAE_DEBUG is set).
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
}
