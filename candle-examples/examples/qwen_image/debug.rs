//! Debug utilities for Qwen-Image tensor comparison and substitution testing.
//!
//! This module provides:
//! - Saving/loading tensors in NumPy format for cross-validation with PyTorch
//! - Tensor statistics for comparison
//! - Optional substitution of PyTorch reference tensors into the pipeline

use anyhow::{anyhow, Result};
use candle::{DType, Device, Tensor};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

// ============================================================================
// Tensor Statistics
// ============================================================================

/// Statistics for comparing tensors between implementations.
#[derive(Debug, Clone)]
pub struct TensorStats {
    pub shape: Vec<usize>,
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
}

impl std::fmt::Display for TensorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "shape={:?} mean={:.6} std={:.6} min={:.6} max={:.6}",
            self.shape, self.mean, self.std, self.min, self.max
        )
    }
}

/// Compute statistics for a tensor.
pub fn tensor_stats(t: &Tensor) -> Result<TensorStats> {
    // Use F32 to match Python (.float() = torch.float32)
    let t_f32 = t.to_dtype(DType::F32)?.flatten_all()?;
    let mean_tensor = t_f32.mean_all()?;
    let mean = mean_tensor.to_scalar::<f32>()?;
    let diff = t_f32.broadcast_sub(&mean_tensor)?;
    let var = (&diff * &diff)?.mean_all()?.to_scalar::<f32>()?;
    let std = var.sqrt();
    let min = t_f32.min(0)?.to_scalar::<f32>()?;
    let max = t_f32.max(0)?.to_scalar::<f32>()?;

    Ok(TensorStats {
        shape: t.dims().to_vec(),
        mean,
        std,
        min,
        max,
    })
}

// ============================================================================
// NumPy Format Save/Load
// ============================================================================

/// Save a tensor in NumPy .npy format.
pub fn save_npy(t: &Tensor, path: &str) -> Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }

    let t = t.to_dtype(DType::F32)?;
    let shape = t.dims();
    let data: Vec<f32> = t.flatten_all()?.to_vec1()?;

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // NumPy .npy format header
    writer.write_all(&[0x93, b'N', b'U', b'M', b'P', b'Y'])?;
    writer.write_all(&[0x01, 0x00])?;

    let shape_str: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let shape_tuple = if shape.len() == 1 {
        format!("({},)", shape_str[0])
    } else {
        format!("({})", shape_str.join(", "))
    };
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': {}}}",
        shape_tuple
    );

    let header_len = header.len();
    let padding_needed = 64 - ((10 + header_len + 1) % 64);
    let padded_header = format!("{}{}\n", header, " ".repeat(padding_needed));

    let header_len_bytes = (padded_header.len() as u16).to_le_bytes();
    writer.write_all(&header_len_bytes)?;
    writer.write_all(padded_header.as_bytes())?;

    for val in data {
        writer.write_all(&val.to_le_bytes())?;
    }

    Ok(())
}

/// Load a tensor from NumPy .npy format.
pub fn load_npy(path: &str, device: &Device) -> Result<Tensor> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 6];
    reader.read_exact(&mut magic)?;
    if magic != [0x93, b'N', b'U', b'M', b'P', b'Y'] {
        return Err(anyhow!("Invalid NumPy magic number"));
    }

    let mut version = [0u8; 2];
    reader.read_exact(&mut version)?;

    let header_len = if version[0] == 1 {
        let mut len_bytes = [0u8; 2];
        reader.read_exact(&mut len_bytes)?;
        u16::from_le_bytes(len_bytes) as usize
    } else {
        let mut len_bytes = [0u8; 4];
        reader.read_exact(&mut len_bytes)?;
        u32::from_le_bytes(len_bytes) as usize
    };

    let mut header_bytes = vec![0u8; header_len];
    reader.read_exact(&mut header_bytes)?;
    let header = String::from_utf8_lossy(&header_bytes);

    let shape = parse_npy_shape(&header)?;
    let total_elements: usize = shape.iter().product();

    let mut data = vec![0f32; total_elements];
    for val in data.iter_mut() {
        let mut bytes = [0u8; 4];
        reader.read_exact(&mut bytes)?;
        *val = f32::from_le_bytes(bytes);
    }

    Ok(Tensor::from_vec(data, shape.as_slice(), device)?)
}

fn parse_npy_shape(header: &str) -> Result<Vec<usize>> {
    let shape_start = header
        .find("'shape':")
        .or_else(|| header.find("\"shape\":"))
        .ok_or_else(|| anyhow!("No shape found in header"))?;

    let after_shape = &header[shape_start..];
    let paren_start = after_shape
        .find('(')
        .ok_or_else(|| anyhow!("No opening paren for shape"))?;
    let paren_end = after_shape
        .find(')')
        .ok_or_else(|| anyhow!("No closing paren for shape"))?;

    let shape_str = &after_shape[paren_start + 1..paren_end];
    let shape: Vec<usize> = shape_str
        .split(',')
        .filter_map(|s| {
            let s = s.trim();
            if s.is_empty() {
                None
            } else {
                s.parse().ok()
            }
        })
        .collect();

    Ok(shape)
}

// ============================================================================
// Debug Context
// ============================================================================

/// Substitution mode for debug context.
#[derive(Debug, Clone)]
pub enum SubstituteMode {
    /// Substitute all available PyTorch reference tensors (default for backward compat)
    All,
    /// Don't substitute any tensors - run full Rust pipeline
    None,
    /// Only substitute tensors with these names
    Only(HashSet<String>),
    /// Substitute all EXCEPT tensors with these names
    Except(HashSet<String>),
}

impl SubstituteMode {
    /// Check if a tensor should be substituted.
    pub fn should_substitute(&self, name: &str) -> bool {
        match self {
            SubstituteMode::All => true,
            SubstituteMode::None => false,
            SubstituteMode::Only(names) => names.contains(name),
            SubstituteMode::Except(names) => !names.contains(name),
        }
    }

    /// Parse a substitution mode from a CLI string.
    ///
    /// Supported formats:
    /// - `"none"`: No substitution, run full Rust pipeline
    /// - `"all"`: Substitute all available PyTorch reference tensors
    /// - `"prompt"`: Only substitute prompt_embeds and prompt_mask
    /// - `"latents"`: Only substitute final_latents and denormalized_latents
    /// - `"vae"`: Substitute VAE-related tensors (image encoding)
    /// - `"vision"`: Substitute vision encoder output
    /// - `"noise"`: Substitute initial noise latents
    /// - `"transformer"`: Substitute transformer inputs (for testing transformer only)
    /// - `"only:name1,name2"`: Only substitute these specific tensors
    /// - `"except:name1,name2"`: Substitute all except these tensors
    pub fn parse(mode_str: &str) -> Result<Self> {
        match mode_str.to_lowercase().as_str() {
            "none" => Ok(SubstituteMode::None),
            "all" => Ok(SubstituteMode::All),
            "prompt" => {
                let names: HashSet<String> = [
                    "prompt_embeds",
                    "prompt_mask",
                    "negative_prompt_embeds",
                    "negative_prompt_mask",
                ]
                .iter()
                .map(|s| s.to_string())
                .collect();
                Ok(SubstituteMode::Only(names))
            }
            "latents" => {
                let names: HashSet<String> = ["final_latents", "denormalized_latents"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                Ok(SubstituteMode::Only(names))
            }
            "vae" => {
                // VAE encoding outputs for edit pipeline
                let names: HashSet<String> = [
                    "image_latents_raw",
                    "image_latents_normalized",
                    "packed_image_latents",
                ]
                .iter()
                .map(|s| s.to_string())
                .collect();
                Ok(SubstituteMode::Only(names))
            }
            "vision" => {
                // Vision encoder output
                let names: HashSet<String> = ["vision_embeds"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                Ok(SubstituteMode::Only(names))
            }
            "vision_input" => {
                // Vision encoder INPUT (pixel values) - use to test vision encoder with identical inputs
                let names: HashSet<String> = ["vision_pixel_values"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                Ok(SubstituteMode::Only(names))
            }
            "noise" => {
                // Initial noise latents
                let names: HashSet<String> =
                    ["noise_latents_unpacked", "packed_noise_latents"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect();
                Ok(SubstituteMode::Only(names))
            }
            "transformer" => {
                // Substitute all inputs to transformer (to test transformer only)
                let names: HashSet<String> = [
                    "prompt_embeds",
                    "prompt_mask",
                    "negative_prompt_embeds",
                    "negative_prompt_mask",
                    "packed_noise_latents",
                    "packed_image_latents",
                ]
                .iter()
                .map(|s| s.to_string())
                .collect();
                Ok(SubstituteMode::Only(names))
            }
            "rope" => {
                // Substitute diffusion RoPE embeddings (test if RoPE is the issue)
                let names: HashSet<String> = [
                    "diffusion_rope_img_freqs",
                    "diffusion_rope_txt_freqs",
                ]
                .iter()
                .map(|s| s.to_string())
                .collect();
                Ok(SubstituteMode::Only(names))
            }
            "block0" => {
                // Substitute block 0 output (test if bug is in block 0 or later)
                let names: HashSet<String> = [
                    "block0_output_img",
                    "block0_output_txt",
                ]
                .iter()
                .map(|s| s.to_string())
                .collect();
                Ok(SubstituteMode::Only(names))
            }
            s if s.starts_with("only:") => {
                let names: HashSet<String> = s[5..]
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                if names.is_empty() {
                    return Err(anyhow!("'only:' requires at least one tensor name"));
                }
                Ok(SubstituteMode::Only(names))
            }
            s if s.starts_with("except:") => {
                let names: HashSet<String> = s[7..]
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                if names.is_empty() {
                    return Err(anyhow!("'except:' requires at least one tensor name"));
                }
                Ok(SubstituteMode::Except(names))
            }
            _ => Err(anyhow!(
                "Invalid substitution mode: '{}'. Use 'none', 'all', 'prompt', 'latents', 'vae', 'vision', 'vision_input', 'noise', 'transformer', 'rope', 'block0', 'only:name1,name2', or 'except:name1,name2'",
                mode_str
            )),
        }
    }
}

/// Debug context for tensor comparison and substitution testing.
///
/// When used with the generate pipeline, this context:
/// 1. Saves Rust tensors to `output_dir` for inspection
/// 2. Compares against PyTorch reference tensors from `reference_dir`
/// 3. Optionally substitutes PyTorch tensors to isolate diverging components
///
/// # Substitution Modes
///
/// The `substitute_mode` field controls which tensors are substituted:
/// - `SubstituteMode::All`: Substitute all available reference tensors (original behavior)
/// - `SubstituteMode::None`: Don't substitute anything - run full Rust pipeline
/// - `SubstituteMode::Only(names)`: Only substitute these specific tensors
/// - `SubstituteMode::Except(names)`: Substitute all except these tensors
///
/// # Example: Isolate Text Encoder
///
/// To test if the transformer is correct (assuming prompt_embeds is the issue):
/// ```ignore
/// ctx.substitute_mode = SubstituteMode::Only(["prompt_embeds", "prompt_mask"].into_iter().map(String::from).collect());
/// ```
///
/// # Example: Test Full Rust Pipeline
///
/// ```ignore
/// ctx.substitute_mode = SubstituteMode::None;
/// ```
pub struct DebugContext {
    reference_dir: String,
    output_dir: String,
    cache: HashMap<String, Tensor>,
    device: Device,
    /// Controls which tensors are substituted with PyTorch references
    pub substitute_mode: SubstituteMode,
    /// Legacy field for backward compatibility (use substitute_mode instead)
    #[deprecated(note = "Use substitute_mode instead")]
    pub substitute: bool,
}

impl DebugContext {
    /// Create a new debug context with default settings (substitute all).
    pub fn new(reference_dir: &str, output_dir: &str, device: &Device) -> Result<Self> {
        Self::with_mode(reference_dir, output_dir, device, SubstituteMode::All)
    }

    /// Create a new debug context with a specific substitution mode.
    pub fn with_mode(
        reference_dir: &str,
        output_dir: &str,
        device: &Device,
        substitute_mode: SubstituteMode,
    ) -> Result<Self> {
        fs::create_dir_all(output_dir)?;
        #[allow(deprecated)]
        Ok(Self {
            reference_dir: reference_dir.to_string(),
            output_dir: output_dir.to_string(),
            cache: HashMap::new(),
            device: device.clone(),
            substitute_mode,
            substitute: true, // Legacy field, kept for API compat
        })
    }

    /// Check if a reference tensor exists.
    pub fn has_reference(&self, name: &str) -> bool {
        let path = format!("{}/{}.npy", self.reference_dir, name);
        Path::new(&path).exists()
    }

    /// Load a reference tensor (with caching).
    pub fn load_reference(&mut self, name: &str) -> Result<Tensor> {
        if let Some(t) = self.cache.get(name) {
            return Ok(t.clone());
        }

        let path = format!("{}/{}.npy", self.reference_dir, name);
        let t = load_npy(&path, &self.device)?;
        self.cache.insert(name.to_string(), t.clone());
        Ok(t)
    }

    /// Debug: Checkpoint a tensor: save, compare, and optionally substitute.
    ///
    /// This is the main API for debug instrumentation:
    /// 1. Saves the Rust tensor to disk
    /// 2. Compares against PyTorch reference (if available)
    /// 3. Returns PyTorch tensor based on `substitute_mode`, otherwise returns original
    pub fn checkpoint(&mut self, name: &str, tensor: Tensor) -> Result<Tensor> {
        self.checkpoint_with_ref(name, name, tensor)
    }

    /// Checkpoint with a different reference name.
    ///
    /// Use this when the Rust and PyTorch checkpoints have different names.
    /// For example, Rust's "packed_latents_step0" corresponds to PyTorch's "initial_latents_packed".
    pub fn checkpoint_with_ref(
        &mut self,
        save_name: &str,
        ref_name: &str,
        tensor: Tensor,
    ) -> Result<Tensor> {
        // Save Rust tensor
        let output_path = format!("{}/{}.npy", self.output_dir, save_name);
        save_npy(&tensor, &output_path)?;

        let rust_stats = tensor_stats(&tensor)?;
        println!("  [DEBUG] {}", save_name);
        println!("    Rust:    {}", rust_stats);

        // Check if we should substitute this tensor
        let should_substitute = self.substitute_mode.should_substitute(ref_name);

        // Compare and optionally substitute (using ref_name for PyTorch lookup)
        if self.has_reference(ref_name) {
            let ref_tensor = self.load_reference(ref_name)?;
            let ref_stats = tensor_stats(&ref_tensor)?;
            println!("    PyTorch: {}", ref_stats);

            // Compute diff (use F32 to match Python)
            if rust_stats.shape == ref_stats.shape {
                let rust_f32 = tensor.to_dtype(DType::F32)?;
                let ref_f32 = ref_tensor.to_dtype(DType::F32)?;
                let diff = (&rust_f32 - &ref_f32)?.abs()?;
                let max_diff = diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
                let mean_diff = diff.flatten_all()?.mean_all()?.to_scalar::<f32>()?;

                let severity = if max_diff > 0.1 {
                    "*** SIGNIFICANT ***"
                } else if max_diff > 0.01 {
                    "** moderate **"
                } else {
                    "OK"
                };
                println!(
                    "    Diff:    max={:.6e} mean={:.6e} [{}]",
                    max_diff, mean_diff, severity
                );
            } else {
                println!("    WARNING: Shape mismatch!");
            }

            // Substitute based on mode
            if should_substitute {
                println!("    -> SUBSTITUTING with PyTorch tensor");
                return Ok(ref_tensor.to_dtype(tensor.dtype())?);
            }
        } else {
            println!("    (no PyTorch reference)");
        }

        Ok(tensor)
    }
}

/// Checkpoint a tensor if debug context is available.
pub fn checkpoint(
    ctx: &mut Option<&mut DebugContext>,
    name: &str,
    tensor: Tensor,
) -> Result<Tensor> {
    match ctx.as_mut() {
        Some(ctx) => ctx.checkpoint(name, tensor),
        None => Ok(tensor),
    }
}

/// Checkpoint with a different reference name.
pub fn checkpoint_ref(
    ctx: &mut Option<&mut DebugContext>,
    save_name: &str,
    ref_name: &str,
    tensor: Tensor,
) -> Result<Tensor> {
    match ctx.as_mut() {
        Some(ctx) => ctx.checkpoint_with_ref(save_name, ref_name, tensor),
        None => Ok(tensor),
    }
}

/// Setup debug directories and create a debug context with default settings (substitute none).
pub fn setup_debug_context(device: &Device) -> Result<DebugContext> {
    setup_debug_context_with_mode(device, SubstituteMode::None)
}

/// Setup debug directories and create a debug context with a specific mode.
pub fn setup_debug_context_with_mode(
    device: &Device,
    mode: SubstituteMode,
) -> Result<DebugContext> {
    let mode_desc = match &mode {
        SubstituteMode::All => "substitute ALL".to_string(),
        SubstituteMode::None => "substitute NONE (full Rust)".to_string(),
        SubstituteMode::Only(names) => format!("substitute ONLY: {:?}", names),
        SubstituteMode::Except(names) => format!("substitute EXCEPT: {:?}", names),
    };
    println!("[DEBUG] Substitution mode: {}", mode_desc);

    DebugContext::with_mode("debug_tensors/pytorch", "debug_tensors/rust", device, mode)
}

/// Convenience: setup debug context for full Rust pipeline (no substitution).
pub fn setup_debug_context_no_substitute(device: &Device) -> Result<DebugContext> {
    setup_debug_context_with_mode(device, SubstituteMode::None)
}

/// Convenience: setup debug context that only substitutes prompt embeddings.
///
/// Use this to isolate whether the transformer is working correctly.
pub fn setup_debug_context_substitute_prompt_only(device: &Device) -> Result<DebugContext> {
    let names: HashSet<String> = ["prompt_embeds", "prompt_mask"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    setup_debug_context_with_mode(device, SubstituteMode::Only(names))
}

/// Convenience: setup debug context that only substitutes final latents.
///
/// Use this to isolate whether the VAE decoder is working correctly.
pub fn setup_debug_context_substitute_latents_only(device: &Device) -> Result<DebugContext> {
    let names: HashSet<String> = ["final_latents", "denormalized_latents"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    setup_debug_context_with_mode(device, SubstituteMode::Only(names))
}

// ============================================================================
// Edit Pipeline Debug Context
// ============================================================================

/// Setup debug context for the EDIT pipeline.
///
/// Uses `debug_tensors/pytorch_edit` as the reference directory
/// (generated by `edit_reference_tensors.py`).
pub fn setup_debug_context_for_edit(device: &Device) -> Result<DebugContext> {
    setup_debug_context_for_edit_with_mode(device, SubstituteMode::None)
}

/// Compare a Rust tensor with a PyTorch reference tensor (for debugging).
///
/// This function loads a reference tensor from the debug directory and compares it
/// with a Rust tensor, printing the comparison results. Useful for comparing
/// intermediate tensors that aren't part of the normal checkpoint flow.
pub fn compare_with_reference(
    reference_dir: &str,
    name: &str,
    rust_tensor: &Tensor,
    device: &Device,
) -> Result<()> {
    let path = format!("{}/{}.npy", reference_dir, name);
    if !std::path::Path::new(&path).exists() {
        println!("  [COMPARE] {}: no PyTorch reference found", name);
        return Ok(());
    }

    let ref_tensor = load_npy(&path, device)?;
    let rust_stats = tensor_stats(rust_tensor)?;
    let ref_stats = tensor_stats(&ref_tensor)?;

    println!("  [COMPARE] {}", name);
    println!("    Rust:    {}", rust_stats);
    println!("    PyTorch: {}", ref_stats);

    if rust_stats.shape == ref_stats.shape {
        let rust_f32 = rust_tensor.to_dtype(DType::F32)?;
        let ref_f32 = ref_tensor.to_dtype(DType::F32)?;
        let diff = (&rust_f32 - &ref_f32)?.abs()?;
        let max_diff = diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean_diff = diff.flatten_all()?.mean_all()?.to_scalar::<f32>()?;

        let severity = if max_diff > 0.1 {
            "*** SIGNIFICANT ***"
        } else if max_diff > 0.01 {
            "** moderate **"
        } else {
            "OK"
        };
        println!(
            "    Diff:    max={:.6e} mean={:.6e} [{}]",
            max_diff, mean_diff, severity
        );
    } else {
        println!("    WARNING: Shape mismatch!");
    }

    Ok(())
}

/// Setup debug context for the EDIT pipeline with a specific mode.
pub fn setup_debug_context_for_edit_with_mode(
    device: &Device,
    mode: SubstituteMode,
) -> Result<DebugContext> {
    let mode_desc = match &mode {
        SubstituteMode::All => "substitute ALL".to_string(),
        SubstituteMode::None => "substitute NONE (full Rust)".to_string(),
        SubstituteMode::Only(names) => format!("substitute ONLY: {:?}", names),
        SubstituteMode::Except(names) => format!("substitute EXCEPT: {:?}", names),
    };
    println!("[DEBUG EDIT] Substitution mode: {}", mode_desc);

    DebugContext::with_mode(
        "debug_tensors/pytorch_edit",
        "debug_tensors/rust_edit",
        device,
        mode,
    )
}