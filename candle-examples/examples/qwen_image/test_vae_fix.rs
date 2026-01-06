//! Quick test for VAE fix - loads PyTorch latents and compares decoded outputs.

use anyhow::Result;
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen_image::{config::VaeConfig, vae::AutoencoderKLQwenImage};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::fs::File;
use std::io::{BufReader, Read};

fn load_npy(path: &str, device: &Device) -> Result<Tensor> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 6];
    reader.read_exact(&mut magic)?;
    if magic != [0x93, b'N', b'U', b'M', b'P', b'Y'] {
        return Err(anyhow::anyhow!("Invalid NumPy magic number"));
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
        .ok_or_else(|| anyhow::anyhow!("No shape found in header"))?;

    let after_shape = &header[shape_start..];
    let paren_start = after_shape
        .find('(')
        .ok_or_else(|| anyhow::anyhow!("No opening paren for shape"))?;
    let paren_end = after_shape
        .find(')')
        .ok_or_else(|| anyhow::anyhow!("No closing paren for shape"))?;

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

fn tensor_stats(t: &Tensor) -> Result<(f32, f32, f32, f32)> {
    let t_f32 = t.to_dtype(DType::F32)?.flatten_all()?;
    let mean = t_f32.mean_all()?.to_scalar::<f32>()?;
    let diff = t_f32.broadcast_sub(&t_f32.mean_all()?)?;
    let var = (&diff * &diff)?.mean_all()?.to_scalar::<f32>()?;
    let std = var.sqrt();
    let min = t_f32.min(0)?.to_scalar::<f32>()?;
    let max = t_f32.max(0)?.to_scalar::<f32>()?;
    Ok((mean, std, min, max))
}

fn main() -> Result<()> {
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = Device::Cpu;

    let dtype = DType::F32;

    println!("Device: {:?}", device);
    println!("\n=== VAE Fix Test ===\n");

    // Load PyTorch reference tensors
    println!("[1/4] Loading PyTorch reference tensors...");
    let pytorch_latents = load_npy("debug_tensors/pytorch/denormalized_latents.npy", &device)?;
    let pytorch_decoded = load_npy("debug_tensors/pytorch/decoded_image.npy", &device)?;

    let (mean, std, min, max) = tensor_stats(&pytorch_latents)?;
    println!(
        "  PyTorch latents: shape={:?}, mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
        pytorch_latents.dims(),
        mean,
        std,
        min,
        max
    );

    let (mean, std, min, max) = tensor_stats(&pytorch_decoded)?;
    println!(
        "  PyTorch decoded: shape={:?}, mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
        pytorch_decoded.dims(),
        mean,
        std,
        min,
        max
    );

    // Load VAE
    println!("\n[2/4] Loading VAE...");
    let api = Api::new()?;
    let repo = api.repo(Repo::new("Qwen/Qwen-Image".to_string(), RepoType::Model));
    let vae_path = repo.get("vae/diffusion_pytorch_model.safetensors")?;

    let vae_config = VaeConfig::default();
    let vb_vae =
        unsafe { VarBuilder::from_mmaped_safetensors(&[vae_path], dtype, &device)? };
    let vae = AutoencoderKLQwenImage::new(&vae_config, vb_vae)?;
    println!("  VAE loaded!");

    // Run VAE decode
    println!("\n[3/4] Running Candle VAE decode...");
    let candle_decoded = vae.decode(&pytorch_latents.to_dtype(dtype)?)?;

    // Extract frame (matches PyTorch: [0][:, :, 0])
    let candle_decoded_frame = candle_decoded.i((.., .., 0, .., ..))?;
    let candle_decoded_frame = candle_decoded_frame.squeeze(0)?; // Remove batch dim for shape [C, H, W]

    let (mean, std, min, max) = tensor_stats(&candle_decoded_frame)?;
    println!(
        "  Candle decoded: shape={:?}, mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
        candle_decoded_frame.dims(),
        mean,
        std,
        min,
        max
    );

    // Compare
    println!("\n[4/4] Comparing outputs...");

    // Make sure shapes match
    let pytorch_decoded_squeezed = if pytorch_decoded.dims().len() == 4 {
        pytorch_decoded.i(0)? // [C, H, W]
    } else {
        pytorch_decoded.clone()
    };

    let (pt_mean, pt_std, _, _) = tensor_stats(&pytorch_decoded_squeezed)?;
    let (rs_mean, rs_std, _, _) = tensor_stats(&candle_decoded_frame)?;

    println!("  PyTorch: mean={:.6}, std={:.6}", pt_mean, pt_std);
    println!("  Candle:  mean={:.6}, std={:.6}", rs_mean, rs_std);

    // Compute diff
    let diff = (&candle_decoded_frame.to_dtype(DType::F32)?
        - &pytorch_decoded_squeezed.to_dtype(DType::F32)?)?
        .abs()?;
    let max_diff = diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    let mean_diff = diff.flatten_all()?.mean_all()?.to_scalar::<f32>()?;

    println!("\n  === DIFF ===");
    println!("  max_diff:  {:.6}", max_diff);
    println!("  mean_diff: {:.6}", mean_diff);

    if max_diff < 0.01 {
        println!("\n  ✓ PASS: VAE outputs match within tolerance!");
    } else if max_diff < 0.1 {
        println!("\n  ~ MODERATE: Some difference but acceptable");
    } else {
        println!("\n  ✗ FAIL: VAE outputs differ significantly!");
    }

    Ok(())
}