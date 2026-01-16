//! Tests for quantized matmul with Q8_1 output (fully quantized pipeline).
//!
//! These tests verify that the Q8_1 output kernels produce results that match
//! the float output kernels within acceptable tolerance. The Q8_1 output path
//! is key to the fully quantized inference pipeline, eliminating unnecessary
//! dequantization between layers.

use candle_core::{
    quantized::{self, GgmlDType},
    Device, Result, Tensor,
};
use rand::prelude::*;

/// Maximum relative error allowed between Q8_1 output and float output.
/// Q8_1 output introduces an additional quantization step, so we allow
/// slightly higher error than the float path.
const MAX_Q8OUT_RELATIVE_ERROR: f32 = 0.05;

/// Maximum absolute error for values close to zero
const MAX_Q8OUT_ABSOLUTE_ERROR: f32 = 0.1;

/// Generate random tensor with values in [-0.5, 0.5]
fn random_tensor(shape: &[usize], device: &Device, seed: u64) -> Result<Tensor> {
    let mut rng = StdRng::seed_from_u64(seed);
    let elem_count: usize = shape.iter().product();
    let data: Vec<f32> = (0..elem_count)
        .map(|_| rng.random::<f32>() - 0.5)
        .collect();
    Tensor::from_vec(data, shape, device)
}

/// Test Q8_1 output matmul against float output for a specific weight dtype.
/// This is CUDA-only since fwd_q8out is only implemented for CUDA.
#[cfg(feature = "cuda")]
fn test_q8out_matmul(
    device: &Device,
    weight_dtype: GgmlDType,
    nrows: usize,  // output dimension
    ncols: usize,  // input dimension (must be multiple of block size)
    batch: usize,  // batch size (1-8)
) -> Result<()> {
    // Skip unsupported dtypes
    if weight_dtype == GgmlDType::F32
        || weight_dtype == GgmlDType::F16
        || weight_dtype == GgmlDType::BF16
    {
        return Ok(());
    }

    // Create random weight matrix and quantize
    let weight_f32 = random_tensor(&[nrows, ncols], device, 42)?;
    let weight_qt = quantized::QTensor::quantize(&weight_f32, weight_dtype)?;

    // Create random input
    let input_f32 = random_tensor(&[batch, ncols], device, 123)?;

    // === Float output path (reference) ===
    let qmatmul = quantized::QMatMul::from_qtensor(weight_qt.clone())?;
    let output_f32 = qmatmul.forward(&input_f32)?;

    // === Q8_1 output path ===
    // Quantize input to Q8_1
    let input_q8_1 = quantized::QTensor::quantize(&input_f32, GgmlDType::Q8_1)?;

    // Call fwd_q8out through the QTensor API
    let output_q8_1 = weight_qt.fwd_q8out(&input_q8_1)?;

    // Dequantize Q8_1 output for comparison
    let output_q8_1_dequant = output_q8_1.dequantize(device)?;

    // === Compare outputs ===
    let out_f32_vec = output_f32.flatten_all()?.to_vec1::<f32>()?;
    let out_q8_vec = output_q8_1_dequant.flatten_all()?.to_vec1::<f32>()?;

    assert_eq!(
        out_f32_vec.len(),
        out_q8_vec.len(),
        "Output size mismatch: f32={}, q8={}",
        out_f32_vec.len(),
        out_q8_vec.len()
    );

    // Calculate error metrics
    let mut max_rel_error = 0.0f32;
    let mut max_abs_error = 0.0f32;
    let mut sum_sq_error = 0.0f32;

    for (i, (&f32_val, &q8_val)) in out_f32_vec.iter().zip(out_q8_vec.iter()).enumerate() {
        let abs_error = (f32_val - q8_val).abs();
        let rel_error = if f32_val.abs() > 1e-6 {
            abs_error / f32_val.abs()
        } else {
            0.0 // Skip relative error for near-zero values
        };

        max_abs_error = max_abs_error.max(abs_error);
        max_rel_error = max_rel_error.max(rel_error);
        sum_sq_error += abs_error * abs_error;

        // Check individual value tolerance
        let within_tolerance = rel_error <= MAX_Q8OUT_RELATIVE_ERROR
            || abs_error <= MAX_Q8OUT_ABSOLUTE_ERROR;

        assert!(
            within_tolerance,
            "Error too large at index {}: f32={}, q8={}, rel_err={}, abs_err={}\n\
             dtype={:?}, shape=({}, {}), batch={}",
            i, f32_val, q8_val, rel_error, abs_error,
            weight_dtype, nrows, ncols, batch
        );
    }

    let rmse = (sum_sq_error / out_f32_vec.len() as f32).sqrt();

    println!(
        "Q8OUT test passed: dtype={:?}, shape=({}, {}), batch={}, max_rel_err={:.4}, max_abs_err={:.4}, rmse={:.6}",
        weight_dtype, nrows, ncols, batch, max_rel_error, max_abs_error, rmse
    );

    Ok(())
}

/// Test all supported weight formats with Q8_1 output
#[cfg(feature = "cuda")]
#[test]
fn test_q8out_all_formats() -> Result<()> {
    let device = Device::new_cuda(0)?;

    // Test matrix sizes (must be compatible with all quantization block sizes)
    // K-quants require multiple of 256, basic quants require multiple of 32
    let test_configs = [
        (64, 256, 1),   // small, single batch
        (128, 512, 2),  // medium, batch 2
        (256, 256, 4),  // square-ish, batch 4
        (512, 256, 8),  // larger, max batch
    ];

    // All weight formats to test
    let weight_dtypes = [
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q8_1,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::Q8K,
    ];

    for dtype in weight_dtypes {
        for &(nrows, ncols, batch) in &test_configs {
            // K-quants require 256-element blocks
            let is_k_quant = matches!(
                dtype,
                GgmlDType::Q2K
                    | GgmlDType::Q3K
                    | GgmlDType::Q4K
                    | GgmlDType::Q5K
                    | GgmlDType::Q6K
                    | GgmlDType::Q8K
            );

            // Skip configs that don't meet block size requirements
            if is_k_quant && ncols % 256 != 0 {
                continue;
            }
            if !is_k_quant && ncols % 32 != 0 {
                continue;
            }

            test_q8out_matmul(&device, dtype, nrows, ncols, batch)?;
        }
    }

    Ok(())
}

/// Test Q8_1 output with Q4_0 weights (most common case)
#[cfg(feature = "cuda")]
#[test]
fn test_q8out_q4_0() -> Result<()> {
    let device = Device::new_cuda(0)?;
    test_q8out_matmul(&device, GgmlDType::Q4_0, 64, 256, 1)?;
    test_q8out_matmul(&device, GgmlDType::Q4_0, 128, 256, 4)?;
    Ok(())
}

/// Test Q8_1 output with Q4K weights (popular K-quant)
#[cfg(feature = "cuda")]
#[test]
fn test_q8out_q4k() -> Result<()> {
    let device = Device::new_cuda(0)?;
    test_q8out_matmul(&device, GgmlDType::Q4K, 64, 256, 1)?;
    test_q8out_matmul(&device, GgmlDType::Q4K, 128, 512, 4)?;
    Ok(())
}

/// Test Q8_1 weights with Q8_1 activations (for attention)
#[cfg(feature = "cuda")]
#[test]
fn test_q8out_q8_1() -> Result<()> {
    let device = Device::new_cuda(0)?;
    test_q8out_matmul(&device, GgmlDType::Q8_1, 64, 256, 1)?;
    test_q8out_matmul(&device, GgmlDType::Q8_1, 128, 256, 4)?;
    Ok(())
}

/// Test Q8K weights with Q8_1 activations
#[cfg(feature = "cuda")]
#[test]
fn test_q8out_q8k() -> Result<()> {
    let device = Device::new_cuda(0)?;
    test_q8out_matmul(&device, GgmlDType::Q8K, 64, 256, 1)?;
    test_q8out_matmul(&device, GgmlDType::Q8K, 128, 512, 4)?;
    Ok(())
}

/// Test edge case: batch size 1 (most common during generation)
#[cfg(feature = "cuda")]
#[test]
fn test_q8out_batch_1() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for dtype in [GgmlDType::Q4_0, GgmlDType::Q4K, GgmlDType::Q8_0] {
        test_q8out_matmul(&device, dtype, 4096, 4096, 1)?;
    }
    Ok(())
}

/// Test various batch sizes from 1 to 8
#[cfg(feature = "cuda")]
#[test]
fn test_q8out_batch_sizes() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let dtype = GgmlDType::Q4_0;

    for batch in 1..=8 {
        test_q8out_matmul(&device, dtype, 128, 256, batch)?;
    }

    Ok(())
}

/// Test that Q8_1 output can be fed back as Q8_1 input (chained inference)
#[cfg(feature = "cuda")]
#[test]
fn test_q8out_chained() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let dtype = GgmlDType::Q4_0;
    let hidden_dim = 256;

    // Create two weight matrices (like two linear layers)
    let weight1_f32 = random_tensor(&[hidden_dim, hidden_dim], &device, 42)?;
    let weight1_qt = quantized::QTensor::quantize(&weight1_f32, dtype)?;

    let weight2_f32 = random_tensor(&[hidden_dim, hidden_dim], &device, 123)?;
    let weight2_qt = quantized::QTensor::quantize(&weight2_f32, dtype)?;

    // Create input and quantize to Q8_1
    let input_f32 = random_tensor(&[1, hidden_dim], &device, 456)?;
    let input_q8_1 = quantized::QTensor::quantize(&input_f32, GgmlDType::Q8_1)?;

    // First layer: input Q8_1 -> output Q8_1
    let output1_q8_1 = weight1_qt.fwd_q8out(&input_q8_1)?;

    // Verify output1 is Q8_1
    assert_eq!(
        output1_q8_1.dtype(),
        GgmlDType::Q8_1,
        "First layer output should be Q8_1"
    );

    // Second layer: feed Q8_1 output directly as input (no dequantization!)
    let output2_q8_1 = weight2_qt.fwd_q8out(&output1_q8_1)?;

    // Verify output2 is Q8_1
    assert_eq!(
        output2_q8_1.dtype(),
        GgmlDType::Q8_1,
        "Second layer output should be Q8_1"
    );

    // Dequantize final output and verify we get valid numbers
    let output2_dequant = output2_q8_1.dequantize(&device)?;
    let output_vec = output2_dequant.flatten_all()?.to_vec1::<f32>()?;

    for (i, val) in output_vec.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Chained Q8_1 output produced non-finite value at index {}: {}",
            i,
            val
        );
    }

    println!(
        "Chained Q8_1 test passed: input {} -> layer1 {} -> layer2 {} elements",
        input_q8_1.shape().elem_count(),
        output1_q8_1.shape().elem_count(),
        output2_q8_1.shape().elem_count()
    );

    Ok(())
}

/// Test that error increases gracefully with more quantization
/// (Q8_1 output should have slightly higher error than float output)
#[cfg(feature = "cuda")]
#[test]
fn test_q8out_error_bounds() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let dtype = GgmlDType::Q4_0;
    let (nrows, ncols, batch) = (256, 256, 1);

    // Create weight and input
    let weight_f32 = random_tensor(&[nrows, ncols], &device, 42)?;
    let weight_qt = quantized::QTensor::quantize(&weight_f32, dtype)?;
    let input_f32 = random_tensor(&[batch, ncols], &device, 123)?;

    // Reference: true float matmul
    let output_true = weight_f32.matmul(&input_f32.t()?)?;
    let output_true = output_true.t()?;

    // Float quantized output
    let qmatmul = quantized::QMatMul::from_qtensor(weight_qt.clone())?;
    let output_float = qmatmul.forward(&input_f32)?;

    // Q8_1 output
    let input_q8_1 = quantized::QTensor::quantize(&input_f32, GgmlDType::Q8_1)?;
    let output_q8_1 = weight_qt.fwd_q8out(&input_q8_1)?;
    let output_q8_1_dequant = output_q8_1.dequantize(&device)?;

    // Calculate RMSEs
    let true_vec = output_true.flatten_all()?.to_vec1::<f32>()?;
    let float_vec = output_float.flatten_all()?.to_vec1::<f32>()?;
    let q8_vec = output_q8_1_dequant.flatten_all()?.to_vec1::<f32>()?;

    let rmse_float: f32 = true_vec
        .iter()
        .zip(float_vec.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
        / true_vec.len() as f32;

    let rmse_q8: f32 = true_vec
        .iter()
        .zip(q8_vec.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
        / true_vec.len() as f32;

    println!(
        "Error comparison: RMSE(float)={:.6}, RMSE(q8out)={:.6}, ratio={:.2}x",
        rmse_float,
        rmse_q8,
        rmse_q8 / rmse_float
    );

    // Q8 output should have higher error but within reasonable bounds
    // Typically 1.1x-2x higher error due to additional quantization
    assert!(
        rmse_q8 < rmse_float * 3.0,
        "Q8 output error ({}) is more than 3x the float error ({})",
        rmse_q8,
        rmse_float
    );

    Ok(())
}

// =====================================================================
// ELEMENT-WISE Q8_1 OPERATION TESTS
// =====================================================================

/// Test add_q8_1 against float addition
#[cfg(feature = "cuda")]
#[test]
fn test_add_q8_1() -> Result<()> {
    let device = Device::new_cuda(0)?;

    // Test various sizes (must be multiple of 32 for Q8_1 block size)
    for size in [32, 128, 256, 1024, 4096] {
        // Create random inputs
        let a_f32 = random_tensor(&[size], &device, 42)?;
        let b_f32 = random_tensor(&[size], &device, 123)?;

        // Reference: float addition
        let expected = (&a_f32 + &b_f32)?;

        // Q8_1 path
        let a_q8 = quantized::QTensor::quantize(&a_f32, GgmlDType::Q8_1)?;
        let b_q8 = quantized::QTensor::quantize(&b_f32, GgmlDType::Q8_1)?;
        let result_q8 = a_q8.add_q8_1(&b_q8)?;

        // Verify output is Q8_1
        assert_eq!(result_q8.dtype(), GgmlDType::Q8_1);

        // Dequantize and compare
        let result_dequant = result_q8.dequantize(&device)?;

        let expected_vec = expected.flatten_all()?.to_vec1::<f32>()?;
        let result_vec = result_dequant.flatten_all()?.to_vec1::<f32>()?;

        // Calculate RMSE
        let rmse: f32 = expected_vec
            .iter()
            .zip(result_vec.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
            / expected_vec.len() as f32;

        // Q8_1 should have low error for addition
        assert!(
            rmse < 0.05,
            "add_q8_1 RMSE {} too high for size {}",
            rmse,
            size
        );

        println!("add_q8_1 size={}: RMSE={:.6}", size, rmse);
    }

    Ok(())
}

/// Test mul_q8_1 against float multiplication
#[cfg(feature = "cuda")]
#[test]
fn test_mul_q8_1() -> Result<()> {
    let device = Device::new_cuda(0)?;

    // Test various sizes (must be multiple of 32 for Q8_1 block size)
    for size in [32, 128, 256, 1024, 4096] {
        // Create random inputs
        let a_f32 = random_tensor(&[size], &device, 42)?;
        let b_f32 = random_tensor(&[size], &device, 123)?;

        // Reference: float multiplication
        let expected = (&a_f32 * &b_f32)?;

        // Q8_1 path
        let a_q8 = quantized::QTensor::quantize(&a_f32, GgmlDType::Q8_1)?;
        let b_q8 = quantized::QTensor::quantize(&b_f32, GgmlDType::Q8_1)?;
        let result_q8 = a_q8.mul_q8_1(&b_q8)?;

        // Verify output is Q8_1
        assert_eq!(result_q8.dtype(), GgmlDType::Q8_1);

        // Dequantize and compare
        let result_dequant = result_q8.dequantize(&device)?;

        let expected_vec = expected.flatten_all()?.to_vec1::<f32>()?;
        let result_vec = result_dequant.flatten_all()?.to_vec1::<f32>()?;

        // Calculate RMSE
        let rmse: f32 = expected_vec
            .iter()
            .zip(result_vec.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
            / expected_vec.len() as f32;

        // Multiplication has higher error due to range expansion
        assert!(
            rmse < 0.1,
            "mul_q8_1 RMSE {} too high for size {}",
            rmse,
            size
        );

        println!("mul_q8_1 size={}: RMSE={:.6}", size, rmse);
    }

    Ok(())
}

/// Test that add_q8_1 can be chained (residual + residual pattern)
#[cfg(feature = "cuda")]
#[test]
fn test_add_q8_1_chained() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let size = 256;

    // Create three random inputs
    let a_f32 = random_tensor(&[size], &device, 42)?;
    let b_f32 = random_tensor(&[size], &device, 123)?;
    let c_f32 = random_tensor(&[size], &device, 456)?;

    // Quantize all
    let a_q8 = quantized::QTensor::quantize(&a_f32, GgmlDType::Q8_1)?;
    let b_q8 = quantized::QTensor::quantize(&b_f32, GgmlDType::Q8_1)?;
    let c_q8 = quantized::QTensor::quantize(&c_f32, GgmlDType::Q8_1)?;

    // Chain: (a + b) + c
    let sum1 = a_q8.add_q8_1(&b_q8)?;
    let sum2 = sum1.add_q8_1(&c_q8)?;

    // Verify all outputs are Q8_1
    assert_eq!(sum1.dtype(), GgmlDType::Q8_1);
    assert_eq!(sum2.dtype(), GgmlDType::Q8_1);

    // Verify final result is finite
    let result = sum2.dequantize(&device)?;
    let result_vec = result.flatten_all()?.to_vec1::<f32>()?;

    for (i, val) in result_vec.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Chained add produced non-finite at {}: {}",
            i,
            val
        );
    }

    println!("Chained add_q8_1 test passed");
    Ok(())
}

/// Test SwiGLU-like pattern: gate_output * up_output
#[cfg(feature = "cuda")]
#[test]
fn test_swiglu_pattern() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let hidden_dim = 256;

    // Simulate gate and up projections (after silu/activation)
    let gate_f32 = random_tensor(&[1, hidden_dim], &device, 42)?;
    let up_f32 = random_tensor(&[1, hidden_dim], &device, 123)?;

    // Reference
    let expected = (&gate_f32 * &up_f32)?;

    // Q8_1 path
    let gate_q8 = quantized::QTensor::quantize(&gate_f32, GgmlDType::Q8_1)?;
    let up_q8 = quantized::QTensor::quantize(&up_f32, GgmlDType::Q8_1)?;
    let result_q8 = gate_q8.mul_q8_1(&up_q8)?;

    // Dequantize and compare
    let result_dequant = result_q8.dequantize(&device)?;

    let expected_vec = expected.flatten_all()?.to_vec1::<f32>()?;
    let result_vec = result_dequant.flatten_all()?.to_vec1::<f32>()?;

    let rmse: f32 = expected_vec
        .iter()
        .zip(result_vec.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
        / expected_vec.len() as f32;

    println!("SwiGLU pattern RMSE: {:.6}", rmse);
    assert!(rmse < 0.1, "SwiGLU pattern RMSE too high: {}", rmse);

    Ok(())
}

/// Test residual connection pattern: x + attn_out
#[cfg(feature = "cuda")]
#[test]
fn test_residual_pattern() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let hidden_dim = 256;

    // Simulate hidden state and attention output
    let hidden_f32 = random_tensor(&[1, hidden_dim], &device, 42)?;
    let attn_out_f32 = random_tensor(&[1, hidden_dim], &device, 123)?;

    // Reference
    let expected = (&hidden_f32 + &attn_out_f32)?;

    // Q8_1 path
    let hidden_q8 = quantized::QTensor::quantize(&hidden_f32, GgmlDType::Q8_1)?;
    let attn_out_q8 = quantized::QTensor::quantize(&attn_out_f32, GgmlDType::Q8_1)?;
    let result_q8 = hidden_q8.add_q8_1(&attn_out_q8)?;

    // Dequantize and compare
    let result_dequant = result_q8.dequantize(&device)?;

    let expected_vec = expected.flatten_all()?.to_vec1::<f32>()?;
    let result_vec = result_dequant.flatten_all()?.to_vec1::<f32>()?;

    let rmse: f32 = expected_vec
        .iter()
        .zip(result_vec.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
        / expected_vec.len() as f32;

    println!("Residual pattern RMSE: {:.6}", rmse);
    assert!(rmse < 0.05, "Residual pattern RMSE too high: {}", rmse);

    Ok(())
}
