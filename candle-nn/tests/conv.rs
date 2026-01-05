#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{test_device, Device, IndexOp, Module, Result, Tensor};

/// Test Conv3d module forward pass
fn conv3d_module(dev: &Device) -> Result<()> {
    // Create input: [batch=1, channels=2, depth=4, height=4, width=4]
    let input = Tensor::arange(0f32, 128., dev)?.reshape((1, 2, 4, 4, 4))?;

    // Create weight: [out_channels=3, in_channels=2, k_d=2, k_h=2, k_w=2]
    let weight = Tensor::arange(0f32, 48., dev)?.reshape((3, 2, 2, 2, 2))?;

    // Create bias: [out_channels=3]
    let bias = Tensor::arange(0f32, 3., dev)?;

    // Test with default config (padding=0, stride=1, dilation=1, groups=1)
    let config = candle_nn::Conv3dConfig::default();
    let conv = candle_nn::Conv3d::new(weight.clone(), Some(bias.clone()), config);

    let output = conv.forward(&input)?;
    assert_eq!(output.dims(), [1, 3, 3, 3, 3]);

    // Verify bias is applied correctly
    let no_bias_conv = candle_nn::Conv3d::new(weight.clone(), None, config);
    let no_bias_output = no_bias_conv.forward(&input)?;

    // The difference should be exactly the bias for each channel
    let diff = (output.i((.., 0, .., .., ..))? - no_bias_output.i((.., 0, .., .., ..))?)?;
    let diff_sum = diff.sum_all()?.to_scalar::<f32>()?;
    // bias[0] = 0, so diff should be 0 * num_elements = 0
    assert!((diff_sum - 0.0).abs() < 1e-5);

    let diff = (output.i((.., 1, .., .., ..))? - no_bias_output.i((.., 1, .., .., ..))?)?;
    let diff_sum = diff.sum_all()?.to_scalar::<f32>()?;
    // bias[1] = 1, broadcasted over 27 elements (3x3x3)
    assert!((diff_sum - 27.0).abs() < 1e-5);

    Ok(())
}

/// Test Conv3d with different configurations
fn conv3d_config(dev: &Device) -> Result<()> {
    let input = Tensor::arange(0f32, 128., dev)?.reshape((1, 2, 4, 4, 4))?;
    let weight = Tensor::arange(0f32, 48., dev)?.reshape((3, 2, 2, 2, 2))?;

    // Test with padding=1
    let config = candle_nn::Conv3dConfig {
        padding: 1,
        stride: 1,
        dilation: 1,
        groups: 1,
    };
    let conv = candle_nn::Conv3d::new(weight.clone(), None, config);
    let output = conv.forward(&input)?;
    // (4 + 2*1 - 2) / 1 + 1 = 5
    assert_eq!(output.dims(), [1, 3, 5, 5, 5]);

    // Test with stride=2
    let config = candle_nn::Conv3dConfig {
        padding: 0,
        stride: 2,
        dilation: 1,
        groups: 1,
    };
    let conv = candle_nn::Conv3d::new(weight.clone(), None, config);
    let output = conv.forward(&input)?;
    // (4 - 2) / 2 + 1 = 2
    assert_eq!(output.dims(), [1, 3, 2, 2, 2]);

    Ok(())
}

/// Test Conv3d with grouped convolution
fn conv3d_groups(dev: &Device) -> Result<()> {
    // Input: [batch=1, channels=4, depth=4, height=4, width=4]
    let input = Tensor::arange(0f32, 256., dev)?.reshape((1, 4, 4, 4, 4))?;
    // Weight: [out_channels=4, in_channels/groups=2, k_d=2, k_h=2, k_w=2]
    let weight = Tensor::arange(0f32, 64., dev)?.reshape((4, 2, 2, 2, 2))?;

    let config = candle_nn::Conv3dConfig {
        padding: 0,
        stride: 1,
        dilation: 1,
        groups: 2,
    };
    let conv = candle_nn::Conv3d::new(weight, None, config);
    let output = conv.forward(&input)?;
    assert_eq!(output.dims(), [1, 4, 3, 3, 3]);

    Ok(())
}

/// Test ConvTranspose3d module forward pass
fn conv_transpose3d_module(dev: &Device) -> Result<()> {
    // Input: [batch=1, channels=2, depth=2, height=2, width=2]
    let input = Tensor::arange(0f32, 16., dev)?.reshape((1, 2, 2, 2, 2))?;

    // Weight: [in_channels=2, out_channels=3, k_d=2, k_h=2, k_w=2]
    let weight = Tensor::arange(0f32, 48., dev)?.reshape((2, 3, 2, 2, 2))?;

    // Bias: [out_channels=3]
    let bias = Tensor::arange(0f32, 3., dev)?;

    let config = candle_nn::ConvTranspose3dConfig::default();
    let conv = candle_nn::ConvTranspose3d::new(weight.clone(), Some(bias.clone()), config);

    let output = conv.forward(&input)?;
    // (2-1)*1 + 1*(2-1) + 1 - 2*0 = 3
    assert_eq!(output.dims(), [1, 3, 3, 3, 3]);

    // Verify bias is applied
    let no_bias_conv = candle_nn::ConvTranspose3d::new(weight.clone(), None, config);
    let no_bias_output = no_bias_conv.forward(&input)?;

    let diff = (output.i((.., 1, .., .., ..))? - no_bias_output.i((.., 1, .., .., ..))?)?;
    let diff_sum = diff.sum_all()?.to_scalar::<f32>()?;
    // bias[1] = 1, broadcasted over 27 elements (3x3x3)
    assert!((diff_sum - 27.0).abs() < 1e-5);

    Ok(())
}

/// Test ConvTranspose3d with stride
fn conv_transpose3d_stride(dev: &Device) -> Result<()> {
    let input = Tensor::arange(0f32, 16., dev)?.reshape((1, 2, 2, 2, 2))?;
    let weight = Tensor::arange(0f32, 48., dev)?.reshape((2, 3, 2, 2, 2))?;

    let config = candle_nn::ConvTranspose3dConfig {
        padding: 0,
        output_padding: 0,
        stride: 2,
        dilation: 1,
        groups: 1,
    };
    let conv = candle_nn::ConvTranspose3d::new(weight, None, config);
    let output = conv.forward(&input)?;
    // (2-1)*2 + 1*(2-1) + 1 - 2*0 = 4
    assert_eq!(output.dims(), [1, 3, 4, 4, 4]);

    Ok(())
}

/// Test ConvTranspose3d with output_padding
fn conv_transpose3d_output_padding(dev: &Device) -> Result<()> {
    let input = Tensor::arange(0f32, 16., dev)?.reshape((1, 2, 2, 2, 2))?;
    let weight = Tensor::arange(0f32, 48., dev)?.reshape((2, 3, 2, 2, 2))?;

    let config = candle_nn::ConvTranspose3dConfig {
        padding: 0,
        output_padding: 1,
        stride: 2,
        dilation: 1,
        groups: 1,
    };
    let conv = candle_nn::ConvTranspose3d::new(weight, None, config);
    let output = conv.forward(&input)?;
    // (2-1)*2 + 1*(2-1) + 1 + 1 - 2*0 = 5
    assert_eq!(output.dims(), [1, 3, 5, 5, 5]);

    Ok(())
}

/// Test ConvTranspose3d with groups
fn conv_transpose3d_groups(dev: &Device) -> Result<()> {
    // Input: [batch=1, channels=4, depth=2, height=2, width=2]
    let input = Tensor::arange(0f32, 32., dev)?.reshape((1, 4, 2, 2, 2))?;
    // Weight: [in_channels=4, out_channels/groups=2, k_d=2, k_h=2, k_w=2]
    let weight = Tensor::arange(0f32, 64., dev)?.reshape((4, 2, 2, 2, 2))?;

    let config = candle_nn::ConvTranspose3dConfig {
        padding: 0,
        output_padding: 0,
        stride: 1,
        dilation: 1,
        groups: 2,
    };
    let conv = candle_nn::ConvTranspose3d::new(weight, None, config);
    let output = conv.forward(&input)?;
    // Output channels: 2 * 2 = 4
    assert_eq!(output.dims(), [1, 4, 3, 3, 3]);

    Ok(())
}

/// Test conv3d helper function with VarBuilder
fn conv3d_var_builder(dev: &Device) -> Result<()> {
    use candle_nn::VarMap;

    let var_map = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle::DType::F32, dev);

    let config = candle_nn::Conv3dConfig::default();
    let conv = candle_nn::conv3d(2, 3, 2, config, vb.pp("conv"))?;

    // Verify weight shape
    assert_eq!(conv.weight().dims(), [3, 2, 2, 2, 2]);
    // Verify bias shape
    assert!(conv.bias().is_some());
    assert_eq!(conv.bias().unwrap().dims(), [3]);

    // Test forward pass
    let input = Tensor::arange(0f32, 128., dev)?.reshape((1, 2, 4, 4, 4))?;
    let output = conv.forward(&input)?;
    assert_eq!(output.dims(), [1, 3, 3, 3, 3]);

    Ok(())
}

/// Test conv3d_no_bias helper function
fn conv3d_no_bias_var_builder(dev: &Device) -> Result<()> {
    use candle_nn::VarMap;

    let var_map = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle::DType::F32, dev);

    let config = candle_nn::Conv3dConfig::default();
    let conv = candle_nn::conv3d_no_bias(2, 3, 2, config, vb.pp("conv"))?;

    // Verify weight shape
    assert_eq!(conv.weight().dims(), [3, 2, 2, 2, 2]);
    // Verify no bias
    assert!(conv.bias().is_none());

    Ok(())
}

/// Test conv_transpose3d helper function
fn conv_transpose3d_var_builder(dev: &Device) -> Result<()> {
    use candle_nn::VarMap;

    let var_map = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle::DType::F32, dev);

    let config = candle_nn::ConvTranspose3dConfig::default();
    let conv = candle_nn::conv_transpose3d(2, 3, 2, config, vb.pp("conv"))?;

    // Verify weight shape: [in_channels, out_channels, k, k, k]
    assert_eq!(conv.weight().dims(), [2, 3, 2, 2, 2]);
    // Verify bias shape
    assert!(conv.bias().is_some());
    assert_eq!(conv.bias().unwrap().dims(), [3]);

    // Test forward pass
    let input = Tensor::arange(0f32, 64., dev)?.reshape((1, 2, 4, 4, 2))?;
    let output = conv.forward(&input)?;
    assert_eq!(output.dims(), [1, 3, 5, 5, 3]);

    Ok(())
}

/// Test conv_transpose3d_no_bias helper function
fn conv_transpose3d_no_bias_var_builder(dev: &Device) -> Result<()> {
    use candle_nn::VarMap;

    let var_map = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle::DType::F32, dev);

    let config = candle_nn::ConvTranspose3dConfig::default();
    let conv = candle_nn::conv_transpose3d_no_bias(2, 3, 2, config, vb.pp("conv"))?;

    // Verify weight shape
    assert_eq!(conv.weight().dims(), [2, 3, 2, 2, 2]);
    // Verify no bias
    assert!(conv.bias().is_none());

    Ok(())
}

test_device!(
    conv3d_module,
    conv3d_module_cpu,
    conv3d_module_gpu,
    conv3d_module_metal
);
test_device!(
    conv3d_config,
    conv3d_config_cpu,
    conv3d_config_gpu,
    conv3d_config_metal
);
test_device!(
    conv3d_groups,
    conv3d_groups_cpu,
    conv3d_groups_gpu,
    conv3d_groups_metal
);
test_device!(
    conv_transpose3d_module,
    conv_transpose3d_module_cpu,
    conv_transpose3d_module_gpu,
    conv_transpose3d_module_metal
);
test_device!(
    conv_transpose3d_stride,
    conv_transpose3d_stride_cpu,
    conv_transpose3d_stride_gpu,
    conv_transpose3d_stride_metal
);
test_device!(
    conv_transpose3d_output_padding,
    conv_transpose3d_output_padding_cpu,
    conv_transpose3d_output_padding_gpu,
    conv_transpose3d_output_padding_metal
);
test_device!(
    conv_transpose3d_groups,
    conv_transpose3d_groups_cpu,
    conv_transpose3d_groups_gpu,
    conv_transpose3d_groups_metal
);
test_device!(
    conv3d_var_builder,
    conv3d_var_builder_cpu,
    conv3d_var_builder_gpu,
    conv3d_var_builder_metal
);
test_device!(
    conv3d_no_bias_var_builder,
    conv3d_no_bias_var_builder_cpu,
    conv3d_no_bias_var_builder_gpu,
    conv3d_no_bias_var_builder_metal
);
test_device!(
    conv_transpose3d_var_builder,
    conv_transpose3d_var_builder_cpu,
    conv_transpose3d_var_builder_gpu,
    conv_transpose3d_var_builder_metal
);
test_device!(
    conv_transpose3d_no_bias_var_builder,
    conv_transpose3d_no_bias_var_builder_cpu,
    conv_transpose3d_no_bias_var_builder_gpu,
    conv_transpose3d_no_bias_var_builder_metal
);
