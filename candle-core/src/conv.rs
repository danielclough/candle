//! 1D, 2D, and 3D Convolutions
//!
use crate::{op::BackpropOp, op::Op, Error, Result, Tensor};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConv1D {
    pub(crate) b_size: usize,
    // Maybe we should have a version without l_in as this bit depends on the input and not only on
    // the weights.
    pub(crate) l_in: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) k_size: usize,
    pub(crate) padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
    pub(crate) cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl ParamsConv1D {
    pub(crate) fn l_out(&self) -> usize {
        (self.l_in + 2 * self.padding - self.dilation * (self.k_size - 1) - 1) / self.stride + 1
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        let l_out = self.l_out();
        vec![self.b_size, self.c_out, l_out]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConvTranspose1D {
    pub(crate) b_size: usize,
    pub(crate) l_in: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) k_size: usize,
    pub(crate) padding: usize,
    pub(crate) output_padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
}

impl ParamsConvTranspose1D {
    pub(crate) fn l_out(&self) -> usize {
        (self.l_in - 1) * self.stride - 2 * self.padding
            + self.dilation * (self.k_size - 1)
            + self.output_padding
            + 1
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        let l_out = self.l_out();
        vec![self.b_size, self.c_out, l_out]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudnnFwdAlgo {
    ImplicitGemm,
    ImplicitPrecompGemm,
    Gemm,
    Direct,
    Fft,
    FftTiling,
    Winograd,
    WinogradNonFused,
    Count,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConv2D {
    pub(crate) b_size: usize,
    pub(crate) i_h: usize,
    pub(crate) i_w: usize,
    pub(crate) k_h: usize,
    pub(crate) k_w: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
    pub cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl ParamsConv2D {
    pub(crate) fn out_h(&self) -> usize {
        (self.i_h + 2 * self.padding - self.dilation * (self.k_h - 1) - 1) / self.stride + 1
    }

    pub(crate) fn out_w(&self) -> usize {
        (self.i_w + 2 * self.padding - self.dilation * (self.k_w - 1) - 1) / self.stride + 1
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        vec![self.b_size, self.c_out, self.out_h(), self.out_w()]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConvTranspose2D {
    pub(crate) b_size: usize,
    pub(crate) i_h: usize,
    pub(crate) i_w: usize,
    pub(crate) k_h: usize,
    pub(crate) k_w: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) padding: usize,
    pub(crate) output_padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
}

impl ParamsConvTranspose2D {
    pub(crate) fn out_h(&self) -> usize {
        (self.i_h - 1) * self.stride + self.dilation * (self.k_h - 1) + self.output_padding + 1
            - 2 * self.padding
    }

    pub(crate) fn out_w(&self) -> usize {
        (self.i_w - 1) * self.stride + self.dilation * (self.k_w - 1) + self.output_padding + 1
            - 2 * self.padding
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        vec![self.b_size, self.c_out, self.out_h(), self.out_w()]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConv3D {
    pub(crate) b_size: usize,
    pub(crate) i_d: usize, // input depth
    pub(crate) i_h: usize, // input height
    pub(crate) i_w: usize, // input width
    pub(crate) k_d: usize, // kernel depth
    pub(crate) k_h: usize, // kernel height
    pub(crate) k_w: usize, // kernel width
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    // Per-dimension parameters (D, H, W)
    pub(crate) padding_d: usize,
    pub(crate) padding_h: usize,
    pub(crate) padding_w: usize,
    pub(crate) stride_d: usize,
    pub(crate) stride_h: usize,
    pub(crate) stride_w: usize,
    pub(crate) dilation_d: usize,
    pub(crate) dilation_h: usize,
    pub(crate) dilation_w: usize,
    pub cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl ParamsConv3D {
    pub(crate) fn out_d(&self) -> usize {
        (self.i_d + 2 * self.padding_d - self.dilation_d * (self.k_d - 1) - 1) / self.stride_d + 1
    }

    pub(crate) fn out_h(&self) -> usize {
        (self.i_h + 2 * self.padding_h - self.dilation_h * (self.k_h - 1) - 1) / self.stride_h + 1
    }

    pub(crate) fn out_w(&self) -> usize {
        (self.i_w + 2 * self.padding_w - self.dilation_w * (self.k_w - 1) - 1) / self.stride_w + 1
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        vec![
            self.b_size,
            self.c_out,
            self.out_d(),
            self.out_h(),
            self.out_w(),
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConvTranspose3D {
    pub(crate) b_size: usize,
    pub(crate) i_d: usize, // input depth
    pub(crate) i_h: usize, // input height
    pub(crate) i_w: usize, // input width
    pub(crate) k_d: usize, // kernel depth
    pub(crate) k_h: usize, // kernel height
    pub(crate) k_w: usize, // kernel width
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    // Per-dimension parameters (D, H, W)
    pub(crate) padding_d: usize,
    pub(crate) padding_h: usize,
    pub(crate) padding_w: usize,
    pub(crate) output_padding_d: usize,
    pub(crate) output_padding_h: usize,
    pub(crate) output_padding_w: usize,
    pub(crate) stride_d: usize,
    pub(crate) stride_h: usize,
    pub(crate) stride_w: usize,
    pub(crate) dilation_d: usize,
    pub(crate) dilation_h: usize,
    pub(crate) dilation_w: usize,
}

impl ParamsConvTranspose3D {
    pub(crate) fn out_d(&self) -> usize {
        (self.i_d - 1) * self.stride_d
            + self.dilation_d * (self.k_d - 1)
            + self.output_padding_d
            + 1
            - 2 * self.padding_d
    }

    pub(crate) fn out_h(&self) -> usize {
        (self.i_h - 1) * self.stride_h
            + self.dilation_h * (self.k_h - 1)
            + self.output_padding_h
            + 1
            - 2 * self.padding_h
    }

    pub(crate) fn out_w(&self) -> usize {
        (self.i_w - 1) * self.stride_w
            + self.dilation_w * (self.k_w - 1)
            + self.output_padding_w
            + 1
            - 2 * self.padding_w
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        vec![
            self.b_size,
            self.c_out,
            self.out_d(),
            self.out_h(),
            self.out_w(),
        ]
    }
}

impl Tensor {
    fn conv1d_single_group(&self, kernel: &Self, params: &ParamsConv1D) -> Result<Self> {
        let storage =
            self.storage()
                .conv1d(self.layout(), &kernel.storage(), kernel.layout(), params)?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::Conv1D {
            arg,
            kernel,
            padding: params.padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }

    /// Applies a 1D convolution over the input tensor.
    pub fn conv1d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        self.conv1d_with_algo(kernel, padding, stride, dilation, groups, None)
    }

    /// Applies a 1D convolution over the input tensor.
    pub fn conv1d_with_algo(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        cudnn_fwd_algo: Option<CudnnFwdAlgo>,
    ) -> Result<Self> {
        let (c_out, c_in_k, k_size) = kernel.dims3()?;
        let (b_size, c_in, l_in) = self.dims3()?;
        if c_in != c_in_k * groups {
            Err(Error::Conv1dInvalidArgs {
                inp_shape: self.shape().clone(),
                k_shape: kernel.shape().clone(),
                padding,
                stride,
                msg: "the number of in-channels on the input doesn't match the kernel size",
            }
            .bt())?
        }

        let params = ParamsConv1D {
            b_size,
            l_in,
            c_out: c_out / groups,
            c_in: c_in / groups,
            k_size,
            padding,
            stride,
            dilation,
            cudnn_fwd_algo,
        };
        if groups == 1 {
            self.conv1d_single_group(kernel, &params)
        } else {
            let blocks = self.chunk(groups, 1)?;
            let kernel = kernel.chunk(groups, 0)?;
            let blocks = blocks
                .iter()
                .zip(&kernel)
                .map(|(block, kernel)| block.conv1d_single_group(kernel, &params))
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&blocks, 1)
        }
    }

    fn conv_transpose1d_single_group(
        &self,
        kernel: &Self,
        params: &ParamsConvTranspose1D,
    ) -> Result<Self> {
        let storage = self.storage().conv_transpose1d(
            self.layout(),
            &kernel.storage(),
            kernel.layout(),
            params,
        )?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::ConvTranspose1D {
            arg,
            kernel,
            padding: params.padding,
            output_padding: params.output_padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }

    /// Applies a 1D transposed convolution over the input tensor.
    pub fn conv_transpose1d(
        &self,
        kernel: &Self,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let (c_in_k, c_out, k_size) = kernel.dims3()?;
        let (b_size, c_in, l_in) = self.dims3()?;
        if c_in != c_in_k {
            crate::bail!("in_channel mismatch between input ({c_in}) and kernel ({c_in_k})")
        }
        if c_in % groups != 0 {
            crate::bail!("in_channel {c_in} is not divisible by the number of groups")
        }
        let params = ParamsConvTranspose1D {
            b_size,
            l_in,
            k_size,
            c_out,
            c_in: c_in / groups,
            padding,
            output_padding,
            stride,
            dilation,
        };
        if groups == 1 {
            self.conv_transpose1d_single_group(kernel, &params)
        } else {
            let blocks = self.chunk(groups, 1)?;
            let kernel = kernel.chunk(groups, 0)?;
            let blocks = blocks
                .iter()
                .zip(&kernel)
                .map(|(block, kernel)| block.conv_transpose1d_single_group(kernel, &params))
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&blocks, 1)
        }
    }

    fn conv2d_single_group(&self, kernel: &Self, params: &ParamsConv2D) -> Result<Self> {
        let storage =
            self.storage()
                .conv2d(self.layout(), &kernel.storage(), kernel.layout(), params)?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::Conv2D {
            arg,
            kernel,
            padding: params.padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }

    /// Applies a 2D convolution over the input tensor.
    pub fn conv2d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        self.conv2d_with_algo(kernel, padding, stride, dilation, groups, None)
    }

    pub fn conv2d_with_algo(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        cudnn_fwd_algo: Option<CudnnFwdAlgo>,
    ) -> Result<Self> {
        let (b_size, c_in, i_h, i_w) = self.dims4()?;
        let (c_out, c_in_k, k_h, k_w) = kernel.dims4()?;
        if c_in != c_in_k * groups {
            crate::bail!(
                "in_channel mismatch between input ({c_in}, groups {groups}) and kernel ({c_in_k})"
            )
        }
        let params = ParamsConv2D {
            b_size,
            i_h,
            i_w,
            k_h,
            k_w,
            c_out: c_out / groups,
            c_in: c_in / groups,
            padding,
            stride,
            dilation,
            cudnn_fwd_algo,
        };
        if groups == 1 {
            self.conv2d_single_group(kernel, &params)
        } else {
            let blocks = self.chunk(groups, 1)?;
            let kernel = kernel.chunk(groups, 0)?;
            let blocks = blocks
                .iter()
                .zip(&kernel)
                .map(|(block, kernel)| block.conv2d_single_group(kernel, &params))
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&blocks, 1)
        }
    }

    /// Applies a 2D transposed convolution over the input tensor.
    pub fn conv_transpose2d(
        &self,
        kernel: &Self,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
    ) -> Result<Self> {
        let (b_size, c_in, i_h, i_w) = self.dims4()?;
        let (c_in_k, c_out, k_h, k_w) = kernel.dims4()?;
        if c_in != c_in_k {
            crate::bail!("in_channel mismatch between input ({c_in}) and kernel ({c_in_k})")
        }
        let params = ParamsConvTranspose2D {
            b_size,
            i_h,
            i_w,
            k_h,
            k_w,
            c_out,
            c_in,
            padding,
            output_padding,
            stride,
            dilation,
        };
        let storage = self.storage().conv_transpose2d(
            self.layout(),
            &kernel.storage(),
            kernel.layout(),
            &params,
        )?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::ConvTranspose2D {
            arg,
            kernel,
            padding: params.padding,
            output_padding: params.output_padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }

    fn conv3d_single_group(&self, kernel: &Self, params: &ParamsConv3D) -> Result<Self> {
        let storage =
            self.storage()
                .conv3d(self.layout(), &kernel.storage(), kernel.layout(), params)?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::Conv3D {
            arg,
            kernel,
            padding_d: params.padding_d,
            padding_h: params.padding_h,
            padding_w: params.padding_w,
            stride_d: params.stride_d,
            stride_h: params.stride_h,
            stride_w: params.stride_w,
            dilation_d: params.dilation_d,
            dilation_h: params.dilation_h,
            dilation_w: params.dilation_w,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }

    /// Applies a 3D convolution over the input tensor.
    ///
    /// The input tensor should have shape `[batch_size, in_channels, depth, height, width]`.
    /// The kernel should have shape `[out_channels, in_channels/groups, kernel_depth, kernel_height, kernel_width]`.
    ///
    /// # Arguments
    /// * `kernel` - The convolution kernel/filter weights
    /// * `padding` - Padding (depth, height, width) to apply to the input
    /// * `stride` - Stride (depth, height, width) for the convolution
    /// * `dilation` - Dilation (depth, height, width) factor for the kernel
    /// * `groups` - Number of groups for grouped convolution
    pub fn conv3d(
        &self,
        kernel: &Self,
        padding: (usize, usize, usize),
        stride: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
    ) -> Result<Self> {
        self.conv3d_with_algo(kernel, padding, stride, dilation, groups, None)
    }

    /// Applies a 3D convolution over the input tensor with an optional cuDNN algorithm.
    pub fn conv3d_with_algo(
        &self,
        kernel: &Self,
        padding: (usize, usize, usize),
        stride: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
        cudnn_fwd_algo: Option<CudnnFwdAlgo>,
    ) -> Result<Self> {
        let (b_size, c_in, i_d, i_h, i_w) = self.dims5()?;
        let (c_out, c_in_k, k_d, k_h, k_w) = kernel.dims5()?;
        if c_in != c_in_k * groups {
            Err(Error::Conv3dInvalidArgs {
                inp_shape: self.shape().clone(),
                k_shape: kernel.shape().clone(),
                padding,
                stride,
                msg: "the number of in-channels on the input doesn't match the kernel size",
            }
            .bt())?
        }
        let (padding_d, padding_h, padding_w) = padding;
        let (stride_d, stride_h, stride_w) = stride;
        let (dilation_d, dilation_h, dilation_w) = dilation;
        let params = ParamsConv3D {
            b_size,
            i_d,
            i_h,
            i_w,
            k_d,
            k_h,
            k_w,
            c_out: c_out / groups,
            c_in: c_in / groups,
            padding_d,
            padding_h,
            padding_w,
            stride_d,
            stride_h,
            stride_w,
            dilation_d,
            dilation_h,
            dilation_w,
            cudnn_fwd_algo,
        };
        if groups == 1 {
            self.conv3d_single_group(kernel, &params)
        } else {
            let blocks = self.chunk(groups, 1)?;
            let kernel = kernel.chunk(groups, 0)?;
            let blocks = blocks
                .iter()
                .zip(&kernel)
                .map(|(block, kernel)| block.conv3d_single_group(kernel, &params))
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&blocks, 1)
        }
    }

    fn conv_transpose3d_single_group(
        &self,
        kernel: &Self,
        params: &ParamsConvTranspose3D,
        groups: usize,
    ) -> Result<Self> {
        let storage = self.storage().conv_transpose3d(
            self.layout(),
            &kernel.storage(),
            kernel.layout(),
            params,
        )?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::ConvTranspose3D {
            arg,
            kernel,
            padding_d: params.padding_d,
            padding_h: params.padding_h,
            padding_w: params.padding_w,
            output_padding_d: params.output_padding_d,
            output_padding_h: params.output_padding_h,
            output_padding_w: params.output_padding_w,
            stride_d: params.stride_d,
            stride_h: params.stride_h,
            stride_w: params.stride_w,
            dilation_d: params.dilation_d,
            dilation_h: params.dilation_h,
            dilation_w: params.dilation_w,
            groups,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }

    /// Applies a 3D transposed convolution over the input tensor.
    ///
    /// The input tensor should have shape `[batch_size, in_channels, depth, height, width]`.
    /// The kernel should have shape `[in_channels, out_channels/groups, kernel_depth, kernel_height, kernel_width]`.
    ///
    /// # Arguments
    /// * `kernel` - The convolution kernel/filter weights
    /// * `padding` - Padding (depth, height, width) to apply to the input
    /// * `output_padding` - Output padding (depth, height, width) to add to the output
    /// * `stride` - Stride (depth, height, width) for the convolution
    /// * `dilation` - Dilation (depth, height, width) factor for the kernel
    /// * `groups` - Number of groups for grouped convolution
    pub fn conv_transpose3d(
        &self,
        kernel: &Self,
        padding: (usize, usize, usize),
        output_padding: (usize, usize, usize),
        stride: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
    ) -> Result<Self> {
        let (b_size, c_in, i_d, i_h, i_w) = self.dims5()?;
        let (c_in_k, c_out, k_d, k_h, k_w) = kernel.dims5()?;
        if c_in != c_in_k {
            crate::bail!("in_channel mismatch between input ({c_in}) and kernel ({c_in_k})")
        }
        if c_in % groups != 0 {
            crate::bail!("in_channel {c_in} is not divisible by the number of groups {groups}")
        }
        let (padding_d, padding_h, padding_w) = padding;
        let (output_padding_d, output_padding_h, output_padding_w) = output_padding;
        let (stride_d, stride_h, stride_w) = stride;
        let (dilation_d, dilation_h, dilation_w) = dilation;
        let params = ParamsConvTranspose3D {
            b_size,
            i_d,
            i_h,
            i_w,
            k_d,
            k_h,
            k_w,
            c_out,
            c_in: c_in / groups,
            padding_d,
            padding_h,
            padding_w,
            output_padding_d,
            output_padding_h,
            output_padding_w,
            stride_d,
            stride_h,
            stride_w,
            dilation_d,
            dilation_h,
            dilation_w,
        };
        if groups == 1 {
            self.conv_transpose3d_single_group(kernel, &params, groups)
        } else {
            let blocks = self.chunk(groups, 1)?;
            let kernel_chunks = kernel.chunk(groups, 0)?;
            let blocks = blocks
                .iter()
                .zip(&kernel_chunks)
                .map(|(block, k)| block.conv_transpose3d_single_group(k, &params, groups))
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&blocks, 1)
        }
    }
}
