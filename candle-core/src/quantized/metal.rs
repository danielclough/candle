use super::{GgmlDType, QStorage};
use crate::backend::BackendStorage;
use crate::{DType, MetalDevice, MetalStorage, Result, Shape, D};
use candle_metal_kernels::metal::Buffer;
use std::sync::Arc;

pub struct QMetalStorage {
    dtype: GgmlDType,
    device: MetalDevice,
    buffer: Arc<Buffer>,
}

impl QMetalStorage {
    pub fn zeros(device: &MetalDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size = elem_count * dtype.type_size() / dtype.block_size();
        let buffer = device.allocate_zeros(size)?;
        Ok(Self {
            buffer,
            device: device.clone(),
            dtype,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<MetalStorage> {
        use crate::quantized::k_quants::GgmlType;

        // Debug flag - set DEBUG_DEQUANT=1 to trace dequantization values
        let debug = std::env::var("DEBUG_DEQUANT").is_ok();

        let buffer = self.device.allocate_buffer(self.buffer.length())?;
        let blit = self.device.blit_command_encoder()?;
        blit.set_label("blit_to_cpu");
        blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
        blit.end_encoding();
        self.device.wait_until_completed()?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();

        if debug {
            eprintln!(
                "[DEBUG DEQUANT] dtype={:?}, elem_count={}, block_len={}, buffer_len={}",
                self.dtype,
                elem_count,
                block_len,
                self.buffer.length()
            );
        }

        match self.dtype {
            GgmlDType::F32 => {
                let vec: Vec<f32> = read_to_vec(&buffer, block_len);
                if debug {
                    eprintln!("[DEBUG DEQUANT] F32 vec.len()={}, first 5: {:?}", vec.len(), &vec[..5.min(vec.len())]);
                }
                f32::to_float(&vec, &mut out);
            }
            GgmlDType::F16 => {
                let vec: Vec<half::f16> = read_to_vec(&buffer, block_len);
                if debug {
                    let first5: Vec<f32> = vec.iter().take(5).map(|v| v.to_f32()).collect();
                    eprintln!("[DEBUG DEQUANT] F16 vec.len()={}, first 5 as f32: {:?}", vec.len(), first5);
                }
                half::f16::to_float(&vec, &mut out);
            }
            GgmlDType::BF16 => {
                let vec: Vec<half::bf16> = read_to_vec(&buffer, block_len);
                if debug {
                    let first5: Vec<f32> = vec.iter().take(5).map(|v| v.to_f32()).collect();
                    let raw5: Vec<u16> = vec.iter().take(5).map(|v| v.to_bits()).collect();
                    eprintln!("[DEBUG DEQUANT] BF16 vec.len()={}, first 5 raw bits: {:?}", vec.len(), raw5);
                    eprintln!("[DEBUG DEQUANT] BF16 first 5 as f32: {:?}", first5);
                }
                half::bf16::to_float(&vec, &mut out);
                if debug {
                    eprintln!("[DEBUG DEQUANT] BF16 after to_float, out first 5: {:?}", &out[..5.min(out.len())]);
                }
            }
            GgmlDType::Q4_0 => {
                let vec: Vec<crate::quantized::BlockQ4_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q4_1 => {
                let vec: Vec<crate::quantized::BlockQ4_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q5_0 => {
                let vec: Vec<crate::quantized::BlockQ5_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q5_1 => {
                let vec: Vec<crate::quantized::BlockQ5_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q8_0 => {
                let vec: Vec<crate::quantized::BlockQ8_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q8_1 => {
                let vec: Vec<crate::quantized::BlockQ8_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q2K => {
                let vec: Vec<crate::quantized::BlockQ2K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ2K::to_float(&vec, &mut out);
            }
            GgmlDType::Q3K => {
                let vec: Vec<crate::quantized::BlockQ3K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ3K::to_float(&vec, &mut out);
            }
            GgmlDType::Q4K => {
                let vec: Vec<crate::quantized::BlockQ4K> = read_to_vec(&buffer, block_len);
                if debug {
                    eprintln!("[DEBUG DEQUANT] Q4K block_len={}", block_len);
                }
                crate::quantized::BlockQ4K::to_float(&vec, &mut out);
                if debug {
                    eprintln!("[DEBUG DEQUANT] Q4K after to_float, out first 10: {:?}", &out[..10.min(out.len())]);
                }
            }
            GgmlDType::Q5K => {
                let vec: Vec<crate::quantized::BlockQ5K> = read_to_vec(&buffer, block_len);
                if debug {
                    eprintln!("[DEBUG DEQUANT] Q5K block_len={}", block_len);
                }
                crate::quantized::BlockQ5K::to_float(&vec, &mut out);
                if debug {
                    eprintln!("[DEBUG DEQUANT] Q5K after to_float, out first 10: {:?}", &out[..10.min(out.len())]);
                }
            }
            GgmlDType::Q6K => {
                let vec: Vec<crate::quantized::BlockQ6K> = read_to_vec(&buffer, block_len);
                if debug {
                    eprintln!("[DEBUG DEQUANT] Q6K block_len={}", block_len);
                }
                crate::quantized::BlockQ6K::to_float(&vec, &mut out);
                if debug {
                    eprintln!("[DEBUG DEQUANT] Q6K after to_float, out first 10: {:?}", &out[..10.min(out.len())]);
                }
            }
            GgmlDType::Q8K => {
                let vec: Vec<crate::quantized::BlockQ8K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8K::to_float(&vec, &mut out);
            }
        }

        if debug {
            let sum: f32 = out.iter().sum();
            let mean = sum / out.len() as f32;
            let min = out.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[DEBUG DEQUANT] Final out: len={}, min={:.6}, max={:.6}, mean={:.6}",
                out.len(),
                min,
                max,
                mean
            );
        }

        let buffer = self.device.new_buffer_with_data(&out)?;
        Ok(MetalStorage::new(
            buffer,
            self.device.clone(),
            elem_count,
            DType::F32,
        ))
    }

    pub fn quantize(&mut self, src: &MetalStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu::<f32>()?;
        let elem_count = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let buffer = self.device.new_buffer_with_data(&qcpu_storage.data()?)?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &MetalStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu::<f32>()?;
        let elem_count = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize_imatrix(&src, imatrix_weights, n_per_row)?;
        let buffer = self.device.new_buffer_with_data(&qcpu_storage.data()?)?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &crate::CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Quantization only happens on CPU for now.
        let elem_count = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
        } else {
            unreachable!()
        }

        let buffer = self.device.new_buffer_with_data(&qcpu_storage.data()?)?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let elem_count = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float(src.as_slice::<f32>()?);
        } else {
            unreachable!()
        }

        let buffer = self.device.new_buffer_with_data(&qcpu_storage.data()?)?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.buffer.length()
    }

    fn fwd_mv(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();

        // We always use a single batch dimension and stack all the tensors in the batch on the
        // second dimension as the implementation in candle-metal-kernels doesn't handle batch
        // properly.
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            n => crate::bail!("Invalid rank {n} for quantized matmul metal"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let io_dtype = storage.dtype();
        let io_ggml_dtype = match io_dtype {
            DType::F32 => candle_metal_kernels::GgmlDType::F32,
            DType::F16 => candle_metal_kernels::GgmlDType::F16,
            DType::BF16 => candle_metal_kernels::GgmlDType::BF16,
            dtype => crate::bail!("Unsupported dtype for quantized matmul: {:?}", dtype),
        };
        let dst = device.new_buffer(dst_shape.elem_count(), io_dtype, "qmatmul")?;
        let encoder = device.command_encoder()?;
        // In some cases it would be better to use the mm variant, though it has its drawbacks
        // around memory alignment.
        for batch_id in 0..m {
            candle_metal_kernels::call_quantized_matmul_mv_t(
                device.device(),
                &encoder,
                device.kernels(),
                self.dtype.into(),
                io_ggml_dtype,
                (1, 1, n, k),
                storage.buffer(),
                (layout.start_offset() + batch_id * k) * io_dtype.size_in_bytes(),
                &self.buffer,
                batch_id * n * io_dtype.size_in_bytes(),
                &dst,
            )
            .map_err(MetalError::from)?;
        }
        let dst_storage = crate::MetalStorage::new(dst, device, dst_shape.elem_count(), io_dtype);
        Ok((dst_storage, dst_shape))
    }

    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let n = self_shape.dim(D::Minus2)?;
        let k = self_shape.dim(D::Minus1)?;
        let mut dst_shape = src_shape.dims().to_vec();

        if src_shape.rank() < self_shape.rank() {
            crate::bail!(
                "input rank ({}) must be >= weight rank ({})",
                src_shape.rank(),
                self_shape.rank()
            )
        }

        if src_shape.dim(D::Minus2)? == 1 {
            return self.fwd_mv(self_shape, storage, layout);
        }

        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let io_dtype = storage.dtype();
        let io_ggml_dtype = match io_dtype {
            DType::F32 => candle_metal_kernels::GgmlDType::F32,
            DType::F16 => candle_metal_kernels::GgmlDType::F16,
            DType::BF16 => candle_metal_kernels::GgmlDType::BF16,
            dtype => crate::bail!("Unsupported dtype for quantized matmul: {:?}", dtype),
        };
        let dst = device.new_buffer(dst_shape.elem_count(), io_dtype, "qmatmul")?;
        let encoder = device.command_encoder()?;

        if self_shape.rank() > 4 {
            crate::bail!("weight rank ({}) must be <= 4", self_shape.rank())
        }
        let src0_l = crate::Layout::contiguous(
            [vec![1; 4 - self_shape.rank()], self_shape.dims().to_vec()].concat(),
        );
        let src0_stride = src0_l
            .stride()
            .iter()
            .map(|x| {
                (*x as f32 * (self.dtype.type_size() as f32 / self.dtype.block_size() as f32))
                    as usize
            })
            .collect::<Vec<_>>();

        if src_shape.rank() > 4 {
            crate::bail!("weight rank ({}) must be <= 4", src_shape.rank())
        }
        let src1_l = crate::Layout::contiguous(
            [vec![1; 4 - src_shape.rank()], src_shape.dims().to_vec()].concat(),
        );

        candle_metal_kernels::call_quantized_matmul_mm_t(
            device.device(),
            &encoder,
            device.kernels(),
            self.dtype.into(),
            io_ggml_dtype,
            src0_l.dims(),
            &src0_stride,
            &self.buffer,
            src1_l.dims(),
            &src1_l
                .stride()
                .iter()
                .map(|x| x * io_dtype.size_in_bytes())
                .collect::<Vec<_>>(),
            storage.buffer(),
            src1_l.start_offset() * io_dtype.size_in_bytes(),
            dst_shape.dims(),
            0,
            &dst,
        )
        .map_err(MetalError::from)?;

        let dst_storage = crate::MetalStorage::new(dst, device, dst_shape.elem_count(), io_dtype);
        Ok((dst_storage, dst_shape))
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let buffer = self.device.allocate_buffer(self.buffer.length())?;
        {
            let blit = self.device.blit_command_encoder()?;
            blit.set_label("blit_to_cpu");
            blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
            blit.end_encoding();
        }
        self.device.wait_until_completed()?;
        Ok(read_to_vec::<u8>(&buffer, self.storage_size_in_bytes()))
    }

    // =========================================================================
    // Q8_1 Operations - Stub implementations for fully quantized pipeline
    // These will be implemented with Metal kernels in a future update.
    // =========================================================================

    /// Forward pass with Q8_1 quantized input and output (for QTensor → QTensor matmul)
    pub fn fwd_q8out(
        &self,
        _self_shape: &Shape,
        _input_storage: &QMetalStorage,
        _input_shape: &Shape,
    ) -> Result<(QMetalStorage, Shape)> {
        crate::bail!("Q8_1 fwd_q8out not yet implemented for Metal - coming soon!")
    }

    /// Forward pass with regular Tensor input and Q8_1 output
    pub fn fwd_q8out_tensor(
        &self,
        _self_shape: &Shape,
        _input_storage: &MetalStorage,
        _input_shape: &Shape,
    ) -> Result<(QMetalStorage, Shape)> {
        crate::bail!("Q8_1 fwd_q8out_tensor not yet implemented for Metal - coming soon!")
    }

    /// Element-wise addition of two Q8_1 tensors
    pub fn add_q8_1(&self, _other: &QMetalStorage, _elem_count: usize) -> Result<QMetalStorage> {
        crate::bail!("Q8_1 add not yet implemented for Metal - coming soon!")
    }

    /// Element-wise multiplication of two Q8_1 tensors
    pub fn mul_q8_1(&self, _other: &QMetalStorage, _elem_count: usize) -> Result<QMetalStorage> {
        crate::bail!("Q8_1 mul not yet implemented for Metal - coming soon!")
    }

    /// SiLU activation in Q8_1 format
    pub fn silu_q8_1(&self, _elem_count: usize) -> Result<QMetalStorage> {
        crate::bail!("Q8_1 silu not yet implemented for Metal - coming soon!")
    }

    /// GELU activation in Q8_1 format
    pub fn gelu_q8_1(&self, _elem_count: usize) -> Result<QMetalStorage> {
        crate::bail!("Q8_1 gelu not yet implemented for Metal - coming soon!")
    }

    /// RMSNorm in Q8_1 format
    pub fn rms_norm_q8_1(
        &self,
        _weight_storage: &MetalStorage,
        _num_rows: usize,
        _hidden_size: usize,
        _eps: f32,
    ) -> Result<QMetalStorage> {
        crate::bail!("Q8_1 rms_norm not yet implemented for Metal - coming soon!")
    }

    /// Softmax in Q8_1 format
    pub fn softmax_q8_1(&self, _num_rows: usize, _seq_len: usize) -> Result<QMetalStorage> {
        crate::bail!("Q8_1 softmax not yet implemented for Metal - coming soon!")
    }

    /// RoPE in Q8_1 format
    pub fn rope_q8_1(
        &self,
        _cos_storage: &MetalStorage,
        _sin_storage: &MetalStorage,
        _batch_heads: usize,
        _seq_len: usize,
        _head_dim: usize,
    ) -> Result<QMetalStorage> {
        crate::bail!("Q8_1 rope not yet implemented for Metal - coming soon!")
    }

    /// Top-k selection from Q8_1 logits
    pub fn topk_q8_1(&self, _vocab_size: usize, _k: usize) -> Result<(Vec<i32>, Vec<f32>)> {
        crate::bail!("Q8_1 topk not yet implemented for Metal - coming soon!")
    }

    /// Argmax from Q8_1 logits
    pub fn argmax_q8_1(&self, _vocab_size: usize) -> Result<(i32, f32)> {
        crate::bail!("Q8_1 argmax not yet implemented for Metal - coming soon!")
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &MetalDevice,
    data: &[T],
) -> Result<QStorage> {
    let buffer = device.new_buffer_with_data(data)?;
    let device = device.clone();
    Ok(QStorage::Metal(QMetalStorage {
        dtype: T::DTYPE,
        device,
        buffer,
    }))
}

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

impl From<GgmlDType> for candle_metal_kernels::GgmlDType {
    fn from(value: GgmlDType) -> Self {
        match value {
            GgmlDType::Q4_0 => candle_metal_kernels::GgmlDType::Q4_0,
            GgmlDType::Q4_1 => candle_metal_kernels::GgmlDType::Q4_1,
            GgmlDType::Q5_0 => candle_metal_kernels::GgmlDType::Q5_0,
            GgmlDType::Q5_1 => candle_metal_kernels::GgmlDType::Q5_1,
            GgmlDType::Q8_0 => candle_metal_kernels::GgmlDType::Q8_0,
            GgmlDType::Q8_1 => candle_metal_kernels::GgmlDType::Q8_1,
            GgmlDType::Q2K => candle_metal_kernels::GgmlDType::Q2K,
            GgmlDType::Q3K => candle_metal_kernels::GgmlDType::Q3K,
            GgmlDType::Q4K => candle_metal_kernels::GgmlDType::Q4K,
            GgmlDType::Q5K => candle_metal_kernels::GgmlDType::Q5K,
            GgmlDType::Q6K => candle_metal_kernels::GgmlDType::Q6K,
            GgmlDType::Q8K => candle_metal_kernels::GgmlDType::Q8K,
            GgmlDType::F16 => candle_metal_kernels::GgmlDType::F16,
            GgmlDType::F32 => candle_metal_kernels::GgmlDType::F32,
            GgmlDType::BF16 => candle_metal_kernels::GgmlDType::BF16,
        }
    }
}
