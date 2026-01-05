use std::borrow::Cow;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    conv::{ParamsConv3D, ParamsConvTranspose3D},
    cpu_backend::{copy_strided_src_, Map2, MatMul},
    Layout, Result, WithDType,
};

pub(super) struct Conv3D<'a>(pub(super) &'a ParamsConv3D);

enum Conv3dImpl {
    TiledIm2Col,
    FullIm2Col,
    Direct,
}

const DEFAULT_CONV3D_IMPL: Conv3dImpl = Conv3dImpl::TiledIm2Col;

impl Map2 for Conv3D<'_> {
    const OP: &'static str = "conv3d";
    fn f<T: WithDType + num_traits::Num + Copy + 'static>(
        &self,
        inp: &[T],
        inp_l: &Layout,
        k: &[T],
        k_l: &Layout,
    ) -> Result<Vec<T>> {
        let p = self.0;

        // Specialization: pick the best algorithm based on parameters.
        // 1x1x1 convolutions with stride=1, padding=0, dilation=1
        if p.k_d == 1
            && p.k_h == 1
            && p.k_w == 1
            && p.stride_d == 1
            && p.stride_h == 1
            && p.stride_w == 1
            && p.padding_d == 0
            && p.padding_h == 0
            && p.padding_w == 0
            && p.dilation_d == 1
            && p.dilation_h == 1
            && p.dilation_w == 1
        {
            return conv3d_1x1x1(p, inp, inp_l, k, k_l);
        } else if p.k_d == 1 && p.k_h == 1 && p.k_w == 1 {
            // Other 1x1x1 convolutions for now are assumed faster with full im2col.
            return conv3d_im2col_gemm(p, inp, inp_l, k, k_l);
        }

        // No fast path, fallback to default general impl.
        match DEFAULT_CONV3D_IMPL {
            Conv3dImpl::TiledIm2Col => conv3d_tiled(p, inp, inp_l, k, k_l),
            Conv3dImpl::Direct => conv3d_direct(p, inp, inp_l, k, k_l),
            Conv3dImpl::FullIm2Col => conv3d_im2col_gemm(p, inp, inp_l, k, k_l),
        }
    }
}

/// Helper to extract 5D strides from a layout's stride slice.
fn dims5_strides(stride: &[usize]) -> Result<(usize, usize, usize, usize, usize)> {
    if stride.len() != 5 {
        crate::bail!("Expected 5D strides, got {}", stride.len());
    }
    Ok((stride[0], stride[1], stride[2], stride[3], stride[4]))
}

/// Fast kernel for 1x1x1 convolutions with stride=1, padding=0, dilation=1
/// These are just matrix multiplications: [c_out, c_in] @ [c_in, b*d*h*w] -> [c_out, b*d*h*w].
fn conv3d_1x1x1<T: WithDType + num_traits::Num + Copy + 'static>(
    p: &ParamsConv3D,
    inp: &[T],
    inp_l: &Layout,
    k: &[T],
    k_l: &Layout,
) -> Result<Vec<T>> {
    let inp = &inp[inp_l.start_offset()..];
    let (inp_s0, inp_s1, inp_s2, inp_s3, inp_s4) = dims5_strides(inp_l.stride())?;
    let k = &k[k_l.start_offset()..];
    let k_stride = k_l.stride();
    let (k_s0, k_s1) = (k_stride[0], k_stride[1]);
    let (out_d, out_h, out_w) = (p.out_d(), p.out_h(), p.out_w());

    let spatial_size = out_d * out_h * out_w;
    let dst = vec![T::zero(); p.b_size * p.c_out * spatial_size];
    let k_reshaped: Cow<[T]> = if k_s0 == p.c_in && k_s1 == 1 {
        // Already contiguous, use slice directly
        Cow::Borrowed(&k[..p.c_out * p.c_in])
    } else {
        // Reshape kernel to [c_out, c_in]
        let mut k_reshaped = Vec::with_capacity(p.c_out * p.c_in);
        (0..p.c_out).for_each(|c_out_idx| {
            (0..p.c_in).for_each(|c_in_idx| {
                let k_idx = c_out_idx * k_s0 + c_in_idx * k_s1;
                k_reshaped.push(k[k_idx]);
            });
        });
        Cow::Owned(k_reshaped)
    };
    let k_layout = Layout::contiguous((p.c_out, p.c_in));

    // Process each batch
    (0..p.b_size).into_par_iter().try_for_each(|b_idx| {
        // Reshape input to [c_in, d*h*w] for this batch
        let mut inp_reshaped = Vec::with_capacity(p.c_in * spatial_size);
        for c_in_idx in 0..p.c_in {
            for d_idx in 0..p.i_d {
                for h_idx in 0..p.i_h {
                    for w_idx in 0..p.i_w {
                        let inp_idx = b_idx * inp_s0
                            + c_in_idx * inp_s1
                            + d_idx * inp_s2
                            + h_idx * inp_s3
                            + w_idx * inp_s4;
                        inp_reshaped.push(inp[inp_idx]);
                    }
                }
            }
        }
        let inp_layout = Layout::contiguous((p.c_in, spatial_size));

        // Perform matmul: [c_out, c_in] @ [c_in, spatial_size] -> [c_out, spatial_size]
        let matmul = MatMul((1, p.c_out, spatial_size, p.c_in));
        let result = matmul.f(&k_reshaped, &k_layout, &inp_reshaped, &inp_layout)?;

        // Copy result to output
        let out_offset = b_idx * p.c_out * spatial_size;
        for (i, r) in result.iter().enumerate() {
            unsafe {
                let ptr = dst.as_ptr().add(out_offset + i) as *mut T;
                *ptr = *r;
            }
        }
        Ok::<(), crate::Error>(())
    })?;

    Ok(dst)
}

/// General tiled convolution implementation using gemm.
///
/// Similar to full im2col, but instead of materializing the full matrix, we process input/output in tiles, in parallel.
fn conv3d_tiled<T: WithDType + num_traits::Num + Copy + 'static>(
    p: &ParamsConv3D,
    inp: &[T],
    inp_l: &Layout,
    k: &[T],
    k_l: &Layout,
) -> Result<Vec<T>> {
    let inp = &inp[inp_l.start_offset()..];
    let (inp_s0, inp_s1, inp_s2, inp_s3, inp_s4) = dims5_strides(inp_l.stride())?;
    let k = &k[k_l.start_offset()..];
    let (k_s0, k_s1, k_s2, k_s3, k_s4) = dims5_strides(k_l.stride())?;
    let (out_d, out_h, out_w) = (p.out_d(), p.out_h(), p.out_w());

    // Output shape: [b_size, c_out, out_d, out_h, out_w].
    let dst = vec![T::zero(); p.b_size * p.c_out * out_d * out_h * out_w];

    // Make contiguous input copy if needed.
    // Target layout: [b, d, h, w, c] for efficient im2col
    let cont_s0 = p.i_d * p.i_h * p.i_w * p.c_in;
    let cont_s1 = p.i_h * p.i_w * p.c_in;
    let cont_s2 = p.i_w * p.c_in;
    let cont_s3 = p.c_in;
    let layout_is_valid = inp_l.stride() == [cont_s0, cont_s1, cont_s2, cont_s3, 1];
    let inp_cont: Cow<[T]> = if layout_is_valid {
        Cow::Borrowed(inp)
    } else {
        let mut inp_cont = vec![T::zero(); p.b_size * p.c_in * p.i_d * p.i_h * p.i_w];
        for b_idx in 0..p.b_size {
            for d_idx in 0..p.i_d {
                for h_idx in 0..p.i_h {
                    for w_idx in 0..p.i_w {
                        for c_idx in 0..p.c_in {
                            let src_idx = b_idx * inp_s0
                                + c_idx * inp_s1
                                + d_idx * inp_s2
                                + h_idx * inp_s3
                                + w_idx * inp_s4;
                            let dst_idx = b_idx * cont_s0
                                + d_idx * cont_s1
                                + h_idx * cont_s2
                                + w_idx * cont_s3
                                + c_idx;
                            inp_cont[dst_idx] = inp[src_idx];
                        }
                    }
                }
            }
        }
        Cow::Owned(inp_cont)
    };

    // shape of k: [c_out, c_in, k_d, k_h, k_w]
    // For matmul, we need flattened k in shape [c_out, k_d * k_h * k_w * c_in]
    let k_size = p.c_in * p.k_d * p.k_h * p.k_w;
    let mut k_flat = Vec::with_capacity(p.c_out * k_size);
    for dst_c_idx in 0..p.c_out {
        for kd in 0..p.k_d {
            for kh in 0..p.k_h {
                for kw in 0..p.k_w {
                    for c_in_idx in 0..p.c_in {
                        let k_idx = dst_c_idx * k_s0
                            + c_in_idx * k_s1
                            + kd * k_s2
                            + kh * k_s3
                            + kw * k_s4;
                        k_flat.push(k[k_idx]);
                    }
                }
            }
        }
    }
    let k_layout = Layout::contiguous((p.c_out, k_size));

    // TILE_SIZE is number of output voxels (out_d * out_h * out_w) per tile.
    const TILE_SIZE: usize = 512;

    let total_out_voxels = out_d * out_h * out_w;

    // Process batches and tiles in parallel using rayon.
    (0..p.b_size).into_par_iter().try_for_each(|b_idx| {
        let inp_offset = b_idx * cont_s0;
        let out_batch_offset = b_idx * (p.c_out * out_d * out_h * out_w);

        let num_tiles = total_out_voxels.div_ceil(TILE_SIZE);
        (0..num_tiles).into_par_iter().try_for_each(|tile_idx| {
            // Determine actual tile size (may be smaller at the end)
            let tile_start = tile_idx * TILE_SIZE;
            let tile_end = (tile_start + TILE_SIZE).min(total_out_voxels);
            let tile_size = tile_end - tile_start;

            // Precompute output coordinates.
            let out_hw = out_h * out_w;
            let out_coords: Vec<_> = (tile_start..tile_end)
                .map(|idx| {
                    let out_z = idx / out_hw;
                    let rem = idx % out_hw;
                    let out_y = rem / out_w;
                    let out_x = rem % out_w;
                    (out_z, out_y, out_x)
                })
                .collect();

            // Build im2col tile: [k_size, tile_size]
            let mut col_tile = vec![T::zero(); k_size * tile_size];

            for (tile_idx, (out_z, out_y, out_x)) in out_coords.iter().enumerate() {
                // Extract the im2col patch for this output position
                for c_in in 0..p.c_in {
                    let mut patch_offset = c_in;
                    for kd in 0..p.k_d {
                        let in_z = (out_z * p.stride_d + kd * p.dilation_d) as isize
                            - p.padding_d as isize;
                        if in_z < 0 || in_z >= p.i_d as isize {
                            patch_offset += p.c_in * p.k_h * p.k_w;
                            continue;
                        }
                        for kh in 0..p.k_h {
                            let in_y = (out_y * p.stride_h + kh * p.dilation_h) as isize
                                - p.padding_h as isize;
                            if in_y < 0 || in_y >= p.i_h as isize {
                                patch_offset += p.c_in * p.k_w;
                                continue;
                            }
                            for kw in 0..p.k_w {
                                let in_x = (out_x * p.stride_w + kw * p.dilation_w) as isize
                                    - p.padding_w as isize;

                                if in_x >= 0 && in_x < p.i_w as isize {
                                    let in_z = in_z as usize;
                                    let in_y = in_y as usize;
                                    let in_x = in_x as usize;
                                    let inp_idx = inp_offset
                                        + in_z * cont_s1
                                        + in_y * cont_s2
                                        + in_x * cont_s3
                                        + c_in;
                                    let col_idx = patch_offset * tile_size + tile_idx;
                                    col_tile[col_idx] = inp_cont[inp_idx];
                                }
                                patch_offset += p.c_in;
                            }
                        }
                    }
                }
            }

            // Now perform matmul: k_flat [c_out, k_size] @ col_tile [k_size, tile_size]
            let matmul = MatMul((1, p.c_out, tile_size, k_size));
            let col_layout = Layout::contiguous((k_size, tile_size));
            let result = matmul.f(&k_flat, &k_layout, &col_tile, &col_layout)?;

            // Copy results to output: result is [c_out, tile_size]
            for (tile_idx, (out_z, out_y, out_x)) in out_coords.iter().enumerate() {
                let dst_base = out_batch_offset + out_z * out_hw + out_y * out_w + out_x;

                for c_out_idx in 0..p.c_out {
                    let dst_idx = dst_base + c_out_idx * (out_d * out_h * out_w);
                    let result_idx = c_out_idx * tile_size + tile_idx;
                    unsafe {
                        let ptr = dst.as_ptr().add(dst_idx) as *mut T;
                        *ptr = result[result_idx];
                    }
                }
            }
            Ok::<(), crate::Error>(())
        })
    })?;

    Ok(dst)
}

/// General direct convolution impl. Decently fast for small inputs and kernels.
fn conv3d_direct<T: WithDType + num_traits::Num + Copy + 'static>(
    p: &ParamsConv3D,
    inp: &[T],
    inp_l: &Layout,
    k: &[T],
    k_l: &Layout,
) -> Result<Vec<T>> {
    let inp = &inp[inp_l.start_offset()..];
    let (inp_s0, inp_s1, inp_s2, inp_s3, inp_s4) = dims5_strides(inp_l.stride())?;
    let k = &k[k_l.start_offset()..];
    let (k_s0, k_s1, k_s2, k_s3, k_s4) = dims5_strides(k_l.stride())?;
    let (out_d, out_h, out_w) = (p.out_d(), p.out_h(), p.out_w());

    // Output shape: [b_size, c_out, out_d, out_h, out_w].
    let dst = vec![T::zero(); p.b_size * p.c_out * out_d * out_h * out_w];

    // Make contiguous input copy if needed.
    let cont_s0 = p.i_d * p.i_h * p.i_w * p.c_in;
    let cont_s1 = p.i_h * p.i_w * p.c_in;
    let cont_s2 = p.i_w * p.c_in;
    let cont_s3 = p.c_in;
    let layout_is_valid = inp_l.stride() == [cont_s0, cont_s1, cont_s2, cont_s3, 1];
    let inp_cont: Cow<[T]> = if layout_is_valid {
        Cow::Borrowed(inp)
    } else {
        let mut inp_cont = vec![T::zero(); p.b_size * p.c_in * p.i_d * p.i_h * p.i_w];
        for b_idx in 0..p.b_size {
            for d_idx in 0..p.i_d {
                for h_idx in 0..p.i_h {
                    for w_idx in 0..p.i_w {
                        for c_idx in 0..p.c_in {
                            let src_idx = b_idx * inp_s0
                                + c_idx * inp_s1
                                + d_idx * inp_s2
                                + h_idx * inp_s3
                                + w_idx * inp_s4;
                            let dst_idx = b_idx * cont_s0
                                + d_idx * cont_s1
                                + h_idx * cont_s2
                                + w_idx * cont_s3
                                + c_idx;
                            inp_cont[dst_idx] = inp[src_idx];
                        }
                    }
                }
            }
        }
        Cow::Owned(inp_cont)
    };
    let inp_cont_len = inp_cont.len();

    // Precompute kernel cache
    let k_cache: Vec<Vec<T>> = (0..p.c_out)
        .map(|dst_c_idx| {
            (0..p.k_d * p.k_h * p.k_w)
                .flat_map(|kdhw| {
                    let kd = kdhw / (p.k_h * p.k_w);
                    let rem = kdhw % (p.k_h * p.k_w);
                    let kh = rem / p.k_w;
                    let kw = rem % p.k_w;
                    (0..p.c_in).map(move |c_in_idx| {
                        k[dst_c_idx * k_s0
                            + c_in_idx * k_s1
                            + kd * k_s2
                            + kh * k_s3
                            + kw * k_s4]
                    })
                })
                .collect()
        })
        .collect();

    for b_idx in 0..p.b_size {
        for offset_d in 0..p.k_d {
            for offset_h in 0..p.k_h {
                for offset_w in 0..p.k_w {
                    let k_offset = (offset_d * p.k_h + offset_h) * p.k_w + offset_w;

                    (0..p.c_out).into_par_iter().for_each(|dst_c_idx| {
                        let k_cont =
                            &k_cache[dst_c_idx][k_offset * p.c_in..(k_offset + 1) * p.c_in];
                        let base_dst_idx = dst_c_idx * out_d * out_h * out_w;
                        let batch_dst_idx =
                            base_dst_idx + b_idx * p.c_out * out_d * out_h * out_w;
                        let batch_src_idx = b_idx * cont_s0;

                        for dst_d in 0..out_d {
                            let src_d = p.stride_d * dst_d + offset_d * p.dilation_d;
                            if src_d < p.padding_d || src_d >= p.i_d + p.padding_d {
                                continue;
                            }
                            let src_d = src_d - p.padding_d;
                            let d_dst_idx = batch_dst_idx + dst_d * out_h * out_w;
                            let d_src_idx = batch_src_idx + src_d * cont_s1;

                            for dst_h in 0..out_h {
                                let src_h = p.stride_h * dst_h + offset_h * p.dilation_h;
                                if src_h < p.padding_h || src_h >= p.i_h + p.padding_h {
                                    continue;
                                }
                                let src_h = src_h - p.padding_h;
                                let h_dst_idx = d_dst_idx + dst_h * out_w;
                                let h_src_idx = d_src_idx + src_h * cont_s2;

                                for dst_w in 0..out_w {
                                    let src_w = p.stride_w * dst_w + offset_w * p.dilation_w;
                                    if src_w < p.padding_w || src_w >= p.i_w + p.padding_w {
                                        continue;
                                    }
                                    let src_w = src_w - p.padding_w;
                                    let dst_idx = h_dst_idx + dst_w;
                                    let inp_idx_1 = h_src_idx + src_w * cont_s3;
                                    let inp_idx_2 = (inp_idx_1 + p.c_in).min(inp_cont_len);
                                    let inp_cont = &inp_cont[inp_idx_1..inp_idx_2];
                                    let mut d = T::zero();
                                    unsafe {
                                        T::vec_dot(
                                            inp_cont.as_ptr(),
                                            k_cont.as_ptr(),
                                            &mut d,
                                            p.c_in,
                                        );
                                        let ptr = dst.as_ptr().add(dst_idx) as *mut T;
                                        *ptr += d;
                                    }
                                }
                            }
                        }
                    });
                }
            }
        }
    }

    Ok(dst)
}

#[allow(clippy::uninit_vec)]
fn alloc_uninit_vec<T: WithDType + Copy + 'static>(size: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(size);
    unsafe { v.set_len(size) };
    v
}

/// Full im2col + gemm convolution implementation.
fn conv3d_im2col_gemm<T: WithDType + num_traits::Num + Copy + 'static>(
    p: &ParamsConv3D,
    inp: &[T],
    inp_l: &Layout,
    kernel: &[T],
    kernel_l: &Layout,
) -> Result<Vec<T>> {
    let (out_d, out_h, out_w) = (p.out_d(), p.out_h(), p.out_w());

    // Build im2col matrix
    // Output shape: [b, d_out * h_out * w_out, c_in * k_d * k_h * k_w]
    let col = im2col_3d(p, inp, inp_l)?;
    let b = p.b_size;
    let n = p.c_out;
    let k = p.c_in * p.k_d * p.k_h * p.k_w;
    let m = out_d * out_h * out_w;
    let col_l = Layout::contiguous((b, m, k));

    let res: Vec<T> = if kernel_l.is_contiguous() {
        let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
            .transpose(1, 2)?
            .broadcast_as((b, k, n))?;
        MatMul((b, m, n, k)).f(&col, &col_l, kernel, &kernel_l)?
    } else {
        // Make the kernel contiguous if not already the case.
        let mut kernel_c = alloc_uninit_vec(kernel_l.shape().elem_count());
        copy_strided_src_(kernel, &mut kernel_c, 0, kernel_l);
        let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
            .transpose(1, 2)?
            .broadcast_as((b, k, n))?;
        MatMul((b, m, n, k)).f(&col, &col_l, &kernel_c, &kernel_l)?
    };

    // Transpose result from [b, d_out*h_out*w_out, c_out] to [b, c_out, d_out, h_out, w_out]
    let res_l = Layout::contiguous((b, out_d, out_h, out_w, p.c_out))
        .transpose(1, 4)? // [b, c_out, h, w, d]
        .transpose(2, 4)? // [b, c_out, d, w, h]
        .transpose(3, 4)?; // [b, c_out, d, h, w]
    let mut res_t = alloc_uninit_vec(res_l.shape().elem_count());
    copy_strided_src_(&res, &mut res_t, 0, &res_l);
    Ok(res_t)
}

/// Im2col for 3D convolution.
/// Transforms input [b, c, d, h, w] to [b, d_out*h_out*w_out, c*k_d*k_h*k_w]
fn im2col_3d<T: WithDType + num_traits::Num + Copy + 'static>(
    p: &ParamsConv3D,
    inp: &[T],
    inp_l: &Layout,
) -> Result<Vec<T>> {
    let (out_d, out_h, out_w) = (p.out_d(), p.out_h(), p.out_w());
    let k_size = p.c_in * p.k_d * p.k_h * p.k_w;
    let spatial_out = out_d * out_h * out_w;

    let inp = &inp[inp_l.start_offset()..];
    let (inp_s0, inp_s1, inp_s2, inp_s3, inp_s4) = dims5_strides(inp_l.stride())?;

    let mut col = vec![T::zero(); p.b_size * spatial_out * k_size];

    for b_idx in 0..p.b_size {
        let batch_offset = b_idx * spatial_out * k_size;

        for od in 0..out_d {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let spatial_idx = (od * out_h + oh) * out_w + ow;
                    let col_offset = batch_offset + spatial_idx * k_size;

                    let mut k_idx = 0;
                    for c in 0..p.c_in {
                        for kd in 0..p.k_d {
                            let id =
                                (od * p.stride_d + kd * p.dilation_d) as isize - p.padding_d as isize;
                            for kh in 0..p.k_h {
                                let ih = (oh * p.stride_h + kh * p.dilation_h) as isize
                                    - p.padding_h as isize;
                                for kw in 0..p.k_w {
                                    let iw = (ow * p.stride_w + kw * p.dilation_w) as isize
                                        - p.padding_w as isize;

                                    col[col_offset + k_idx] =
                                        if id >= 0
                                            && id < p.i_d as isize
                                            && ih >= 0
                                            && ih < p.i_h as isize
                                            && iw >= 0
                                            && iw < p.i_w as isize
                                        {
                                            let inp_idx = b_idx * inp_s0
                                                + c * inp_s1
                                                + (id as usize) * inp_s2
                                                + (ih as usize) * inp_s3
                                                + (iw as usize) * inp_s4;
                                            inp[inp_idx]
                                        } else {
                                            T::zero()
                                        };
                                    k_idx += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(col)
}

pub(super) struct ConvTranspose3D<'a>(pub(super) &'a ParamsConvTranspose3D);

impl Map2 for ConvTranspose3D<'_> {
    const OP: &'static str = "conv_transpose3d";
    fn f<T: WithDType + num_traits::Num + Copy + 'static>(
        &self,
        inp: &[T],
        inp_l: &Layout,
        k: &[T],
        k_l: &Layout,
    ) -> Result<Vec<T>> {
        let p = self.0;
        let inp = &inp[inp_l.start_offset()..];
        let (inp_s0, inp_s1, inp_s2, inp_s3, inp_s4) = dims5_strides(inp_l.stride())?;
        let k = &k[k_l.start_offset()..];
        let (k_s0, k_s1, k_s2, k_s3, k_s4) = dims5_strides(k_l.stride())?;
        let (out_d, out_h, out_w) = (p.out_d(), p.out_h(), p.out_w());

        // Output shape: [b_size, c_out, out_d, out_h, out_w].
        let dst = vec![T::zero(); p.b_size * p.c_out * out_d * out_h * out_w];
        let dst_s0 = p.c_out * out_d * out_h * out_w;
        let dst_s1 = out_d * out_h * out_w;
        let dst_s2 = out_h * out_w;
        let dst_s3 = out_w;
        let dst_s4 = 1;

        // Make contiguous input copy: [b, i_d, i_h, i_w, c_in]
        let cont_s0 = p.i_d * p.i_h * p.i_w * p.c_in;
        let cont_s1 = p.i_h * p.i_w * p.c_in;
        let cont_s2 = p.i_w * p.c_in;
        let cont_s3 = p.c_in;
        let mut inp_cont = vec![T::zero(); p.b_size * p.c_in * p.i_d * p.i_h * p.i_w];
        for b_idx in 0..p.b_size {
            for d_idx in 0..p.i_d {
                for h_idx in 0..p.i_h {
                    for w_idx in 0..p.i_w {
                        for c_idx in 0..p.c_in {
                            let src_idx = b_idx * inp_s0
                                + c_idx * inp_s1
                                + d_idx * inp_s2
                                + h_idx * inp_s3
                                + w_idx * inp_s4;
                            let dst_idx = b_idx * cont_s0
                                + d_idx * cont_s1
                                + h_idx * cont_s2
                                + w_idx * cont_s3
                                + c_idx;
                            inp_cont[dst_idx] = inp[src_idx];
                        }
                    }
                }
            }
        }

        for k_d in 0..p.k_d {
            for k_h in 0..p.k_h {
                for k_w in 0..p.k_w {
                    (0..p.c_out).into_par_iter().for_each(|dst_c_idx| {
                        // Collect kernel values for this output channel
                        let k_cont: Vec<T> = (0..p.c_in)
                            .map(|c_in_idx| {
                                k[c_in_idx * k_s0
                                    + dst_c_idx * k_s1
                                    + k_d * k_s2
                                    + k_h * k_s3
                                    + k_w * k_s4]
                            })
                            .collect();

                        for b_idx in 0..p.b_size {
                            for inp_d in 0..p.i_d {
                                for inp_h in 0..p.i_h {
                                    for inp_w in 0..p.i_w {
                                        let out_d_pos =
                                            inp_d * p.stride_d + k_d * p.dilation_d;
                                        let out_h_pos =
                                            inp_h * p.stride_h + k_h * p.dilation_h;
                                        let out_w_pos =
                                            inp_w * p.stride_w + k_w * p.dilation_w;

                                        if out_d_pos < p.padding_d
                                            || out_h_pos < p.padding_h
                                            || out_w_pos < p.padding_w
                                        {
                                            continue;
                                        }
                                        let out_d_idx = out_d_pos - p.padding_d;
                                        let out_h_idx = out_h_pos - p.padding_h;
                                        let out_w_idx = out_w_pos - p.padding_w;

                                        if out_d_idx < out_d
                                            && out_h_idx < out_h
                                            && out_w_idx < out_w
                                        {
                                            let inp_cont = &inp_cont[b_idx * cont_s0
                                                + inp_d * cont_s1
                                                + inp_h * cont_s2
                                                + inp_w * cont_s3..];
                                            let dst_idx = b_idx * dst_s0
                                                + dst_c_idx * dst_s1
                                                + out_d_idx * dst_s2
                                                + out_h_idx * dst_s3
                                                + out_w_idx * dst_s4;
                                            let mut d = T::zero();
                                            unsafe {
                                                T::vec_dot(
                                                    inp_cont.as_ptr(),
                                                    k_cont.as_ptr(),
                                                    &mut d,
                                                    p.c_in,
                                                );
                                                let ptr = dst.as_ptr().add(dst_idx) as *mut T;
                                                *ptr += d;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    });
                }
            }
        }
        Ok(dst)
    }
}
