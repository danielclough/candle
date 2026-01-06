#!/usr/bin/env python3
"""
Debug VAE layer by layer to find divergence point.

This script loads the VAE, runs specific layers with controlled inputs,
and saves intermediate outputs for comparison with Rust.
"""

import os
import sys
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
import torch.nn.functional as F

# Add diffusers to path
diffusers_path = Path(__file__).parent.parent.parent.parent / "diffusers" / "src"
if diffusers_path.exists():
    sys.path.insert(0, str(diffusers_path))


def save_tensor(name: str, tensor: torch.Tensor, output_dir: str):
    """Save a PyTorch tensor as NumPy .npy file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.npy")
    arr = tensor.detach().cpu().float().numpy()
    np.save(path, arr)
    print(f"  Saved {name}: shape={arr.shape}, mean={arr.mean():.6f}, std={arr.std():.6f}")


def load_tensor(name: str, input_dir: str) -> torch.Tensor:
    """Load a tensor from NumPy .npy file."""
    path = os.path.join(input_dir, f"{name}.npy")
    arr = np.load(path)
    return torch.from_numpy(arr)


def tensor_stats(name: str, tensor: torch.Tensor):
    """Print tensor statistics."""
    t = tensor.detach().float()
    print(f"  {name}: shape={list(t.shape)}, mean={t.mean():.6f}, std={t.std():.6f}, "
          f"min={t.min():.6f}, max={t.max():.6f}")


def main():
    device = torch.device("cpu")  # Use CPU for consistency
    dtype = torch.float32
    output_dir = "debug_tensors/vae_layers"

    print("=" * 60)
    print("VAE Layer-by-Layer Debug")
    print("=" * 60)

    # Load the VAE
    print("\n[1/5] Loading VAE...")
    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage.from_pretrained(
        "Qwen/Qwen-Image",
        subfolder="vae",
        torch_dtype=dtype,
    ).to(device)
    vae.eval()
    print("  VAE loaded!")

    # Load the denormalized latents from PyTorch reference
    print("\n[2/5] Loading reference latents...")
    ref_dir = "debug_tensors/pytorch"

    if not os.path.exists(f"{ref_dir}/denormalized_latents.npy"):
        print("  ERROR: Reference latents not found. Run generate_reference_tensors.py first!")
        return

    z = load_tensor("denormalized_latents", ref_dir).to(device, dtype)
    tensor_stats("Input z", z)

    # ===========================================================================
    # Test 1: post_quant_conv
    # ===========================================================================
    print("\n[3/5] Testing post_quant_conv...")
    with torch.no_grad():
        z_post = vae.post_quant_conv(z)
    tensor_stats("After post_quant_conv", z_post)
    save_tensor("vae_post_quant_conv", z_post, output_dir)

    # ===========================================================================
    # Test 2: conv_in
    # ===========================================================================
    print("\n[4/5] Testing decoder.conv_in...")
    with torch.no_grad():
        x = vae.decoder.conv_in(z_post)
    tensor_stats("After conv_in", x)
    save_tensor("vae_conv_in", x, output_dir)

    # ===========================================================================
    # Test 3: Mid block - first resnet
    # ===========================================================================
    print("\n[5/5] Testing mid_block components...")

    # First resnet in mid_block
    with torch.no_grad():
        # Test individual components of first resnet
        resnet0 = vae.decoder.mid_block.resnets[0]

        # Shortcut
        h_short = resnet0.conv_shortcut(x)
        tensor_stats("  mid.res0 shortcut", h_short)
        save_tensor("vae_mid_res0_shortcut", h_short, output_dir)

        # First norm
        x_norm1 = resnet0.norm1(x)
        tensor_stats("  mid.res0 norm1", x_norm1)
        save_tensor("vae_mid_res0_norm1", x_norm1, output_dir)

        # First activation
        x_act1 = F.silu(x_norm1)
        tensor_stats("  mid.res0 silu1", x_act1)
        save_tensor("vae_mid_res0_silu1", x_act1, output_dir)

        # First conv
        x_conv1 = resnet0.conv1(x_act1)
        tensor_stats("  mid.res0 conv1", x_conv1)
        save_tensor("vae_mid_res0_conv1", x_conv1, output_dir)

        # Second norm
        x_norm2 = resnet0.norm2(x_conv1)
        tensor_stats("  mid.res0 norm2", x_norm2)
        save_tensor("vae_mid_res0_norm2", x_norm2, output_dir)

        # Second activation
        x_act2 = F.silu(x_norm2)
        tensor_stats("  mid.res0 silu2", x_act2)

        # Second conv
        x_conv2 = resnet0.conv2(x_act2)
        tensor_stats("  mid.res0 conv2", x_conv2)

        # Full resnet0 output
        x_res0 = x_conv2 + h_short
        tensor_stats("  mid.res0 output", x_res0)
        save_tensor("vae_mid_res0_output", x_res0, output_dir)

        # Now test attention block
        print("\n  Testing attention block...")
        attn = vae.decoder.mid_block.attentions[0]

        # Get attention intermediate values
        identity = x_res0.clone()
        b, c, t, h, w = x_res0.shape

        # Reshape for attention
        x_for_attn = x_res0.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        tensor_stats("  attn input reshape", x_for_attn)
        save_tensor("vae_attn_input_reshape", x_for_attn, output_dir)

        # Norm
        x_normed = attn.norm(x_for_attn)
        tensor_stats("  attn norm", x_normed)
        save_tensor("vae_attn_norm", x_normed, output_dir)

        # QKV
        qkv = attn.to_qkv(x_normed)
        tensor_stats("  attn qkv", qkv)
        save_tensor("vae_attn_qkv", qkv, output_dir)

        # Reshape QKV
        qkv_reshaped = qkv.reshape(b * t, 1, c * 3, -1)
        qkv_permuted = qkv_reshaped.permute(0, 1, 3, 2).contiguous()
        tensor_stats("  attn qkv_permuted", qkv_permuted)
        save_tensor("vae_attn_qkv_permuted", qkv_permuted, output_dir)

        # Split to Q, K, V
        q, k, v = qkv_permuted.chunk(3, dim=-1)
        tensor_stats("  attn Q", q)
        tensor_stats("  attn K", k)
        tensor_stats("  attn V", v)
        save_tensor("vae_attn_q", q, output_dir)
        save_tensor("vae_attn_k", k, output_dir)
        save_tensor("vae_attn_v", v, output_dir)

        # Compute attention manually to see intermediate values
        # Q @ K^T
        k_t = k.transpose(-2, -1)  # Transpose last two dims only!
        tensor_stats("  attn K^T", k_t)
        save_tensor("vae_attn_k_transposed", k_t, output_dir)

        # Scale factor
        scale = c ** 0.5
        print(f"  Attention scale factor: {scale}")

        # QK^T / scale
        attn_weights = torch.matmul(q, k_t) / scale
        tensor_stats("  attn weights (pre-softmax)", attn_weights)
        save_tensor("vae_attn_weights_pre_softmax", attn_weights, output_dir)

        # Softmax
        attn_weights_softmax = F.softmax(attn_weights, dim=-1)
        tensor_stats("  attn weights (post-softmax)", attn_weights_softmax)
        save_tensor("vae_attn_weights_post_softmax", attn_weights_softmax, output_dir)

        # Attention output
        attn_out = torch.matmul(attn_weights_softmax, v)
        tensor_stats("  attn output", attn_out)
        save_tensor("vae_attn_out", attn_out, output_dir)

        # Now compare with PyTorch's scaled_dot_product_attention
        attn_out_sdpa = F.scaled_dot_product_attention(q, k, v)
        tensor_stats("  attn output (SDPA)", attn_out_sdpa)

        # Check if they match
        diff = (attn_out - attn_out_sdpa).abs()
        print(f"  Manual vs SDPA max diff: {diff.max():.6e}")

        # Continue with output projection
        out = attn_out.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        tensor_stats("  attn output reshape", out)
        save_tensor("vae_attn_out_reshape", out, output_dir)

        out = attn.proj(out)
        tensor_stats("  attn proj", out)
        save_tensor("vae_attn_proj", out, output_dir)

        # Reshape back
        out = out.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        tensor_stats("  attn final reshape", out)

        # Add residual
        out = out + identity
        tensor_stats("  attn + residual", out)
        save_tensor("vae_attn_output", out, output_dir)

    # ===========================================================================
    # Test full decode
    # ===========================================================================
    print("\n[6/5] Testing full decode...")
    with torch.no_grad():
        decoded = vae.decode(z, return_dict=False)[0]
    tensor_stats("Full decoded output", decoded)
    save_tensor("vae_decoded_full", decoded, output_dir)

    # Extract single frame
    decoded_frame = decoded[:, :, 0]
    tensor_stats("Decoded frame 0", decoded_frame)
    save_tensor("vae_decoded_frame0", decoded_frame, output_dir)

    print("\n" + "=" * 60)
    print("Layer debug tensors saved to:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
