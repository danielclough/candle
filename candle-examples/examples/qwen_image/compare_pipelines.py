#!/usr/bin/env python3
"""
Compare outputs from official pipeline vs manual implementation to find differences.
Uses the same fixed initial latents to isolate algorithmic differences from RNG differences.
"""

import os
import sys
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

diffusers_path = Path(__file__).parent.parent.parent.parent / "diffusers" / "src"
if diffusers_path.exists():
    sys.path.insert(0, str(diffusers_path))

transformers_path = Path(__file__).parent.parent.parent.parent / "transformers" / "src"
if transformers_path.exists():
    sys.path.insert(0, str(transformers_path))

import torch
import numpy as np

def main():
    device = "cpu"  # Use CPU for reproducibility
    dtype = torch.float32

    print("=" * 60)
    print("Comparing Official Pipeline vs Manual Implementation")
    print("=" * 60)
    print(f"Device: {device}, dtype: {dtype}")

    # Create fixed initial latents for both pipelines
    torch.manual_seed(42)
    height, width = 512, 512
    latent_h, latent_w = height // 8, width // 8

    # Official pipeline shape: [B, T, C, H, W] = [1, 1, 16, 64, 64]
    initial_latents_official = torch.randn(1, 1, 16, latent_h, latent_w, device=device, dtype=dtype)

    # Manual script shape: [B, C, T, H, W] = [1, 16, 1, 64, 64]
    # This is the SAME tensor, just different dimension order
    initial_latents_manual = initial_latents_official.permute(0, 2, 1, 3, 4)

    print(f"\nInitial latents:")
    print(f"  Official shape: {initial_latents_official.shape}")
    print(f"  Manual shape:   {initial_latents_manual.shape}")
    print(f"  Are they the same data? {torch.allclose(initial_latents_official.permute(0, 2, 1, 3, 4), initial_latents_manual)}")

    # Load official pipeline
    print("\nLoading official pipeline...")
    from diffusers import QwenImagePipeline

    pipe = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=dtype)
    pipe = pipe.to(device)

    # Extract components for manual comparison
    scheduler = pipe.scheduler
    transformer = pipe.transformer
    vae = pipe.vae

    # Pack latents like official pipeline does
    packed_official = pipe._pack_latents(
        initial_latents_official,
        batch_size=1,
        num_channels=16,
        height=latent_h,
        width=latent_w
    )
    print(f"\nPacked latents (official): {packed_official.shape}")

    # Pack latents like manual script does
    def pack_manual(latents, h, w, patch_size=2):
        if latents.ndim == 4:
            latents = latents.unsqueeze(2)
        batch_size, num_channels, num_frames, height, width = latents.shape
        latents = latents.view(batch_size, num_channels, height // patch_size, patch_size, width // patch_size, patch_size)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // patch_size) * (width // patch_size), num_channels * patch_size * patch_size)
        return latents

    packed_manual = pack_manual(initial_latents_manual, latent_h, latent_w)
    print(f"Packed latents (manual):   {packed_manual.shape}")

    # Compare packed latents
    if packed_official.shape == packed_manual.shape:
        diff = (packed_official - packed_manual).abs()
        print(f"\nPacked latents comparison:")
        print(f"  Max diff: {diff.max().item():.6e}")
        print(f"  Mean diff: {diff.mean().item():.6e}")
        if diff.max().item() > 1e-5:
            print("  *** PACK FUNCTION DIFFERS! ***")
    else:
        print(f"\n*** SHAPE MISMATCH: {packed_official.shape} vs {packed_manual.shape} ***")

    # Compare scheduler setup
    packed_h, packed_w = latent_h // 2, latent_w // 2
    image_seq_len = packed_h * packed_w

    # Official mu calculation (from pipeline)
    def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return image_seq_len * m + b

    mu = calculate_shift(image_seq_len)
    print(f"\nScheduler mu: {mu:.6f}")
    print(f"Image seq len: {image_seq_len}")

    # Setup scheduler
    scheduler.set_timesteps(20, device=device, mu=mu)
    print(f"Timesteps: {scheduler.timesteps[:5].tolist()}...")

    print("\n" + "=" * 60)
    print("If packed latents match but final images differ,")
    print("the issue is in the denoising loop or VAE decode.")
    print("=" * 60)


if __name__ == "__main__":
    main()
