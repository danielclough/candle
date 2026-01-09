#!/usr/bin/env python3
"""
Compare Rust vs PyTorch vision preprocessing step by step.
"""

import sys
sys.path.insert(0, 'transformers/src')
sys.path.insert(0, 'diffusers/src')

import os
import numpy as np
from PIL import Image
import torch

# Load the reference pixel values if available
pytorch_ref = None
ref_path = 'debug_tensors/pytorch/vision_pixel_values.npy'
if os.path.exists(ref_path):
    pytorch_ref = np.load(ref_path)
    print(f"PyTorch vision_pixel_values shape: {pytorch_ref.shape}")
    print(f"PyTorch: mean={pytorch_ref.mean():.6f}, std={pytorch_ref.std():.6f}")
    print(f"PyTorch: min={pytorch_ref.min():.6f}, max={pytorch_ref.max():.6f}")
else:
    print(f"No PyTorch reference at {ref_path}")

# Load the image
image_path = "qwen_image_output.png"
img = Image.open(image_path)
print(f"\nOriginal image: {img.size} ({img.mode})")

# Resize to 252x252 (what Rust does for 256x256 output constrained by vision max_pixels)
target_height, target_width = 252, 252
resized_bicubic = img.resize((target_width, target_height), Image.Resampling.BICUBIC)
resized_lanczos = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
resized_bilinear = img.resize((target_width, target_height), Image.Resampling.BILINEAR)

# Convert to numpy
rgb_bicubic = np.array(resized_bicubic.convert('RGB'))
rgb_lanczos = np.array(resized_lanczos.convert('RGB'))
rgb_bilinear = np.array(resized_bilinear.convert('RGB'))

print(f"\nResized to {target_width}x{target_height}:")
print(f"  BICUBIC:  mean={rgb_bicubic.mean():.4f}, std={rgb_bicubic.std():.4f}")
print(f"  LANCZOS:  mean={rgb_lanczos.mean():.4f}, std={rgb_lanczos.std():.4f}")
print(f"  BILINEAR: mean={rgb_bilinear.mean():.4f}, std={rgb_bilinear.std():.4f}")

# Normalize with CLIP constants
IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711])

def normalize_image(rgb_data: np.ndarray) -> np.ndarray:
    """Normalize to CLIP format: (pixel/255 - mean) / std"""
    normalized = rgb_data.astype(np.float32) / 255.0
    normalized = (normalized - IMAGE_MEAN) / IMAGE_STD
    # Convert to channels-first: (H, W, C) -> (C, H, W)
    normalized = normalized.transpose(2, 0, 1)
    return normalized

norm_bicubic = normalize_image(rgb_bicubic)
norm_lanczos = normalize_image(rgb_lanczos)
norm_bilinear = normalize_image(rgb_bilinear)

print(f"\nAfter normalization (C, H, W):")
print(f"  BICUBIC:  mean={norm_bicubic.mean():.6f}, std={norm_bicubic.std():.6f}, min={norm_bicubic.min():.6f}, max={norm_bicubic.max():.6f}")
print(f"  LANCZOS:  mean={norm_lanczos.mean():.6f}, std={norm_lanczos.std():.6f}, min={norm_lanczos.min():.6f}, max={norm_lanczos.max():.6f}")
print(f"  BILINEAR: mean={norm_bilinear.mean():.6f}, std={norm_bilinear.std():.6f}, min={norm_bilinear.min():.6f}, max={norm_bilinear.max():.6f}")

# Now use HuggingFace processor
from transformers import Qwen2VLImageProcessor

processor = Qwen2VLImageProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct')
print(f"\nHuggingFace processor config:")
print(f"  resample: {processor.resample} (3=BICUBIC)")
print(f"  image_mean: {processor.image_mean}")
print(f"  image_std: {processor.image_std}")

# Process with constrained max_pixels (same as Rust)
max_pixels = 256 * 256  # 65536
processor_constrained = Qwen2VLImageProcessor.from_pretrained(
    'Qwen/Qwen2-VL-7B-Instruct',
    max_pixels=max_pixels
)

# Process the original image (not the resized one)
result = processor_constrained(images=img, return_tensors="pt")
hf_pixel_values = result['pixel_values'].numpy()
hf_grid_thw = result['image_grid_thw'].numpy()

print(f"\nHuggingFace processor output:")
print(f"  pixel_values shape: {hf_pixel_values.shape}")
print(f"  grid_thw: {hf_grid_thw}")
print(f"  pixel_values: mean={hf_pixel_values.mean():.6f}, std={hf_pixel_values.std():.6f}")
print(f"  pixel_values: min={hf_pixel_values.min():.6f}, max={hf_pixel_values.max():.6f}")

# Compare with PyTorch reference if available
if pytorch_ref is not None:
    print(f"\n=== Comparison with PyTorch reference ===")
    print(f"PyTorch:     mean={pytorch_ref.mean():.6f}, std={pytorch_ref.std():.6f}, min={pytorch_ref.min():.6f}, max={pytorch_ref.max():.6f}")
    print(f"HuggingFace: mean={hf_pixel_values.mean():.6f}, std={hf_pixel_values.std():.6f}, min={hf_pixel_values.min():.6f}, max={hf_pixel_values.max():.6f}")

    # Check if HuggingFace matches PyTorch reference
    if pytorch_ref.shape == hf_pixel_values.shape:
        diff = np.abs(pytorch_ref - hf_pixel_values)
        print(f"\nHF vs PyTorch ref diff: max={diff.max():.6f}, mean={diff.mean():.6f}")
    else:
        print(f"\nShape mismatch: PyTorch {pytorch_ref.shape} vs HF {hf_pixel_values.shape}")
