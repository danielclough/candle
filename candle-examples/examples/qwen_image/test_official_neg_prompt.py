#!/usr/bin/env python3
"""
Diagnostic test: Does the official diffusers QwenImageEditPipeline work with negative prompts?

If this works: Bug is in our reference script or Rust implementation
If this fails: Bug is upstream in diffusers
"""

import argparse
import sys
from pathlib import Path

# Add local diffusers and transformers to path (QwenImageEditPipeline isn't in public diffusers yet)
SCRIPT_DIR = Path(__file__).parent.resolve()
LOCAL_DIFFUSERS = SCRIPT_DIR.parent.parent.parent / "diffusers" / "src"
LOCAL_TRANSFORMERS = SCRIPT_DIR.parent.parent.parent / "transformers" / "src"

if LOCAL_DIFFUSERS.exists():
    sys.path.insert(0, str(LOCAL_DIFFUSERS))
    print(f"Using local diffusers from: {LOCAL_DIFFUSERS}")

if LOCAL_TRANSFORMERS.exists():
    sys.path.insert(0, str(LOCAL_TRANSFORMERS))
    print(f"Using local transformers from: {LOCAL_TRANSFORMERS}")

import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Test official diffusers pipeline with negative prompts")
    parser.add_argument("--input-image", type=str, required=True, help="Input image path")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, NOT mps - has issues)")
    parser.add_argument("--steps", type=int, default=5, help="Number of inference steps (fewer = faster)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Auto-detect device (MPS has dimension mismatch bugs with this model)
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"  # Skip MPS - has known issues with QwenImageEditPipeline

    if device == "mps":
        print("WARNING: MPS has issues with this pipeline, falling back to CPU")
        device = "cpu"

    print(f"Using device: {device}")

    # Load pipeline
    from diffusers import QwenImageEditPipeline

    # Use float32 for CPU (bfloat16 may not be supported)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("Loading QwenImageEditPipeline...")
    pipe = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        torch_dtype=dtype
    ).to(device)

    # For CPU, also ensure text encoder is float32
    if device == "cpu":
        pipe.text_encoder = pipe.text_encoder.to(torch.float32)

    # Load input image
    print(f"Loading input image: {args.input_image}")
    image = Image.open(args.input_image).convert("RGB")
    orig_width, orig_height = image.size
    print(f"  Image size: {image.size}")

    # Use explicit 256x256 (matching reference script command)
    # Pass original image - let pipeline handle resize via height/width params
    width = 256
    height = 256
    print(f"  Output dimensions: {width}x{height} (pipeline will resize)")

    prompt = "Make the sky psychedelic"
    negative_prompt = "low resolution, low quality, deformed limbs, deformed fingers, oversaturated, waxy appearance, face without details, overly smooth, AI-like appearance, chaotic composition, blurred text, distorted"

    # Generator device (must be 'cpu' for CPU inference)
    gen_device = "cpu" if device == "cpu" else device

    # Test 1: Without negative prompt (baseline) - should work fine
    print(f"\n=== Test 1: WITHOUT negative prompt (baseline) ===")
    print(f"  Prompt: {prompt}")
    print(f"  CFG scale: 4.0 (no negative prompt = no CFG applied)")
    print(f"  Steps: {args.steps}, Seed: {args.seed}")

    result_no_neg = pipe(
        prompt=prompt,
        image=image,
        height=height,
        width=width,
        true_cfg_scale=4.0,  # CFG scale, but no negative prompt means no CFG
        num_inference_steps=args.steps,
        generator=torch.Generator(device=gen_device).manual_seed(args.seed),
    ).images[0]

    output_path_no_neg = f"{args.output_dir}/test_official_no_neg.png"
    result_no_neg.save(output_path_no_neg)
    print(f"  Saved: {output_path_no_neg}")

    # Test 2: With negative prompt
    print(f"\n=== Test 2: WITH negative prompt ===")
    print(f"  Prompt: {prompt}")
    print(f"  Negative: {negative_prompt[:50]}...")
    print(f"  CFG scale: 4.0")
    print(f"  Steps: {args.steps}, Seed: {args.seed}")

    result_with_neg = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        height=height,
        width=width,
        true_cfg_scale=4.0,
        num_inference_steps=args.steps,
        generator=torch.Generator(device=gen_device).manual_seed(args.seed),
    ).images[0]

    output_path_with_neg = f"{args.output_dir}/test_official_with_neg.png"
    result_with_neg.save(output_path_with_neg)
    print(f"  Saved: {output_path_with_neg}")

    print("\n=== RESULTS ===")
    print(f"Without negative prompt: {output_path_no_neg}")
    print(f"With negative prompt:    {output_path_with_neg}")
    print("\nCheck the images:")
    print("  - If WITH negative prompt is clean: BUG IS IN OUR CODE")
    print("  - If WITH negative prompt is noisy: BUG IS UPSTREAM IN DIFFUSERS")


if __name__ == "__main__":
    main()
