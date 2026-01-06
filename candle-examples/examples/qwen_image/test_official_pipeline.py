#!/usr/bin/env python3
"""
Test the official QwenImagePipeline to verify it produces valid output.
"""

import os
import sys
from pathlib import Path

# Enable MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Add local diffusers to path
diffusers_path = Path(__file__).parent.parent.parent.parent / "diffusers" / "src"
if diffusers_path.exists():
    sys.path.insert(0, str(diffusers_path))

transformers_path = Path(__file__).parent.parent.parent.parent / "transformers" / "src"
if transformers_path.exists():
    sys.path.insert(0, str(transformers_path))

import torch


def main():
    # Use CPU to avoid MPS issues - slower but reliable
    device = "cpu"
    dtype = torch.float32

    print(f"Device: {device}, dtype: {dtype}")
    print("Loading pipeline...")

    from diffusers import QwenImagePipeline

    pipe = QwenImagePipeline.from_pretrained(
        "Qwen/Qwen-Image",
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)

    print("Pipeline loaded!")

    prompt = "A serene mountain landscape"
    print(f"Generating image for: '{prompt}'")

    # Use seed for reproducibility
    generator = torch.Generator(device=device).manual_seed(42)

    result = pipe(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=20,
        generator=generator,
        true_cfg_scale=1.0,  # No CFG for simplicity
    )

    image = result.images[0]
    output_path = "official_pipeline_output.png"
    image.save(output_path)
    print(f"Image saved to: {output_path}")


if __name__ == "__main__":
    main()
