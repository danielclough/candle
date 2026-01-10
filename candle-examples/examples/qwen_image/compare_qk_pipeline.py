#!/usr/bin/env python3
"""
Compare Q/K tensors at each pipeline stage between PyTorch and Rust.

This script helps identify where the Q/K divergence occurs in the attention pipeline:
- Stage 1 (proj): After linear projection, before reshape
- Stage 2 (norm): After QkNorm (RMSNorm), before RoPE
- Stage 3 (rope): After RoPE application

Usage:
    # First, generate PyTorch reference tensors:
    uv run candle-examples/examples/qwen_image/edit_reference_tensors.py \
        --input-image input.png --prompt "Make the sky blue" --steps 1

    # Then, run Rust with Q/K saving enabled:
    QWEN_SAVE_QK=1 QWEN_DEBUG=1 cargo run --release \
        --features metal,accelerate --example qwen_image -- \
        edit --input-image input.png --prompt "Make the sky blue" --steps 1

    # Finally, compare:
    uv run candle-examples/examples/qwen_image/compare_qk_pipeline.py
"""

import argparse
import os
from pathlib import Path
import numpy as np


def load_tensor(path: Path) -> np.ndarray | None:
    """Load a tensor from .npy file, return None if not found."""
    if path.exists():
        return np.load(path).astype(np.float32)
    return None


def tensor_stats(arr: np.ndarray) -> dict:
    """Compute statistics for a tensor."""
    return {
        "shape": arr.shape,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def compare_tensors(name: str, py_tensor: np.ndarray, rs_tensor: np.ndarray) -> dict:
    """Compare two tensors and return comparison statistics."""
    # Handle shape mismatch
    if py_tensor.shape != rs_tensor.shape:
        return {
            "name": name,
            "match": False,
            "error": f"Shape mismatch: PyTorch={py_tensor.shape}, Rust={rs_tensor.shape}",
            "py_stats": tensor_stats(py_tensor),
            "rs_stats": tensor_stats(rs_tensor),
        }

    diff = np.abs(py_tensor - rs_tensor)
    rel_diff = diff / (np.abs(py_tensor) + 1e-8)

    return {
        "name": name,
        "match": True,
        "shape": py_tensor.shape,
        "max_diff": float(diff.max()),
        "mean_diff": float(diff.mean()),
        "max_rel_diff": float(rel_diff.max()),
        "mean_rel_diff": float(rel_diff.mean()),
        "py_stats": tensor_stats(py_tensor),
        "rs_stats": tensor_stats(rs_tensor),
    }


def print_comparison(result: dict, verbose: bool = False):
    """Print comparison result with color coding."""
    name = result["name"]

    if not result["match"]:
        print(f"\n{name}: {result['error']}")
        return

    max_diff = result["max_diff"]
    mean_diff = result["mean_diff"]

    # Color coding based on difference magnitude
    if max_diff < 0.001:
        status = "OK"
    elif max_diff < 0.01:
        status = "WARN"
    elif max_diff < 0.1:
        status = "WARN+"
    else:
        status = "FAIL"

    print(f"\n{name} [{status}]:")
    print(f"  Shape: {result['shape']}")
    print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    if verbose:
        py_s = result["py_stats"]
        rs_s = result["rs_stats"]
        print(f"  PyTorch: mean={py_s['mean']:.6f}, std={py_s['std']:.6f}, "
              f"min={py_s['min']:.6f}, max={py_s['max']:.6f}")
        print(f"  Rust:    mean={rs_s['mean']:.6f}, std={rs_s['std']:.6f}, "
              f"min={rs_s['min']:.6f}, max={rs_s['max']:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Q/K tensors between PyTorch and Rust at each pipeline stage"
    )
    parser.add_argument(
        "--pytorch-dir",
        type=str,
        default="debug_tensors/pytorch_edit",
        help="Directory with PyTorch reference tensors",
    )
    parser.add_argument(
        "--rust-dir",
        type=str,
        default="debug_tensors/rust_edit",
        help="Directory with Rust tensors",
    )
    parser.add_argument(
        "--pass",
        dest="cfg_pass",
        choices=["pos", "neg", "both"],
        default="pos",
        help="Which CFG pass to compare (pos=positive, neg=negative, both)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed statistics for each tensor",
    )
    args = parser.parse_args()

    py_dir = Path(args.pytorch_dir)
    rs_dir = Path(args.rust_dir)

    if not py_dir.exists():
        print(f"Error: PyTorch tensor directory not found: {py_dir}")
        print("Run edit_reference_tensors.py first to generate PyTorch tensors.")
        return 1

    if not rs_dir.exists():
        print(f"Error: Rust tensor directory not found: {rs_dir}")
        print("Run Rust with QWEN_SAVE_QK=1 to generate Rust tensors.")
        return 1

    passes = ["pos", "neg"] if args.cfg_pass == "both" else [args.cfg_pass]

    for cfg_pass in passes:
        suffix = f"_{cfg_pass}"

        print(f"\n{'=' * 70}")
        print(f" Q/K PIPELINE COMPARISON - {cfg_pass.upper()} PASS")
        print(f"{'=' * 70}")

        # =========================================================================
        # Stage 1: After Projection (before reshape)
        # =========================================================================
        print(f"\n{'-' * 40}")
        print("STAGE 1: After Linear Projection")
        print(f"{'-' * 40}")

        # Mapping: PyTorch uses block0_internal_*_proj_pos.npy, Rust uses block0_proj_*_pos.npy
        stage1_tensors = [
            ("img_q_proj", f"block0_internal_img_q_proj{suffix}.npy", f"block0_proj_img_q{suffix}.npy"),
            ("img_k_proj", f"block0_internal_img_k_proj{suffix}.npy", f"block0_proj_img_k{suffix}.npy"),
            ("txt_q_proj", f"block0_internal_txt_q_proj{suffix}.npy", f"block0_proj_txt_q{suffix}.npy"),
            ("txt_k_proj", f"block0_internal_txt_k_proj{suffix}.npy", f"block0_proj_txt_k{suffix}.npy"),
        ]

        stage1_results = []
        for name, py_file, rs_file in stage1_tensors:
            py_tensor = load_tensor(py_dir / py_file)
            rs_tensor = load_tensor(rs_dir / rs_file)

            if py_tensor is None:
                print(f"\n{name}: PyTorch tensor not found ({py_file})")
                continue
            if rs_tensor is None:
                print(f"\n{name}: Rust tensor not found ({rs_file})")
                continue

            result = compare_tensors(name, py_tensor, rs_tensor)
            stage1_results.append(result)
            print_comparison(result, args.verbose)

        # =========================================================================
        # Stage 2: After QkNorm (before RoPE)
        # =========================================================================
        print(f"\n{'-' * 40}")
        print("STAGE 2: After QkNorm (RMSNorm)")
        print(f"{'-' * 40}")

        stage2_tensors = [
            ("img_q_after_norm", f"block0_internal_img_q_after_norm{suffix}.npy", f"block0_norm_img_q{suffix}.npy"),
            ("img_k_after_norm", f"block0_internal_img_k_after_norm{suffix}.npy", f"block0_norm_img_k{suffix}.npy"),
            ("txt_q_after_norm", f"block0_internal_txt_q_after_norm{suffix}.npy", f"block0_norm_txt_q{suffix}.npy"),
            ("txt_k_after_norm", f"block0_internal_txt_k_after_norm{suffix}.npy", f"block0_norm_txt_k{suffix}.npy"),
        ]

        stage2_results = []
        for name, py_file, rs_file in stage2_tensors:
            py_tensor = load_tensor(py_dir / py_file)
            rs_tensor = load_tensor(rs_dir / rs_file)

            if py_tensor is None:
                print(f"\n{name}: PyTorch tensor not found ({py_file})")
                continue
            if rs_tensor is None:
                print(f"\n{name}: Rust tensor not found ({rs_file})")
                continue

            result = compare_tensors(name, py_tensor, rs_tensor)
            stage2_results.append(result)
            print_comparison(result, args.verbose)

        # =========================================================================
        # Stage 3: After RoPE
        # =========================================================================
        print(f"\n{'-' * 40}")
        print("STAGE 3: After RoPE")
        print(f"{'-' * 40}")

        stage3_tensors = [
            ("img_q_after_rope", f"block0_internal_img_q_after_rope{suffix}.npy", f"block0_rope_img_q{suffix}.npy"),
            ("img_k_after_rope", f"block0_internal_img_k_after_rope{suffix}.npy", f"block0_rope_img_k{suffix}.npy"),
            ("txt_q_after_rope", f"block0_internal_txt_q_after_rope{suffix}.npy", f"block0_rope_txt_q{suffix}.npy"),
            ("txt_k_after_rope", f"block0_internal_txt_k_after_rope{suffix}.npy", f"block0_rope_txt_k{suffix}.npy"),
        ]

        stage3_results = []
        for name, py_file, rs_file in stage3_tensors:
            py_tensor = load_tensor(py_dir / py_file)
            rs_tensor = load_tensor(rs_dir / rs_file)

            if py_tensor is None:
                print(f"\n{name}: PyTorch tensor not found ({py_file})")
                continue
            if rs_tensor is None:
                print(f"\n{name}: Rust tensor not found ({rs_file})")
                continue

            result = compare_tensors(name, py_tensor, rs_tensor)
            stage3_results.append(result)
            print_comparison(result, args.verbose)

        # =========================================================================
        # Stage 4: Attention outputs (already saved by existing debug)
        # =========================================================================
        print(f"\n{'-' * 40}")
        print("STAGE 4: Attention Outputs")
        print(f"{'-' * 40}")

        stage4_tensors = [
            ("attn_weights", f"block0_internal_attn_weights{suffix}.npy", f"block0_attn_weights{suffix}.npy"),
            ("attn_probs", f"block0_internal_attn_probs{suffix}.npy", f"block0_attn_probs{suffix}.npy"),
        ]

        stage4_results = []
        for name, py_file, rs_file in stage4_tensors:
            py_tensor = load_tensor(py_dir / py_file)
            rs_tensor = load_tensor(rs_dir / rs_file)

            if py_tensor is None:
                print(f"\n{name}: PyTorch tensor not found ({py_file})")
                continue
            if rs_tensor is None:
                print(f"\n{name}: Rust tensor not found ({rs_file})")
                continue

            result = compare_tensors(name, py_tensor, rs_tensor)
            stage4_results.append(result)
            print_comparison(result, args.verbose)

        # =========================================================================
        # RoPE Frequencies (not per-pass)
        # =========================================================================
        if cfg_pass == passes[0]:  # Only show once
            print(f"\n{'-' * 40}")
            print("RoPE FREQUENCIES")
            print(f"{'-' * 40}")

            rope_tensors = [
                ("img_freqs", "diffusion_rope_img_freqs.npy", "diffusion_rope_img_freqs.npy"),
                ("txt_freqs", "diffusion_rope_txt_freqs.npy", "diffusion_rope_txt_freqs.npy"),
            ]

            for name, py_file, rs_file in rope_tensors:
                py_tensor = load_tensor(py_dir / py_file)
                rs_tensor = load_tensor(rs_dir / rs_file)

                if py_tensor is None:
                    print(f"\n{name}: PyTorch tensor not found ({py_file})")
                    continue
                if rs_tensor is None:
                    print(f"\n{name}: Rust tensor not found ({rs_file})")
                    continue

                result = compare_tensors(name, py_tensor, rs_tensor)
                print_comparison(result, args.verbose)

        # =========================================================================
        # Summary
        # =========================================================================
        print(f"\n{'=' * 70}")
        print(f" SUMMARY - {cfg_pass.upper()} PASS")
        print(f"{'=' * 70}")

        all_results = stage1_results + stage2_results + stage3_results + stage4_results

        if not all_results:
            print("\nNo tensors found for comparison!")
            print("Make sure both PyTorch and Rust tensor directories contain the expected files.")
            continue

        # Find first stage with significant divergence
        divergence_stage = None
        for stage_name, results in [
            ("Stage 1 (proj)", stage1_results),
            ("Stage 2 (norm)", stage2_results),
            ("Stage 3 (rope)", stage3_results),
            ("Stage 4 (attn)", stage4_results),
        ]:
            max_diffs = [r.get("max_diff", 0) for r in results if r.get("match", False)]
            if max_diffs and max(max_diffs) > 0.01:
                if divergence_stage is None:
                    divergence_stage = stage_name

        if divergence_stage:
            print(f"\n*** DIVERGENCE FIRST DETECTED AT: {divergence_stage} ***")
        else:
            print(f"\n ALL stages match within tolerance (max_diff < 0.01)")

        # Print stage-by-stage summary
        for stage_name, results in [
            ("Stage 1 (proj)", stage1_results),
            ("Stage 2 (norm)", stage2_results),
            ("Stage 3 (rope)", stage3_results),
            ("Stage 4 (attn)", stage4_results),
        ]:
            max_diffs = [r.get("max_diff", float("inf")) for r in results if r.get("match", False)]
            if max_diffs:
                worst = max(max_diffs)
                status = "OK" if worst < 0.001 else ("WARN" if worst < 0.01 else "FAIL")
                print(f"  {stage_name}: max_diff={worst:.6f} [{status}]")
            else:
                print(f"  {stage_name}: No data")

    print()
    return 0


if __name__ == "__main__":
    exit(main())
