#!/usr/bin/env python3
"""
Save intermediate tensors from NeMo Parakeet models for debugging/comparison with Candle.

Saves tensors in binary format compatible with Candle's load_tensor_from_binary():
- 4 bytes: ndim (u32 little-endian)
- ndim * 4 bytes: shape (u32 each)
- prod(shape) * 4 bytes: data (f32 little-endian, row-major)

Usage:
    uv run save_nemo_intermediates.py --model ctc-1b --audio text/en-Carter_man.wav
    uv run save_nemo_intermediates.py --model rnnt-1b --audio text/en-Carter_man.wav
"""
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "nemo_toolkit[asr]>=2.0.0",
#     "torch>=2.0.0",
#     "soundfile",
#     "onnx>=1.15.0,<1.17.0",
#     "ml_dtypes>=0.3.0,<0.5.0",
# ]
# ///

import argparse
import struct
from pathlib import Path

import torch
import numpy as np

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

import nemo.collections.asr as nemo_asr


MODEL_VARIANTS = {
    "tdt-v2": "nvidia/parakeet-tdt-0.6b-v2",
    "tdt-v3": "nvidia/parakeet-tdt-0.6b-v3",
    "rnnt-1b": "nvidia/parakeet-rnnt-1.1b",
    "ctc-1b": "nvidia/parakeet-ctc-1.1b",
}


def save_tensor_binary(tensor: torch.Tensor, path: Path):
    """Save tensor in Candle-compatible binary format."""
    # Convert to numpy (f32, contiguous)
    arr = tensor.detach().cpu().float().numpy()
    arr = np.ascontiguousarray(arr)

    with open(path, 'wb') as f:
        # Write ndim
        f.write(struct.pack('<I', arr.ndim))
        # Write shape
        for dim in arr.shape:
            f.write(struct.pack('<I', dim))
        # Write data (f32 little-endian)
        f.write(arr.tobytes())

    print(f"  Saved {path.name}: shape={arr.shape}, dtype={arr.dtype}")


def save_intermediates(model, audio_path: Path, output_dir: Path):
    """Run model and save intermediate tensors."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {audio_path}")
    print(f"Output dir: {output_dir}")

    # Storage for captured tensors
    captured = {}
    hooks = []

    # Find the preprocessor and encoder components
    preprocessor = getattr(model, 'preprocessor', None)
    encoder = getattr(model, 'encoder', None)

    if preprocessor is None:
        print("Warning: No preprocessor found")
    if encoder is None:
        print("Warning: No encoder found")
        return

    # Hook to capture mel spectrogram from preprocessor
    def capture_mel(module, input, output):
        if isinstance(output, tuple):
            mel, mel_len = output
        else:
            mel = output
        captured['mel'] = mel.clone()
        print(f"  Captured mel: shape={mel.shape}")

    # Hook to capture encoder output
    def capture_encoder_output(module, input, output):
        if isinstance(output, tuple):
            enc_out, enc_len = output
        else:
            enc_out = output
        captured['encoder_output'] = enc_out.clone()
        print(f"  Captured encoder_output: shape={enc_out.shape}")

    # Hook to capture subsampling output (pre_encode)
    def capture_subsampling(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            captured[name] = out.clone()
            print(f"  Captured {name}: shape={out.shape}")
        return hook

    # Register hooks
    if preprocessor is not None:
        hooks.append(preprocessor.register_forward_hook(capture_mel))

    # For FastConformer, the subsampling is in pre_encode
    if hasattr(encoder, 'pre_encode'):
        hooks.append(encoder.pre_encode.register_forward_hook(capture_subsampling('subsampling')))

    hooks.append(encoder.register_forward_hook(capture_encoder_output))

    # Also capture individual conformer blocks if available
    if hasattr(encoder, 'layers'):
        for i, layer in enumerate(encoder.layers):
            def make_block_hook(idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        out = output[0]
                    else:
                        out = output
                    captured[f'block_{idx:02d}'] = out.clone()
                return hook
            hooks.append(layer.register_forward_hook(make_block_hook(i)))

    # Run transcription to trigger hooks
    print("\nRunning model forward pass...")
    try:
        result = model.transcribe([str(audio_path)])
        if isinstance(result, tuple):
            transcription = result[0][0] if result[0] else ""
        else:
            transcription = result[0] if result else ""

        if hasattr(transcription, 'text'):
            transcription = transcription.text

        print(f"\nTranscription: {transcription}")
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

    # Save captured tensors
    print(f"\nSaving {len(captured)} tensors to {output_dir}...")

    for name, tensor in captured.items():
        # Remove batch dimension for saving (Candle will add it back)
        if tensor.dim() >= 1 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        save_tensor_binary(tensor, output_dir / f"{name}.bin")

    # Save transcription for reference
    with open(output_dir / "transcription.txt", 'w') as f:
        f.write(str(transcription))
    print(f"  Saved transcription.txt")

    # Print summary statistics
    print("\n=== Tensor Statistics ===")
    for name, tensor in captured.items():
        t = tensor.float()
        print(f"{name}:")
        print(f"  shape: {list(tensor.shape)}")
        print(f"  min: {t.min().item():.4f}, max: {t.max().item():.4f}")
        print(f"  mean: {t.mean().item():.4f}, std: {t.std().item():.4f}")
        positive_pct = (t > 0).float().mean().item() * 100
        print(f"  positive: {positive_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Save NeMo Parakeet intermediate tensors for Candle comparison"
    )
    parser.add_argument(
        "--model", "-m",
        choices=list(MODEL_VARIANTS.keys()),
        required=True,
        help="Model variant to use",
    )
    parser.add_argument(
        "--audio", "-a",
        type=Path,
        default=None,
        help="Audio file to process (default: first WAV in text/)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory (default: nemo_intermediates/<model>)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # Find audio file
    if args.audio:
        audio_path = args.audio
    else:
        # Default to first WAV in text/
        text_dir = script_dir / "text"
        wav_files = sorted(text_dir.glob("*.wav"))
        if not wav_files:
            print("No WAV files found in text/ directory")
            return
        audio_path = wav_files[0]

    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = script_dir / "nemo_intermediates" / args.model

    # Load model
    hf_name = MODEL_VARIANTS[args.model]
    print(f"Loading model: {args.model} ({hf_name})")
    model = nemo_asr.models.ASRModel.from_pretrained(hf_name)
    print("Model loaded successfully!")

    # Print model structure info
    print(f"\nModel type: {type(model).__name__}")
    if hasattr(model, 'encoder'):
        print(f"Encoder type: {type(model.encoder).__name__}")
        if hasattr(model.encoder, 'layers'):
            print(f"Encoder layers: {len(model.encoder.layers)}")

    # Save intermediates
    save_intermediates(model, audio_path, output_dir)

    print(f"\n=== Done! ===")
    print(f"Files saved to: {output_dir}")
    print(f"\nTo test encoder substitution in Candle:")
    print(f"  cargo run --example parakeet --release --features parakeet -- \\")
    print(f"    --model-variant ctc1b \\")
    print(f"    --input {audio_path} \\")
    print(f"    --compare-nemo {output_dir} \\")
    print(f"    --substitute-encoder")


if __name__ == "__main__":
    main()
