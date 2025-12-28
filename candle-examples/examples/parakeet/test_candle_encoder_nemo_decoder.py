#!/usr/bin/env python3
"""
Test Candle encoder output with NeMo decoder (bi-directional testing).

This loads the Candle encoder output from binary files and runs it through
NeMo's CTC decoder to verify if the Candle encoder produces valid output.

Usage:
    uv run test_candle_encoder_nemo_decoder.py --candle-dir candle_intermediates
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

import warnings
warnings.filterwarnings("ignore")

import nemo.collections.asr as nemo_asr


def load_tensor_binary(path: Path) -> torch.Tensor:
    """Load tensor from Candle binary format."""
    with open(path, 'rb') as f:
        # Read ndim
        ndim = struct.unpack('<I', f.read(4))[0]

        # Read shape
        shape = []
        for _ in range(ndim):
            dim = struct.unpack('<I', f.read(4))[0]
            shape.append(dim)

        # Read data
        total_elements = 1
        for d in shape:
            total_elements *= d

        data = np.frombuffer(f.read(total_elements * 4), dtype=np.float32)
        data = data.reshape(shape)

    return torch.from_numpy(data.copy())


def main():
    parser = argparse.ArgumentParser(
        description="Test Candle encoder output with NeMo decoder"
    )
    parser.add_argument(
        "--candle-dir", "-c",
        type=Path,
        required=True,
        help="Directory containing candle_encoder_output.bin",
    )
    parser.add_argument(
        "--model", "-m",
        choices=["ctc-1b", "rnnt-1b"],
        default="ctc-1b",
        help="Model variant to use for decoding",
    )
    args = parser.parse_args()

    # Load Candle encoder output
    enc_path = args.candle_dir / "candle_encoder_output.bin"
    if not enc_path.exists():
        print(f"Error: {enc_path} not found")
        print("Run Candle with --save-encoder first:")
        print("  cargo run --example parakeet --release --features parakeet -- \\")
        print("    --model-variant ctc1b --input audio.wav --save-encoder candle_intermediates")
        return

    print(f"Loading Candle encoder output from {enc_path}")
    candle_enc = load_tensor_binary(enc_path)
    print(f"  Shape: {candle_enc.shape}")  # Should be [features, time]
    print(f"  RMS: {candle_enc.pow(2).mean().sqrt().item():.4f}")
    print(f"  Range: [{candle_enc.min().item():.4f}, {candle_enc.max().item():.4f}]")

    # Add batch dim - NeMo CTC decoder expects [batch, features, time]
    candle_enc = candle_enc.unsqueeze(0)  # [1, features, time]
    print(f"  After adding batch: {candle_enc.shape}")

    # Load NeMo model
    model_map = {
        "ctc-1b": "nvidia/parakeet-ctc-1.1b",
        "rnnt-1b": "nvidia/parakeet-rnnt-1.1b",
    }
    hf_name = model_map[args.model]
    print(f"\nLoading NeMo model: {hf_name}")
    model = nemo_asr.models.ASRModel.from_pretrained(hf_name)
    model.eval()

    # Get the decoder/head
    print("\nRunning CTC decoding on Candle encoder output...")

    # For CTC models, we need to project encoder output to vocab size
    # and then decode
    with torch.no_grad():
        # The encoder output needs to go through the decoder (CTC head)
        # In NeMo CTC, decoder contains the linear projection
        if hasattr(model, 'decoder'):
            # Project to vocabulary
            logits = model.decoder.decoder_layers(candle_enc)
            print(f"Logits shape: {logits.shape}")  # [batch, vocab, time]

            # Transpose to [batch, time, vocab] for log_softmax and argmax
            logits = logits.transpose(1, 2)  # [batch, time, vocab]
            print(f"Logits after transpose: {logits.shape}")

            # Get log probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Greedy decoding - argmax over vocab dimension
            predictions = log_probs.argmax(dim=-1)  # [batch, time]
            print(f"Predictions shape: {predictions.shape}")
            print(f"Predictions: {predictions[0].tolist()[:50]}...")

            # Decode to text using tokenizer
            # Remove repeated tokens and blanks (CTC decoding)
            tokens = predictions[0].tolist()
            vocab_size = logits.shape[-1]
            blank_id = 0  # CTC blank is usually 0

            # Simple CTC collapse
            collapsed = []
            prev = None
            for t in tokens:
                if t != prev and t != blank_id:
                    collapsed.append(t)
                prev = t

            print(f"Collapsed tokens ({len(collapsed)}): {collapsed[:50]}...")

            # Try to use model's decoding
            try:
                # Create fake lengths tensor
                lengths = torch.tensor([candle_enc.shape[1]])

                # Use model's greedy decoding
                hypotheses = model.decoding.ctc_decoder_predictions_tensor(
                    log_probs,
                    decoder_lengths=lengths,
                    return_hypotheses=False
                )
                if hypotheses:
                    print(f"\n=== NeMo Transcription (from Candle encoder) ===")
                    print(hypotheses[0])
                    print("=" * 50)
            except Exception as e:
                print(f"Could not use model's decoding: {e}")

                # Manual decoding using tokenizer
                try:
                    text = model.tokenizer.ids_to_text(collapsed)
                    print(f"\n=== Manual Transcription ===")
                    print(text)
                    print("=" * 50)
                except Exception as e2:
                    print(f"Manual decoding failed: {e2}")

    print("\nDone!")


if __name__ == "__main__":
    main()
