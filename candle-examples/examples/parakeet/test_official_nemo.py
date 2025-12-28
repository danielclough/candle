#!/usr/bin/env python3
"""
Test official NVIDIA NeMo Parakeet models against the same audio files.
Compare results with Candle implementation across all model variants.

Supported models:
- parakeet-tdt-0.6b-v2: TDT decoder, 1024 vocab, English only
- parakeet-tdt-0.6b-v3: TDT decoder, 8192 vocab, 25 languages
- parakeet-rnnt-1.1b: RNN-T decoder, 1024 vocab, English only
- parakeet-ctc-1.1b: CTC decoder, 1024 vocab, English only
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
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

import nemo.collections.asr as nemo_asr


# Model variants with their HuggingFace names
MODEL_VARIANTS = {
    "tdt-v2": "nvidia/parakeet-tdt-0.6b-v2",
    "tdt-v3": "nvidia/parakeet-tdt-0.6b-v3",
    "rnnt-1b": "nvidia/parakeet-rnnt-1.1b",
    "ctc-1b": "nvidia/parakeet-ctc-1.1b",
}


@dataclass
class TranscriptionResult:
    """Result of transcribing a single audio file."""
    filename: str
    expected: str
    transcription: str
    match: str  # "EXACT", "PARTIAL", "DIFFERENT"
    word_accuracy: float = 0.0


@dataclass
class ModelResults:
    """Results for a single model variant."""
    model_name: str
    hf_name: str
    results: list = field(default_factory=list)
    load_error: Optional[str] = None


def calculate_word_accuracy(expected: str, actual: str) -> float:
    """Calculate word-level accuracy between expected and actual transcriptions."""
    expected_words = set(expected.lower().split())
    actual_words = set(actual.lower().split())

    if not expected_words:
        return 100.0 if not actual_words else 0.0

    common = len(expected_words & actual_words)
    return (common / len(expected_words)) * 100


def extract_transcription(hyp) -> str:
    """Extract text from various NeMo hypothesis formats."""
    if hyp is None:
        return ""
    if isinstance(hyp, str):
        return hyp
    if hasattr(hyp, 'text'):
        return hyp.text
    return str(hyp)


def transcribe_with_model(model, audio_files: list) -> list:
    """Transcribe audio files using a NeMo model."""
    results = []

    for audio_file in audio_files:
        try:
            transcriptions = model.transcribe([str(audio_file)])

            # Handle different return formats
            if isinstance(transcriptions, tuple):
                hyp = transcriptions[0][0] if transcriptions[0] else None
            else:
                hyp = transcriptions[0] if transcriptions else None

            text = extract_transcription(hyp)
            results.append(text)
        except Exception as e:
            print(f"  Error transcribing {audio_file.name}: {e}")
            results.append("")

    return results


def test_model_variant(
    variant_name: str,
    hf_name: str,
    wav_files: list,
    expected_texts: dict,
) -> ModelResults:
    """Test a single model variant against all audio files."""

    model_results = ModelResults(model_name=variant_name, hf_name=hf_name)

    print(f"\n{'='*60}")
    print(f"  Testing: {variant_name} ({hf_name})")
    print(f"{'='*60}")

    # Load the model
    try:
        print(f"Loading model...")
        model = nemo_asr.models.ASRModel.from_pretrained(hf_name)
        print(f"Model loaded successfully!")
    except Exception as e:
        error_msg = f"Failed to load model: {e}"
        print(f"ERROR: {error_msg}")
        model_results.load_error = error_msg
        return model_results

    # Transcribe all files
    transcriptions = transcribe_with_model(model, wav_files)

    # Compare results
    for wav_file, transcription in zip(wav_files, transcriptions):
        basename = wav_file.stem
        expected = expected_texts.get(basename, "")

        # Normalize for comparison
        expected_norm = expected.lower().strip()
        actual_norm = transcription.lower().strip()

        word_accuracy = calculate_word_accuracy(expected, transcription)

        if expected_norm == actual_norm:
            match = "EXACT"
        elif word_accuracy >= 90:
            match = "PARTIAL"
        else:
            match = "DIFFERENT"

        result = TranscriptionResult(
            filename=basename,
            expected=expected,
            transcription=transcription,
            match=match,
            word_accuracy=word_accuracy,
        )
        model_results.results.append(result)

        print(f"\n  {basename}:")
        print(f"    Expected: {expected}")
        print(f"    NeMo:     {transcription}")
        print(f"    Match:    {match} ({word_accuracy:.1f}% word accuracy)")

    # Clean up model to free memory
    del model

    return model_results


def print_summary(all_results: list):
    """Print summary of all model results."""

    print(f"\n{'='*80}")
    print(f"  SUMMARY: NeMo Official Model Results")
    print(f"{'='*80}")

    # Header
    print(f"\n{'Model':<15} {'Loaded':<8} {'Exact':<8} {'Partial':<8} {'Different':<10} {'Avg Acc':<10}")
    print("-" * 70)

    for model_result in all_results:
        if model_result.load_error:
            print(f"{model_result.model_name:<15} {'FAIL':<8} {'-':<8} {'-':<8} {'-':<10} {'-':<10}")
            continue

        exact = sum(1 for r in model_result.results if r.match == "EXACT")
        partial = sum(1 for r in model_result.results if r.match == "PARTIAL")
        different = sum(1 for r in model_result.results if r.match == "DIFFERENT")
        avg_acc = sum(r.word_accuracy for r in model_result.results) / len(model_result.results) if model_result.results else 0

        print(f"{model_result.model_name:<15} {'OK':<8} {exact:<8} {partial:<8} {different:<10} {avg_acc:.1f}%")

    # Detailed transcription comparison
    print(f"\n{'='*80}")
    print(f"  Transcription Comparison Across Models")
    print(f"{'='*80}")

    # Get all filenames
    filenames = set()
    for model_result in all_results:
        for r in model_result.results:
            filenames.add(r.filename)

    for filename in sorted(filenames):
        print(f"\n{filename}:")
        for model_result in all_results:
            if model_result.load_error:
                continue
            for r in model_result.results:
                if r.filename == filename:
                    status = "EXACT" if r.match == "EXACT" else f"{r.word_accuracy:.0f}%"
                    print(f"  {model_result.model_name:<12}: {r.transcription} [{status}]")


def save_results_json(all_results: list, output_path: Path):
    """Save results to JSON for comparison with Candle."""

    data = {
        "tool": "nemo",
        "models": {}
    }

    for model_result in all_results:
        model_data = {
            "hf_name": model_result.hf_name,
            "loaded": model_result.load_error is None,
            "error": model_result.load_error,
            "transcriptions": {}
        }

        for r in model_result.results:
            model_data["transcriptions"][r.filename] = {
                "expected": r.expected,
                "transcription": r.transcription,
                "match": r.match,
                "word_accuracy": r.word_accuracy,
            }

        data["models"][model_result.model_name] = model_data

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test official NeMo Parakeet models against audio files"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_VARIANTS.keys()) + ["all"],
        default=["all"],
        help="Model variants to test (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results (optional)",
    )
    args = parser.parse_args()

    # Determine which models to test
    if "all" in args.models:
        models_to_test = list(MODEL_VARIANTS.keys())
    else:
        models_to_test = args.models

    script_dir = Path(__file__).parent
    text_dir = script_dir / "text"

    print("=" * 60)
    print("  Official NeMo Parakeet Multi-Model Test Suite")
    print("=" * 60)
    print(f"\nModels to test: {', '.join(models_to_test)}")
    print(f"Audio directory: {text_dir}")

    # Find all WAV files and their expected transcriptions
    wav_files = sorted(text_dir.glob("*.wav"))
    expected_texts = {}

    for wav_file in wav_files:
        txt_file = text_dir / f"{wav_file.stem}.txt"
        if txt_file.exists():
            expected_texts[wav_file.stem] = txt_file.read_text().strip()

    print(f"Found {len(wav_files)} audio files, {len(expected_texts)} with expected transcriptions")

    # Filter to only files with expected text
    wav_files = [f for f in wav_files if f.stem in expected_texts]

    if not wav_files:
        print("ERROR: No test files found!")
        sys.exit(1)

    # Test each model variant
    all_results = []

    for variant_name in models_to_test:
        hf_name = MODEL_VARIANTS[variant_name]
        result = test_model_variant(variant_name, hf_name, wav_files, expected_texts)
        all_results.append(result)

    # Print summary
    print_summary(all_results)

    # Save results if requested
    if args.output:
        save_results_json(all_results, args.output)
    else:
        # Default output path
        default_output = script_dir / "nemo_results.json"
        save_results_json(all_results, default_output)


if __name__ == "__main__":
    main()
