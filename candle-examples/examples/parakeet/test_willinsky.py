#!/usr/bin/env python3
"""Test Willinsky audio file with all models - NeMo vs Candle comparison"""

# Redirect temp directory to disk BEFORE any imports that use tempfile
# (NeMo extracts 4GB+ models to temp, which overflows RAM-backed /tmp)
import os
import tempfile
_nemo_tmp = os.path.expanduser("~/.cache/nemo_tmp")
os.makedirs(_nemo_tmp, exist_ok=True)
os.environ['TMPDIR'] = _nemo_tmp
tempfile.tempdir = _nemo_tmp

import subprocess
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Use the WAV file (converted from OGG for compatibility)
AUDIO_FILE = "willinsky_150s.wav"
SCRIPT_DIR = Path(__file__).parent
CANDLE_ROOT = SCRIPT_DIR.parent.parent.parent

MODELS = {
    "tdt-v2": "nvidia/parakeet-tdt-0.6b-v2",
    "tdt-v3": "nvidia/parakeet-tdt-0.6b-v3",
    "rnnt-1b": "nvidia/parakeet-rnnt-1.1b",
    "ctc-1b": "nvidia/parakeet-ctc-1.1b",
}

CANDLE_VARIANTS = {
    "tdt-v2": "tdt-v2",
    "tdt-v3": "tdt-v3",
    "rnnt-1b": "rnnt1b",
    "ctc-1b": "ctc1b",
}

def cleanup_nemo_tmp():
    """Clean up NeMo temp extraction files."""
    import shutil
    tmp_dir = Path(_nemo_tmp)
    for item in tmp_dir.iterdir():
        try:
            # NeMo creates tmp* directories for extraction
            if item.name.startswith('tmp') and item.is_dir():
                shutil.rmtree(item)
        except:
            pass

def run_nemo(model_name: str, hf_name: str, audio: str) -> str:
    """Run NeMo transcription"""
    import nemo.collections.asr as nemo_asr

    try:
        print(f"  Loading model...", flush=True)
        model = nemo_asr.models.ASRModel.from_pretrained(hf_name)
        print(f"  Transcribing...", flush=True)
        result = model.transcribe([audio], verbose=False)
        if isinstance(result, list) and len(result) > 0:
            item = result[0]
            # Handle Hypothesis object (has .text attribute) or string
            if hasattr(item, 'text'):
                text = item.text
            else:
                text = str(item)
        else:
            text = str(result)

        # Clean up to free /tmp space before next model
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass
        import gc
        gc.collect()

        # Delete NeMo's temp extraction files from /tmp
        cleanup_nemo_tmp()

        return text
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"ERROR: {e}"

def run_candle(variant: str, audio: str) -> str:
    """Run Candle transcription"""
    cmd = [
        "cargo", "run", "--example", "parakeet", "--release", "--features", "parakeet",
        "--", "--model-variant", variant, "--input", audio
    ]
    try:
        print(f"  Running cargo...", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(CANDLE_ROOT))
        if result.returncode != 0:
            return f"ERROR: {result.stderr[-500:]}"
        # Extract transcription from output
        lines = result.stdout.split('\n')
        in_transcription = False
        transcription = []
        for line in lines:
            if "--- Transcription ---" in line:
                in_transcription = True
                continue
            if "---------------------" in line and in_transcription:
                break
            if in_transcription:
                transcription.append(line)
        return '\n'.join(transcription).strip()
    except subprocess.TimeoutExpired:
        return "ERROR: Timeout after 600s"
    except Exception as e:
        return f"ERROR: {e}"

def word_error_rate(ref: str, hyp: str) -> float:
    """Calculate WER between reference and hypothesis"""
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()

    r, h = len(ref_words), len(hyp_words)
    if r == 0:
        return 1.0 if h > 0 else 0.0

    d = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1):
        d[i][0] = i
    for j in range(h + 1):
        d[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    return d[r][h] / r

def main():
    audio_path = SCRIPT_DIR / AUDIO_FILE
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        print("Please convert the OGG file first:")
        print(f"  ffmpeg -i 060123-John.Willinsky-The.Economics.of.Knowledge.as.a.Public.Good.ogg -ar 16000 -ac 1 willinsky_150s.wav")
        sys.exit(1)

    print(f"Testing: {AUDIO_FILE}")
    print(f"Path: {audio_path}")
    print("=" * 80)

    results = {}

    for model_name, hf_name in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({hf_name})")
        print(f"{'='*60}")

        # Run NeMo
        print(f"NeMo:")
        nemo_result = run_nemo(model_name, hf_name, str(audio_path))
        nemo_words = len(nemo_result.split())
        print(f"  Result: {nemo_words} words")
        print(f"  First 150 chars: {nemo_result[:150]}...")

        # Run Candle
        print(f"Candle:")
        candle_result = run_candle(CANDLE_VARIANTS[model_name], str(audio_path))
        candle_words = len(candle_result.split())
        print(f"  Result: {candle_words} words")
        print(f"  First 150 chars: {candle_result[:150]}...")

        # Compare
        if nemo_result.startswith("ERROR") or candle_result.startswith("ERROR"):
            match = "ERROR"
            wer = -1
        elif nemo_result.lower().strip() == candle_result.lower().strip():
            match = "EXACT"
            wer = 0.0
        else:
            wer = word_error_rate(nemo_result, candle_result)
            match = f"WER={wer*100:.1f}%"

        print(f"Match: {match}")

        results[model_name] = {
            "nemo": nemo_result,
            "nemo_words": nemo_words,
            "candle": candle_result,
            "candle_words": candle_words,
            "match": match,
            "wer": wer
        }

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Model':<12} {'NeMo Words':<12} {'Candle Words':<14} {'Match'}")
    print("-" * 60)
    for model_name, r in results.items():
        print(f"{model_name:<12} {r['nemo_words']:<12} {r['candle_words']:<14} {r['match']}")

    # Save results
    output_file = SCRIPT_DIR / "willinsky_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()