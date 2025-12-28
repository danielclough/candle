#!/usr/bin/env python3
"""
Compare NeMo vs Candle transcriptions token-by-token.

This script:
1. Runs NeMo on an audio file and captures the token sequence
2. Runs Candle on the same audio and captures the token sequence
3. Compares the sequences to find where they diverge

Usage:
    uv run compare_tokens.py --model tdt-v2 --audio willinsky_150s.wav
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
import subprocess
import re
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import torch
import nemo.collections.asr as nemo_asr


MODEL_VARIANTS = {
    "tdt-v2": ("nvidia/parakeet-tdt-0.6b-v2", "tdt-v2"),
    "tdt-v3": ("nvidia/parakeet-tdt-0.6b-v3", "tdt-v3"),
    "rnnt-1b": ("nvidia/parakeet-rnnt-1.1b", "rnnt1b"),
    "ctc-1b": ("nvidia/parakeet-ctc-1.1b", "ctc1b"),
}


def get_nemo_tokens(model, audio_path: str):
    """Get token sequence from NeMo model."""

    # For TDT/RNNT models, we need to access the decoding internals
    result = model.transcribe([audio_path], return_hypotheses=True)

    if hasattr(result[0], 'y_sequence'):
        # TDT/RNNT models return y_sequence
        tokens = result[0].y_sequence.tolist()
    elif hasattr(result[0], 'text'):
        # Try to get from hypothesis
        text = result[0].text
        # Re-encode using tokenizer
        tokens = model.tokenizer.text_to_ids(text)
    else:
        tokens = []

    text = str(result[0].text) if hasattr(result[0], 'text') else str(result[0])
    return tokens, text


def get_candle_output(variant: str, audio_path: str, script_dir: Path):
    """Run Candle and capture the transcription."""
    candle_root = script_dir.parent.parent.parent

    cmd = [
        "cargo", "run", "--example", "parakeet", "--release", "--features", "parakeet",
        "--", "--model-variant", variant, "--input", audio_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(candle_root))

    if result.returncode != 0:
        print(f"Candle error:\n{result.stderr[-1000:]}")
        return None, None

    # Extract transcription
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

    text = '\n'.join(transcription).strip()

    # Try to extract token count
    tokens_match = re.search(r'Tokens: (\d+)', result.stdout)
    token_count = int(tokens_match.group(1)) if tokens_match else len(text.split())

    return token_count, text


def word_diff(text1: str, text2: str):
    """Show word-level differences between two texts."""
    words1 = text1.lower().split()
    words2 = text2.lower().split()

    # Simple LCS-based diff
    m, n = len(words1), len(words2)

    # Build DP table for LCS
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i-1] == words2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # Backtrack to find diff
    diffs = []
    i, j = m, n

    while i > 0 and j > 0:
        if words1[i-1] == words2[j-1]:
            i -= 1
            j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            diffs.append(('NEMO_ONLY', i-1, words1[i-1]))
            i -= 1
        else:
            diffs.append(('CANDLE_ONLY', j-1, words2[j-1]))
            j -= 1

    while i > 0:
        diffs.append(('NEMO_ONLY', i-1, words1[i-1]))
        i -= 1
    while j > 0:
        diffs.append(('CANDLE_ONLY', j-1, words2[j-1]))
        j -= 1

    return list(reversed(diffs))


def main():
    parser = argparse.ArgumentParser(description="Compare NeMo vs Candle tokens")
    parser.add_argument("--model", "-m", choices=list(MODEL_VARIANTS.keys()), default="tdt-v2")
    parser.add_argument("--audio", "-a", type=str, default="willinsky_150s.wav")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    audio_path = script_dir / args.audio

    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return

    hf_name, candle_variant = MODEL_VARIANTS[args.model]

    print(f"Model: {args.model}")
    print(f"Audio: {audio_path}")
    print("=" * 80)

    # Get NeMo output
    print("\n[1/2] Running NeMo...")
    model = nemo_asr.models.ASRModel.from_pretrained(hf_name)
    nemo_tokens, nemo_text = get_nemo_tokens(model, str(audio_path))
    print(f"NeMo tokens: {len(nemo_tokens)}")
    print(f"NeMo text ({len(nemo_text.split())} words): {nemo_text[:200]}...")

    # Get Candle output
    print("\n[2/2] Running Candle...")
    candle_token_count, candle_text = get_candle_output(candle_variant, str(audio_path), script_dir)
    if candle_text is None:
        print("Candle failed!")
        return
    print(f"Candle tokens: {candle_token_count}")
    print(f"Candle text ({len(candle_text.split())} words): {candle_text[:200]}...")

    # Compare texts
    print("\n" + "=" * 80)
    print("WORD-LEVEL DIFFERENCES")
    print("=" * 80)

    diffs = word_diff(nemo_text, candle_text)

    if not diffs:
        print("✓ EXACT MATCH!")
    else:
        print(f"Found {len(diffs)} differences:\n")

        # Group consecutive diffs
        nemo_missing = [d for d in diffs if d[0] == 'NEMO_ONLY']
        candle_missing = [d for d in diffs if d[0] == 'CANDLE_ONLY']

        print(f"Words in NeMo but not Candle ({len(nemo_missing)}):")
        for _, idx, word in nemo_missing[:20]:
            print(f"  - '{word}' (position ~{idx})")
        if len(nemo_missing) > 20:
            print(f"  ... and {len(nemo_missing) - 20} more")

        print(f"\nWords in Candle but not NeMo ({len(candle_missing)}):")
        for _, idx, word in candle_missing[:20]:
            print(f"  + '{word}' (position ~{idx})")
        if len(candle_missing) > 20:
            print(f"  ... and {len(candle_missing) - 20} more")

    # Calculate WER
    nemo_words = nemo_text.lower().split()
    candle_words = candle_text.lower().split()

    # Simple WER calculation
    r, h = len(nemo_words), len(candle_words)
    d = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1):
        d[i][0] = i
    for j in range(h + 1):
        d[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if nemo_words[i-1] == candle_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])

    wer = d[r][h] / r if r > 0 else 0

    print(f"\n" + "=" * 80)
    print(f"WER: {wer*100:.2f}% ({d[r][h]} edits / {r} reference words)")
    print("=" * 80)


if __name__ == "__main__":
    main()
