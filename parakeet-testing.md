# Plan: Add Integration Tests for Parakeet ASR Features

## Overview

Create a Rust integration test example (like the Python tests `test_official_nemo.py`, `test_willinsky.py`) that validates all features documented in FEATURES.md by running actual model inference against test audio files.

## Test Strategy

Create an **example binary** (`parakeet_test`) that:
1. Loads the Parakeet model (TDT-v3 by default, optionally other variants)
2. Runs inference on the existing test audio files (`en-*.wav`)
3. Compares against expected transcriptions (`en-*.txt`)
4. Tests all FEATURES.md capabilities and reports pass/fail

## Test File Location

**File:** `candle-examples/examples/parakeet_test.rs` (new example binary)

Run via:
```bash
cargo run --example parakeet_test --release --features parakeet
```

Optionally with model variant:
```bash
cargo run --example parakeet_test --release --features parakeet -- --model-variant rnnt-1b
```

## Features to Test

### 1. Basic Decoding (TDT, RNNT, CTC)
- ✅ Greedy decoding produces valid transcription
- ✅ Beam search produces valid transcription
- ✅ Output matches expected text (WER < 10%)

### 2. Timestamps
- ✅ `compute_timestamps()` returns valid TimestampInfo
- ✅ Token timestamps increase monotonically
- ✅ Timestamps fall within audio duration

### 3. Confidence Scores
- ✅ Log probabilities are negative (valid range)
- ✅ `word_info()` returns aggregated word confidences
- ✅ All aggregation methods work (Product, Mean, Min)

### 4. Batch Inference
- ✅ `forward_batch()` processes multiple files
- ✅ Results match single-file inference
- ✅ Batch beam search works

### 5. Streaming/Chunked Inference
- ✅ Long audio is chunked correctly
- ✅ Chunk overlap produces smooth boundaries
- ✅ Final transcription is coherent

### 6. N-gram LM Fusion
- ✅ ARPA file loads correctly
- ✅ LM scoring works (test with mock LM data)
- ✅ LM fusion doesn't break transcription

### 7. MALSD Beam Search (TDT only)
- ✅ MALSD produces valid output
- ✅ MALSD respects beam width

## Test Data

Use existing test files in `candle-examples/examples/parakeet/text/`:
- `en-Carter_man.wav` + `.txt`
- `en-Davis_man.wav` + `.txt`
- `en-Emma_woman.wav` + `.txt`
- `en-Frank_man.wav` + `.txt`
- `en-Grace_woman.wav` + `.txt`
- `en-Mike_man.wav` + `.txt`

## Implementation Outline

```rust
// tests.rs - Integration test for Parakeet features

use candle_transformers::models::parakeet::*;

fn calculate_wer(reference: &str, hypothesis: &str) -> f32 { ... }

fn load_test_model(device: &Device) -> Result<(Parakeet, SimpleDecoder)> { ... }

fn load_test_audio(path: &str, num_mel_bins: usize, device: &Device) -> Result<Tensor> { ... }

// Feature tests:

fn test_greedy_decoding() { ... }
fn test_beam_search() { ... }
fn test_timestamps() { ... }
fn test_word_timestamps_with_confidence() { ... }
fn test_confidence_aggregation_methods() { ... }
fn test_batch_inference() { ... }
fn test_streaming_inference() { ... }
fn test_ngram_lm_parsing() { ... }
fn test_lm_fusion() { ... }
fn test_malsd_beam_search() { ... }

fn main() {
    // Run all tests, report results
}
```

## Success Criteria

1. All tests pass with WER < 10% on test audio
2. Tests complete in reasonable time (< 60s for TDT model)
3. Tests validate all FEATURES.md capabilities
4. Tests produce clear pass/fail output

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `candle-examples/examples/parakeet_test.rs` | **Create** | New integration test example binary |
| `candle-examples/Cargo.toml` | Modify | Add `[[example]] name = "parakeet_test"` with parakeet feature |

## Output Format

Similar to Python tests - print results in a clear table:

```
============================================================
  Parakeet Feature Test Suite (TDT-v3)
============================================================

Audio: en-Carter_man.wav
  Expected: "The quick brown fox jumps over the lazy dog."
  Result:   "The quick brown fox jumps over the lazy dog."
  WER: 0.0%  [PASS]

Test: Greedy Decoding .............. PASS
Test: Beam Search (width=5) ........ PASS
Test: Timestamps ................... PASS
Test: Word Timestamps .............. PASS
Test: Confidence Scores ............ PASS
Test: Batch Inference (3 files) .... PASS
Test: Streaming (30s chunks) ....... PASS
Test: MALSD Beam Search ............ PASS

============================================================
  SUMMARY: 8/8 tests passed
============================================================
```

## Approach

Create `parakeet_test.rs` as a standalone example that mirrors the Python test style - simple, practical, and validates real model behavior against the test audio files in `text/`.
