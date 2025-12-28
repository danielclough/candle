# Parakeet ASR Features - Implementation Status

## Overview

This document tracks the implementation status of NeMo TDT features in the Candle Parakeet implementation.

## Current State

**What we have (all completed!):**
- ✅ Greedy decoding (TDT, RNNT, CTC)
- ✅ Basic beam search (TDT, RNNT, CTC)
- ✅ Correct transcription output (0% WER on TDT-v2)
- ✅ **Timestamps** (token/word level)
- ✅ **Confidence scores** (token/word level with aggregation)
- ✅ **Batched inference** (multiple files in one run)
- ✅ **Streaming/chunked inference** (for long audio)
- ✅ **N-gram LM shallow fusion** (ARPA format parser + beam search integration)
- ✅ **MALSD beam search** (alignment-length synchronous decoding)

---

## Feature Details

### Phase 1-3: Rich Decoding Results with Timestamps and Confidence ✅

**Data structures implemented in `mod.rs`:**
```rust
pub struct TokenInfo {
    pub token_id: u32,
    pub log_prob: f32,       // Log probability
    pub start_frame: usize,  // Start frame index
    pub end_frame: usize,    // End frame index
}

pub struct DecodingResult {
    pub tokens: Vec<TokenInfo>,
    pub total_log_prob: f32,
    pub num_frames: usize,
}

pub struct TimestampInfo {
    pub start_time_sec: f32,
    pub end_time_sec: f32,
    pub confidence: f32,
}

pub struct WordInfo {
    pub word: String,
    pub start_time_sec: f32,
    pub end_time_sec: f32,
    pub confidence: f32,
    pub token_indices: Vec<usize>,
}

pub enum ConfidenceAggregation {
    Product,  // exp(sum(log_probs))
    Mean,     // exp(mean(log_probs))
    Min,      // min(exp(log_probs))
}
```

**CLI Options:**
```bash
# Token-level timestamps
cargo run --example parakeet --features parakeet -- \
    --input audio.wav --timestamps

# Word-level timestamps
cargo run --example parakeet --features parakeet -- \
    --input audio.wav --word-timestamps

# With confidence scores
cargo run --example parakeet --features parakeet -- \
    --input audio.wav --word-timestamps --confidence

# Choose aggregation method
cargo run --example parakeet --features parakeet -- \
    --input audio.wav --word-timestamps --confidence --confidence-method mean
```

---

### Phase 4: Batched Inference ✅

Process multiple audio files efficiently in a single run.

**Implementation in `model.rs`:**
```rust
impl Parakeet {
    pub fn forward_batch(&self, mels: &[&Tensor]) -> Result<Vec<DecodingResult>>;
    pub fn forward_batch_beam(&self, mels: &[&Tensor], beam_width: usize) -> Result<Vec<DecodingResult>>;
}
```

**CLI Usage:**
```bash
# Process multiple files
cargo run --example parakeet --features parakeet -- \
    --input file1.wav --input file2.wav --input file3.wav

# With word timestamps for all
cargo run --example parakeet --features parakeet -- \
    --input file1.wav --input file2.wav --word-timestamps
```

---

### Phase 5: Streaming/Chunked Inference ✅

Process long audio files in overlapping chunks.

**CLI Options:**
```bash
# Enable streaming mode
cargo run --example parakeet --features parakeet -- \
    --input long_audio.wav --streaming

# Custom chunk settings
cargo run --example parakeet --features parakeet -- \
    --input long_audio.wav --streaming \
    --chunk-duration 60.0 --chunk-overlap 10.0
```

**Default settings:**
- Chunk duration: 30 seconds
- Overlap: 5 seconds

---

### Phase 6: N-gram Language Model ✅

Pure Rust ARPA format parser with beam search integration.

**Implementation in `lm.rs`:**
```rust
pub struct NgramLM {
    order: usize,
    vocab: HashMap<String, u32>,
    ngrams: HashMap<Vec<u32>, (f32, f32)>,  // (log10_prob, backoff)
}

impl NgramLM {
    pub fn load_arpa(path: &Path) -> Result<Self>;
    pub fn score(&self, context: &[u32], next_word: u32) -> f32;
}
```

**LM Fusion in Beam Search (TDT only):**

The `decode_beam_with_lm` method implements shallow fusion:
- Tracks word boundaries using SentencePiece `▁` prefix convention
- Applies LM scoring when a word is completed
- Combined score: `acoustic_score + lm_weight * lm_score`

**CLI Usage:**
```bash
# Load LM for beam search
cargo run --example parakeet --features parakeet -- \
    --input audio.wav --beam-width 5 --lm domain.arpa --lm-weight 0.5
```

---

### Phase 7: MALSD Beam Search ✅

Modified Alignment-Length Synchronous Decoding for TDT decoder.

**Implementation in `tdt.rs`:**
```rust
impl TdtDecoder {
    pub fn decode_malsd(&self, encoder_output: &Tensor, beam_width: usize) -> Result<DecodingResult>;
}
```

**Algorithm:**
- Synchronizes beams by alignment length k = t + u (time + tokens)
- Processes all beams at same alignment level together
- Can improve accuracy for some inputs

**CLI Option:**
```bash
cargo run --example parakeet --features parakeet -- \
    --input audio.wav --beam-width 5 --malsd
```

---

## Usage Examples

### Basic Transcription
```bash
cargo run --example parakeet --release --features parakeet -- \
    --input audio.wav
```

### With Timestamps and Confidence
```bash
cargo run --example parakeet --release --features parakeet -- \
    --input audio.wav --word-timestamps --confidence
```

### Batch Processing with Beam Search
```bash
cargo run --example parakeet --release --features parakeet -- \
    --input file1.wav --input file2.wav --beam-width 5
```

### Long Audio with Streaming
```bash
cargo run --example parakeet --release --features parakeet -- \
    --input long_lecture.wav --streaming --word-timestamps
```

### With Language Model
```bash
cargo run --example parakeet --release --features parakeet -- \
    --input audio.wav --beam-width 5 --lm domain.arpa --lm-weight 0.5
```

### MALSD Beam Search (TDT only)
```bash
cargo run --example parakeet --release --features parakeet -- \
    --input audio.wav --beam-width 5 --malsd
```

---

## API Reference

### Decoder Trait
```rust
pub trait Decoder {
    fn decode_greedy(&self, encoder_output: &Tensor) -> Result<Vec<u32>>;
    fn decode_beam(&self, encoder_output: &Tensor, beam_width: usize) -> Result<Vec<u32>>;
    fn decode_greedy_with_info(&self, encoder_output: &Tensor) -> Result<DecodingResult>;
    fn decode_beam_with_info(&self, encoder_output: &Tensor, beam_width: usize) -> Result<DecodingResult>;
}
```

### Parakeet Model
```rust
impl Parakeet {
    // Standard inference
    pub fn forward(&self, mel: &Tensor) -> Result<Vec<u32>>;
    pub fn forward_beam(&self, mel: &Tensor, beam_width: usize) -> Result<Vec<u32>>;

    // Rich results
    pub fn forward_with_info(&self, mel: &Tensor) -> Result<DecodingResult>;
    pub fn forward_beam_with_info(&self, mel: &Tensor, beam_width: usize) -> Result<DecodingResult>;

    // MALSD (TDT only)
    pub fn forward_malsd(&self, mel: &Tensor, beam_width: usize) -> Result<DecodingResult>;

    // LM fusion (TDT only)
    pub fn forward_beam_with_lm(
        &self, mel: &Tensor, beam_width: usize,
        lm: &NgramLM, vocab: &[String], lm_weight: f32
    ) -> Result<DecodingResult>;

    // Batch processing
    pub fn forward_batch(&self, mels: &[&Tensor]) -> Result<Vec<DecodingResult>>;
    pub fn forward_batch_beam(&self, mels: &[&Tensor], beam_width: usize) -> Result<Vec<DecodingResult>>;
}
```

### DecodingResult Methods
```rust
impl DecodingResult {
    pub fn token_ids(&self) -> Vec<u32>;
    pub fn compute_timestamps(&self) -> Vec<TimestampInfo>;
    pub fn word_info(&self, vocab: &[String], aggregation: ConfidenceAggregation) -> Vec<WordInfo>;
}
```

---

## Files Modified

**Library (candle-transformers/src/models/parakeet/):**
- `mod.rs` - Result types, timestamp helpers, ConfidenceAggregation
- `model.rs` - Decoder trait, batch methods, MALSD wrapper, LM fusion wrapper
- `tdt.rs` - Rich results, MALSD beam search, LM-fused beam search
- `rnnt.rs` - Rich results
- `ctc.rs` - Rich results
- `lm.rs` - N-gram LM implementation

**Example (candle-examples/examples/parakeet/):**
- `main.rs` - CLI options, batch mode, streaming, LM/MALSD integration, output formats

---

## Constants

```rust
pub const FRAME_DURATION_MS: f32 = 80.0;  // 8 × 10ms hop
pub const SAMPLE_RATE: usize = 16000;
pub const HOP_LENGTH: usize = 160;
```

Each encoder frame represents 80ms of audio (8× subsampling factor × 10ms hop length).
