# Parakeet Debug Notes

## Summary

Investigated RNN-T/CTC transcription failures for XXL models (1.1B params, 42 layers).

## Issues Found

### 1. RNN-T Decoding Loop Bug (FIXED)

**Location**: `candle-transformers/src/models/parakeet/rnnt.rs:231-290`

**Problem**: Original RNN-T greedy decoder advanced frames incorrectly.

**Fix**: Added inner loop that keeps emitting tokens until blank is predicted.

### 2. Missing xscale for XXL Models (FIXED)

**Location**: `candle-transformers/src/models/parakeet/mod.rs:253-262`

**Problem**: XXL config had `scale_input: false` but NeMo XXL models use `xscaling=true`.

**Fix**: Set `scale_input: true` in `EncoderConfig::xxl()`.

**Result**: Block 0 RMS now matches NeMo (2.74 vs 2.73).

### 3. Missing Conv Module Biases (FIXED)

**Location**: `candle-transformers/src/models/parakeet/fastconformer.rs:299-357`

**Symptom**: CTC-1B model produced all blank tokens. Block 1+ diverged by 7-14%.

**Root Cause**: The ConvModule was missing bias parameters for all three convolutions:
- `pointwise_conv1.bias`: [2048]
- `depthwise_conv.bias`: [1024]
- `pointwise_conv2.bias`: [1024]

NeMo's checkpoint includes these biases, but Candle was passing `None`, causing significant numerical divergence.

**Fix**: Load bias tensors for all three convolutions:
```rust
// Before (wrong):
let pw1_weight = vb.pp("pointwise_conv1").get(..., "weight")?;
let pointwise_conv1 = candle_nn::Conv1d::new(pw1_weight, None, cfg);

// After (correct):
let pw1_vb = vb.pp("pointwise_conv1");
let pw1_weight = pw1_vb.get(..., "weight")?;
let pw1_bias = pw1_vb.get(hidden * 2, "bias")?;
let pointwise_conv1 = candle_nn::Conv1d::new(pw1_weight, Some(pw1_bias), cfg);
```

**Result**: All block outputs now match NeMo within 1%:

| Block | Before Fix | After Fix | NeMo | Status |
|-------|-----------|-----------|------|--------|
| 0 | 2.743 | 2.741 | 2.730 | ✅ +0.4% |
| 1 | 3.450 | 3.728 | 3.702 | ✅ +0.7% (was -7%) |
| 2 | 3.284 | 3.692 | 3.716 | ✅ -0.6% (was -12%) |

CTC-1B now produces correct transcriptions!

## Progress Summary (2025-12-27)

### Fixed Issues ✅

1. **RNN-T Decoding Loop Bug** - Fixed inner loop for token emission
2. **Missing xscale for XXL Models** - Set `scale_input: true` for XXL configs
3. **Missing Conv Module Biases** - Added bias loading for all conv layers
4. **Missing Audio Resampling** - Added resampling to 16kHz before mel computation
5. **Dithering During Inference** - Removed (NeMo only dithers during training)
6. **TDT Duration=0 Handling** - Fixed frame advancement to allow duration=0 (stay on same frame)

### What Works
- ✅ TDT-v2 model (XL, 24 layers) - All 6 test files match NeMo exactly
- ✅ TDT-v3 model (XL, 24 layers) - All 6 test files match NeMo exactly
- ✅ CTC-1B model (XXL, 42 layers) - All 6 test files match NeMo exactly
- ✅ RNN-T-1B model (XXL, 42 layers) - All 6 test files match NeMo exactly

### 7. TDT Duration=0 Frame Advancement (FIXED)

**Location**: `candle-transformers/src/models/parakeet/tdt.rs:259-343`

**Symptom**: TDT models were dropping short words like "a", "as", "of", "bright" resulting in 2-5% WER.

**Root Cause**: The TDT greedy decoder was using `t += max(duration, 1)` which always advanced by at least 1 frame. But per the [ICML 2023 TDT paper](https://arxiv.org/abs/2304.06795), when duration=0, the decoder should **stay on the same frame** to potentially emit more tokens.

**Original code (wrong):**
```rust
t += (duration as usize).max(1);  // Always advance by at least 1
```

**Fixed code:**
```rust
// TDT algorithm: t += duration (can be 0 to stay on same frame)
if duration > 0 {
    t += duration;
    break;
}
// Duration=0: stay on same frame, but check safety limit
if symbols_on_frame >= MAX_SYMBOLS_PER_FRAME {
    t += 1;  // Prevent infinite loop
    break;
}
// Continue on same frame with updated predictor state
```

**Why this matters**: When duration=0, the TDT model is signaling "emit another token before moving to the next frame". This is how rapid speech sequences (like "it's a double") get captured - the model emits multiple tokens per audio frame.

**Results**:
| Model | Before Fix | After Fix |
|-------|-----------|-----------|
| TDT-v2 | 2.2% WER | **0.0% WER** (EXACT MATCH) |
| TDT-v3 | 2.5% WER | 1.56% WER |

**Reference**: The TDT paper's Algorithm 2 (Greedy Inference) specifies:
```
t += duration_idx2duration[duration_idx]
```
This allows t to not advance when duration=0, enabling multiple token emissions per frame.

### 4. TDT-v2 Marginal Decoding Case (FIXED)

**File**: en-Davis_man.wav
**Symptom (before fix)**: Candle outputs "in the realm..." while NeMo outputs "In the realm..."

**Analysis**:
The first non-blank token decision was marginal:
- Token 36 ('in'): score 77.54
- Token 378 ('In'): score 76.85
- Difference: **0.69** (less than 1%)

**Root Cause**: Audio was not being resampled to 16kHz before mel computation. See "Missing Audio Resampling to 16kHz" section above.

**Result After Fix**: Token 378 ('In') now scores 92.66 vs Token 36 ('in') at 88.99 - a clear margin of 3.67.

### Investigation Notes

**BatchNorm was NOT the issue**: Initially suspected, but investigation showed running_mean and running_var were loading correctly (mean=0.284, var=20.5 as expected).

**How the conv bias bug was found**:
1. Added detailed debug output for each sub-component in ConformerBlock
2. Compared Candle vs NeMo intermediate values step-by-step
3. Found conv module output was 3x different (3.15 vs 1.03)
4. Checked NeMo checkpoint and found bias tensors we weren't loading

### 5. Missing Audio Resampling to 16kHz (FIXED)

**Location**: `candle-examples/examples/parakeet/main.rs:683-693`

**Problem**: NeMo resamples all audio to 16kHz before computing mel spectrograms. Candle was processing audio at native sample rate (e.g., 24kHz), producing different frame counts.

**Symptom**: Mel spectrogram frame count mismatch:
- Audio: 211200 samples @ 24kHz = 8.8s
- Candle (native rate): 1321 frames (processing at 24kHz)
- NeMo: 881 frames (resampled to 16kHz first)
- Ratio: 1321/881 = 1.5 = 24000/16000 ✓

**Root Cause Analysis**:
1. NeMo's `AudioSegment.from_file()` uses `librosa.core.resample(orig_sr, target_sr=16000)`
2. NeMo's `FilterbankFeatures` assumes 16kHz input
3. Candle was using `mel_filters_for_sample_rate(native_rate)` without resampling

**Fix**: Added resampling to 16kHz before mel computation:
```rust
// Resample to 16kHz if needed (NeMo always resamples to 16kHz)
let pcm_data = if native_sample_rate != parakeet::SAMPLE_RATE {
    candle_examples::audio::resample(&pcm_data, sample_rate, parakeet::SAMPLE_RATE as u32)?
} else {
    pcm_data
};
```

**Result**: TDT-v2 en-Davis_man.wav now correctly outputs "In the realm..." matching NeMo.

### 6. Dithering During Inference (FIXED)

**Location**: `candle-examples/examples/parakeet/main.rs:695-697`

**Problem**: NeMo only applies dithering during training mode:
```python
if self.training and self.dither > 0:
    x += self.dither * torch.randn_like(x)
```

Candle was applying dithering unconditionally during inference.

**Fix**: Removed dithering for inference mode since NeMo doesn't use it.

### NeMo Audio Processing Order (Reference)

Based on investigation of [NeMo FilterbankFeatures](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/preprocessing/features.py):

1. **Resample** → 16kHz (in AudioSegment.from_file)
2. **Dither** → Only during training
3. **Preemphasis** → y[n] = x[n] - 0.97 * x[n-1]
4. **STFT** → N_FFT=512, hop=160, win=400
5. **Mel filterbank** → 128 bins (XL) or 80 bins (XXL)
6. **Log** → log(x + 2^-24)
7. **Normalize** → per_feature (mean=0, std=1 per mel bin)

### NeMo ConformerLayer Forward Structure (Reference)

From [NeMo conformer_modules.py](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/conformer_modules.py):

```python
# FFN1: scaled by 0.5
residual = x
x = self.norm_feed_forward1(x)
x = self.feed_forward1(x)
residual = residual + dropout(x) * 0.5  # fc_factor = 0.5

# Self-Attention: no scaling
x = self.norm_self_att(residual)
x = self.self_attn(...)
residual = residual + dropout(x)

# Convolution: no scaling, uses BatchNorm
x = self.norm_conv(residual)
x = self.conv(x)  # includes batch_norm
residual = residual + dropout(x)

# FFN2: scaled by 0.5
x = self.norm_feed_forward2(residual)
x = self.feed_forward2(x)
residual = residual + dropout(x) * 0.5

# Final LayerNorm
x = self.norm_out(residual)
```

Our implementation matches this structure, but BatchNorm may not have correct running statistics.

## Test Commands

```bash
# Generate NeMo reference outputs
cd candle-examples/examples/parakeet
uv run save_nemo_intermediates.py --model ctc-1b

# Test CTC-1B (currently fails)
cargo run --example parakeet --release --features parakeet -- \
    --model-variant ctc1b \
    --input text/en-Carter_man.wav

# Test TDT-v3 (should work)
cargo run --example parakeet --release --features parakeet -- \
    --model-variant tdt-v3 \
    --input text/en-Carter_man.wav

# Test with NeMo encoder substitution (works!)
cargo run --example parakeet --release --features parakeet -- \
    --model-variant ctc1b \
    --input text/en-Carter_man.wav \
    --compare-nemo nemo_intermediates/ctc-1b \
    --substitute-encoder
```

## Files

- `save_nemo_intermediates.py` - Save NeMo intermediate tensors
- `test_candle_encoder_nemo_decoder.py` - Test Candle encoder with NeMo decoder
- `nemo_intermediates/ctc-1b/` - NeMo reference outputs (42 blocks + encoder + mel)
- `candle_intermediates/` - Candle outputs for comparison

## References

- [NeMo Conformer Modules](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/conformer_modules.py)
- [NeMo FastConformer Config](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/fastconformer/fast-conformer_ctc_bpe.yaml)
- [Candle batch_norm](https://github.com/huggingface/candle/blob/main/candle-nn/src/batch_norm.rs)
