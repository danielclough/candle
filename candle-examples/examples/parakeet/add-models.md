# Parakeet Multi-Model Support Implementation Plan

## Overview

Add support for all NVIDIA Parakeet ASR model variants to the Candle implementation:

| Model | Params | Encoder | Decoder | Vocab | Languages |
|-------|--------|---------|---------|-------|-----------|
| parakeet-tdt-0.6b-v2 | 600M | FastConformer XL (24 layers) | TDT | 1024 | English |
| parakeet-tdt-0.6b-v3 | 600M | FastConformer XL (24 layers) | TDT | 8192 | 25 languages |
| parakeet-rnnt-1.1b | 1.1B | FastConformer XXL (42 layers) | RNN-T | 1024 | English |
| parakeet-ctc-1.1b | 1.1B | FastConformer XXL (42 layers) | CTC | 1024 | English |

## Architecture Differences (Validated Against NeMo)

### Encoder Variants
- **XL (0.6B models)**: 24 layers, conv_kernel_size=9, 128 mel bins
- **XXL (1.1B models)**: 42 layers, conv_kernel_size=9, 80 mel bins (NOTE: kernel=9, not 5 as originally documented)

### Decoder Types

#### 1. TDT (current - already implemented)
- **Joint output**: `[vocab_size + 1 + num_durations]` = `[8193 + 5]` for v3
- **Split**: Token logits at `[:-num_durations]`, duration logits at `[-num_durations:]`
- **Frame advance**: `last_frame + predicted_duration`
- **NeMo source**: `tdt_beam_decoding.py`

#### 2. RNN-T
- **Joint output**: `[vocab_size + 1]` only (no duration)
- **Prediction network**: Same 2-layer LSTM as TDT
- **Frame advance**: Always +1 (no skipping)
- **NeMo source**: `rnnt.py`, `rnnt_greedy_decoding.py`

#### 3. CTC
- **Decoder**: Single Conv1d(encoder_dim, vocab_size+1, kernel_size=1) - equivalent to Linear
- **Greedy**: `predictions.max(dim=-1)` then collapse blanks/repeats
- **Blank collapsing**: Remove consecutive duplicates and blank tokens
- **NeMo source**: `conv_asr.py`, `ctc_greedy_decoding.py`

---

## Implementation Phases

### Phase 1: Configuration Refactoring
**Files:** `candle-transformers/src/models/parakeet/mod.rs`

1. Add `ModelVariant` enum:
   ```rust
   pub enum ModelVariant {
       TdtV2,
       TdtV3,
       Rnnt1b,
       Ctc1b,
   }
   ```

2. Add encoder configs for XL and XXL:
   ```rust
   impl EncoderConfig {
       pub fn xl() -> Self { /* 24 layers, kernel=9 */ }
       pub fn xxl() -> Self { /* 42 layers, kernel=5 */ }
   }
   ```

3. Add decoder configs for each variant:
   ```rust
   impl DecoderConfig {
       pub fn tdt_v2() -> Self { /* vocab=1024 */ }
       pub fn tdt_v3() -> Self { /* vocab=8192 */ }
       pub fn rnnt_1b() -> Self { /* vocab=1024, no duration */ }
       pub fn ctc_1b() -> Self { /* vocab=1024 */ }
   }
   ```

4. Add `DecoderType` enum:
   ```rust
   pub enum DecoderType {
       Tdt,
       Rnnt,
       Ctc,
   }
   ```

### Phase 2: RNN-T Decoder Implementation
**Files:** `candle-transformers/src/models/parakeet/rnnt.rs` (new file)

RNN-T is nearly identical to TDT, but without duration prediction:

1. **RnntPredictor**: Identical to TdtPredictor (2-layer LSTM with embedding)
   - Embedding: `(vocab_size + 1) → predictor_hidden_size`
   - LSTM layers: hidden_size=640

2. **RnntJointNetwork**: Same structure, smaller output
   - encoder_proj: `1024 → 640`
   - predictor_proj: `640 → 640`
   - output_linear: `640 → (vocab_size + 1)` (no duration logits)

3. **Greedy decoding** (validated from NeMo):
   ```rust
   while t < num_frames {
       let (token_logits,) = joint.forward(enc[t], pred_state);
       let token = argmax(token_logits);
       if token != blank_id {
           tokens.push(token);
           pred_state = predictor.step(token);
       }
       t += 1;  // Always advance by 1 (key difference from TDT)
   }
   ```

4. **Beam search**: Same as TDT but without duration scoring

### Phase 3: CTC Decoder Implementation
**Files:** `candle-transformers/src/models/parakeet/ctc.rs` (new file)

CTC is much simpler - no recurrent predictor (validated from NeMo `conv_asr.py`):

1. **CtcDecoder struct**:
   ```rust
   pub struct CtcDecoder {
       // NeMo uses Conv1d(kernel_size=1) which is equivalent to Linear
       output_proj: Linear,  // encoder_dim (1024) → vocab_size + 1 (1025)
       blank_id: u32,        // Always vocab_size (1024)
   }
   ```

2. **Greedy decoding** (validated from NeMo `ctc_greedy_decoding.py`):
   ```rust
   fn decode_greedy(&self, encoder_output: &Tensor) -> Vec<u32> {
       // Project: [batch, time, 1024] → [batch, time, 1025]
       let logits = self.output_proj.forward(encoder_output)?;

       // Argmax per frame: [batch, time]
       let predictions = logits.argmax(D::Minus1)?;

       // Collapse blanks and repeats
       let mut tokens = Vec::new();
       let mut prev = self.blank_id;
       for t in 0..num_frames {
           let token = predictions[t];
           if token != prev && token != self.blank_id {
               tokens.push(token);
           }
           prev = token;
       }
       tokens
   }
   ```

3. **Beam search with prefix merging**:
   ```rust
   fn decode_beam(&self, encoder_output: &Tensor, beam_width: usize) -> Vec<u32> {
       // CTC beam search tracks prefixes, not individual tokens
       // Each beam has: prefix (tokens), p_blank (ends in blank), p_non_blank
       // Merge beams with same prefix text
   }
   ```

   Key CTC beam search concepts (from NeMo):
   - Track `(prefix, p_blank, p_non_blank)` tuples
   - When extending with blank: `p_blank_new = (p_blank + p_non_blank) * p_t(blank)`
   - When extending with token c:
     - If c == last(prefix): only use p_blank (can't repeat without blank)
     - Otherwise: use (p_blank + p_non_blank)
   - Merge beams with identical prefixes by summing probabilities

### Phase 4: Unified Model Interface
**Files:** `candle-transformers/src/models/parakeet/model.rs`

Create decoder trait and unified model:

```rust
pub trait Decoder {
    fn decode_greedy(&self, encoder_output: &Tensor) -> Result<Vec<u32>>;
    fn decode_beam(&self, encoder_output: &Tensor, beam_width: usize) -> Result<Vec<u32>>;
}

pub enum DecoderImpl {
    Tdt(TdtDecoder),
    Rnnt(RnntDecoder),
    Ctc(CtcDecoder),
}

impl Parakeet {
    pub fn load(vb: VarBuilder, config: Config) -> Result<Self> {
        let encoder = FastConformerEncoder::load(...)?;
        let decoder = match config.decoder_type {
            DecoderType::Tdt => DecoderImpl::Tdt(TdtDecoder::load(...)?),
            DecoderType::Rnnt => DecoderImpl::Rnnt(RnntDecoder::load(...)?),
            DecoderType::Ctc => DecoderImpl::Ctc(CtcDecoder::load(...)?),
        };
        Ok(Self { encoder, decoder, config })
    }
}
```

### Phase 5: CLI Updates
**Files:** `candle-examples/examples/parakeet/main.rs`

1. Add `--model-variant` argument:
   ```rust
   #[arg(long, value_enum, default_value = "tdt-v3")]
   model_variant: ModelVariant,
   ```

2. Update HuggingFace download logic to use variant-specific repo/file
3. Update `load_nemo_model()` to accept variant and load appropriate decoder
4. Make vocab offset calculation dynamic based on config

### Phase 6: Testing
**Files:** `candle-examples/examples/parakeet/test_transcriptions.sh`

1. Update test script to accept model variant argument
2. Run tests against all model variants
3. Compare with official NeMo outputs for each variant

---

## File Changes Summary

### New Files
- `candle-transformers/src/models/parakeet/rnnt.rs` - RNN-T decoder
- `candle-transformers/src/models/parakeet/ctc.rs` - CTC decoder

### Modified Files
- `candle-transformers/src/models/parakeet/mod.rs` - Add configs, enums, exports
- `candle-transformers/src/models/parakeet/model.rs` - Unified decoder interface
- `candle-transformers/src/models/parakeet/fastconformer.rs` - XXL encoder support (if needed)
- `candle-examples/examples/parakeet/main.rs` - CLI and loading updates

---

## Implementation Order

1. **Phase 1**: Config refactoring (foundation for everything)
2. **Phase 2**: RNN-T decoder (similar to existing TDT)
3. **Phase 3**: CTC decoder (simpler, good validation)
4. **Phase 4**: Unified model interface
5. **Phase 5**: CLI updates
6. **Phase 6**: Testing

---

## Key Technical Notes

### Weight Mapping (Validated from NeMo)

**TDT/RNN-T models:**
```
encoder.pre_encode.*           → ConvSubsampling
encoder.layers.{i}.*           → ConformerBlock[i]
decoder.prediction_network.*   → Predictor (embedding + LSTM)
decoder.joint_network.*        → Joint (encoder_proj, predictor_proj, output)
```

**CTC models:**
```
encoder.pre_encode.*           → ConvSubsampling
encoder.layers.{i}.*           → ConformerBlock[i]
decoder.decoder_layers.*       → Single Linear projection (Conv1d kernel=1)
```

**Key difference:** CTC has no `prediction_network` or `joint_network` - just `decoder_layers`

### Vocabulary Handling
- v2/rnnt/ctc: 1024 tokens, SentencePiece
- v3: 8192 tokens (multilingual), SentencePiece
- Blank token always at vocab_size index

### Encoder Compatibility
- All models use same FastConformer architecture
- Only difference: num_layers (24 vs 42) and conv_kernel_size (9 vs 5)
- Current implementation handles this via config - no code changes needed

---

## Estimated Effort

| Phase | Complexity | Files Changed |
|-------|------------|---------------|
| Phase 1 | Low | 1 |
| Phase 2 | Medium | 2 |
| Phase 3 | Low | 2 |
| Phase 4 | Medium | 1 |
| Phase 5 | Low | 1 |
| Phase 6 | Low | 1 |

Total: ~500-700 lines of new code
