//! Top-level Parakeet ASR Model
//!
//! Supports multiple decoder types: TDT, RNN-T, and CTC.

use candle::{Result, Tensor};
use candle_nn::VarBuilder;

use super::ctc::CtcDecoder;
use super::fastconformer::FastConformerEncoder;
use super::rnnt::RnntDecoder;
use super::tdt::TdtDecoder;
use super::{Config, DecoderType, DecodingResult};

/// Decoder trait for unified interface
///
/// All decoder types (TDT, RNN-T, CTC) implement this trait.
/// The `_with_info` variants return rich results with timestamps and confidence.
pub trait Decoder {
    /// Greedy decoding - fastest but less accurate
    fn decode_greedy(&self, encoder_output: &Tensor) -> Result<Vec<u32>>;

    /// Beam search decoding - slower but more accurate
    fn decode_beam(&self, encoder_output: &Tensor, beam_width: usize) -> Result<Vec<u32>>;

    /// Greedy decoding with rich results (timestamps, confidence)
    fn decode_greedy_with_info(&self, encoder_output: &Tensor) -> Result<DecodingResult>;

    /// Beam search decoding with rich results (timestamps, confidence)
    fn decode_beam_with_info(
        &self,
        encoder_output: &Tensor,
        beam_width: usize,
    ) -> Result<DecodingResult>;
}

impl Decoder for TdtDecoder {
    fn decode_greedy(&self, encoder_output: &Tensor) -> Result<Vec<u32>> {
        self.decode_greedy(encoder_output)
    }

    fn decode_beam(&self, encoder_output: &Tensor, beam_width: usize) -> Result<Vec<u32>> {
        self.decode_beam(encoder_output, beam_width)
    }

    fn decode_greedy_with_info(&self, encoder_output: &Tensor) -> Result<DecodingResult> {
        self.decode_greedy_with_info(encoder_output)
    }

    fn decode_beam_with_info(
        &self,
        encoder_output: &Tensor,
        beam_width: usize,
    ) -> Result<DecodingResult> {
        self.decode_beam_with_info(encoder_output, beam_width)
    }
}

impl Decoder for RnntDecoder {
    fn decode_greedy(&self, encoder_output: &Tensor) -> Result<Vec<u32>> {
        self.decode_greedy(encoder_output)
    }

    fn decode_beam(&self, encoder_output: &Tensor, beam_width: usize) -> Result<Vec<u32>> {
        self.decode_beam(encoder_output, beam_width)
    }

    fn decode_greedy_with_info(&self, encoder_output: &Tensor) -> Result<DecodingResult> {
        self.decode_greedy_with_info(encoder_output)
    }

    fn decode_beam_with_info(
        &self,
        encoder_output: &Tensor,
        beam_width: usize,
    ) -> Result<DecodingResult> {
        self.decode_beam_with_info(encoder_output, beam_width)
    }
}

impl Decoder for CtcDecoder {
    fn decode_greedy(&self, encoder_output: &Tensor) -> Result<Vec<u32>> {
        self.decode_greedy(encoder_output)
    }

    fn decode_beam(&self, encoder_output: &Tensor, beam_width: usize) -> Result<Vec<u32>> {
        self.decode_beam(encoder_output, beam_width)
    }

    fn decode_greedy_with_info(&self, encoder_output: &Tensor) -> Result<DecodingResult> {
        self.decode_greedy_with_info(encoder_output)
    }

    fn decode_beam_with_info(
        &self,
        encoder_output: &Tensor,
        beam_width: usize,
    ) -> Result<DecodingResult> {
        self.decode_beam_with_info(encoder_output, beam_width)
    }
}

/// Decoder implementation enum for static dispatch
///
/// This avoids the overhead of dynamic dispatch (trait objects) while
/// providing a unified interface for all decoder types.
#[derive(Debug, Clone)]
pub enum DecoderImpl {
    Tdt(TdtDecoder),
    Rnnt(RnntDecoder),
    Ctc(CtcDecoder),
}

impl Decoder for DecoderImpl {
    fn decode_greedy(&self, encoder_output: &Tensor) -> Result<Vec<u32>> {
        match self {
            DecoderImpl::Tdt(d) => d.decode_greedy(encoder_output),
            DecoderImpl::Rnnt(d) => d.decode_greedy(encoder_output),
            DecoderImpl::Ctc(d) => d.decode_greedy(encoder_output),
        }
    }

    fn decode_beam(&self, encoder_output: &Tensor, beam_width: usize) -> Result<Vec<u32>> {
        match self {
            DecoderImpl::Tdt(d) => d.decode_beam(encoder_output, beam_width),
            DecoderImpl::Rnnt(d) => d.decode_beam(encoder_output, beam_width),
            DecoderImpl::Ctc(d) => d.decode_beam(encoder_output, beam_width),
        }
    }

    fn decode_greedy_with_info(&self, encoder_output: &Tensor) -> Result<DecodingResult> {
        match self {
            DecoderImpl::Tdt(d) => d.decode_greedy_with_info(encoder_output),
            DecoderImpl::Rnnt(d) => d.decode_greedy_with_info(encoder_output),
            DecoderImpl::Ctc(d) => d.decode_greedy_with_info(encoder_output),
        }
    }

    fn decode_beam_with_info(
        &self,
        encoder_output: &Tensor,
        beam_width: usize,
    ) -> Result<DecodingResult> {
        match self {
            DecoderImpl::Tdt(d) => d.decode_beam_with_info(encoder_output, beam_width),
            DecoderImpl::Rnnt(d) => d.decode_beam_with_info(encoder_output, beam_width),
            DecoderImpl::Ctc(d) => d.decode_beam_with_info(encoder_output, beam_width),
        }
    }
}

impl DecoderImpl {
    /// Load the appropriate decoder based on decoder type
    pub fn load(
        vb: VarBuilder,
        config: &Config,
    ) -> Result<Self> {
        match config.decoder_type {
            DecoderType::Tdt => {
                let decoder = TdtDecoder::load(vb, &config.encoder, &config.decoder)?;
                Ok(DecoderImpl::Tdt(decoder))
            }
            DecoderType::Rnnt => {
                let decoder = RnntDecoder::load(vb, &config.encoder, &config.decoder)?;
                Ok(DecoderImpl::Rnnt(decoder))
            }
            DecoderType::Ctc => {
                let decoder = CtcDecoder::load(vb, &config.encoder, &config.decoder)?;
                Ok(DecoderImpl::Ctc(decoder))
            }
        }
    }

    /// Returns the decoder type
    pub fn decoder_type(&self) -> DecoderType {
        match self {
            DecoderImpl::Tdt(_) => DecoderType::Tdt,
            DecoderImpl::Rnnt(_) => DecoderType::Rnnt,
            DecoderImpl::Ctc(_) => DecoderType::Ctc,
        }
    }

    /// MALSD beam search (only available for TDT decoder)
    ///
    /// Modified Alignment-Length Synchronous Decoding synchronizes beams
    /// by alignment length instead of time, potentially improving accuracy.
    pub fn decode_malsd(
        &self,
        encoder_output: &Tensor,
        beam_width: usize,
    ) -> Result<DecodingResult> {
        match self {
            DecoderImpl::Tdt(d) => d.decode_malsd(encoder_output, beam_width),
            DecoderImpl::Rnnt(_) | DecoderImpl::Ctc(_) => {
                candle::bail!("MALSD decoding is only available for TDT decoder")
            }
        }
    }

    /// Beam search with N-gram LM shallow fusion (only available for TDT decoder)
    ///
    /// Applies LM scoring at word boundaries for improved accuracy on domain-specific tasks.
    pub fn decode_beam_with_lm(
        &self,
        encoder_output: &Tensor,
        beam_width: usize,
        lm: &super::lm::NgramLM,
        vocab: &[String],
        lm_weight: f32,
    ) -> Result<DecodingResult> {
        match self {
            DecoderImpl::Tdt(d) => {
                d.decode_beam_with_lm(encoder_output, beam_width, lm, vocab, lm_weight)
            }
            DecoderImpl::Rnnt(_) | DecoderImpl::Ctc(_) => {
                candle::bail!("LM fusion is currently only available for TDT decoder")
            }
        }
    }
}

/// Parakeet ASR Model
///
/// Supports multiple model variants:
/// - TDT v2/v3: Token-Duration Transducer with frame skipping
/// - RNN-T: Standard transducer without duration prediction
/// - CTC: Simple frame-wise classification
#[derive(Debug, Clone)]
pub struct Parakeet {
    pub encoder: FastConformerEncoder,
    pub decoder: DecoderImpl,
    pub config: Config,
}

impl Parakeet {
    /// Load model from weights with automatic decoder selection
    ///
    /// The decoder type is determined by `config.decoder_type`.
    pub fn load(vb: VarBuilder, config: Config) -> Result<Self> {
        // NeMo weight structure:
        // - encoder.* - FastConformer encoder
        // - decoder.* - Predictor (for TDT/RNN-T) or decoder_layers (for CTC)
        // - joint.* - Joint network (for TDT/RNN-T only)
        let encoder = FastConformerEncoder::load(vb.pp("encoder"), &config.encoder)?;
        let decoder = DecoderImpl::load(vb.clone(), &config)?;

        Ok(Self {
            encoder,
            decoder,
            config,
        })
    }

    /// Load model with specific TDT decoder (for backwards compatibility)
    pub fn load_tdt(vb: VarBuilder, config: Config) -> Result<Self> {
        let encoder = FastConformerEncoder::load(vb.pp("encoder"), &config.encoder)?;
        let decoder = TdtDecoder::load(vb.clone(), &config.encoder, &config.decoder)?;

        Ok(Self {
            encoder,
            decoder: DecoderImpl::Tdt(decoder),
            config,
        })
    }

    /// Run inference on mel spectrogram
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram tensor of shape [batch, num_mel_bins, time]
    ///
    /// # Returns
    /// Vector of token IDs (use tokenizer to decode to text)
    pub fn forward(&self, mel: &Tensor) -> Result<Vec<u32>> {
        let enc_output = self.encoder.forward(mel, None)?;
        self.decoder.decode_greedy(&enc_output)
    }

    /// Run inference with beam search decoding
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram tensor of shape [batch, num_mel_bins, time]
    /// * `beam_width` - Number of beams for beam search
    ///
    /// # Returns
    /// Vector of token IDs (use tokenizer to decode to text)
    pub fn forward_beam(&self, mel: &Tensor, beam_width: usize) -> Result<Vec<u32>> {
        let enc_output = self.encoder.forward(mel, None)?;
        self.decoder.decode_beam(&enc_output, beam_width)
    }

    /// Run inference with rich results (timestamps, confidence)
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram tensor of shape [batch, num_mel_bins, time]
    ///
    /// # Returns
    /// DecodingResult with tokens, timestamps, and confidence scores
    pub fn forward_with_info(&self, mel: &Tensor) -> Result<DecodingResult> {
        let enc_output = self.encoder.forward(mel, None)?;
        self.decoder.decode_greedy_with_info(&enc_output)
    }

    /// Run inference with beam search and rich results
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram tensor of shape [batch, num_mel_bins, time]
    /// * `beam_width` - Number of beams for beam search
    ///
    /// # Returns
    /// DecodingResult with tokens, timestamps, and confidence scores
    pub fn forward_beam_with_info(
        &self,
        mel: &Tensor,
        beam_width: usize,
    ) -> Result<DecodingResult> {
        let enc_output = self.encoder.forward(mel, None)?;
        self.decoder.decode_beam_with_info(&enc_output, beam_width)
    }

    /// Run inference with MALSD beam search (TDT only)
    ///
    /// MALSD (Modified Alignment-Length Synchronous Decoding) synchronizes
    /// beams by alignment length k = t + u instead of just time t.
    /// This can improve accuracy for some inputs.
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram tensor of shape [batch, num_mel_bins, time]
    /// * `beam_width` - Number of beams for beam search
    ///
    /// # Returns
    /// DecodingResult with tokens, timestamps, and confidence scores
    ///
    /// # Errors
    /// Returns error if decoder is not TDT (MALSD only works with TDT)
    pub fn forward_malsd(
        &self,
        mel: &Tensor,
        beam_width: usize,
    ) -> Result<DecodingResult> {
        let enc_output = self.encoder.forward(mel, None)?;
        self.decoder.decode_malsd(&enc_output, beam_width)
    }

    /// Run inference with beam search and N-gram LM fusion (TDT only)
    ///
    /// Applies shallow fusion with an N-gram language model during beam search.
    /// LM scores are applied at word boundaries (SentencePiece `▁` prefix).
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram tensor of shape [batch, num_mel_bins, time]
    /// * `beam_width` - Number of beams for beam search
    /// * `lm` - N-gram language model
    /// * `vocab` - ASR vocabulary (token ID -> token string)
    /// * `lm_weight` - Weight for LM scores (typically 0.3-0.7)
    ///
    /// # Returns
    /// DecodingResult with tokens, timestamps, and combined acoustic+LM scores
    pub fn forward_beam_with_lm(
        &self,
        mel: &Tensor,
        beam_width: usize,
        lm: &super::lm::NgramLM,
        vocab: &[String],
        lm_weight: f32,
    ) -> Result<DecodingResult> {
        let enc_output = self.encoder.forward(mel, None)?;
        self.decoder
            .decode_beam_with_lm(&enc_output, beam_width, lm, vocab, lm_weight)
    }

    /// Get the encoder output without decoding
    ///
    /// Useful for debugging or custom decoding strategies.
    pub fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        self.encoder.forward(mel, None)
    }

    /// Returns the decoder type for this model
    pub fn decoder_type(&self) -> DecoderType {
        self.decoder.decoder_type()
    }

    /// Process multiple audio samples in a batch
    ///
    /// # Arguments
    /// * `mels` - Slice of mel spectrogram tensors, each of shape [1, num_mel_bins, time]
    ///
    /// # Returns
    /// Vector of DecodingResults, one per input
    ///
    /// The encoder processes all inputs in a single batch for efficiency.
    /// Decoding is done per-sequence (transducer decoders don't batch well).
    pub fn forward_batch(&self, mels: &[&Tensor]) -> Result<Vec<DecodingResult>> {
        if mels.is_empty() {
            return Ok(vec![]);
        }

        // Get dimensions
        let num_mel_bins = mels[0].dim(1)?;
        let max_time = mels.iter().map(|m| m.dim(2).unwrap_or(0)).max().unwrap_or(0);
        let device = mels[0].device();

        // Pad all mels to max_time and stack into batch
        let mut padded_mels = Vec::with_capacity(mels.len());
        let mut original_lengths = Vec::with_capacity(mels.len());

        for mel in mels {
            let time = mel.dim(2)?;
            original_lengths.push(time);

            if time < max_time {
                // Pad with zeros on the time dimension
                let padding = Tensor::zeros((1, num_mel_bins, max_time - time), mel.dtype(), device)?;
                let mel_squeezed = mel.squeeze(0)?; // [mel_bins, time]
                let padding_squeezed = padding.squeeze(0)?;
                let padded = Tensor::cat(&[&mel_squeezed, &padding_squeezed], 1)?; // [mel_bins, time]
                padded_mels.push(padded);
            } else {
                padded_mels.push(mel.squeeze(0)?.clone());
            }
        }

        // Stack into batch tensor: [batch, mel_bins, time]
        let batch_mel = Tensor::stack(&padded_mels.iter().collect::<Vec<_>>(), 0)?;

        // Run encoder on batch
        let enc_output = self.encoder.forward(&batch_mel, None)?;
        // enc_output: [batch, time/8, hidden]

        // Decode each sequence
        let batch_size = mels.len();
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            // Extract single sequence: [1, time/8, hidden]
            let seq_output = enc_output.narrow(0, i, 1)?;

            // Calculate actual output length (after subsampling)
            let subsampling = self.config.encoder.subsampling_factor;
            let actual_output_len = (original_lengths[i] + subsampling - 1) / subsampling;

            // Trim to actual length (remove padding)
            let seq_output = if actual_output_len < seq_output.dim(1)? {
                seq_output.narrow(1, 0, actual_output_len)?
            } else {
                seq_output
            };

            // Decode
            let result = self.decoder.decode_greedy_with_info(&seq_output)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Process multiple audio samples in a batch with beam search
    ///
    /// Same as `forward_batch` but uses beam search decoding.
    pub fn forward_batch_beam(
        &self,
        mels: &[&Tensor],
        beam_width: usize,
    ) -> Result<Vec<DecodingResult>> {
        if mels.is_empty() {
            return Ok(vec![]);
        }

        // Get dimensions
        let num_mel_bins = mels[0].dim(1)?;
        let max_time = mels.iter().map(|m| m.dim(2).unwrap_or(0)).max().unwrap_or(0);
        let device = mels[0].device();

        // Pad all mels to max_time and stack into batch
        let mut padded_mels = Vec::with_capacity(mels.len());
        let mut original_lengths = Vec::with_capacity(mels.len());

        for mel in mels {
            let time = mel.dim(2)?;
            original_lengths.push(time);

            if time < max_time {
                let padding = Tensor::zeros((1, num_mel_bins, max_time - time), mel.dtype(), device)?;
                let mel_squeezed = mel.squeeze(0)?;
                let padding_squeezed = padding.squeeze(0)?;
                let padded = Tensor::cat(&[&mel_squeezed, &padding_squeezed], 1)?;
                padded_mels.push(padded);
            } else {
                padded_mels.push(mel.squeeze(0)?.clone());
            }
        }

        let batch_mel = Tensor::stack(&padded_mels.iter().collect::<Vec<_>>(), 0)?;
        let enc_output = self.encoder.forward(&batch_mel, None)?;

        let batch_size = mels.len();
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let seq_output = enc_output.narrow(0, i, 1)?;
            let subsampling = self.config.encoder.subsampling_factor;
            let actual_output_len = (original_lengths[i] + subsampling - 1) / subsampling;

            let seq_output = if actual_output_len < seq_output.dim(1)? {
                seq_output.narrow(1, 0, actual_output_len)?
            } else {
                seq_output
            };

            let result = self.decoder.decode_beam_with_info(&seq_output, beam_width)?;
            results.push(result);
        }

        Ok(results)
    }
}
