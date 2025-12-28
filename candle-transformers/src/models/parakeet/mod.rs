//! Parakeet ASR Model Implementation
//!
//! NVIDIA Parakeet models are FastConformer-based ASR models with various decoder types:
//!
//! | Model | Params | Encoder | Decoder | Vocab | Languages |
//! |-------|--------|---------|---------|-------|-----------|
//! | parakeet-tdt-0.6b-v2 | 600M | FastConformer XL (24 layers) | TDT | 1024 | English |
//! | parakeet-tdt-0.6b-v3 | 600M | FastConformer XL (24 layers) | TDT | 8192 | 25 languages |
//! | parakeet-rnnt-1.1b | 1.1B | FastConformer XXL (42 layers) | RNN-T | 1024 | English |
//! | parakeet-ctc-1.1b | 1.1B | FastConformer XXL (42 layers) | CTC | 1024 | English |
//!
//! - [NVIDIA Parakeet TDT v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
//! - [FastConformer Paper](https://arxiv.org/abs/2305.05084)
//! - [TDT Paper](https://arxiv.org/abs/2304.06795)

pub mod audio;
pub mod ctc;
pub mod fastconformer;
pub mod lm;
pub mod model;
pub mod rnnt;
pub mod tdt;

use candle_nn::Activation;
use serde::Deserialize;

/// Supported Parakeet model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelVariant {
    /// parakeet-tdt-0.6b-v2: 600M params, 24 layers, 1024 vocab, English only
    TdtV2,
    /// parakeet-tdt-0.6b-v3: 600M params, 24 layers, 8192 vocab, 25 languages
    #[default]
    TdtV3,
    /// parakeet-rnnt-1.1b: 1.1B params, 42 layers, 1024 vocab, English only
    Rnnt1b,
    /// parakeet-ctc-1.1b: 1.1B params, 42 layers, 1024 vocab, English only
    Ctc1b,
}

impl ModelVariant {
    /// Returns the HuggingFace repository name for this model variant
    pub fn repo_name(&self) -> &'static str {
        match self {
            ModelVariant::TdtV2 => "nvidia/parakeet-tdt-0.6b-v2",
            ModelVariant::TdtV3 => "nvidia/parakeet-tdt-0.6b-v3",
            ModelVariant::Rnnt1b => "nvidia/parakeet-rnnt-1.1b",
            ModelVariant::Ctc1b => "nvidia/parakeet-ctc-1.1b",
        }
    }

    /// Returns the .nemo filename for this model variant
    pub fn nemo_filename(&self) -> &'static str {
        match self {
            ModelVariant::TdtV2 => "parakeet-tdt-0.6b-v2.nemo",
            ModelVariant::TdtV3 => "parakeet-tdt-0.6b-v3.nemo",
            ModelVariant::Rnnt1b => "parakeet-rnnt-1.1b.nemo",
            ModelVariant::Ctc1b => "parakeet-ctc-1.1b.nemo",
        }
    }

    /// Returns the decoder type for this model variant
    pub fn decoder_type(&self) -> DecoderType {
        match self {
            ModelVariant::TdtV2 | ModelVariant::TdtV3 => DecoderType::Tdt,
            ModelVariant::Rnnt1b => DecoderType::Rnnt,
            ModelVariant::Ctc1b => DecoderType::Ctc,
        }
    }

    /// Returns the appropriate Config for this model variant
    pub fn config(&self) -> Config {
        match self {
            ModelVariant::TdtV2 => Config::parakeet_tdt_0_6b_v2(),
            ModelVariant::TdtV3 => Config::parakeet_tdt_0_6b_v3(),
            ModelVariant::Rnnt1b => Config::parakeet_rnnt_1_1b(),
            ModelVariant::Ctc1b => Config::parakeet_ctc_1_1b(),
        }
    }
}

/// Decoder architecture types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
pub enum DecoderType {
    /// Token-Duration Transducer: RNN-T with duration prediction for frame skipping
    #[default]
    Tdt,
    /// RNN Transducer: Standard transducer without duration prediction
    Rnnt,
    /// Connectionist Temporal Classification: Simple frame-wise classification
    Ctc,
}

// Audio processing constants
pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 512;
pub const HOP_LENGTH: usize = 160;
pub const WIN_LENGTH: usize = 400;
pub const N_MELS: usize = 128;
pub const PREEMPHASIS: f32 = 0.97;

// Sample rate validation bounds
// 8kHz minimum: telephony quality, reduced frequency range (Nyquist=4kHz)
// 48kHz maximum: tested limit, professional audio quality
pub const MIN_SAMPLE_RATE: usize = 8000;
pub const MAX_SAMPLE_RATE: usize = 48000;

// Dithering amount (NeMo default)
pub const DITHER_AMOUNT: f32 = 1e-5;

// Frame-to-time conversion constant
// Each encoder frame = subsampling_factor (8) × hop_length (160) / sample_rate (16000) = 80ms
pub const FRAME_DURATION_MS: f32 = 80.0;

// ============================================================================
// Rich Decoding Result Types
// ============================================================================

/// Per-token information from decoding
///
/// Contains the token ID, its log probability (confidence), and frame timing.
/// Frame indices can be converted to timestamps using `FRAME_DURATION_MS`.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenInfo {
    /// The vocabulary token ID
    pub token_id: u32,
    /// Log probability of this token (higher = more confident)
    pub log_prob: f32,
    /// Encoder frame where this token started
    pub start_frame: usize,
    /// Encoder frame where this token ended
    pub end_frame: usize,
}

/// Rich decoding result with metadata
///
/// Contains the full sequence of tokens with timing and confidence information.
#[derive(Clone, Debug)]
pub struct DecodingResult {
    /// Sequence of tokens with metadata
    pub tokens: Vec<TokenInfo>,
    /// Sum of all token log probabilities
    pub total_log_prob: f32,
    /// Total encoder frames processed
    pub num_frames: usize,
}

impl DecodingResult {
    /// Create a new empty result
    pub fn new(num_frames: usize) -> Self {
        Self {
            tokens: Vec::new(),
            total_log_prob: 0.0,
            num_frames,
        }
    }

    /// Extract just token IDs (for backward compatibility)
    pub fn token_ids(&self) -> Vec<u32> {
        self.tokens.iter().map(|t| t.token_id).collect()
    }

    /// Convert frame indices to timestamps
    pub fn compute_timestamps(&self) -> Vec<TimestampInfo> {
        self.tokens
            .iter()
            .map(|t| TimestampInfo {
                start_time_sec: t.start_frame as f32 * FRAME_DURATION_MS / 1000.0,
                end_time_sec: t.end_frame as f32 * FRAME_DURATION_MS / 1000.0,
                confidence: t.log_prob.exp(),
            })
            .collect()
    }

    /// Compute word-level information from tokens
    ///
    /// Groups consecutive tokens into words and aggregates confidence.
    /// Uses SentencePiece convention: word starts with `▁` (U+2581).
    pub fn word_info(&self, vocab: &[String], aggregation: ConfidenceAggregation) -> Vec<WordInfo> {
        if self.tokens.is_empty() {
            return Vec::new();
        }

        let mut words = Vec::new();
        let mut current_word = String::new();
        let mut current_log_probs = Vec::new();
        let mut current_token_indices = Vec::new();
        let mut word_start_frame = 0;
        let mut word_end_frame = 0;

        for (idx, token_info) in self.tokens.iter().enumerate() {
            let token_text = vocab
                .get(token_info.token_id as usize)
                .map(|s| s.as_str())
                .unwrap_or("");

            // Check if this token starts a new word (SentencePiece: starts with ▁)
            let starts_new_word = token_text.starts_with('▁') || token_text.starts_with("##");

            if starts_new_word && !current_word.is_empty() {
                // Save the previous word
                let confidence = aggregate_confidence(&current_log_probs, aggregation);
                words.push(WordInfo {
                    word: current_word.clone(),
                    start_time_sec: word_start_frame as f32 * FRAME_DURATION_MS / 1000.0,
                    end_time_sec: word_end_frame as f32 * FRAME_DURATION_MS / 1000.0,
                    confidence,
                    token_indices: current_token_indices.clone(),
                });

                // Start a new word
                current_word.clear();
                current_log_probs.clear();
                current_token_indices.clear();
                word_start_frame = token_info.start_frame;
            }

            if current_word.is_empty() {
                word_start_frame = token_info.start_frame;
            }

            // Add token text to current word (strip the ▁ prefix)
            let clean_text = token_text
                .trim_start_matches('▁')
                .trim_start_matches("##");
            current_word.push_str(clean_text);
            current_log_probs.push(token_info.log_prob);
            current_token_indices.push(idx);
            word_end_frame = token_info.end_frame;
        }

        // Don't forget the last word
        if !current_word.is_empty() {
            let confidence = aggregate_confidence(&current_log_probs, aggregation);
            words.push(WordInfo {
                word: current_word,
                start_time_sec: word_start_frame as f32 * FRAME_DURATION_MS / 1000.0,
                end_time_sec: word_end_frame as f32 * FRAME_DURATION_MS / 1000.0,
                confidence,
                token_indices: current_token_indices,
            });
        }

        words
    }
}

/// Timestamp information for a token
#[derive(Clone, Debug, PartialEq)]
pub struct TimestampInfo {
    /// Start time in seconds
    pub start_time_sec: f32,
    /// End time in seconds
    pub end_time_sec: f32,
    /// Confidence (probability, not log probability)
    pub confidence: f32,
}

/// Word-level information with timing and confidence
#[derive(Clone, Debug, PartialEq)]
pub struct WordInfo {
    /// The word text
    pub word: String,
    /// Start time in seconds
    pub start_time_sec: f32,
    /// End time in seconds
    pub end_time_sec: f32,
    /// Aggregated confidence for the word
    pub confidence: f32,
    /// Indices of tokens that make up this word
    pub token_indices: Vec<usize>,
}

/// Confidence aggregation methods for word-level confidence
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConfidenceAggregation {
    /// Product of probabilities: exp(sum(log_probs))
    #[default]
    Product,
    /// Geometric mean: exp(mean(log_probs))
    Mean,
    /// Minimum probability: min(exp(log_probs))
    Min,
}

/// Helper function to aggregate log probabilities into a confidence score
fn aggregate_confidence(log_probs: &[f32], method: ConfidenceAggregation) -> f32 {
    if log_probs.is_empty() {
        return 0.0;
    }
    match method {
        ConfidenceAggregation::Product => {
            let sum: f32 = log_probs.iter().sum();
            sum.exp()
        }
        ConfidenceAggregation::Mean => {
            let mean = log_probs.iter().sum::<f32>() / log_probs.len() as f32;
            mean.exp()
        }
        ConfidenceAggregation::Min => log_probs.iter().map(|lp| lp.exp()).fold(f32::INFINITY, f32::min),
    }
}

// ============================================================================
// Configuration Types
// ============================================================================

/// FastConformer encoder configuration
/// Values from HuggingFace Transformers ParakeetEncoderConfig
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EncoderConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "default_true")]
    pub attention_bias: bool,
    #[serde(default = "default_true")]
    pub convolution_bias: bool,
    #[serde(default = "default_conv_kernel_size")]
    pub conv_kernel_size: usize,
    #[serde(default = "default_subsampling_factor")]
    pub subsampling_factor: usize,
    #[serde(default = "default_subsampling_conv_channels")]
    pub subsampling_conv_channels: usize,
    #[serde(default = "default_num_mel_bins")]
    pub num_mel_bins: usize,
    #[serde(default = "default_subsampling_conv_kernel_size")]
    pub subsampling_conv_kernel_size: usize,
    #[serde(default = "default_subsampling_conv_stride")]
    pub subsampling_conv_stride: usize,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
    #[serde(default = "default_zero")]
    pub dropout_positions: f64,
    #[serde(default = "default_dropout")]
    pub activation_dropout: f64,
    #[serde(default = "default_dropout")]
    pub attention_dropout: f64,
    #[serde(default = "default_dropout")]
    pub layerdrop: f64,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_true")]
    pub scale_input: bool,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
}

// Default value functions for serde
fn default_hidden_size() -> usize {
    1024
}
fn default_num_hidden_layers() -> usize {
    24
}
fn default_num_attention_heads() -> usize {
    8
}
fn default_intermediate_size() -> usize {
    4096
}
fn default_hidden_act() -> Activation {
    Activation::Silu
}
fn default_true() -> bool {
    true
}
fn default_conv_kernel_size() -> usize {
    9
}
fn default_subsampling_factor() -> usize {
    8
}
fn default_subsampling_conv_channels() -> usize {
    256
}
fn default_num_mel_bins() -> usize {
    128 // NeMo uses 128 mel features
}
fn default_subsampling_conv_kernel_size() -> usize {
    3
}
fn default_subsampling_conv_stride() -> usize {
    2
}
fn default_dropout() -> f64 {
    0.1
}
fn default_zero() -> f64 {
    0.0
}
fn default_max_position_embeddings() -> usize {
    5000
}
fn default_layer_norm_eps() -> f64 {
    1e-5
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 8,
            intermediate_size: 4096,
            hidden_act: Activation::Silu,
            attention_bias: true,
            convolution_bias: true,
            conv_kernel_size: 9,
            subsampling_factor: 8,
            subsampling_conv_channels: 256,
            num_mel_bins: 128,
            subsampling_conv_kernel_size: 3,
            subsampling_conv_stride: 2,
            dropout: 0.1,
            dropout_positions: 0.0,
            activation_dropout: 0.1,
            attention_dropout: 0.1,
            layerdrop: 0.1,
            max_position_embeddings: 5000,
            scale_input: false,  // NeMo parakeet-tdt-0.6b-v3 has xscaling=False
            layer_norm_eps: 1e-5,
        }
    }
}

impl EncoderConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// FastConformer XL configuration (600M models: TDT v2, TDT v3)
    /// 24 layers, conv_kernel_size=9
    pub fn xl() -> Self {
        Self {
            num_hidden_layers: 24,
            conv_kernel_size: 9,
            ..Self::default()
        }
    }

    /// FastConformer XXL configuration (1.1B models: RNN-T, CTC)
    /// 42 layers, conv_kernel_size=9, 80 mel bins (different from XL's 128)
    /// NeMo XXL models have xscaling=true (multiply input by sqrt(d_model))
    pub fn xxl() -> Self {
        Self {
            num_hidden_layers: 42,
            conv_kernel_size: 9,  // Same as XL (verified from weights)
            num_mel_bins: 80,     // XXL uses 80 mel bins vs XL's 128
            scale_input: true,    // NeMo XXL has xscaling=true
            ..Self::default()
        }
    }
}

/// TDT decoder configuration
/// Values from NeMo (HuggingFace Transformers only has CTC decoder)
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DecoderConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_predictor_hidden_size")]
    pub predictor_hidden_size: usize,
    #[serde(default = "default_joint_hidden_size")]
    pub joint_hidden_size: usize,
    #[serde(default = "default_max_duration")]
    pub max_duration: usize,
    #[serde(default = "default_blank_id")]
    pub blank_id: u32,
}

fn default_vocab_size() -> usize {
    8192
}
fn default_predictor_hidden_size() -> usize {
    640
}
fn default_joint_hidden_size() -> usize {
    640
}
fn default_max_duration() -> usize {
    4
}
fn default_blank_id() -> u32 {
    8192  // blank is at vocab_size (index after all vocab tokens)
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8192,
            predictor_hidden_size: 640,
            joint_hidden_size: 640,
            max_duration: 4,
            blank_id: 8192,  // blank is at vocab_size
        }
    }
}

impl DecoderConfig {
    /// TDT v2 decoder: 1024 vocab (English only), with duration prediction
    pub fn tdt_v2() -> Self {
        Self {
            vocab_size: 1024,
            predictor_hidden_size: 640,
            joint_hidden_size: 640,
            max_duration: 4,
            blank_id: 1024,  // blank is at vocab_size
        }
    }

    /// TDT v3 decoder: 8192 vocab (multilingual), with duration prediction
    pub fn tdt_v3() -> Self {
        Self::default()
    }

    /// RNN-T decoder: 1024 vocab (English only), no duration prediction
    pub fn rnnt_1b() -> Self {
        Self {
            vocab_size: 1024,
            predictor_hidden_size: 640,
            joint_hidden_size: 640,
            max_duration: 0,  // RNN-T has no duration prediction
            blank_id: 1024,   // blank is at vocab_size
        }
    }

    /// CTC decoder: 1024 vocab (English only), no predictor network
    pub fn ctc_1b() -> Self {
        Self {
            vocab_size: 1024,
            predictor_hidden_size: 0,  // CTC has no predictor
            joint_hidden_size: 0,      // CTC has no joint network
            max_duration: 0,           // CTC has no duration
            blank_id: 1024,            // blank is at vocab_size
        }
    }
}

/// Full Parakeet model configuration
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub encoder: EncoderConfig,
    pub decoder: DecoderConfig,
    #[serde(default)]
    pub decoder_type: DecoderType,
}

impl Default for Config {
    fn default() -> Self {
        Self::parakeet_tdt_0_6b_v3()
    }
}

impl Config {
    /// Configuration for parakeet-tdt-0.6b-v2 model
    /// 600M params, 24 layers (XL), 1024 vocab, English only
    pub fn parakeet_tdt_0_6b_v2() -> Self {
        Self {
            encoder: EncoderConfig::xl(),
            decoder: DecoderConfig::tdt_v2(),
            decoder_type: DecoderType::Tdt,
        }
    }

    /// Configuration for parakeet-tdt-0.6b-v3 model
    /// 600M params, 24 layers (XL), 8192 vocab, 25 languages
    pub fn parakeet_tdt_0_6b_v3() -> Self {
        Self {
            encoder: EncoderConfig::xl(),
            decoder: DecoderConfig::tdt_v3(),
            decoder_type: DecoderType::Tdt,
        }
    }

    /// Configuration for parakeet-rnnt-1.1b model
    /// 1.1B params, 42 layers (XXL), 1024 vocab, English only
    pub fn parakeet_rnnt_1_1b() -> Self {
        Self {
            encoder: EncoderConfig::xxl(),
            decoder: DecoderConfig::rnnt_1b(),
            decoder_type: DecoderType::Rnnt,
        }
    }

    /// Configuration for parakeet-ctc-1.1b model
    /// 1.1B params, 42 layers (XXL), 1024 vocab, English only
    pub fn parakeet_ctc_1_1b() -> Self {
        Self {
            encoder: EncoderConfig::xxl(),
            decoder: DecoderConfig::ctc_1b(),
            decoder_type: DecoderType::Ctc,
        }
    }
}

// Re-export audio processing utilities
pub use audio::{dither, mel_filters_for_sample_rate, mel_filters_for_sample_rate_and_bins, normalize_per_feature, pcm_to_mel, pcm_to_mel_at_sample_rate};

// Re-export model components
pub use ctc::CtcDecoder;
pub use fastconformer::FastConformerEncoder;
pub use lm::NgramLM;
pub use model::{Decoder, DecoderImpl, Parakeet};
pub use rnnt::RnntDecoder;
pub use tdt::TdtDecoder;

// Re-export enums for convenience
pub use DecoderType::{Ctc, Rnnt, Tdt};
pub use ModelVariant::{Ctc1b, Rnnt1b, TdtV2, TdtV3};

// Note: Rich result types (TokenInfo, DecodingResult, TimestampInfo, WordInfo,
// ConfidenceAggregation, FRAME_DURATION_MS) are already public at the module level
