//! Parakeet ASR Example
//!
//! NVIDIA Parakeet models are FastConformer-based ASR models with various decoder types.
//!
//! Supported models:
//! - tdt-v2: 600M params, 24 layers, English only
//! - tdt-v3: 600M params, 24 layers, 25 languages (default)
//! - rnnt-1b: 1.1B params, 42 layers, English only
//! - ctc-1b: 1.1B params, 42 layers, English only
//!
//! ```bash
//! # Download default model (TDT v3) and transcribe audio
//! cargo run --example parakeet --release --features parakeet -- --input audio.wav
//!
//! # Use a specific model variant
//! cargo run --example parakeet --release --features parakeet -- \
//!     --input audio.wav --model-variant rnnt-1b
//!
//! # Use local model
//! cargo run --example parakeet --release --features parakeet -- \
//!     --input audio.wav --model /path/to/model.nemo --model-variant ctc-1b
//!
//! # Run on CPU
//! cargo run --example parakeet --release --features parakeet -- --input audio.wav --cpu
//! ```

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle::{Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;

use candle_transformers::models::parakeet::{self, Decoder, ModelVariant, Parakeet};

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
enum ModelVariantArg {
    /// TDT v2: 600M params, 24 layers, 1024 vocab, English only
    TdtV2,
    /// TDT v3: 600M params, 24 layers, 8192 vocab, 25 languages
    #[default]
    TdtV3,
    /// RNN-T 1.1B: 1.1B params, 42 layers, 1024 vocab, English only
    Rnnt1b,
    /// CTC 1.1B: 1.1B params, 42 layers, 1024 vocab, English only
    Ctc1b,
}

impl From<ModelVariantArg> for ModelVariant {
    fn from(arg: ModelVariantArg) -> Self {
        match arg {
            ModelVariantArg::TdtV2 => ModelVariant::TdtV2,
            ModelVariantArg::TdtV3 => ModelVariant::TdtV3,
            ModelVariantArg::Rnnt1b => ModelVariant::Rnnt1b,
            ModelVariantArg::Ctc1b => ModelVariant::Ctc1b,
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Parakeet ASR - supports TDT, RNN-T, and CTC models")]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Model variant to use.
    #[arg(long, value_enum, default_value = "tdt-v3")]
    model_variant: ModelVariantArg,

    /// Path to .nemo model file. If not provided, downloads from HuggingFace.
    #[arg(long)]
    model: Option<String>,

    /// Input audio file (WAV or other supported format).
    /// Can specify multiple times for batch processing: --input file1.wav --input file2.wav
    #[arg(long, required = true)]
    input: Vec<String>,

    /// Use beam search decoding with specified beam width.
    #[arg(long)]
    beam_width: Option<usize>,

    /// Enable tracing for performance profiling.
    #[arg(long)]
    tracing: bool,

    /// Debug: list weight names and shapes from the model
    #[arg(long)]
    debug_weights: bool,

    /// Path to pre-computed mel spectrogram (binary format from save_nemo_mel.py).
    /// If provided, uses this instead of computing mel from audio.
    #[arg(long)]
    mel: Option<String>,

    /// Directory containing NeMo intermediate tensors for substitution testing.
    /// Created by save_nemo_intermediates.py
    #[arg(long)]
    compare_nemo: Option<String>,

    /// Substitution testing: inject NeMo encoder output instead of running Candle encoder.
    /// Use with --compare-nemo to load encoder_output.bin
    #[arg(long)]
    substitute_encoder: bool,

    /// Debug: scale encoder output by this factor (e.g., 100 to test if decoder works with larger signal)
    #[arg(long)]
    scale_encoder: Option<f64>,

    /// Compare each encoder block output against NeMo intermediates.
    /// Use with --compare-nemo to load block_XX.bin files
    #[arg(long)]
    compare_blocks: bool,

    /// Substitute NeMo subsampling output instead of running Candle subsampling.
    /// Use with --compare-nemo to load subsampling.bin
    #[arg(long)]
    substitute_subsampling: bool,

    /// Substitute a specific NeMo block output. Runs Candle up to block N-1,
    /// then injects NeMo block N output and continues.
    #[arg(long)]
    substitute_block: Option<usize>,

    /// Save Candle encoder output to directory for bi-directional testing.
    /// Creates candle_encoder_output.bin in the specified directory.
    #[arg(long)]
    save_encoder: Option<String>,

    /// Output result as JSON (includes transcription, timing, and file info).
    #[arg(long)]
    json: bool,

    /// Disable result caching. By default, results are cached in ~/.cache/candle/parakeet/results/
    /// and reused when the same audio file and model variant are requested.
    #[arg(long)]
    no_cache: bool,

    /// Enable timestamp output for each token.
    #[arg(long)]
    timestamps: bool,

    /// Enable word-level timestamp output (groups tokens into words).
    #[arg(long)]
    word_timestamps: bool,

    /// Show confidence scores for tokens/words.
    #[arg(long)]
    confidence: bool,

    /// Confidence aggregation method for words: product, mean, or min.
    #[arg(long, value_enum, default_value = "product")]
    confidence_method: ConfidenceMethodArg,

    /// Enable chunked/streaming processing for long audio files.
    /// Splits audio into overlapping chunks and processes them independently.
    #[arg(long)]
    streaming: bool,

    /// Chunk duration in seconds for streaming mode.
    #[arg(long, default_value = "30.0")]
    chunk_duration: f32,

    /// Overlap duration in seconds for streaming mode.
    #[arg(long, default_value = "5.0")]
    chunk_overlap: f32,

    /// Path to N-gram language model in ARPA format for LM fusion.
    /// Applied during beam search decoding.
    #[arg(long)]
    lm: Option<String>,

    /// LM fusion weight (alpha). Score = acoustic + alpha * lm_score.
    #[arg(long, default_value = "0.5")]
    lm_weight: f32,

    /// Use MALSD (Modified Alignment-Length Synchronous Decoding) for beam search.
    /// Synchronizes beams by alignment length instead of time, which can improve
    /// accuracy for some audio inputs.
    #[arg(long)]
    malsd: bool,
}

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
enum ConfidenceMethodArg {
    #[default]
    Product,
    Mean,
    Min,
}

impl From<ConfidenceMethodArg> for parakeet::ConfidenceAggregation {
    fn from(arg: ConfidenceMethodArg) -> Self {
        match arg {
            ConfidenceMethodArg::Product => parakeet::ConfidenceAggregation::Product,
            ConfidenceMethodArg::Mean => parakeet::ConfidenceAggregation::Mean,
            ConfidenceMethodArg::Min => parakeet::ConfidenceAggregation::Min,
        }
    }
}

/// Load a tensor from binary format (created by save_nemo_intermediates.py)
///
/// Format:
/// - 4 bytes: ndim (u32 little-endian)
/// - ndim * 4 bytes: shape (u32 each)
/// - prod(shape) * 4 bytes: data (f32 little-endian, row-major)
fn load_tensor_from_binary(path: &std::path::Path, device: &Device) -> Result<Tensor> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;

    // Read ndim
    let mut header = [0u8; 4];
    file.read_exact(&mut header)?;
    let ndim = u32::from_le_bytes(header) as usize;

    // Read shape
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        let mut dim_bytes = [0u8; 4];
        file.read_exact(&mut dim_bytes)?;
        shape.push(u32::from_le_bytes(dim_bytes) as usize);
    }

    // Calculate total elements
    let total_elements: usize = shape.iter().product();

    // Read data
    let mut data = vec![0u8; total_elements * 4];
    file.read_exact(&mut data)?;

    let values: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(Tensor::from_vec(values, shape, device)?)
}

/// Save a tensor to binary format (same format as load_tensor_from_binary)
fn save_tensor_to_binary(tensor: &Tensor, path: &std::path::Path) -> Result<()> {
    use std::io::Write;

    let tensor = tensor.to_dtype(candle::DType::F32)?;
    let shape = tensor.dims();
    let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;

    let mut file = std::fs::File::create(path)?;

    // Write ndim
    file.write_all(&(shape.len() as u32).to_le_bytes())?;

    // Write shape
    for &dim in shape {
        file.write_all(&(dim as u32).to_le_bytes())?;
    }

    // Write data
    for val in data {
        file.write_all(&val.to_le_bytes())?;
    }

    Ok(())
}

/// Compare two tensors and print statistics
fn compare_tensors(name: &str, candle: &Tensor, nemo: &Tensor) -> Result<()> {
    // Check shapes match
    if candle.dims() != nemo.dims() {
        println!("  {} SHAPE MISMATCH: candle={:?}, nemo={:?}",
                 name, candle.dims(), nemo.dims());
        return Ok(());
    }

    let diff = (candle - nemo)?;
    let abs_diff = diff.abs()?;

    let max_diff: f32 = abs_diff.flatten_all()?.max(0)?.to_scalar()?;
    let mean_diff: f32 = abs_diff.flatten_all()?.mean_all()?.to_scalar()?;
    let rms_diff: f32 = diff.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;

    let candle_rms: f32 = candle.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
    let nemo_rms: f32 = nemo.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;

    let relative_error = if nemo_rms > 1e-6 { rms_diff / nemo_rms } else { rms_diff };

    let status = if relative_error < 0.01 {
        "OK"
    } else if relative_error < 0.1 {
        "WARN"
    } else {
        "FAIL"
    };

    println!("  {} [{}]: max_diff={:.6}, mean_diff={:.6}, rms_diff={:.6}, rel_err={:.4}",
             name, status, max_diff, mean_diff, rms_diff, relative_error);
    println!("    candle_rms={:.4}, nemo_rms={:.4}", candle_rms, nemo_rms);

    Ok(())
}

/// Load mel spectrogram from binary format (created by save_nemo_mel.py)
///
/// Format:
/// - 4 bytes: n_mels (u32 little-endian)
/// - 4 bytes: n_frames (u32 little-endian)
/// - n_mels * n_frames * 4 bytes: mel values (f32 little-endian, row-major)
fn load_mel_from_binary(path: &std::path::Path, device: &Device) -> Result<Tensor> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;

    let n_mels = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let n_frames = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;

    println!("Loading mel from {:?}: shape=({}, {})", path, n_mels, n_frames);

    let mut data = vec![0u8; n_mels * n_frames * 4];
    file.read_exact(&mut data)?;

    // Convert bytes to f32
    let mel: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    // Create tensor with shape [1, n_mels, n_frames]
    let mel = Tensor::from_vec(mel, (1, n_mels, n_frames), device)?;

    // Debug: print mel stats
    let mel_min: f32 = mel.flatten_all()?.min(0)?.to_scalar()?;
    let mel_max: f32 = mel.flatten_all()?.max(0)?.to_scalar()?;
    let mel_mean: f32 = mel.flatten_all()?.mean_all()?.to_scalar()?;
    let mel_rms: f32 = mel.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
    println!("Mel stats: min={:.3}, max={:.3}, mean={:.3}, rms={:.3}", mel_min, mel_max, mel_mean, mel_rms);

    Ok(mel)
}

/// Get the cache directory for extracted nemo archives.
/// Uses ~/.cache/candle/parakeet/ on Unix, or a temp-based fallback.
fn get_nemo_cache_dir() -> Result<std::path::PathBuf> {
    // Try XDG_CACHE_HOME first, then HOME/.cache, then temp directory fallback
    let cache_dir = if let Some(xdg_cache) = std::env::var_os("XDG_CACHE_HOME") {
        std::path::PathBuf::from(xdg_cache).join("candle").join("parakeet")
    } else if let Some(home) = std::env::var_os("HOME") {
        std::path::PathBuf::from(home).join(".cache").join("candle").join("parakeet")
    } else if let Some(userprofile) = std::env::var_os("USERPROFILE") {
        // Windows fallback
        std::path::PathBuf::from(userprofile).join(".cache").join("candle").join("parakeet")
    } else {
        // Final fallback to system temp with a stable subdirectory
        std::env::temp_dir().join("candle-parakeet-cache")
    };
    std::fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir)
}

/// Extract a .nemo archive to a persistent cache directory.
/// Returns the path to the extracted contents (reuses cache if already extracted).
fn extract_nemo_archive(nemo_path: &std::path::Path) -> Result<std::path::PathBuf> {
    use flate2::read::GzDecoder;
    use tar::Archive;

    // Determine cache location based on the nemo filename
    let nemo_name = nemo_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let cache_dir = get_nemo_cache_dir()?;
    let extract_dir = cache_dir.join(nemo_name);

    // Check if already extracted (weights file exists)
    let weights_marker = extract_dir.join("model_weights.ckpt");
    if weights_marker.exists() {
        println!("Using cached extraction at {:?}", extract_dir);
        return Ok(extract_dir);
    }

    // Clean up partial extraction if present
    if extract_dir.exists() {
        std::fs::remove_dir_all(&extract_dir)?;
    }
    std::fs::create_dir_all(&extract_dir)?;

    println!("Extracting to cache: {:?}", extract_dir);

    let file = std::fs::File::open(nemo_path)?;

    // Try gzip first, fall back to plain tar
    let mut buf = [0u8; 2];
    {
        use std::io::Read;
        let mut peek_file = std::fs::File::open(nemo_path)?;
        peek_file.read_exact(&mut buf)?;
    }

    if buf == [0x1f, 0x8b] {
        // Gzip magic bytes
        let decoder = GzDecoder::new(file);
        let mut archive = Archive::new(decoder);
        archive.unpack(&extract_dir)?;
    } else {
        // Plain tar
        let mut archive = Archive::new(file);
        archive.unpack(&extract_dir)?;
    }

    Ok(extract_dir)
}

fn debug_weights(nemo_path: &std::path::Path) -> Result<()> {
    use candle::pickle::PthTensors;

    println!("Extracting .nemo archive...");
    let extract_dir = extract_nemo_archive(nemo_path)?;
    let weights_path = extract_dir.join("model_weights.ckpt");

    println!("Loading weights from {:?}...", weights_path);
    let pth = PthTensors::new(&weights_path, None)?;

    // Check norm_out weights for blocks 20-23 and 40-41 (XXL)
    println!("\n=== norm_out weight stats ===");
    for layer_idx in [0, 10, 20, 21, 22, 23, 40, 41] {
        let weight_name = format!("encoder.layers.{}.norm_out.weight", layer_idx);
        let bias_name = format!("encoder.layers.{}.norm_out.bias", layer_idx);

        if let Ok(Some(w)) = pth.get(&weight_name) {
            let w = w.to_dtype(candle::DType::F32)?;
            let min: f32 = w.min(0)?.to_scalar()?;
            let max: f32 = w.max(0)?.to_scalar()?;
            let mean: f32 = w.mean_all()?.to_scalar()?;
            let rms: f32 = w.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            println!("Layer {:2} norm_out.weight: min={:.4}, max={:.4}, mean={:.4}, rms={:.4}",
                     layer_idx, min, max, mean, rms);
        }
        if let Ok(Some(b)) = pth.get(&bias_name) {
            let b = b.to_dtype(candle::DType::F32)?;
            let min: f32 = b.min(0)?.to_scalar()?;
            let max: f32 = b.max(0)?.to_scalar()?;
            let mean: f32 = b.mean_all()?.to_scalar()?;
            println!("Layer {:2} norm_out.bias:   min={:.4}, max={:.4}, mean={:.4}",
                     layer_idx, min, max, mean);
        }
    }

    // Check BatchNorm running stats for each block
    println!("\n=== BatchNorm running stats ===");
    for layer_idx in [0, 10, 20, 21, 22, 23] {
        let rm_name = format!("encoder.layers.{}.conv.batch_norm.running_mean", layer_idx);
        let rv_name = format!("encoder.layers.{}.conv.batch_norm.running_var", layer_idx);
        let w_name = format!("encoder.layers.{}.conv.batch_norm.weight", layer_idx);
        let b_name = format!("encoder.layers.{}.conv.batch_norm.bias", layer_idx);

        print!("Layer {:2} batch_norm: ", layer_idx);

        if let Ok(Some(rm)) = pth.get(&rm_name) {
            let rm = rm.to_dtype(candle::DType::F32)?;
            let rms: f32 = rm.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            let min: f32 = rm.min(0)?.to_scalar()?;
            let max: f32 = rm.max(0)?.to_scalar()?;
            print!("running_mean(rms={:.4}, [{:.4},{:.4}]) ", rms, min, max);
        } else {
            print!("running_mean=MISSING ");
        }

        if let Ok(Some(rv)) = pth.get(&rv_name) {
            let rv = rv.to_dtype(candle::DType::F32)?;
            let rms: f32 = rv.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            let min: f32 = rv.min(0)?.to_scalar()?;
            let max: f32 = rv.max(0)?.to_scalar()?;
            print!("running_var(rms={:.4}, [{:.4},{:.4}])", rms, min, max);
        } else {
            print!("running_var=MISSING");
        }
        println!();
    }

    // Check pointwise_conv2 weights for each block (debugging Block 23 explosion)
    println!("\n=== Pointwise Conv2 weights ===");
    for layer_idx in [0, 10, 20, 21, 22, 23] {
        let w_name = format!("encoder.layers.{}.conv.pointwise_conv2.weight", layer_idx);
        if let Ok(Some(w)) = pth.get(&w_name) {
            let w = w.to_dtype(candle::DType::F32)?;
            let rms: f32 = w.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            let min: f32 = w.flatten_all()?.min(0)?.to_scalar()?;
            let max: f32 = w.flatten_all()?.max(0)?.to_scalar()?;
            println!("Layer {:2} pointwise_conv2: rms={:.4}, range=[{:.4},{:.4}]",
                     layer_idx, rms, min, max);
        }
    }

    // Check joint network weights
    println!("\n=== Joint network weights ===");
    for name in ["joint.enc.weight", "joint.enc.bias", "joint.pred.weight", "joint.pred.bias"] {
        if let Ok(Some(w)) = pth.get(name) {
            let w = w.to_dtype(candle::DType::F32)?;
            let min: f32 = w.flatten_all()?.min(0)?.to_scalar()?;
            let max: f32 = w.flatten_all()?.max(0)?.to_scalar()?;
            let rms: f32 = w.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            println!("{}: shape={:?}, min={:.4}, max={:.4}, rms={:.4}", name, w.dims(), min, max, rms);
        }
    }

    // Check pre_encode.out weights (subsampling output projection)
    println!("\n=== Pre-encode output projection ===");
    if let Ok(Some(w)) = pth.get("encoder.pre_encode.out.weight") {
        let w = w.to_dtype(candle::DType::F32)?;
        let min: f32 = w.flatten_all()?.min(0)?.to_scalar()?;
        let max: f32 = w.flatten_all()?.max(0)?.to_scalar()?;
        let rms: f32 = w.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
        println!("out.weight: shape={:?}, min={:.4}, max={:.4}, rms={:.4}", w.dims(), min, max, rms);
    }
    if let Ok(Some(b)) = pth.get("encoder.pre_encode.out.bias") {
        let b = b.to_dtype(candle::DType::F32)?;
        let min: f32 = b.min(0)?.to_scalar()?;
        let max: f32 = b.max(0)?.to_scalar()?;
        let rms: f32 = b.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
        println!("out.bias: shape={:?}, min={:.4}, max={:.4}, rms={:.4}", b.dims(), min, max, rms);
    }

    println!("\nWeight tensors:");
    let mut names: Vec<_> = pth.tensor_infos().keys().collect();
    names.sort();
    for name in names.iter() {
        let info = &pth.tensor_infos()[*name];
        // Load the tensor to get its shape
        if let Ok(tensor) = pth.get(*name) {
            if let Some(t) = tensor {
                println!("  {}: {:?}", name, t.dims());
            } else {
                println!("  {}: (not found)", name);
            }
        } else {
            println!("  {}: dtype={:?}", name, info.dtype);
        }
    }
    if names.len() > 100 {
        println!("  ... and {} more", names.len() - 100);
    }
    println!("\nTotal: {} tensors", names.len());

    Ok(())
}

/// Simple vocab-based decoder for SentencePiece tokens
struct SimpleDecoder {
    vocab: Vec<String>,
    /// Number of special tokens before BPE tokens in the model vocabulary.
    /// For v3: 8192 - 7918 = 274 special tokens
    /// For v2/rnnt/ctc: 1024 - vocab_len special tokens
    special_token_offset: usize,
}

impl SimpleDecoder {
    fn load(vocab_path: &std::path::Path, total_vocab_size: usize) -> Result<Self> {
        use std::io::BufRead;
        let file = std::fs::File::open(vocab_path)?;
        let reader = std::io::BufReader::new(file);
        let vocab: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();

        // Calculate special token offset based on total vocab size from config
        let special_token_offset = total_vocab_size.saturating_sub(vocab.len());
        println!("Vocab has {} tokens, total model vocab: {}, special token offset: {}",
                 vocab.len(), total_vocab_size, special_token_offset);

        Ok(Self { vocab, special_token_offset })
    }

    fn decode(&self, tokens: &[u32]) -> String {
        let mut result = String::new();
        for &id in tokens {
            // Adjust for special token offset
            let adjusted_id = match (id as usize).checked_sub(self.special_token_offset) {
                Some(idx) => idx,
                None => continue,
            };
            let token_str = match self.vocab.get(adjusted_id) {
                Some(s) => s,
                None => continue,
            };

            // Handle ## continuation markers (BERT-style word pieces)
            // Tokens starting with ## are continuations (append without space)
            // Other tokens are word starts (add space before, except at start)
            if let Some(suffix) = token_str.strip_prefix("##") {
                result.push_str(suffix);
            } else {
                // Also handle SentencePiece ▁ (U+2581) for word boundaries
                let token_str = token_str.replace('▁', " ");
                if !result.is_empty() && !token_str.starts_with(' ') {
                    result.push(' ');
                }
                result.push_str(&token_str);
            }
        }
        result.trim().to_string()
    }

    /// Get the vocabulary for word-level processing.
    /// Returns vocab adjusted with special token offset prepended as empty strings.
    fn get_adjusted_vocab(&self) -> Vec<String> {
        // Prepend empty strings for special tokens so indices align
        let mut adjusted = vec![String::new(); self.special_token_offset];
        adjusted.extend(self.vocab.iter().cloned());
        adjusted
    }

    /// Decode a single token to its string representation
    fn decode_token(&self, id: u32) -> Option<&str> {
        let adjusted_id = (id as usize).checked_sub(self.special_token_offset)?;
        self.vocab.get(adjusted_id).map(|s| s.as_str())
    }
}

/// Load audio file and compute mel spectrogram
fn load_audio_as_mel(
    path: &str,
    num_mel_bins: usize,
    device: &Device,
) -> Result<(Tensor, f64)> {
    let (pcm_data, sample_rate) = candle_examples::audio::pcm_decode(path)?;
    let native_sample_rate = sample_rate as usize;

    if native_sample_rate < parakeet::MIN_SAMPLE_RATE {
        anyhow::bail!(
            "Sample rate {}Hz too low for {}. Minimum: {}Hz",
            native_sample_rate,
            path,
            parakeet::MIN_SAMPLE_RATE
        );
    }

    let audio_duration = pcm_data.len() as f64 / native_sample_rate as f64;

    // Resample to 16kHz if needed
    let pcm_data = if native_sample_rate != parakeet::SAMPLE_RATE {
        candle_examples::audio::resample(&pcm_data, sample_rate, parakeet::SAMPLE_RATE as u32)?
    } else {
        pcm_data
    };

    // Compute mel spectrogram
    let mel_filters = parakeet::mel_filters_for_sample_rate_and_bins(parakeet::SAMPLE_RATE, num_mel_bins);
    let mut mel = parakeet::pcm_to_mel(&pcm_data, &mel_filters);
    let num_frames = mel.len() / num_mel_bins;

    // Apply per-feature normalization
    parakeet::normalize_per_feature(&mut mel, num_mel_bins, num_frames);

    let mel = Tensor::from_vec(mel, (1, num_mel_bins, num_frames), device)?;
    Ok((mel, audio_duration))
}

fn load_nemo_model(
    nemo_path: &std::path::Path,
    variant: ModelVariant,
    device: &Device,
) -> Result<(Parakeet, SimpleDecoder)> {
    println!("Extracting .nemo archive...");
    let extract_dir = extract_nemo_archive(nemo_path)?;

    // List contents for debugging
    let contents: Vec<_> = std::fs::read_dir(&extract_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    println!("Archive contents: {:?}", contents);

    // Load PyTorch weights
    let weights_path = extract_dir.join("model_weights.ckpt");
    if !weights_path.exists() {
        anyhow::bail!(
            "model_weights.ckpt not found in .nemo archive. Contents: {:?}",
            contents
        );
    }

    println!("Loading weights from {:?}...", weights_path);
    let tensors = candle::pickle::read_all(&weights_path)?;

    // Debug: print some weight names to understand NeMo structure
    println!("Sample weight names (first 20):");
    for (i, (name, _)) in tensors.iter().take(20).enumerate() {
        println!("  {}: {}", i, name);
    }
    // Check for batch_norm weight names
    let bn_keys: Vec<_> = tensors.iter()
        .filter(|(k, _)| k.contains("batch_norm") || k.contains("bn"))
        .map(|(k, _)| k.as_str())
        .take(10)
        .collect();
    println!("BatchNorm related keys: {:?}", bn_keys);

    // Check for pre_encode weight names
    let pre_encode_keys: Vec<_> = tensors.iter()
        .filter(|(k, _)| k.contains("pre_encode"))
        .map(|(k, t)| format!("{}: {:?}", k, t.dims()))
        .collect();
    println!("Pre-encode keys ({}):", pre_encode_keys.len());
    for key in &pre_encode_keys {
        println!("  {}", key);
    }

    let tensors: std::collections::HashMap<String, Tensor> = tensors.into_iter().collect();
    let vb = VarBuilder::from_tensors(tensors, candle::DType::F32, device);

    // Load config based on variant
    let config = variant.config();
    println!("Using config for {:?}: encoder layers={}, vocab_size={}, decoder={:?}",
             variant,
             config.encoder.num_hidden_layers,
             config.decoder.vocab_size,
             config.decoder_type);

    // Load model
    let model = Parakeet::load(vb, config.clone())?;

    // Find vocab file for simple decoding
    let vocab_path = contents
        .iter()
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with("vocab.txt"))
        })
        .ok_or_else(|| anyhow::anyhow!("vocab.txt not found in .nemo archive"))?;

    println!("Loading vocabulary from {:?}...", vocab_path);
    let decoder = SimpleDecoder::load(vocab_path, config.decoder.vocab_size)?;

    Ok((model, decoder))
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;
    use std::io::Write;

    let args = Args::parse();

    // Open debug log file
    let debug_log_path = std::path::Path::new("candle-examples/examples/parakeet/debug_output.log");
    let mut debug_log = std::fs::File::create(debug_log_path)?;
    writeln!(debug_log, "=== Parakeet Debug Log ===")?;
    writeln!(debug_log, "Audio files: {:?}", args.input)?;
    writeln!(debug_log, "")?;

    // Check for batch mode
    let batch_mode = args.input.len() > 1;
    if batch_mode {
        println!("Batch mode: processing {} audio files", args.input.len());
    }

    // Set up tracing if requested
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let device = candle_examples::device(args.cpu)?;
    println!("Using device: {:?}", device);

    // Get model variant
    let variant: ModelVariant = args.model_variant.into();
    println!("Model variant: {:?}", variant);

    // Load or download model
    let model_path = match args.model {
        Some(p) => PathBuf::from(p),
        None => {
            println!("Downloading model from HuggingFace: {} ...", variant.repo_name());
            let api = Api::new()?;
            let repo = api.repo(Repo::new(
                variant.repo_name().to_string(),
                RepoType::Model,
            ));
            repo.get(variant.nemo_filename())?
        }
    };

    // Debug mode: just list weights and exit
    if args.debug_weights {
        return debug_weights(&model_path);
    }

    println!("Loading model from {:?}...", model_path);
    let (model, decoder) = load_nemo_model(&model_path, variant, &device)?;

    // Get the number of mel bins from the model config (128 for XL, 80 for XXL)
    let num_mel_bins = model.config.encoder.num_mel_bins;
    println!("Using {} mel bins for audio processing", num_mel_bins);

    // Load language model if specified
    let lm = if let Some(lm_path) = &args.lm {
        println!("Loading N-gram LM from {:?}...", lm_path);
        let lm = parakeet::NgramLM::load_arpa(std::path::Path::new(lm_path))?;
        println!(
            "LM loaded: order={}, vocab_size={}, weight={}",
            lm.order(),
            lm.vocab_size(),
            args.lm_weight
        );
        if args.beam_width.is_none() {
            eprintln!("Warning: --lm requires --beam-width for LM fusion to take effect");
        }
        if model.decoder_type() != parakeet::DecoderType::Tdt {
            eprintln!("Warning: LM fusion is currently only supported for TDT decoder");
        }
        Some(lm)
    } else {
        None
    };

    // Get vocabulary for LM integration
    let vocab_for_lm = decoder.get_adjusted_vocab();

    // Handle batch mode separately - process multiple files efficiently
    if batch_mode {
        println!("\n=== Batch Processing {} Files ===", args.input.len());
        let batch_start = std::time::Instant::now();

        // Load all audio files and compute mel spectrograms
        let mut mels = Vec::new();
        let mut durations = Vec::new();
        let mut file_names = Vec::new();

        for path in &args.input {
            println!("Loading {}...", path);
            match load_audio_as_mel(path, num_mel_bins, &device) {
                Ok((mel, duration)) => {
                    mels.push(mel);
                    durations.push(duration);
                    file_names.push(path.clone());
                }
                Err(e) => {
                    eprintln!("Error loading {}: {}", path, e);
                }
            }
        }

        if mels.is_empty() {
            anyhow::bail!("No audio files could be loaded");
        }

        // Convert to refs for batch processing
        let mel_refs: Vec<&Tensor> = mels.iter().collect();

        // Run batch inference
        println!("\nRunning batch inference...");
        let infer_start = std::time::Instant::now();
        let results = match args.beam_width {
            Some(beam_width) => {
                println!("Using beam search with width {}", beam_width);
                model.forward_batch_beam(&mel_refs, beam_width)?
            }
            None => model.forward_batch(&mel_refs)?,
        };
        let infer_elapsed = infer_start.elapsed();

        // Print results for each file
        println!("\n=== Batch Results ===");
        let total_duration: f64 = durations.iter().sum();
        for (i, result) in results.iter().enumerate() {
            let text = decoder.decode(&result.token_ids());
            println!("\n[{}] {}", i + 1, file_names[i]);
            println!("  Text: {}", text);
            println!("  Duration: {:.2}s, Tokens: {}", durations[i], result.tokens.len());

            if args.word_timestamps || args.confidence {
                let vocab = decoder.get_adjusted_vocab();
                let aggregation: parakeet::ConfidenceAggregation = args.confidence_method.into();
                let words = result.word_info(&vocab, aggregation);
                if args.word_timestamps {
                    for word_info in &words {
                        if args.confidence {
                            println!(
                                "    [{:5.2}s - {:5.2}s] {} ({:.1}%)",
                                word_info.start_time_sec,
                                word_info.end_time_sec,
                                word_info.word,
                                word_info.confidence * 100.0
                            );
                        } else {
                            println!(
                                "    [{:5.2}s - {:5.2}s] {}",
                                word_info.start_time_sec, word_info.end_time_sec, word_info.word
                            );
                        }
                    }
                }
            }
        }

        let batch_elapsed = batch_start.elapsed();
        println!("\n=== Batch Summary ===");
        println!("Files processed: {}", results.len());
        println!("Total audio duration: {:.2}s", total_duration);
        println!("Inference time: {:?}", infer_elapsed);
        println!("Total time (including I/O): {:?}", batch_elapsed);
        println!("Real-time factor: {:.2}x", infer_elapsed.as_secs_f64() / total_duration);

        return Ok(());
    }

    // Get mel spectrogram - either from file or compute from audio (single file mode)
    let (mel, audio_duration) = if let Some(mel_path) = &args.mel {
        // Load pre-computed mel spectrogram from binary file
        println!("Loading pre-computed mel spectrogram from {:?}...", mel_path);
        let mel = load_mel_from_binary(std::path::Path::new(mel_path), &device)?;
        // Estimate audio duration from mel frames
        let num_frames = mel.dim(2)?;
        let duration = num_frames as f64 * parakeet::HOP_LENGTH as f64 / parakeet::SAMPLE_RATE as f64;
        println!("Mel spectrogram: {:?} (estimated {:.2}s)", mel.dims(), duration);
        (mel, duration)
    } else {
        // Load and preprocess audio (first file for single mode, or use batch processing for multiple)
        let input_file = &args.input[0];
        println!("Loading audio from {:?}...", input_file);
        let (pcm_data, sample_rate) = candle_examples::audio::pcm_decode(input_file)?;
        let native_sample_rate = sample_rate as usize;

        // Validate sample rate (NeMo processes at native rate without resampling)
        if native_sample_rate < parakeet::MIN_SAMPLE_RATE {
            anyhow::bail!(
                "Sample rate {}Hz too low. Minimum supported: {}Hz",
                native_sample_rate,
                parakeet::MIN_SAMPLE_RATE
            );
        }
        if native_sample_rate > parakeet::MAX_SAMPLE_RATE {
            eprintln!(
                "Warning: Sample rate {}Hz exceeds tested range ({}Hz). Results may vary.",
                native_sample_rate,
                parakeet::MAX_SAMPLE_RATE
            );
        }

        // Validate input
        if pcm_data.is_empty() {
            anyhow::bail!("Empty audio input");
        }
        if pcm_data.len() < parakeet::WIN_LENGTH {
            eprintln!(
                "Warning: Audio ({} samples) shorter than window length ({}). Results may be unreliable.",
                pcm_data.len(),
                parakeet::WIN_LENGTH
            );
        }

        let audio_duration = pcm_data.len() as f64 / native_sample_rate as f64;
        println!(
            "Audio loaded: {} samples at {}Hz ({:.2}s)",
            pcm_data.len(),
            native_sample_rate,
            audio_duration,
        );

        // Resample to 16kHz if needed (NeMo always resamples to 16kHz before mel computation)
        let pcm_data = if native_sample_rate != parakeet::SAMPLE_RATE {
            let expected_len = (pcm_data.len() as f64 * parakeet::SAMPLE_RATE as f64 / native_sample_rate as f64).round() as usize;
            let resampled = candle_examples::audio::resample(&pcm_data, sample_rate, parakeet::SAMPLE_RATE as u32)?;
            println!(
                "Resampled {}Hz→{}Hz: {} → {} samples (expected {}, diff={})",
                native_sample_rate, parakeet::SAMPLE_RATE,
                pcm_data.len(), resampled.len(), expected_len,
                resampled.len() as i64 - expected_len as i64
            );
            resampled
        } else {
            pcm_data
        };

        // Note: NeMo only applies dithering during training, not inference
        // So we skip dithering here for inference-mode compatibility
        let pcm_data = pcm_data;

        // Compute mel spectrogram at 16kHz (NeMo's target sample rate)
        let mel_filters = parakeet::mel_filters_for_sample_rate_and_bins(parakeet::SAMPLE_RATE, num_mel_bins);
        let mut mel = parakeet::pcm_to_mel(&pcm_data, &mel_filters);
        let num_frames = mel.len() / num_mel_bins;

        // Apply per-feature normalization (NeMo normalize: per_feature)
        parakeet::normalize_per_feature(&mut mel, num_mel_bins, num_frames);

        let mel = Tensor::from_vec(mel, (1, num_mel_bins, num_frames), &device)?;
        println!("Mel spectrogram: {:?}", mel.dims());

        // Debug: print mel stats
        let mel_min: f32 = mel.flatten_all()?.min(0)?.to_scalar()?;
        let mel_max: f32 = mel.flatten_all()?.max(0)?.to_scalar()?;
        let mel_mean: f32 = mel.flatten_all()?.mean_all()?.to_scalar()?;
        println!("Mel stats: min={:.3}, max={:.3}, mean={:.3}", mel_min, mel_max, mel_mean);

        (mel, audio_duration)
    };

    // Load NeMo intermediates if needed for substitution or comparison
    let nemo_subsampling = if args.substitute_subsampling {
        if let Some(ref nemo_dir) = args.compare_nemo {
            let path = std::path::Path::new(nemo_dir).join("subsampling.bin");
            if path.exists() {
                let tensor = load_tensor_from_binary(&path, &device)?;
                // Add batch dimension if needed
                let tensor = if tensor.dims().len() == 2 {
                    tensor.unsqueeze(0)?
                } else {
                    tensor
                };
                Some(tensor)
            } else {
                anyhow::bail!("--substitute-subsampling requires subsampling.bin in --compare-nemo directory");
            }
        } else {
            anyhow::bail!("--substitute-subsampling requires --compare-nemo directory");
        }
    } else {
        None
    };

    let nemo_block = if let Some(block_idx) = args.substitute_block {
        if let Some(ref nemo_dir) = args.compare_nemo {
            let path = std::path::Path::new(nemo_dir).join(format!("block_{:02}.bin", block_idx));
            if path.exists() {
                let tensor = load_tensor_from_binary(&path, &device)?;
                // Add batch dimension if needed
                let tensor = if tensor.dims().len() == 2 {
                    tensor.unsqueeze(0)?
                } else {
                    tensor
                };
                Some((block_idx, tensor))
            } else {
                anyhow::bail!("--substitute-block {} requires block_{:02}.bin in --compare-nemo directory", block_idx, block_idx);
            }
        } else {
            anyhow::bail!("--substitute-block requires --compare-nemo directory");
        }
    } else {
        None
    };

    // Run encoder with optional block outputs for comparison/substitution
    let (candle_enc_output, block_outputs) = if args.compare_blocks || args.substitute_subsampling || args.substitute_block.is_some() {
        println!("\n=== Running encoder with block tracking ===");
        let (subsampling, blocks, final_output) = model.encoder.forward_with_block_outputs(
            &mel,
            None,
            nemo_subsampling.as_ref(),
            nemo_block.as_ref().map(|(idx, t)| (*idx, t)),
        )?;

        // Print subsampling output stats
        let sub_rms: f32 = subsampling.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
        println!("Candle subsampling RMS: {:.4}", sub_rms);

        (final_output, Some(blocks))
    } else {
        // Normal forward pass
        let output = model.encoder.forward(&mel, None)?;
        (output, None)
    };

    let enc_min: f32 = candle_enc_output.flatten_all()?.min(0)?.to_scalar()?;
    let enc_max: f32 = candle_enc_output.flatten_all()?.max(0)?.to_scalar()?;
    let enc_mean: f32 = candle_enc_output.flatten_all()?.mean_all()?.to_scalar()?;
    println!("Candle encoder output: {:?}", candle_enc_output.dims());
    println!("Candle encoder stats: min={:.3}, max={:.3}, mean={:.3}", enc_min, enc_max, enc_mean);

    // Save encoder output if requested (for bi-directional testing)
    if let Some(ref save_dir) = args.save_encoder {
        let save_path = std::path::Path::new(save_dir);
        std::fs::create_dir_all(save_path)?;

        // Save encoder output - squeeze batch dim and transpose to [features, time] for NeMo compatibility
        let enc_to_save = candle_enc_output.squeeze(0)?; // [time, features]
        let enc_to_save = enc_to_save.t()?; // [features, time] - NeMo format
        let enc_file = save_path.join("candle_encoder_output.bin");
        save_tensor_to_binary(&enc_to_save, &enc_file)?;
        println!("Saved Candle encoder output to {:?} (shape: {:?})", enc_file, enc_to_save.dims());

        // Also save mel spectrogram
        let mel_to_save = mel.squeeze(0)?; // [features, time]
        let mel_file = save_path.join("candle_mel.bin");
        save_tensor_to_binary(&mel_to_save, &mel_file)?;
        println!("Saved Candle mel to {:?} (shape: {:?})", mel_file, mel_to_save.dims());
    }

    // Per-block comparison if requested
    if args.compare_blocks {
        if let Some(ref nemo_dir) = args.compare_nemo {
            if let Some(ref blocks) = block_outputs {
                println!("\n=== Per-Block Comparison (Candle vs NeMo) ===");
                println!("{:>8} {:>12} {:>12} {:>10} {:>10} {:>8}", "Block", "Candle_RMS", "NeMo_RMS", "Ratio", "Rel_Err", "Status");
                println!("{}", "-".repeat(70));

                let nemo_path = std::path::Path::new(nemo_dir);
                for (i, block_output) in blocks.iter().enumerate() {
                    let block_path = nemo_path.join(format!("block_{:02}.bin", i));
                    if block_path.exists() {
                        let nemo_block = load_tensor_from_binary(&block_path, &device)?;
                        let nemo_block = if nemo_block.dims().len() == 2 {
                            nemo_block.unsqueeze(0)?
                        } else {
                            nemo_block
                        };

                        let candle_rms: f32 = block_output.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
                        let nemo_rms: f32 = nemo_block.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;

                        // Compute difference
                        let diff = (block_output - &nemo_block)?;
                        let rms_diff: f32 = diff.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
                        let rel_err = if nemo_rms > 1e-6 { rms_diff / nemo_rms } else { rms_diff };

                        let ratio = candle_rms / nemo_rms;
                        let status = if rel_err < 0.01 {
                            "OK"
                        } else if rel_err < 0.05 {
                            "WARN"
                        } else if rel_err < 0.15 {
                            "DIVERGE"
                        } else {
                            "FAIL"
                        };

                        println!("{:>8} {:>12.4} {:>12.4} {:>10.3}x {:>10.4} {:>8}",
                                 i, candle_rms, nemo_rms, ratio, rel_err, status);
                    }
                }
                println!("{}", "=".repeat(70));
            }
        } else {
            println!("Warning: --compare-blocks requires --compare-nemo directory");
        }
    }

    // Substitution testing: optionally replace encoder output with NeMo's
    let enc_output = if args.substitute_encoder {
        if let Some(ref nemo_dir) = args.compare_nemo {
            let nemo_path = std::path::Path::new(nemo_dir);
            let enc_out_path = nemo_path.join("encoder_output.bin");
            if enc_out_path.exists() {
                println!("\n=== SUBSTITUTION: Using NeMo encoder output ===");
                let nemo_enc = load_tensor_from_binary(&enc_out_path, &device)?;
                // NeMo saves as [features, time] but we need [time, features]
                let nemo_enc = if nemo_enc.dims().len() == 2 {
                    nemo_enc.t()?.unsqueeze(0)?
                } else if nemo_enc.dims().len() == 3 && nemo_enc.dim(1)? > nemo_enc.dim(2)? {
                    // [1, features, time] -> [1, time, features]
                    nemo_enc.transpose(1, 2)?
                } else {
                    nemo_enc
                };
                println!("NeMo encoder shape (after transpose): {:?}", nemo_enc.dims());
                let nemo_rms: f32 = nemo_enc.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
                println!("NeMo encoder RMS: {:.6}", nemo_rms);
                println!("Candle encoder RMS: {:.6}", candle_enc_output.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?);
                println!("If transcription works now, the bug is in the encoder.");
                println!("If still fails, the bug is in the decoder/joint network.");
                println!("================================================\n");
                nemo_enc
            } else {
                anyhow::bail!("--substitute-encoder requires encoder_output.bin in --compare-nemo directory");
            }
        } else {
            anyhow::bail!("--substitute-encoder requires --compare-nemo directory");
        }
    } else {
        candle_enc_output.clone()
    };

    // Debug: optionally scale encoder output to test decoder with larger signal
    let enc_output = if let Some(scale) = args.scale_encoder {
        println!("\n=== DEBUG: Scaling encoder output by {}x ===", scale);
        let orig_rms: f32 = enc_output.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
        let scaled = (enc_output * scale)?;
        let scaled_rms: f32 = scaled.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
        println!("Scaled encoder RMS: {:.6} (was {:.6})", scaled_rms, orig_rms);
        println!("This tests if the decoder works with a larger encoder signal.");
        println!("==============================================\n");
        scaled
    } else {
        enc_output
    };

    // Compare with NeMo intermediates if directory provided
    if let Some(ref nemo_dir) = args.compare_nemo {
        println!("\n=== Substitution Testing Comparison ===");
        let nemo_path = std::path::Path::new(nemo_dir);

        // Compare encoder output
        let enc_out_path = nemo_path.join("encoder_output.bin");
        if enc_out_path.exists() {
            let nemo_enc = load_tensor_from_binary(&enc_out_path, &device)?;
            // Add batch dimension if needed
            let nemo_enc = if nemo_enc.dims().len() == 2 {
                nemo_enc.unsqueeze(0)?
            } else {
                nemo_enc
            };
            compare_tensors("encoder_output", &candle_enc_output, &nemo_enc)?;
        } else {
            println!("  encoder_output.bin not found");
        }

        // Compare mel if not using pre-loaded
        if args.mel.is_none() {
            let mel_path = nemo_path.join("mel.bin");
            if mel_path.exists() {
                let nemo_mel = load_tensor_from_binary(&mel_path, &device)?;
                let nemo_mel = if nemo_mel.dims().len() == 2 {
                    nemo_mel.unsqueeze(0)?
                } else {
                    nemo_mel
                };
                compare_tensors("mel_spectrogram", &mel, &nemo_mel)?;
            }
        }

        println!("===========================================\n");
    }

    // Debug: print encoder output at key frames
    let num_frames = enc_output.dim(1)?;
    println!("\nEncoder output at key frames:");
    for &frame in &[0usize, 5, 10, 15, 19, 20, 25, 30, 40, 50] {
        if frame < num_frames {
            let frame_data = enc_output.narrow(1, frame, 1)?.squeeze(1)?;
            let frame_rms: f32 = frame_data.sqr()?.mean_all()?.sqrt()?.to_scalar()?;
            let frame_min: f32 = frame_data.flatten_all()?.min(0)?.to_scalar()?;
            let frame_max: f32 = frame_data.flatten_all()?.max(0)?.to_scalar()?;
            println!("  Frame {:3}: rms={:.4}, range=[{:.3}, {:.3}]", frame, frame_rms, frame_min, frame_max);
        }
    }

    // Check if we're using modified encoder (disables some features)
    let use_modified_encoder = args.substitute_encoder
        || args.scale_encoder.is_some()
        || args.substitute_subsampling
        || args.substitute_block.is_some();

    // Handle streaming mode for long audio
    if args.streaming && !use_modified_encoder {
        println!("\n=== Streaming Mode ===");
        println!("Chunk duration: {:.1}s, Overlap: {:.1}s", args.chunk_duration, args.chunk_overlap);

        // Calculate chunk parameters in samples
        let chunk_samples = (args.chunk_duration * parakeet::SAMPLE_RATE as f32) as usize;
        let overlap_samples = (args.chunk_overlap * parakeet::SAMPLE_RATE as f32) as usize;
        let hop_samples = chunk_samples - overlap_samples;

        // We need the original PCM data - re-load it
        let input_file = &args.input[0];
        let (pcm_data, sample_rate) = candle_examples::audio::pcm_decode(input_file)?;
        let pcm_data = if sample_rate as usize != parakeet::SAMPLE_RATE {
            candle_examples::audio::resample(&pcm_data, sample_rate, parakeet::SAMPLE_RATE as u32)?
        } else {
            pcm_data
        };

        let total_samples = pcm_data.len();
        let num_chunks = (total_samples.saturating_sub(overlap_samples) + hop_samples - 1) / hop_samples;
        println!("Audio: {} samples ({:.2}s), {} chunks", total_samples, audio_duration, num_chunks);

        let start = std::time::Instant::now();
        let mel_filters = parakeet::mel_filters_for_sample_rate_and_bins(parakeet::SAMPLE_RATE, num_mel_bins);

        let mut all_tokens = Vec::new();
        let mut all_results = Vec::new();

        for chunk_idx in 0..num_chunks {
            let start_sample = chunk_idx * hop_samples;
            let end_sample = (start_sample + chunk_samples).min(total_samples);
            let chunk_data = &pcm_data[start_sample..end_sample];

            let chunk_start_time = start_sample as f32 / parakeet::SAMPLE_RATE as f32;
            println!("  Chunk {}/{}: samples {}..{} ({:.2}s-{:.2}s)",
                chunk_idx + 1, num_chunks, start_sample, end_sample,
                chunk_start_time, end_sample as f32 / parakeet::SAMPLE_RATE as f32);

            // Compute mel spectrogram for this chunk
            let mut mel_data = parakeet::pcm_to_mel(chunk_data, &mel_filters);
            let num_frames = mel_data.len() / num_mel_bins;
            parakeet::normalize_per_feature(&mut mel_data, num_mel_bins, num_frames);
            let chunk_mel = Tensor::from_vec(mel_data, (1, num_mel_bins, num_frames), &device)?;

            // Run inference on this chunk
            let chunk_result = model.forward_with_info(&chunk_mel)?;

            // Adjust timestamps to account for chunk position
            let mut adjusted_tokens = Vec::new();
            let frame_offset = (start_sample as f32 / parakeet::SAMPLE_RATE as f32 * 1000.0 / parakeet::FRAME_DURATION_MS) as usize;

            for mut token_info in chunk_result.tokens {
                token_info.start_frame += frame_offset;
                token_info.end_frame += frame_offset;
                adjusted_tokens.push(token_info);
            }

            // For overlapping regions, skip tokens that fall within the overlap
            // (they'll be captured by the next chunk with better context)
            if chunk_idx < num_chunks - 1 {
                let overlap_start_frame = frame_offset + (chunk_samples - overlap_samples) as usize * 1000 / (parakeet::FRAME_DURATION_MS as usize * parakeet::SAMPLE_RATE);
                adjusted_tokens.retain(|t| t.start_frame < overlap_start_frame || chunk_idx == num_chunks - 1);
            }

            all_tokens.extend(adjusted_tokens.iter().map(|t| t.token_id));
            all_results.extend(adjusted_tokens);
        }

        let elapsed = start.elapsed();

        // Decode tokens to text
        let text = decoder.decode(&all_tokens);

        println!("\n--- Transcription (Streaming) ---");
        println!("{}", text);
        println!("---------------------------------");
        println!("Chunks: {}, Tokens: {}", num_chunks, all_tokens.len());
        println!("Inference time: {:?}", elapsed);
        println!("Real-time factor: {:.2}x", elapsed.as_secs_f64() / audio_duration);

        // Print word timestamps if requested
        if args.word_timestamps {
            println!("\n--- Word Timestamps ---");
            let result = parakeet::DecodingResult {
                tokens: all_results,
                total_log_prob: 0.0, // Not meaningful for chunked
                num_frames: (audio_duration * 1000.0 / parakeet::FRAME_DURATION_MS as f64) as usize,
            };
            let vocab = decoder.get_adjusted_vocab();
            let aggregation: parakeet::ConfidenceAggregation = args.confidence_method.into();
            let words = result.word_info(&vocab, aggregation);
            for word_info in &words {
                println!(
                    "[{:5.2}s - {:5.2}s] {}",
                    word_info.start_time_sec, word_info.end_time_sec, word_info.word
                );
            }
        }

        return Ok(());
    }

    // Run inference using the (possibly substituted/scaled) encoder output
    // Validate MALSD usage
    if args.malsd {
        if args.beam_width.is_none() {
            eprintln!("Warning: --malsd requires --beam-width. Defaulting to beam width 5.");
        }
        if model.decoder_type() != parakeet::DecoderType::Tdt {
            anyhow::bail!("--malsd is only supported for TDT decoder (use --model-variant tdt-v2 or tdt-v3)");
        }
    }

    println!("Running transcription...");
    let start = std::time::Instant::now();
    // Check if we need rich results (timestamps or confidence)
    // MALSD always returns rich results
    let need_rich_results = args.timestamps || args.word_timestamps || args.confidence || args.malsd;

    let (tokens, result_opt) = if need_rich_results {
        // Use _with_info methods to get timestamps and confidence
        let result = if use_modified_encoder {
            match args.beam_width {
                Some(beam_width) if args.malsd => {
                    println!("Using MALSD beam search with width {} on modified encoder output", beam_width);
                    model.decoder.decode_malsd(&enc_output, beam_width)?
                }
                Some(beam_width) => {
                    println!("Using beam search with width {} on modified encoder output (with timestamps)", beam_width);
                    model.decoder.decode_beam_with_info(&enc_output, beam_width)?
                }
                None if args.malsd => {
                    println!("Using MALSD beam search with width 5 on modified encoder output");
                    model.decoder.decode_malsd(&enc_output, 5)?
                }
                None => model.decoder.decode_greedy_with_info(&enc_output)?,
            }
        } else {
            match args.beam_width {
                Some(beam_width) if args.malsd => {
                    println!("Using MALSD beam search with width {}", beam_width);
                    model.forward_malsd(&mel, beam_width)?
                }
                Some(beam_width) if lm.is_some() => {
                    println!("Using beam search with width {} + LM (weight={}) (with timestamps)", beam_width, args.lm_weight);
                    model.forward_beam_with_lm(&mel, beam_width, lm.as_ref().unwrap(), &vocab_for_lm, args.lm_weight)?
                }
                Some(beam_width) => {
                    println!("Using beam search with width {} (with timestamps)", beam_width);
                    model.forward_beam_with_info(&mel, beam_width)?
                }
                None if args.malsd => {
                    println!("Using MALSD beam search with width 5");
                    model.forward_malsd(&mel, 5)?
                }
                None => model.forward_with_info(&mel)?,
            }
        };
        let tokens = result.token_ids();
        (tokens, Some(result))
    } else {
        // Standard path without rich results
        let tokens = if use_modified_encoder {
            match args.beam_width {
                Some(beam_width) => {
                    println!("Using beam search with width {} on modified encoder output", beam_width);
                    model.decoder.decode_beam(&enc_output, beam_width)?
                }
                None => model.decoder.decode_greedy(&enc_output)?,
            }
        } else {
            match args.beam_width {
                Some(beam_width) if lm.is_some() => {
                    println!("Using beam search with width {} + LM (weight={})", beam_width, args.lm_weight);
                    let result = model.forward_beam_with_lm(&mel, beam_width, lm.as_ref().unwrap(), &vocab_for_lm, args.lm_weight)?;
                    result.token_ids()
                }
                Some(beam_width) => {
                    println!("Using beam search with width {}", beam_width);
                    model.forward_beam(&mel, beam_width)?
                }
                None => model.forward(&mel)?,
            }
        };
        (tokens, None)
    };
    let elapsed = start.elapsed();

    // Decode tokens to text
    let text = decoder.decode(&tokens);

    println!("\n--- Transcription ---");
    println!("{}", text);
    println!("---------------------");
    println!("Tokens: {}", tokens.len());
    println!("Inference time: {:?}", elapsed);
    println!("Real-time factor: {:.2}x", elapsed.as_secs_f64() / audio_duration);

    // Print timestamps and/or confidence if requested
    if let Some(result) = result_opt {
        if args.timestamps && !args.word_timestamps {
            // Token-level timestamps
            println!("\n--- Token Timestamps ---");
            let timestamps = result.compute_timestamps();
            for (token_info, ts) in result.tokens.iter().zip(timestamps.iter()) {
                let token_text = decoder.decode_token(token_info.token_id).unwrap_or("<unk>");
                if args.confidence {
                    println!(
                        "[{:5.2}s - {:5.2}s] {:20} (conf: {:5.1}%)",
                        ts.start_time_sec,
                        ts.end_time_sec,
                        token_text,
                        ts.confidence * 100.0
                    );
                } else {
                    println!(
                        "[{:5.2}s - {:5.2}s] {}",
                        ts.start_time_sec, ts.end_time_sec, token_text
                    );
                }
            }
            println!("Total log probability: {:.4}", result.total_log_prob);
        }

        if args.word_timestamps {
            // Word-level timestamps with aggregated confidence
            println!("\n--- Word Timestamps ---");
            let vocab = decoder.get_adjusted_vocab();
            let aggregation: parakeet::ConfidenceAggregation = args.confidence_method.into();
            let words = result.word_info(&vocab, aggregation);
            for word_info in &words {
                if args.confidence {
                    println!(
                        "[{:5.2}s - {:5.2}s] {:20} (conf: {:5.1}%)",
                        word_info.start_time_sec,
                        word_info.end_time_sec,
                        word_info.word,
                        word_info.confidence * 100.0
                    );
                } else {
                    println!(
                        "[{:5.2}s - {:5.2}s] {}",
                        word_info.start_time_sec, word_info.end_time_sec, word_info.word
                    );
                }
            }
            println!(
                "Words: {}, Aggregation: {:?}",
                words.len(),
                aggregation
            );
        }

        if args.confidence && !args.timestamps && !args.word_timestamps {
            // Just show overall confidence
            let avg_confidence = if result.tokens.is_empty() {
                0.0
            } else {
                (result.total_log_prob / result.tokens.len() as f32).exp()
            };
            println!(
                "\nAverage token confidence: {:.1}%",
                avg_confidence * 100.0
            );
        }
    }

    Ok(())
}
