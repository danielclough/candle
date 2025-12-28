//! Audio processing for Parakeet TDT
//!
//! Mel spectrogram computation with preemphasis filter.
//! Adapted from whisper/audio.rs with Parakeet-specific parameters.

use super::{HOP_LENGTH, N_FFT, N_MELS, PREEMPHASIS, WIN_LENGTH};

pub trait Float:
    num_traits::Float + num_traits::FloatConst + num_traits::NumAssign + Send + Sync
{
}

impl Float for f32 {}
impl Float for f64 {}

/// Apply dithering (tiny noise) to prevent numerical issues with silent audio
///
/// NeMo applies dithering before mel spectrogram computation to avoid log(0)
/// and other numerical issues with very quiet or silent audio segments.
///
/// # Arguments
/// * `samples` - Audio samples to dither (modified in place)
/// * `amount` - Dithering amount (typically 1e-5)
pub fn dither(samples: &mut [f32], amount: f32) {
    use rand::Rng;
    let mut rng = rand::rng();
    for sample in samples.iter_mut() {
        *sample += rng.random_range(-amount..amount);
    }
}

/// Apply preemphasis filter: y[n] = x[n] - coeff * x[n-1]
pub fn preemphasis<T: Float>(samples: &[T], coeff: T) -> Vec<T> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(samples.len());
    result.push(samples[0]);
    for i in 1..samples.len() {
        result.push(samples[i] - coeff * samples[i - 1]);
    }
    result
}

/// Generate mel filterbank matrix for given parameters
/// Based on librosa.filters.mel with HTK formula and Slaney normalization
///
/// Uses continuous frequency weights (like librosa) rather than discrete bin indices,
/// which handles the case where mel bands are narrower than FFT bins.
pub fn mel_filters() -> Vec<f32> {
    mel_filters_for_sample_rate(super::SAMPLE_RATE)
}

/// Generate mel filterbank matrix for a specific sample rate (default 128 mel bins)
///
/// NeMo's preprocessor uses librosa.filters.mel with fmax=Nyquist.
/// This means the mel bins span the full frequency range of the input sample rate.
pub fn mel_filters_for_sample_rate(sample_rate: usize) -> Vec<f32> {
    mel_filters_for_sample_rate_and_bins(sample_rate, N_MELS)
}

/// Generate mel filterbank matrix for a specific sample rate and number of mel bins
///
/// NeMo's preprocessor uses librosa.filters.mel with fmax=Nyquist.
/// XL models (0.6B) use 128 mel bins, XXL models (1.1B) use 80 mel bins.
pub fn mel_filters_for_sample_rate_and_bins(sample_rate: usize, n_mels: usize) -> Vec<f32> {
    let n_fft = N_FFT;
    let sample_rate = sample_rate as f32;
    let f_min = 0.0f32;
    // Use full Nyquist frequency for the sample rate (like NeMo/librosa)
    let f_max = sample_rate / 2.0;

    let n_freqs = n_fft / 2 + 1; // 257

    // Convert Hz to mel scale (Slaney formula - librosa default with htk=False)
    // Linear below 1000 Hz, logarithmic above
    const F_MIN: f32 = 0.0;
    const F_SP: f32 = 200.0 / 3.0; // Linear spacing: 66.67 Hz per mel
    const MIN_LOG_HZ: f32 = 1000.0; // Transition frequency
    const MIN_LOG_MEL: f32 = (MIN_LOG_HZ - F_MIN) / F_SP; // = 15 mels
    const LOGSTEP: f32 = 0.06875177742094912; // ln(6.4) / 27

    fn hz_to_mel(hz: f32) -> f32 {
        if hz < MIN_LOG_HZ {
            (hz - F_MIN) / F_SP
        } else {
            MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOGSTEP
        }
    }
    fn mel_to_hz(mel: f32) -> f32 {
        if mel < MIN_LOG_MEL {
            F_MIN + F_SP * mel
        } else {
            MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * LOGSTEP).exp()
        }
    }

    // Create mel points in mel scale (n_mels + 2 points for triangular filters)
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    // Convert mel points to Hz
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // FFT bin frequencies
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|k| k as f32 * sample_rate / n_fft as f32)
        .collect();

    // Create filterbank matrix using librosa's continuous weight approach
    // For each mel filter m, compute triangular weights based on frequency distance
    let mut filters = vec![0.0f32; n_mels * n_freqs];

    for m in 0..n_mels {
        let f_left = hz_points[m];
        let f_center = hz_points[m + 1];
        let f_right = hz_points[m + 2];

        // Slaney normalization
        let enorm = 2.0 / (f_right - f_left);

        for k in 0..n_freqs {
            let freq = fft_freqs[k];

            // Rising slope: (freq - f_left) / (f_center - f_left)
            // Falling slope: (f_right - freq) / (f_right - f_center)
            // Take minimum of both (triangular filter)
            let lower_slope = if f_center > f_left {
                (freq - f_left) / (f_center - f_left)
            } else {
                0.0
            };

            let upper_slope = if f_right > f_center {
                (f_right - freq) / (f_right - f_center)
            } else {
                0.0
            };

            // Triangular filter: min of slopes, but must be >= 0
            let weight = lower_slope.min(upper_slope).max(0.0);
            filters[m * n_freqs + k] = enorm * weight;
        }
    }

    filters
}

// FFT implementation adapted from whisper/audio.rs
fn fft<T: Float>(inp: &[T]) -> Vec<T> {
    let n = inp.len();
    let zero = T::zero();
    if n == 1 {
        return vec![inp[0], zero];
    }
    if n % 2 == 1 {
        return dft(inp);
    }
    let mut out = vec![zero; n * 2];

    let mut even = Vec::with_capacity(n / 2);
    let mut odd = Vec::with_capacity(n / 2);

    for (i, &inp) in inp.iter().enumerate() {
        if i % 2 == 0 {
            even.push(inp)
        } else {
            odd.push(inp);
        }
    }

    let even_fft = fft(&even);
    let odd_fft = fft(&odd);

    let two_pi = T::PI() + T::PI();
    let n_t = T::from(n).unwrap();
    for k in 0..n / 2 {
        let k_t = T::from(k).unwrap();
        let theta = two_pi * k_t / n_t;
        let re = theta.cos();
        let im = -theta.sin();

        let re_odd = odd_fft[2 * k];
        let im_odd = odd_fft[2 * k + 1];

        out[2 * k] = even_fft[2 * k] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

        out[2 * (k + n / 2)] = even_fft[2 * k] - re * re_odd + im * im_odd;
        out[2 * (k + n / 2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
    out
}

fn dft<T: Float>(inp: &[T]) -> Vec<T> {
    let zero = T::zero();
    let n = inp.len();
    let two_pi = T::PI() + T::PI();

    let mut out = Vec::with_capacity(2 * n);
    let n_t = T::from(n).unwrap();
    for k in 0..n {
        let k_t = T::from(k).unwrap();
        let mut re = zero;
        let mut im = zero;

        for (j, &inp) in inp.iter().enumerate() {
            let j_t = T::from(j).unwrap();
            let angle = two_pi * k_t * j_t / n_t;
            re += inp * angle.cos();
            im -= inp * angle.sin();
        }

        out.push(re);
        out.push(im);
    }
    out
}

/// Compute log mel spectrogram
fn log_mel_spectrogram_<T: Float>(
    samples: &[T],
    filters: &[T],
    fft_size: usize,
    hop_length: usize,
    win_length: usize,
    n_mel: usize,
) -> Vec<T> {
    let zero = T::zero();
    let two_pi = T::PI() + T::PI();
    let half = T::from(0.5).unwrap();
    let one = T::from(1.0).unwrap();

    // Create Hann window of win_length, zero-padded to fft_size
    // NeMo uses periodic=False (symmetric window): cos(2π*i/(N-1))
    let win_length_minus_one = T::from(win_length - 1).unwrap();
    let hann: Vec<T> = (0..fft_size)
        .map(|i| {
            if i < win_length {
                half * (one - ((two_pi * T::from(i).unwrap()) / win_length_minus_one).cos())
            } else {
                zero
            }
        })
        .collect();

    let n_frames = samples.len().saturating_sub(win_length) / hop_length + 1;
    if n_frames == 0 {
        return vec![zero; n_mel];
    }

    let n_fft = fft_size / 2 + 1;
    let mut mel = vec![zero; n_frames * n_mel];
    let mut fft_in = vec![zero; fft_size];

    for i in 0..n_frames {
        let offset = i * hop_length;

        // Apply Hann window
        for j in 0..fft_size {
            if offset + j < samples.len() {
                fft_in[j] = hann[j] * samples[offset + j];
            } else {
                fft_in[j] = zero;
            }
        }

        // FFT
        let mut fft_out = fft(&fft_in);

        // Calculate magnitude squared
        for j in 0..fft_size {
            fft_out[j] = fft_out[2 * j] * fft_out[2 * j] + fft_out[2 * j + 1] * fft_out[2 * j + 1];
        }

        // Apply mel filterbank and take log
        // NeMo uses log(x + guard) with guard = 2^-24 ≈ 5.96e-8 (not clamp!)
        let log_guard = T::from(5.960464477539063e-8).unwrap(); // 2^-24
        for j in 0..n_mel {
            let mut sum = zero;
            for k in 0..n_fft {
                sum += fft_out[k] * filters[j * n_fft + k];
            }
            // NeMo: log(x + guard), not log(max(x, guard))
            mel[j * n_frames + i] = (sum + log_guard).ln();
        }
    }

    mel
}

/// Normalize mel spectrogram per feature channel (NeMo per_feature normalization)
///
/// For each mel bin, computes mean and std across time dimension,
/// then normalizes to approximately zero mean and unit variance.
/// Uses unbiased std estimator (N-1 denominator) matching NeMo.
pub fn normalize_per_feature(mel: &mut [f32], n_mels: usize, n_frames: usize) {
    const EPSILON: f32 = 1e-5;

    if n_frames <= 1 {
        return; // Cannot compute std with single frame
    }

    for j in 0..n_mels {
        // Compute mean for this mel bin across time
        let mut sum = 0.0f32;
        for i in 0..n_frames {
            sum += mel[j * n_frames + i];
        }
        let mean = sum / n_frames as f32;

        // Compute std with unbiased estimator (N-1)
        let mut var_sum = 0.0f32;
        for i in 0..n_frames {
            let diff = mel[j * n_frames + i] - mean;
            var_sum += diff * diff;
        }
        let std = (var_sum / (n_frames - 1) as f32).sqrt() + EPSILON;

        // Normalize
        for i in 0..n_frames {
            mel[j * n_frames + i] = (mel[j * n_frames + i] - mean) / std;
        }
    }
}

/// Convert PCM audio to mel spectrogram for Parakeet
///
/// Applies preemphasis filter and computes log mel spectrogram
/// with Parakeet-specific parameters (N_FFT=512, HOP_LENGTH=160, etc.)
///
/// Uses STFT centering (pads signal by n_fft//2 on each side) to match NeMo behavior.
///
/// Note: This does NOT apply per-feature normalization. Call `normalize_per_feature`
/// separately if normalization is needed (NeMo uses `normalize: per_feature` by default).
pub fn pcm_to_mel(samples: &[f32], filters: &[f32]) -> Vec<f32> {
    // Apply preemphasis filter
    let samples = preemphasis(samples, PREEMPHASIS);

    // Derive n_mels from filter matrix dimensions
    // filters shape: [n_mels, n_freqs] where n_freqs = N_FFT / 2 + 1
    let n_freqs = N_FFT / 2 + 1;
    let n_mels = filters.len() / n_freqs;

    // Apply STFT centering: pad signal by n_fft//2 on each side
    // This matches NeMo's AudioToMelSpectrogramPreprocessor behavior (center=True)
    let pad_len = N_FFT / 2;
    let mut padded = vec![0.0f32; pad_len + samples.len() + pad_len];
    padded[pad_len..pad_len + samples.len()].copy_from_slice(&samples);

    // Compute log mel spectrogram on padded signal
    log_mel_spectrogram_(&padded, filters, N_FFT, HOP_LENGTH, WIN_LENGTH, n_mels)
}

/// Convert PCM audio to mel spectrogram at the original sample rate
///
/// Unlike `pcm_to_mel`, this function does NOT expect audio to be resampled to 16kHz.
/// Instead, it uses the same hop of 160 samples regardless of sample rate,
/// which matches NeMo's AudioToMelSpectrogramPreprocessor behavior.
///
/// This produces more frames for higher sample rates:
/// - 16kHz: hop = 160 samples = 10ms → same as pcm_to_mel
/// - 24kHz: hop = 160 samples = 6.67ms → 1.5x more frames
/// - 48kHz: hop = 160 samples = 3.33ms → 3x more frames
///
/// The mel filterbank should be created with `mel_filters_for_sample_rate(sample_rate)`.
pub fn pcm_to_mel_at_sample_rate(samples: &[f32], filters: &[f32], _sample_rate: usize) -> Vec<f32> {
    // Apply preemphasis filter
    let samples = preemphasis(samples, PREEMPHASIS);

    // Derive n_mels from filter matrix dimensions
    let n_freqs = N_FFT / 2 + 1;
    let n_mels = filters.len() / n_freqs;

    // Apply STFT centering: pad signal by n_fft//2 on each side
    let pad_len = N_FFT / 2;
    let mut padded = vec![0.0f32; pad_len + samples.len() + pad_len];
    padded[pad_len..pad_len + samples.len()].copy_from_slice(&samples);

    // Use fixed hop of 160 samples (same as NeMo)
    // This means higher sample rates produce more frames
    let hop_length = HOP_LENGTH; // 160 samples regardless of sample rate

    // Keep the same FFT and window parameters
    // NeMo uses the same n_fft=512, win_length=400 regardless of sample rate
    // The mel filterbank handles the frequency mapping adjustment
    log_mel_spectrogram_(&padded, filters, N_FFT, hop_length, WIN_LENGTH, n_mels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preemphasis() {
        let samples = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = preemphasis(&samples, 0.97);
        assert_eq!(result.len(), 4);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - (2.0 - 0.97 * 1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_mel_filters_shape() {
        let filters = mel_filters();
        let n_freqs = N_FFT / 2 + 1; // 257
        assert_eq!(filters.len(), N_MELS * n_freqs);
    }

    #[test]
    fn test_pcm_to_mel() {
        let samples = vec![0.0f32; 16000]; // 1 second of silence
        let filters = mel_filters();
        let mel = pcm_to_mel(&samples, &filters);
        // With STFT centering, signal is padded by N_FFT/2 on each side
        // Padded length: 16000 + N_FFT = 16512
        // Expected frames: (16512 - 400) / 160 + 1 = 101
        let padded_len = 16000 + N_FFT;
        let expected_frames = (padded_len - WIN_LENGTH) / HOP_LENGTH + 1;
        assert_eq!(mel.len(), N_MELS * expected_frames);
    }

    #[test]
    fn test_mel_filters_no_zero_sum() {
        // Verify that no mel filter has zero sum (except possibly the first one at 0 Hz)
        let filters = mel_filters();
        let n_freqs = N_FFT / 2 + 1;
        let mut zero_sum_count = 0;

        for m in 0..N_MELS {
            let mut sum = 0.0f32;
            for k in 0..n_freqs {
                sum += filters[m * n_freqs + k];
            }
            if sum == 0.0 {
                zero_sum_count += 1;
            }
        }

        // At most 1 zero-sum filter (the first one at 0 Hz is expected to be zero)
        assert!(
            zero_sum_count <= 1,
            "Too many zero-sum mel filters: {} (should be at most 1)",
            zero_sum_count
        );
    }

    #[test]
    fn test_normalize_per_feature() {
        // Create a simple 2x4 mel spectrogram (2 mel bins, 4 frames)
        // mel[bin * n_frames + frame]
        let mut mel = vec![
            1.0, 2.0, 3.0, 4.0, // bin 0: mean=2.5, std~=1.29
            10.0, 20.0, 30.0, 40.0, // bin 1: mean=25, std~=12.9
        ];
        let n_mels = 2;
        let n_frames = 4;

        normalize_per_feature(&mut mel, n_mels, n_frames);

        // After normalization, each bin should have ~zero mean
        let mean0: f32 = mel[0..4].iter().sum::<f32>() / 4.0;
        let mean1: f32 = mel[4..8].iter().sum::<f32>() / 4.0;
        assert!(mean0.abs() < 1e-5, "Bin 0 mean should be ~0, got {}", mean0);
        assert!(mean1.abs() < 1e-5, "Bin 1 mean should be ~0, got {}", mean1);

        // And ~unit variance (std close to 1, accounting for N-1 estimator and epsilon)
        let var0: f32 = mel[0..4].iter().map(|x| x * x).sum::<f32>() / 4.0;
        let var1: f32 = mel[4..8].iter().map(|x| x * x).sum::<f32>() / 4.0;
        // With N-1 denominator and epsilon, variance should be close to (N-1)/N ≈ 0.75
        assert!(
            (var0 - 0.75).abs() < 0.1,
            "Bin 0 variance should be ~0.75, got {}",
            var0
        );
        assert!(
            (var1 - 0.75).abs() < 0.1,
            "Bin 1 variance should be ~0.75, got {}",
            var1
        );
    }

    #[test]
    fn test_mel_filters_multiple_sample_rates() {
        // Test that filterbanks work correctly at various sample rates
        // Zero-sum filter counts should match librosa.filters.mel with htk=True, norm='slaney'
        // (verified against librosa 0.10.x)
        let expected_zero_sum = [
            (8000, 0),
            (16000, 1),
            (22050, 3),
            (24000, 4),
            (44100, 10),
            (48000, 11),
        ];

        for &(sr, expected) in &expected_zero_sum {
            let filters = mel_filters_for_sample_rate(sr);
            let n_freqs = N_FFT / 2 + 1;
            assert_eq!(
                filters.len(),
                N_MELS * n_freqs,
                "Wrong filterbank shape at {}Hz",
                sr
            );

            // Count zero-sum filters and verify they match librosa
            let mut zero_sum = 0;
            for m in 0..N_MELS {
                let sum: f32 = (0..n_freqs).map(|k| filters[m * n_freqs + k]).sum();
                if sum == 0.0 {
                    zero_sum += 1;
                }
            }
            assert_eq!(
                zero_sum, expected,
                "Zero-sum filter count at {}Hz: got {} expected {} (matching librosa)",
                sr, zero_sum, expected
            );
        }
    }

    #[test]
    fn test_pcm_to_mel_short_audio() {
        let filters = mel_filters();
        // Test with audio shorter than win_length (400 samples)
        let short_audio = vec![0.1f32; 100];
        let mel = pcm_to_mel(&short_audio, &filters);
        // Should produce at least 1 frame due to STFT centering
        assert!(!mel.is_empty(), "Should produce at least 1 frame for short audio");
        assert_eq!(
            mel.len() % N_MELS,
            0,
            "Mel length should be multiple of N_MELS"
        );
    }

    #[test]
    fn test_pcm_to_mel_24khz_frame_count() {
        // 1 second at 24kHz = 24000 samples
        // With STFT centering: padded = 24000 + 512 = 24512 samples
        // Expected frames = (24512 - 400) / 160 + 1 = 151
        let samples = vec![0.0f32; 24000];
        let filters = mel_filters_for_sample_rate(24000);
        let mel = pcm_to_mel(&samples, &filters);
        let frames = mel.len() / N_MELS;

        // At 24kHz with hop=160, we get 1.5x more frames than at 16kHz
        // 16kHz: 1 second = 100 frames (16000/160)
        // 24kHz: 1 second = 150 frames (24000/160)
        assert!(
            frames >= 150 && frames <= 155,
            "Expected ~151 frames at 24kHz for 1s audio, got {}",
            frames
        );
    }

    #[test]
    fn test_dither() {
        let mut samples = vec![0.0f32; 1000];
        dither(&mut samples, 1e-5);

        // After dithering, samples should be non-zero (with high probability)
        let non_zero_count = samples.iter().filter(|&&x| x != 0.0).count();
        assert!(
            non_zero_count > 900,
            "Dithering should modify most samples, only {} were non-zero",
            non_zero_count
        );

        // All values should be within dither range
        for &sample in &samples {
            assert!(
                sample.abs() <= 1e-5,
                "Dithered value {} exceeds range",
                sample
            );
        }
    }
}
