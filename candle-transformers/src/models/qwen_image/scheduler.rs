//! Flow Match Euler Discrete Scheduler for Qwen-Image.
//!
//! This module implements the FlowMatchEulerDiscreteScheduler used for diffusion
//! inference in Qwen-Image. Key concepts:
//!
//! - **Flow Matching**: A training paradigm where the model learns a velocity field
//!   that describes the flow from noise to data.
//!
//! - **Euler Discrete**: Uses discrete Euler steps to integrate the learned velocity
//!   field during inference.
//!
//! - **Dynamic Shifting**: Resolution-aware sigma scheduling that adjusts the noise
//!   levels based on image size. Larger images get more aggressive shifting to
//!   maintain quality.
//!
//! # Noise Schedule
//!
//! The scheduler uses the rectified flow formulation:
//! ```text
//! x_t = (1 - sigma) * x_0 + sigma * noise
//! ```
//!
//! During inference, sigmas go from 1.0 (pure noise) to 0.0 (clean image).

use super::config::SchedulerConfig;
use candle::{Result, Tensor};

/// Flow Match Euler Discrete Scheduler.
///
/// Implements the flow matching scheduler with dynamic shift support for
/// resolution-dependent sigma scheduling.
///
/// # Example
///
/// ```ignore
/// let mut scheduler = FlowMatchEulerDiscreteScheduler::new(&config);
/// scheduler.set_timesteps(50, None, Some(mu));
///
/// for step in 0..50 {
///     let noise_pred = model.forward(&latents, scheduler.timesteps()[step], ...)?;
///     latents = scheduler.step(&noise_pred, &latents)?;
/// }
/// ```
#[derive(Debug, Clone)]
pub struct FlowMatchEulerDiscreteScheduler {
    /// Number of training timesteps (typically 1000).
    num_train_timesteps: usize,

    /// Whether to use dynamic shifting based on image resolution.
    use_dynamic_shifting: bool,

    /// Base shift value for the smallest supported resolution.
    base_shift: f64,

    /// Maximum shift value for the largest supported resolution.
    _max_shift: f64,

    /// Base image sequence length for shift interpolation.
    _base_image_seq_len: usize,

    /// Maximum image sequence length for shift interpolation.
    _max_image_seq_len: usize,

    /// Current sigma values (noise levels) for each step.
    sigmas: Vec<f64>,

    /// Current timestep values (sigmas × num_train_timesteps).
    timesteps: Vec<f64>,

    /// Current step index during inference.
    step_index: usize,

    /// Starting index for img2img (skip initial denoising steps).
    begin_index: usize,
}

impl FlowMatchEulerDiscreteScheduler {
    /// Create a new scheduler with the given configuration.
    pub fn new(config: &SchedulerConfig) -> Self {
        Self {
            num_train_timesteps: config.num_train_timesteps,
            use_dynamic_shifting: config.use_dynamic_shifting,
            base_shift: config.base_shift,
            _max_shift: config.max_shift,
            _base_image_seq_len: config.base_image_seq_len,
            _max_image_seq_len: config.max_image_seq_len,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            step_index: 0,
            begin_index: 0,
        }
    }

    /// Create a scheduler with default configuration.
    pub fn default_config() -> Self {
        Self::new(&SchedulerConfig::default())
    }

    /// Time shift function for dynamic shifting.
    ///
    /// Maps uniform timesteps to resolution-dependent timesteps using exponential
    /// shifting. This biases towards larger noise levels for larger images.
    ///
    /// # Arguments
    /// * `mu` - Shift parameter computed from `calculate_shift`
    /// * `t` - Input sigma value in [0, 1]
    ///
    /// # Returns
    /// Shifted sigma value
    pub fn time_shift(&self, mu: f64, t: f64) -> f64 {
        if t <= 0.0 || t >= 1.0 {
            return t;
        }
        let exp_mu = mu.exp();
        exp_mu / (exp_mu + (1.0 / t - 1.0))
    }

    /// Set timesteps for inference.
    ///
    /// # Arguments
    /// * `num_inference_steps` - Number of denoising steps
    /// * `sigmas` - Optional custom sigma schedule (overrides default linspace)
    /// * `mu` - Optional shift parameter for dynamic shifting
    pub fn set_timesteps(
        &mut self,
        num_inference_steps: usize,
        sigmas: Option<&[f64]>,
        mu: Option<f64>,
    ) {
        // Generate or use provided sigmas
        let mut sigmas = if let Some(s) = sigmas {
            s.to_vec()
        } else {
            // Match PyTorch: linspace from sigma_max (1.0) to sigma_min (1/num_train_timesteps)
            // This ensures the schedule reaches near-zero noise at the final step
            let sigma_max = 1.0;
            let sigma_min = 1.0 / self.num_train_timesteps as f64;
            (0..num_inference_steps)
                .map(|i| {
                    sigma_max
                        - (sigma_max - sigma_min) * i as f64
                            / (num_inference_steps - 1).max(1) as f64
                })
                .collect()
        };

        // Apply dynamic shifting if enabled
        if self.use_dynamic_shifting {
            let mu = mu.unwrap_or(self.base_shift);
            sigmas = sigmas.iter().map(|&s| self.time_shift(mu, s)).collect();
        }

        // Append final sigma = 0.0
        sigmas.push(0.0);

        // Compute timesteps = sigmas × num_train_timesteps
        self.timesteps = sigmas
            .iter()
            .map(|&s| s * self.num_train_timesteps as f64)
            .collect();

        self.sigmas = sigmas;
        self.step_index = self.begin_index;
    }

    /// Get current sigma values.
    pub fn sigmas(&self) -> &[f64] {
        &self.sigmas
    }

    /// Get current timestep values.
    pub fn timesteps(&self) -> &[f64] {
        &self.timesteps
    }

    /// Get the current step index.
    pub fn step_index(&self) -> usize {
        self.step_index
    }

    /// Get the number of inference steps.
    pub fn num_inference_steps(&self) -> usize {
        self.sigmas.len().saturating_sub(1)
    }

    /// Euler step: integrate the velocity field for one step.
    ///
    /// Uses the formula: x_{t+1} = x_t + dt × v(x_t, t)
    ///
    /// # Arguments
    /// * `model_output` - Predicted velocity from the model
    /// * `sample` - Current noisy sample
    ///
    /// # Returns
    /// Denoised sample for the next step
    pub fn step(&mut self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let sigma = self.sigmas[self.step_index];
        let sigma_next = self.sigmas[self.step_index + 1];
        let dt = sigma_next - sigma; // Negative since sigma decreases

        // Euler step: x_{t+1} = x_t + dt × v(x_t, t)
        let prev_sample = (sample + (model_output * dt)?)?;

        self.step_index += 1;
        Ok(prev_sample)
    }

    /// Scale noise for initial latent preparation (img2img / inpainting).
    ///
    /// Uses the flow matching interpolation formula:
    /// x_t = (1 - sigma) × x_0 + sigma × noise
    ///
    /// # Arguments
    /// * `sample` - Clean sample (image latents)
    /// * `noise` - Random noise
    /// * `sigma` - Noise level at starting step
    ///
    /// # Returns
    /// Noised sample at the specified sigma level
    pub fn scale_noise(&self, sample: &Tensor, noise: &Tensor, sigma: f64) -> Result<Tensor> {
        // x_t = (1 - sigma) × x_0 + sigma × noise
        let clean_weight = 1.0 - sigma;
        (sample * clean_weight)? + (noise * sigma)?
    }

    /// Set the starting index for img2img inference.
    ///
    /// When using strength < 1.0 for img2img, we skip initial denoising steps.
    /// Call this before `set_timesteps` to start from a later step.
    ///
    /// # Arguments
    /// * `begin_index` - Step to start denoising from
    pub fn set_begin_index(&mut self, begin_index: usize) {
        self.begin_index = begin_index;
        self.step_index = begin_index;
    }

    /// Reset the scheduler for a new inference run.
    pub fn reset(&mut self) {
        self.step_index = self.begin_index;
    }
}

/// Calculate the dynamic shift parameter `mu` based on image sequence length.
///
/// Linear interpolation between base_shift and max_shift based on sequence length:
/// ```text
/// mu = (seq_len - base_seq_len) / (max_seq_len - base_seq_len) × (max_shift - base_shift) + base_shift
/// ```
///
/// # Arguments
/// * `image_seq_len` - Number of latent tokens (height/2 × width/2 for packed latents)
/// * `base_seq_len` - Base sequence length (typically 256)
/// * `max_seq_len` - Maximum sequence length (typically 4096)
/// * `base_shift` - Shift value at base sequence length (typically 0.5)
/// * `max_shift` - Shift value at max sequence length (typically 1.15)
///
/// # Returns
/// Interpolated shift parameter `mu`
///
/// # Example
///
/// ```ignore
/// // For a 1024×1024 image with 8× spatial compression and 2× packing:
/// // latent = 128×128, packed = 64×64, seq_len = 4096
/// let mu = calculate_shift(4096, 256, 4096, 0.5, 1.15);
/// assert!((mu - 1.15).abs() < 0.01);
/// ```
pub fn calculate_shift(
    image_seq_len: usize,
    base_seq_len: usize,
    max_seq_len: usize,
    base_shift: f64,
    max_shift: f64,
) -> f64 {
    // Linear interpolation: mu = m × seq_len + b
    let m = (max_shift - base_shift) / (max_seq_len - base_seq_len) as f64;
    let b = base_shift - m * base_seq_len as f64;
    image_seq_len as f64 * m + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_shift() {
        // At base sequence length, shift should be base_shift
        let mu = calculate_shift(256, 256, 4096, 0.5, 1.15);
        assert!((mu - 0.5).abs() < 1e-6);

        // At max sequence length, shift should be max_shift
        let mu = calculate_shift(4096, 256, 4096, 0.5, 1.15);
        assert!((mu - 1.15).abs() < 1e-6);

        // Midpoint should be midpoint shift
        let mu = calculate_shift(2176, 256, 4096, 0.5, 1.15);
        assert!((mu - 0.825).abs() < 0.01);
    }

    #[test]
    fn test_scheduler_timesteps() {
        let config = SchedulerConfig::default();
        let scheduler = FlowMatchEulerDiscreteScheduler::new(&config);

        // Set 10 inference steps without dynamic shifting
        let mut scheduler_no_shift = scheduler.clone();
        scheduler_no_shift.use_dynamic_shifting = false;
        scheduler_no_shift.set_timesteps(10, None, None);

        // Should have 11 sigmas (10 steps + final 0.0)
        assert_eq!(scheduler_no_shift.sigmas().len(), 11);
        assert_eq!(scheduler_no_shift.timesteps().len(), 11);

        // First sigma should be 1.0, last should be 0.0
        assert!((scheduler_no_shift.sigmas()[0] - 1.0).abs() < 1e-6);
        assert!((scheduler_no_shift.sigmas()[10] - 0.0).abs() < 1e-6);

        // Second sigma should be linspace from 1.0 to sigma_min
        // For 10 steps: sigma[1] = 1.0 - (1.0 - 0.001) * 1/9 ≈ 0.889
        let sigma_min = 1.0 / 1000.0; // num_train_timesteps = 1000
        let expected_sigma_1 = 1.0 - (1.0 - sigma_min) / 9.0;
        assert!(
            (scheduler_no_shift.sigmas()[1] - expected_sigma_1).abs() < 1e-6,
            "sigma[1] should be {}, got {}",
            expected_sigma_1,
            scheduler_no_shift.sigmas()[1]
        );
    }

    #[test]
    fn test_scheduler_sigmas_match_pytorch() {
        // Test that 5-step schedule with dynamic shifting matches PyTorch exactly
        let config = SchedulerConfig::default();
        let mut scheduler = FlowMatchEulerDiscreteScheduler::new(&config);

        // mu = 0.5 for 256×256 image (image_seq_len = 256)
        scheduler.set_timesteps(5, None, Some(0.5));

        // PyTorch reference values (from scheduling_flow_match_euler_discrete.py)
        let pytorch_sigmas = [1.0, 0.832, 0.623, 0.356, 0.0016, 0.0];

        for (i, &py_sigma) in pytorch_sigmas.iter().enumerate() {
            let rust_sigma = scheduler.sigmas()[i];
            let diff = (rust_sigma - py_sigma).abs();
            assert!(
                diff < 0.001,
                "sigma[{}]: Rust={:.4}, PyTorch={:.4}, diff={:.6}",
                i,
                rust_sigma,
                py_sigma,
                diff
            );
        }
    }

    #[test]
    fn test_time_shift() {
        let scheduler = FlowMatchEulerDiscreteScheduler::default_config();

        // At mu=0, time_shift should return t unchanged
        let t = scheduler.time_shift(0.0, 0.5);
        assert!((t - 0.5).abs() < 1e-6);

        // At larger mu, output should be larger (biased towards higher noise)
        let t_shifted = scheduler.time_shift(1.0, 0.5);
        assert!(t_shifted > 0.5);
    }

    #[test]
    fn test_scheduler_step() -> Result<()> {
        use candle::Device;

        let device = Device::Cpu;
        let config = SchedulerConfig::default();
        let mut scheduler = FlowMatchEulerDiscreteScheduler::new(&config);
        scheduler.use_dynamic_shifting = false;
        scheduler.set_timesteps(10, None, None);

        // Create dummy tensors
        let sample = Tensor::randn(0f32, 1f32, (1, 64, 64), &device)?;
        let model_output = Tensor::randn(0f32, 1f32, (1, 64, 64), &device)?;

        // Take a step
        let prev_sample = scheduler.step(&model_output, &sample)?;
        assert_eq!(prev_sample.dims(), sample.dims());
        assert_eq!(scheduler.step_index(), 1);

        Ok(())
    }

    #[test]
    fn test_scale_noise() -> Result<()> {
        use candle::Device;

        let device = Device::Cpu;
        let scheduler = FlowMatchEulerDiscreteScheduler::default_config();

        let sample = Tensor::ones((1, 4, 4), candle::DType::F32, &device)?;
        let noise = Tensor::zeros((1, 4, 4), candle::DType::F32, &device)?;

        // At sigma=0, should return clean sample
        let noised = scheduler.scale_noise(&sample, &noise, 0.0)?;
        let diff = (&sample - &noised)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-6);

        // At sigma=1, should return pure noise
        let noised = scheduler.scale_noise(&sample, &noise, 1.0)?;
        let max_val = noised.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(max_val < 1e-6); // noise was zeros

        Ok(())
    }
}
