//! Token-Duration Transducer (TDT) Decoder for Parakeet
//!
//! TDT is the key innovation that makes Parakeet fast. Unlike standard RNNT
//! that processes frame-by-frame, TDT predicts both a token AND a duration,
//! allowing it to skip multiple frames at once (~2.8x faster inference).
//!
//! The predictor uses a 2-layer LSTM to maintain context from previously
//! emitted tokens, which is critical for proper decoding.

use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{embedding, Embedding, Linear, VarBuilder};
use candle_nn::rnn::{LSTMConfig, LSTMState, LSTM, RNN};

use super::{DecoderConfig, DecodingResult, EncoderConfig, TokenInfo};

fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    let bias = vb.get(out_dim, "bias")?;
    Ok(Linear::new(weight, Some(bias)))
}

/// LSTM state for the 2-layer predictor
#[derive(Debug, Clone)]
pub struct PredictorState {
    pub layer0: LSTMState,
    pub layer1: LSTMState,
}

impl PredictorState {
    pub fn zero(hidden_size: usize, batch_size: usize, dtype: DType, device: &Device) -> Result<Self> {
        let zeros = Tensor::zeros((batch_size, hidden_size), dtype, device)?;
        Ok(Self {
            layer0: LSTMState::new(zeros.clone(), zeros.clone()),
            layer1: LSTMState::new(zeros.clone(), zeros),
        })
    }
}

/// TDT Predictor with 2-layer LSTM
///
/// NeMo structure:
/// - decoder.prediction.embed.weight: [vocab_size+1, hidden_size]
/// - decoder.prediction.lstm.weight_ih_l0/l1, weight_hh_l0/l1, bias_ih_l0/l1, bias_hh_l0/l1
#[derive(Debug, Clone)]
pub struct TdtPredictor {
    embedding: Embedding,
    lstm0: LSTM,
    lstm1: LSTM,
    hidden_size: usize,
    device: Device,
    dtype: DType,
}

impl TdtPredictor {
    pub fn load(vb: VarBuilder, cfg: &DecoderConfig) -> Result<Self> {
        // NeMo: decoder.prediction.embed.weight: [8193, 640]
        // vocab_size + 1 for blank/padding
        let embedding = embedding(
            cfg.vocab_size + 1,
            cfg.predictor_hidden_size,
            vb.pp("embed"),
        )?;

        // NeMo: decoder.prediction.dec_rnn.lstm - 2-layer LSTM
        let lstm_vb = vb.pp("dec_rnn").pp("lstm");

        // Layer 0: input is embedding, output is hidden_size
        let config0 = LSTMConfig {
            layer_idx: 0,
            ..Default::default()
        };
        let lstm0 = LSTM::new(
            cfg.predictor_hidden_size,
            cfg.predictor_hidden_size,
            config0,
            lstm_vb.clone(),
        )?;

        // Layer 1: input is hidden_size, output is hidden_size
        let config1 = LSTMConfig {
            layer_idx: 1,
            ..Default::default()
        };
        let lstm1 = LSTM::new(
            cfg.predictor_hidden_size,
            cfg.predictor_hidden_size,
            config1,
            lstm_vb,
        )?;

        let device = vb.device().clone();
        let dtype = vb.dtype();

        Ok(Self {
            embedding,
            lstm0,
            lstm1,
            hidden_size: cfg.predictor_hidden_size,
            device,
            dtype,
        })
    }

    /// Get initial zero state for the LSTM
    pub fn zero_state(&self, batch_size: usize) -> Result<PredictorState> {
        PredictorState::zero(self.hidden_size, batch_size, self.dtype, &self.device)
    }

    /// Forward pass through embedding and LSTM layers
    /// Returns (output, new_state)
    pub fn forward(&self, tokens: &Tensor, state: &PredictorState) -> Result<(Tensor, PredictorState)> {
        // tokens: [batch, 1] - last emitted token
        let embedded = self.embedding.forward(tokens)?;

        // embedded: [batch, 1, hidden_size] -> squeeze to [batch, hidden_size]
        let embedded = embedded.squeeze(1)?;

        // LSTM layer 0
        let state0 = self.lstm0.step(&embedded, &state.layer0)?;

        // LSTM layer 1
        let state1 = self.lstm1.step(&state0.h, &state.layer1)?;

        // Output is the hidden state of layer 1, unsqueeze to [batch, 1, hidden_size]
        let output = state1.h.unsqueeze(1)?;

        let new_state = PredictorState {
            layer0: state0,
            layer1: state1,
        };

        Ok((output, new_state))
    }
}

/// TDT Joint Network
/// NeMo structure: enc projection, pred projection, joint_net with combined output
#[derive(Debug, Clone)]
pub struct TdtJointNetwork {
    encoder_proj: Linear,
    predictor_proj: Linear,
    output_linear: Linear, // Combined token + duration output
    vocab_size: usize,
    num_durations: usize,
}

impl TdtJointNetwork {
    pub fn load(vb: VarBuilder, encoder_cfg: &EncoderConfig, decoder_cfg: &DecoderConfig) -> Result<Self> {
        // NeMo: joint.enc.weight: [640, 1024], joint.enc.bias: [640]
        let encoder_proj = linear(
            encoder_cfg.hidden_size,
            decoder_cfg.joint_hidden_size,
            vb.pp("enc"),
        )?;

        // NeMo: joint.pred.weight: [640, 640], joint.pred.bias: [640]
        let predictor_proj = linear(
            decoder_cfg.predictor_hidden_size,
            decoder_cfg.joint_hidden_size,
            vb.pp("pred"),
        )?;

        // NeMo: joint.joint_net.2.weight: [8198, 640]
        // 8198 = 8193 tokens (8192 vocab + 1 blank) + 5 durations
        let num_tokens = decoder_cfg.vocab_size + 1;  // +1 for blank
        let num_durations = decoder_cfg.max_duration + 1;
        let num_outputs = num_tokens + num_durations;
        let output_linear = linear(
            decoder_cfg.joint_hidden_size,
            num_outputs,
            vb.pp("joint_net.2"),
        )?;

        // Debug: verify weight loading (use dynamic blank index)
        let blank_idx = decoder_cfg.vocab_size;  // blank is at vocab_size
        println!("\n=== Joint Network Weight Verification ===");
        println!("vocab_size={}, blank_idx={}, num_outputs={}", decoder_cfg.vocab_size, blank_idx, num_outputs);
        let enc_w = encoder_proj.weight();
        let out_w = output_linear.weight();
        let out_b = output_linear.bias().unwrap();

        // Check enc weight first values
        let enc_first5: Vec<f32> = enc_w.i((0, 0..5))?.to_vec1()?;
        println!("enc_weight[0, :5]: {:?}", enc_first5);

        // Check output weight for blank (dynamic index)
        let blank_row: Vec<f32> = out_w.i((blank_idx, 0..5))?.to_vec1()?;
        println!("out_weight[{}, :5] (blank): {:?}", blank_idx, blank_row);

        // Check biases for blank
        let blank_bias: f32 = out_b.i(blank_idx)?.to_scalar()?;
        println!("out_bias[{}] (blank): {:.4}", blank_idx, blank_bias);
        println!("=========================================\n");

        Ok(Self {
            encoder_proj,
            predictor_proj,
            output_linear,
            vocab_size: num_tokens,  // includes blank
            num_durations,
        })
    }

    pub fn forward(
        &self,
        encoder_state: &Tensor,
        predictor_state: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Project to joint space
        let enc = self.encoder_proj.forward(encoder_state)?;
        let pred = self.predictor_proj.forward(predictor_state)?;

        // Combine (additive) and apply activation
        let sum = (enc + pred)?;
        let joint = sum.relu()?;

        // Get combined output
        let output = self.output_linear.forward(&joint)?;

        // Split into token logits and duration logits
        // output: [batch, seq, vocab_size + num_durations + 1]
        let token_logits = output.narrow(D::Minus1, 0, self.vocab_size)?;
        let duration_logits = output.narrow(D::Minus1, self.vocab_size, self.num_durations)?;

        Ok((token_logits, duration_logits))
    }
}

/// TDT Decoder - full decoder with greedy decoding
#[derive(Debug, Clone)]
pub struct TdtDecoder {
    predictor: TdtPredictor,
    joint: TdtJointNetwork,
    blank_id: u32,
    max_duration: usize,
}

impl TdtDecoder {
    pub fn load(vb: VarBuilder, encoder_cfg: &EncoderConfig, decoder_cfg: &DecoderConfig) -> Result<Self> {
        // NeMo structure: decoder.prediction.*, joint.*
        let predictor = TdtPredictor::load(vb.pp("decoder").pp("prediction"), decoder_cfg)?;
        let joint = TdtJointNetwork::load(vb.pp("joint"), encoder_cfg, decoder_cfg)?;

        Ok(Self {
            predictor,
            joint,
            blank_id: decoder_cfg.blank_id,
            max_duration: decoder_cfg.max_duration,
        })
    }

    /// Greedy TDT decoding - the key speedup over standard RNNT
    ///
    /// The TDT decoder works as follows (per the ICML 2023 TDT paper):
    /// 1. Initialize predictor by passing blank token through LSTM
    /// 2. At each frame, combine encoder state with predictor state in joint network
    /// 3. If non-blank token is emitted, update predictor with new token
    /// 4. Advance frame by predicted duration (can be 0, allowing multiple tokens per frame)
    ///
    /// Key difference from standard RNNT: TDT predicts both token AND duration.
    /// When duration=0, we stay on the same frame and can emit more tokens.
    /// This allows capturing rapid speech sequences that span multiple tokens per frame.
    pub fn decode_greedy(&self, encoder_output: &Tensor) -> Result<Vec<u32>> {
        let num_frames = encoder_output.dim(1)?;
        let device = encoder_output.device();

        let mut tokens = Vec::new();
        let mut t = 0;

        // Initialize LSTM state with zeros
        let mut lstm_state = self.predictor.zero_state(1)?;

        // IMPORTANT: Pass blank token through LSTM to get proper initial state.
        // Even though blank embedding is zeros, the LSTM biases produce meaningful
        // output that significantly affects joint network predictions.
        let blank_token = Tensor::new(&[self.blank_id], device)?.unsqueeze(0)?;
        let (pred_output_init, lstm_state_init) = self.predictor.forward(&blank_token, &lstm_state)?;
        let mut pred_output = pred_output_init;
        lstm_state = lstm_state_init;

        // Maximum symbols per frame to prevent infinite loops when duration=0
        const MAX_SYMBOLS_PER_FRAME: usize = 10;

        while t < num_frames {
            // Get encoder state at frame t
            let enc_state = encoder_output.narrow(1, t, 1)?;

            // Track symbols emitted on current frame (for safety limit)
            let mut symbols_on_frame = 0;

            loop {
                // Joint network produces token and duration distributions
                let (token_logits, duration_logits) = self.joint.forward(&enc_state, &pred_output)?;

                // Get best token (greedy)
                // token_logits: [batch=1, seq=1, vocab_size] -> squeeze to get [vocab_size]
                let token = token_logits
                    .squeeze(0)?
                    .squeeze(0)?
                    .argmax(D::Minus1)?
                    .to_scalar::<u32>()?;

                // Get predicted duration (index into durations array [0,1,2,3,4])
                let duration = duration_logits
                    .squeeze(0)?
                    .squeeze(0)?
                    .argmax(D::Minus1)?
                    .to_scalar::<u32>()? as usize;

                if token != self.blank_id {
                    tokens.push(token);
                    symbols_on_frame += 1;

                    // Update predictor state with emitted token
                    let pred_input = Tensor::new(&[token], device)?.unsqueeze(0)?;
                    let (new_output, new_state) = self.predictor.forward(&pred_input, &lstm_state)?;
                    pred_output = new_output;
                    lstm_state = new_state;
                }

                // Advance frame by predicted duration
                // Key TDT insight: duration=0 means stay on same frame for more tokens
                if duration > 0 {
                    t += duration;
                    break;
                }

                // Duration is 0 - stay on same frame, but check safety limit
                if symbols_on_frame >= MAX_SYMBOLS_PER_FRAME {
                    // Force advance to prevent infinite loop
                    t += 1;
                    break;
                }

                // If blank was predicted with duration=0, that's unusual but advance anyway
                if token == self.blank_id {
                    t += 1;
                    break;
                }

                // Otherwise continue on same frame with updated predictor state
            }
        }

        Ok(tokens)
    }

    /// Beam search decoding for better accuracy
    ///
    /// Each beam tracks:
    /// - tokens: sequence of emitted non-blank tokens
    /// - lstm_state: LSTM hidden state after last token
    /// - pred_output: predictor output corresponding to current state
    /// - score: accumulated log probability
    /// - frame: current frame position
    /// - symbols_on_frame: count of symbols emitted on current frame (for duration=0)
    pub fn decode_beam(
        &self,
        encoder_output: &Tensor,
        beam_width: usize,
    ) -> Result<Vec<u32>> {
        let num_frames = encoder_output.dim(1)?;
        let device = encoder_output.device();

        // Maximum symbols per frame to prevent infinite loops when duration=0
        const MAX_SYMBOLS_PER_FRAME: usize = 10;

        struct BeamState {
            tokens: Vec<u32>,
            lstm_state: PredictorState,
            pred_output: Tensor,
            score: f32,
            frame: usize,
            symbols_on_frame: usize,
        }

        impl Clone for BeamState {
            fn clone(&self) -> Self {
                Self {
                    tokens: self.tokens.clone(),
                    lstm_state: self.lstm_state.clone(),
                    pred_output: self.pred_output.clone(),
                    score: self.score,
                    frame: self.frame,
                    symbols_on_frame: self.symbols_on_frame,
                }
            }
        }

        // Initialize LSTM state with zeros, then pass blank token to get proper initial state
        let initial_lstm_state = self.predictor.zero_state(1)?;
        let blank_token = Tensor::new(&[self.blank_id], device)?.unsqueeze(0)?;
        let (initial_pred_output, initial_lstm_state) =
            self.predictor.forward(&blank_token, &initial_lstm_state)?;

        let mut beams = vec![BeamState {
            tokens: vec![],
            lstm_state: initial_lstm_state,
            pred_output: initial_pred_output,
            score: 0.0,
            frame: 0,
            symbols_on_frame: 0,
        }];

        while beams.iter().any(|b| b.frame < num_frames) {
            let mut new_beams = Vec::new();

            for beam in &beams {
                if beam.frame >= num_frames {
                    new_beams.push(beam.clone());
                    continue;
                }

                let enc_state = encoder_output.narrow(1, beam.frame, 1)?;
                let (token_logits, duration_logits) =
                    self.joint.forward(&enc_state, &beam.pred_output)?;

                let token_log_probs =
                    candle_nn::ops::log_softmax(&token_logits.squeeze(1)?, D::Minus1)?;
                let duration_log_probs =
                    candle_nn::ops::log_softmax(&duration_logits.squeeze(1)?, D::Minus1)?;

                let token_log_probs_vec: Vec<f32> = token_log_probs.flatten_all()?.to_vec1()?;
                let mut token_scores: Vec<(usize, f32)> = token_log_probs_vec
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| (i, s))
                    .collect();
                token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                token_scores.truncate(beam_width);

                for (token_idx, token_score) in token_scores {
                    let token = token_idx as u32;

                    for d in 0..=self.max_duration {
                        let dur_score: f32 = duration_log_probs.i((0, d))?.to_scalar()?;
                        let total_score = beam.score + token_score + dur_score;

                        let (new_tokens, new_lstm_state, new_pred_output, new_symbols) = if token
                            != self.blank_id
                        {
                            // Emit token and update predictor
                            let mut tokens = beam.tokens.clone();
                            tokens.push(token);
                            let pred_input = Tensor::new(&[token], device)?.unsqueeze(0)?;
                            let (pred_output, lstm_state) =
                                self.predictor.forward(&pred_input, &beam.lstm_state)?;
                            (tokens, lstm_state, pred_output, beam.symbols_on_frame + 1)
                        } else {
                            // Blank - keep same state
                            (
                                beam.tokens.clone(),
                                beam.lstm_state.clone(),
                                beam.pred_output.clone(),
                                beam.symbols_on_frame,
                            )
                        };

                        // Calculate new frame based on duration
                        // Duration=0 means stay on same frame (unless we hit safety limit)
                        let (new_frame, reset_symbols) = if d > 0 {
                            (beam.frame + d, true)
                        } else if new_symbols >= MAX_SYMBOLS_PER_FRAME || token == self.blank_id {
                            // Safety limit or blank with duration=0: force advance
                            (beam.frame + 1, true)
                        } else {
                            // Stay on same frame
                            (beam.frame, false)
                        };

                        new_beams.push(BeamState {
                            tokens: new_tokens,
                            lstm_state: new_lstm_state,
                            pred_output: new_pred_output,
                            score: total_score,
                            frame: new_frame,
                            symbols_on_frame: if reset_symbols { 0 } else { new_symbols },
                        });
                    }
                }
            }

            new_beams.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            new_beams.truncate(beam_width);
            beams = new_beams;
        }

        Ok(beams.first().map(|b| b.tokens.clone()).unwrap_or_default())
    }

    /// Greedy TDT decoding with rich results (timestamps, confidence)
    ///
    /// Same algorithm as `decode_greedy` but captures frame indices and log probabilities
    /// for each emitted token.
    pub fn decode_greedy_with_info(&self, encoder_output: &Tensor) -> Result<DecodingResult> {
        let num_frames = encoder_output.dim(1)?;
        let device = encoder_output.device();

        let mut result = DecodingResult::new(num_frames);
        let mut t = 0;

        // Initialize LSTM state with zeros
        let mut lstm_state = self.predictor.zero_state(1)?;

        // Pass blank token through LSTM to get proper initial state
        let blank_token = Tensor::new(&[self.blank_id], device)?.unsqueeze(0)?;
        let (pred_output_init, lstm_state_init) = self.predictor.forward(&blank_token, &lstm_state)?;
        let mut pred_output = pred_output_init;
        lstm_state = lstm_state_init;

        // Maximum symbols per frame to prevent infinite loops when duration=0
        const MAX_SYMBOLS_PER_FRAME: usize = 10;

        while t < num_frames {
            // Get encoder state at frame t
            let enc_state = encoder_output.narrow(1, t, 1)?;

            // Track symbols emitted on current frame (for safety limit)
            let mut symbols_on_frame = 0;

            loop {
                // Joint network produces token and duration distributions
                let (token_logits, duration_logits) = self.joint.forward(&enc_state, &pred_output)?;

                // Compute log softmax for confidence scores
                let token_log_probs =
                    candle_nn::ops::log_softmax(&token_logits.squeeze(0)?.squeeze(0)?, D::Minus1)?;

                // Get best token (greedy)
                let token = token_log_probs.argmax(D::Minus1)?.to_scalar::<u32>()?;

                // Get the log probability for this token
                let log_prob: f32 = token_log_probs.i(token as usize)?.to_scalar()?;

                // Get predicted duration (index into durations array [0,1,2,3,4])
                let duration = duration_logits
                    .squeeze(0)?
                    .squeeze(0)?
                    .argmax(D::Minus1)?
                    .to_scalar::<u32>()? as usize;

                if token != self.blank_id {
                    // Record token with frame info and confidence
                    let start_frame = t;
                    let end_frame = t + duration.max(1); // End frame is at least 1 after start

                    result.tokens.push(TokenInfo {
                        token_id: token,
                        log_prob,
                        start_frame,
                        end_frame,
                    });
                    result.total_log_prob += log_prob;

                    symbols_on_frame += 1;

                    // Update predictor state with emitted token
                    let pred_input = Tensor::new(&[token], device)?.unsqueeze(0)?;
                    let (new_output, new_state) = self.predictor.forward(&pred_input, &lstm_state)?;
                    pred_output = new_output;
                    lstm_state = new_state;
                }

                // Advance frame by predicted duration
                if duration > 0 {
                    t += duration;
                    break;
                }

                // Duration is 0 - stay on same frame, but check safety limit
                if symbols_on_frame >= MAX_SYMBOLS_PER_FRAME {
                    t += 1;
                    break;
                }

                // If blank was predicted with duration=0, advance anyway
                if token == self.blank_id {
                    t += 1;
                    break;
                }

                // Otherwise continue on same frame with updated predictor state
            }
        }

        Ok(result)
    }

    /// Beam search decoding with rich results (timestamps, confidence)
    ///
    /// Same algorithm as `decode_beam` but captures frame indices and log probabilities.
    pub fn decode_beam_with_info(
        &self,
        encoder_output: &Tensor,
        beam_width: usize,
    ) -> Result<DecodingResult> {
        let num_frames = encoder_output.dim(1)?;
        let device = encoder_output.device();

        // Maximum symbols per frame to prevent infinite loops when duration=0
        const MAX_SYMBOLS_PER_FRAME: usize = 10;

        struct BeamState {
            tokens: Vec<TokenInfo>,
            lstm_state: PredictorState,
            pred_output: Tensor,
            score: f32,
            frame: usize,
            symbols_on_frame: usize,
        }

        impl Clone for BeamState {
            fn clone(&self) -> Self {
                Self {
                    tokens: self.tokens.clone(),
                    lstm_state: self.lstm_state.clone(),
                    pred_output: self.pred_output.clone(),
                    score: self.score,
                    frame: self.frame,
                    symbols_on_frame: self.symbols_on_frame,
                }
            }
        }

        // Initialize LSTM state with zeros, then pass blank token to get proper initial state
        let initial_lstm_state = self.predictor.zero_state(1)?;
        let blank_token = Tensor::new(&[self.blank_id], device)?.unsqueeze(0)?;
        let (initial_pred_output, initial_lstm_state) =
            self.predictor.forward(&blank_token, &initial_lstm_state)?;

        let mut beams = vec![BeamState {
            tokens: vec![],
            lstm_state: initial_lstm_state,
            pred_output: initial_pred_output,
            score: 0.0,
            frame: 0,
            symbols_on_frame: 0,
        }];

        while beams.iter().any(|b| b.frame < num_frames) {
            let mut new_beams = Vec::new();

            for beam in &beams {
                if beam.frame >= num_frames {
                    new_beams.push(beam.clone());
                    continue;
                }

                let enc_state = encoder_output.narrow(1, beam.frame, 1)?;
                let (token_logits, duration_logits) =
                    self.joint.forward(&enc_state, &beam.pred_output)?;

                let token_log_probs =
                    candle_nn::ops::log_softmax(&token_logits.squeeze(1)?, D::Minus1)?;
                let duration_log_probs =
                    candle_nn::ops::log_softmax(&duration_logits.squeeze(1)?, D::Minus1)?;

                let token_log_probs_vec: Vec<f32> = token_log_probs.flatten_all()?.to_vec1()?;
                let mut token_scores: Vec<(usize, f32)> = token_log_probs_vec
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| (i, s))
                    .collect();
                token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                token_scores.truncate(beam_width);

                for (token_idx, token_score) in token_scores {
                    let token = token_idx as u32;

                    for d in 0..=self.max_duration {
                        let dur_score: f32 = duration_log_probs.i((0, d))?.to_scalar()?;
                        let total_score = beam.score + token_score + dur_score;

                        let (new_tokens, new_lstm_state, new_pred_output, new_symbols) = if token
                            != self.blank_id
                        {
                            // Emit token and update predictor
                            let mut tokens = beam.tokens.clone();

                            let start_frame = beam.frame;
                            let end_frame = beam.frame + d.max(1);

                            tokens.push(TokenInfo {
                                token_id: token,
                                log_prob: token_score,
                                start_frame,
                                end_frame,
                            });

                            let pred_input = Tensor::new(&[token], device)?.unsqueeze(0)?;
                            let (pred_output, lstm_state) =
                                self.predictor.forward(&pred_input, &beam.lstm_state)?;
                            (tokens, lstm_state, pred_output, beam.symbols_on_frame + 1)
                        } else {
                            // Blank - keep same state
                            (
                                beam.tokens.clone(),
                                beam.lstm_state.clone(),
                                beam.pred_output.clone(),
                                beam.symbols_on_frame,
                            )
                        };

                        // Calculate new frame based on duration
                        let (new_frame, reset_symbols) = if d > 0 {
                            (beam.frame + d, true)
                        } else if new_symbols >= MAX_SYMBOLS_PER_FRAME || token == self.blank_id {
                            (beam.frame + 1, true)
                        } else {
                            (beam.frame, false)
                        };

                        new_beams.push(BeamState {
                            tokens: new_tokens,
                            lstm_state: new_lstm_state,
                            pred_output: new_pred_output,
                            score: total_score,
                            frame: new_frame,
                            symbols_on_frame: if reset_symbols { 0 } else { new_symbols },
                        });
                    }
                }
            }

            new_beams.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            new_beams.truncate(beam_width);
            beams = new_beams;
        }

        let best_beam = beams.first();
        Ok(DecodingResult {
            tokens: best_beam.map(|b| b.tokens.clone()).unwrap_or_default(),
            total_log_prob: best_beam.map(|b| b.score).unwrap_or(0.0),
            num_frames,
        })
    }

    /// MALSD (Modified Alignment-Length Synchronous Decoding)
    ///
    /// Unlike standard beam search which synchronizes by time, MALSD synchronizes
    /// by alignment length k = t + u (time frame + number of emitted tokens).
    /// This allows all beams at the same alignment level to be processed together,
    /// potentially improving accuracy.
    ///
    /// Reference: "Alignment-Length Synchronous Decoding for RNN Transducer"
    pub fn decode_malsd(&self, encoder_output: &Tensor, beam_width: usize) -> Result<DecodingResult> {
        let num_frames = encoder_output.dim(1)?;
        let device = encoder_output.device();

        // Maximum alignment length: T + max possible tokens
        // In practice, we can't emit more tokens than frames
        let max_alignment = num_frames * 2;

        // Maximum symbols per alignment step (to prevent runaway)
        const MAX_SYMBOLS_PER_ALIGNMENT: usize = 5;

        #[derive(Clone)]
        struct MalsdBeamState {
            tokens: Vec<TokenInfo>,
            lstm_state: PredictorState,
            pred_output: Tensor,
            score: f32,
            frame: usize,      // t: current time frame
            num_tokens: usize, // u: number of tokens emitted
        }

        // Initialize
        let initial_lstm_state = self.predictor.zero_state(1)?;
        let blank_token = Tensor::new(&[self.blank_id], device)?.unsqueeze(0)?;
        let (initial_pred_output, initial_lstm_state) =
            self.predictor.forward(&blank_token, &initial_lstm_state)?;

        // Beams organized by alignment length k = t + u
        let mut beams_by_alignment: Vec<Vec<MalsdBeamState>> = vec![Vec::new(); max_alignment + 1];

        beams_by_alignment[0].push(MalsdBeamState {
            tokens: vec![],
            lstm_state: initial_lstm_state,
            pred_output: initial_pred_output,
            score: 0.0,
            frame: 0,
            num_tokens: 0,
        });

        let mut best_completed: Option<MalsdBeamState> = None;

        // Process alignment levels in order
        for k in 0..max_alignment {
            let current_beams = std::mem::take(&mut beams_by_alignment[k]);

            for beam in current_beams {
                // Check if this beam is complete (all frames processed)
                if beam.frame >= num_frames {
                    if best_completed.is_none()
                        || beam.score > best_completed.as_ref().unwrap().score
                    {
                        best_completed = Some(beam);
                    }
                    continue;
                }

                let enc_state = encoder_output.narrow(1, beam.frame, 1)?;
                let (token_logits, duration_logits) =
                    self.joint.forward(&enc_state, &beam.pred_output)?;

                let token_log_probs =
                    candle_nn::ops::log_softmax(&token_logits.squeeze(1)?, D::Minus1)?;
                let duration_log_probs =
                    candle_nn::ops::log_softmax(&duration_logits.squeeze(1)?, D::Minus1)?;

                let token_log_probs_vec: Vec<f32> = token_log_probs.flatten_all()?.to_vec1()?;
                let mut token_scores: Vec<(usize, f32)> = token_log_probs_vec
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| (i, s))
                    .collect();
                token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                token_scores.truncate(beam_width);

                for (token_idx, token_score) in token_scores {
                    let token = token_idx as u32;

                    // Consider all durations
                    for d in 0..=self.max_duration {
                        let dur_score: f32 = duration_log_probs.i((0, d))?.to_scalar()?;
                        let total_score = beam.score + token_score + dur_score;

                        let (new_tokens, new_lstm_state, new_pred_output, new_num_tokens) =
                            if token != self.blank_id {
                                let mut tokens = beam.tokens.clone();
                                tokens.push(TokenInfo {
                                    token_id: token,
                                    log_prob: token_score,
                                    start_frame: beam.frame,
                                    end_frame: beam.frame + d.max(1),
                                });

                                let pred_input = Tensor::new(&[token], device)?.unsqueeze(0)?;
                                let (pred_output, lstm_state) =
                                    self.predictor.forward(&pred_input, &beam.lstm_state)?;
                                (tokens, lstm_state, pred_output, beam.num_tokens + 1)
                            } else {
                                (
                                    beam.tokens.clone(),
                                    beam.lstm_state.clone(),
                                    beam.pred_output.clone(),
                                    beam.num_tokens,
                                )
                            };

                        let new_frame = if d > 0 { beam.frame + d } else { beam.frame + 1 };

                        // Calculate new alignment: k' = t' + u'
                        let new_alignment = new_frame + new_num_tokens;

                        if new_alignment <= max_alignment {
                            let target_beams = &mut beams_by_alignment[new_alignment];

                            // Keep beams bounded
                            target_beams.push(MalsdBeamState {
                                tokens: new_tokens,
                                lstm_state: new_lstm_state,
                                pred_output: new_pred_output,
                                score: total_score,
                                frame: new_frame,
                                num_tokens: new_num_tokens,
                            });

                            // Prune if too many beams at this alignment
                            if target_beams.len() > beam_width * MAX_SYMBOLS_PER_ALIGNMENT {
                                target_beams.sort_by(|a, b| {
                                    b.score.partial_cmp(&a.score).unwrap()
                                });
                                target_beams.truncate(beam_width);
                            }
                        }
                    }
                }
            }

            // Prune beams at alignment k+1 after processing k
            if k + 1 <= max_alignment {
                let target = &mut beams_by_alignment[k + 1];
                if target.len() > beam_width {
                    target.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
                    target.truncate(beam_width);
                }
            }
        }

        // Return best completed beam
        let best_beam = best_completed;
        Ok(DecodingResult {
            tokens: best_beam
                .as_ref()
                .map(|b| b.tokens.clone())
                .unwrap_or_default(),
            total_log_prob: best_beam.as_ref().map(|b| b.score).unwrap_or(0.0),
            num_frames,
        })
    }

    /// Beam search with N-gram LM shallow fusion
    ///
    /// Applies LM scoring at word boundaries (tokens starting with `▁`).
    /// The final score is: acoustic_score + lm_weight * lm_score
    ///
    /// # Arguments
    /// * `encoder_output` - Encoder output tensor
    /// * `beam_width` - Number of beams to keep
    /// * `lm` - N-gram language model
    /// * `vocab` - ASR vocabulary (token ID -> token string)
    /// * `lm_weight` - Weight for LM scores (typically 0.3-0.7)
    pub fn decode_beam_with_lm(
        &self,
        encoder_output: &Tensor,
        beam_width: usize,
        lm: &super::lm::NgramLM,
        vocab: &[String],
        lm_weight: f32,
    ) -> Result<DecodingResult> {
        use super::lm::NgramLM;

        let num_frames = encoder_output.dim(1)?;
        let device = encoder_output.device();

        const MAX_SYMBOLS_PER_FRAME: usize = 10;
        const MAX_LM_CONTEXT: usize = 4; // Keep last N words for LM context

        #[derive(Clone)]
        struct LmBeamState {
            tokens: Vec<TokenInfo>,
            lstm_state: PredictorState,
            pred_output: Tensor,
            acoustic_score: f32,
            lm_score: f32,
            frame: usize,
            symbols_on_frame: usize,
            // For LM scoring
            current_word: String,       // Word being built from tokens
            lm_context: Vec<u32>,       // Previous LM word IDs
        }

        impl LmBeamState {
            fn total_score(&self, lm_weight: f32) -> f32 {
                self.acoustic_score + lm_weight * self.lm_score
            }
        }

        // Helper to score a completed word with LM
        fn score_word(lm: &NgramLM, context: &[u32], word: &str) -> (f32, Option<u32>) {
            // Look up word in LM vocab
            if let Some(word_id) = lm.word_to_id(word) {
                let score = lm.score(context, word_id);
                // Convert log10 to natural log for consistency
                (NgramLM::log10_to_ln(score), Some(word_id))
            } else {
                // Unknown word - use small penalty
                (-5.0, None)
            }
        }

        // Initialize
        let initial_lstm_state = self.predictor.zero_state(1)?;
        let blank_token = Tensor::new(&[self.blank_id], device)?.unsqueeze(0)?;
        let (initial_pred_output, initial_lstm_state) =
            self.predictor.forward(&blank_token, &initial_lstm_state)?;

        let mut beams = vec![LmBeamState {
            tokens: vec![],
            lstm_state: initial_lstm_state,
            pred_output: initial_pred_output,
            acoustic_score: 0.0,
            lm_score: 0.0,
            frame: 0,
            symbols_on_frame: 0,
            current_word: String::new(),
            lm_context: vec![],
        }];

        while beams.iter().any(|b| b.frame < num_frames) {
            let mut new_beams = Vec::new();

            for beam in &beams {
                if beam.frame >= num_frames {
                    new_beams.push(beam.clone());
                    continue;
                }

                let enc_state = encoder_output.narrow(1, beam.frame, 1)?;
                let (token_logits, duration_logits) =
                    self.joint.forward(&enc_state, &beam.pred_output)?;

                let token_log_probs =
                    candle_nn::ops::log_softmax(&token_logits.squeeze(1)?, D::Minus1)?;
                let duration_log_probs =
                    candle_nn::ops::log_softmax(&duration_logits.squeeze(1)?, D::Minus1)?;

                let token_log_probs_vec: Vec<f32> = token_log_probs.flatten_all()?.to_vec1()?;
                let mut token_scores: Vec<(usize, f32)> = token_log_probs_vec
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| (i, s))
                    .collect();
                token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                token_scores.truncate(beam_width * 2); // Keep more candidates for LM diversity

                for (token_idx, token_score) in token_scores {
                    let token = token_idx as u32;

                    for d in 0..=self.max_duration {
                        let dur_score: f32 = duration_log_probs.i((0, d))?.to_scalar()?;
                        let new_acoustic_score = beam.acoustic_score + token_score + dur_score;

                        let (
                            new_tokens,
                            new_lstm_state,
                            new_pred_output,
                            new_symbols,
                            new_current_word,
                            new_lm_context,
                            new_lm_score,
                        ) = if token != self.blank_id {
                            // Get token text
                            let token_text = vocab
                                .get(token as usize)
                                .map(|s| s.as_str())
                                .unwrap_or("");

                            let mut new_word = beam.current_word.clone();
                            let mut new_context = beam.lm_context.clone();
                            let mut lm_score = beam.lm_score;

                            // Check if this token starts a new word (SentencePiece convention)
                            let is_word_start = token_text.starts_with('▁');

                            if is_word_start && !new_word.is_empty() {
                                // Complete the previous word and score it
                                let (word_score, word_id) =
                                    score_word(lm, &new_context, &new_word.to_lowercase());
                                lm_score += word_score;

                                // Update context
                                if let Some(id) = word_id {
                                    new_context.push(id);
                                    if new_context.len() > MAX_LM_CONTEXT {
                                        new_context.remove(0);
                                    }
                                }

                                // Start new word (strip the ▁ prefix)
                                new_word = token_text.trim_start_matches('▁').to_string();
                            } else {
                                // Continue building current word
                                new_word.push_str(token_text.trim_start_matches('▁'));
                            }

                            // Create token info
                            let mut tokens = beam.tokens.clone();
                            tokens.push(TokenInfo {
                                token_id: token,
                                log_prob: token_score,
                                start_frame: beam.frame,
                                end_frame: beam.frame + d.max(1),
                            });

                            let pred_input = Tensor::new(&[token], device)?.unsqueeze(0)?;
                            let (pred_output, lstm_state) =
                                self.predictor.forward(&pred_input, &beam.lstm_state)?;

                            (
                                tokens,
                                lstm_state,
                                pred_output,
                                beam.symbols_on_frame + 1,
                                new_word,
                                new_context,
                                lm_score,
                            )
                        } else {
                            // Blank - keep same state
                            (
                                beam.tokens.clone(),
                                beam.lstm_state.clone(),
                                beam.pred_output.clone(),
                                beam.symbols_on_frame,
                                beam.current_word.clone(),
                                beam.lm_context.clone(),
                                beam.lm_score,
                            )
                        };

                        let (new_frame, reset_symbols) = if d > 0 {
                            (beam.frame + d, true)
                        } else if new_symbols >= MAX_SYMBOLS_PER_FRAME || token == self.blank_id {
                            (beam.frame + 1, true)
                        } else {
                            (beam.frame, false)
                        };

                        new_beams.push(LmBeamState {
                            tokens: new_tokens,
                            lstm_state: new_lstm_state,
                            pred_output: new_pred_output,
                            acoustic_score: new_acoustic_score,
                            lm_score: new_lm_score,
                            frame: new_frame,
                            symbols_on_frame: if reset_symbols { 0 } else { new_symbols },
                            current_word: new_current_word,
                            lm_context: new_lm_context,
                        });
                    }
                }
            }

            // Sort by combined score and prune
            new_beams.sort_by(|a, b| {
                b.total_score(lm_weight)
                    .partial_cmp(&a.total_score(lm_weight))
                    .unwrap()
            });
            new_beams.truncate(beam_width);
            beams = new_beams;
        }

        // Score final word for completed beams
        for beam in &mut beams {
            if !beam.current_word.is_empty() {
                let (word_score, _) =
                    score_word(lm, &beam.lm_context, &beam.current_word.to_lowercase());
                beam.lm_score += word_score;
            }
        }

        // Return best beam
        beams.sort_by(|a, b| {
            b.total_score(lm_weight)
                .partial_cmp(&a.total_score(lm_weight))
                .unwrap()
        });

        let best_beam = beams.first();
        Ok(DecodingResult {
            tokens: best_beam.map(|b| b.tokens.clone()).unwrap_or_default(),
            total_log_prob: best_beam
                .map(|b| b.total_score(lm_weight))
                .unwrap_or(0.0),
            num_frames,
        })
    }
}
