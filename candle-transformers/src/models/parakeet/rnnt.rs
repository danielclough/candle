//! RNN Transducer (RNN-T) Decoder for Parakeet
//!
//! RNN-T is the standard transducer architecture without duration prediction.
//! Unlike TDT which can skip frames, RNN-T processes every frame sequentially.
//!
//! The predictor uses a 2-layer LSTM (identical to TDT) to maintain context
//! from previously emitted tokens.

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

/// RNN-T Predictor with 2-layer LSTM
///
/// Identical to TDT predictor - the difference is in the joint network output.
///
/// NeMo structure:
/// - decoder.prediction.embed.weight: [vocab_size+1, hidden_size]
/// - decoder.prediction.lstm.weight_ih_l0/l1, weight_hh_l0/l1, bias_ih_l0/l1, bias_hh_l0/l1
#[derive(Debug, Clone)]
pub struct RnntPredictor {
    embedding: Embedding,
    lstm0: LSTM,
    lstm1: LSTM,
    hidden_size: usize,
    device: Device,
    dtype: DType,
}

impl RnntPredictor {
    pub fn load(vb: VarBuilder, cfg: &DecoderConfig) -> Result<Self> {
        // NeMo: decoder.prediction.embed.weight: [vocab_size+1, hidden_size]
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

/// RNN-T Joint Network
///
/// Unlike TDT, RNN-T joint network only outputs token logits (no duration).
/// Output size: vocab_size + 1 (including blank)
#[derive(Debug, Clone)]
pub struct RnntJointNetwork {
    encoder_proj: Linear,
    predictor_proj: Linear,
    output_linear: Linear,
    vocab_size: usize, // includes blank
}

impl RnntJointNetwork {
    pub fn load(vb: VarBuilder, encoder_cfg: &EncoderConfig, decoder_cfg: &DecoderConfig) -> Result<Self> {
        // NeMo: joint.enc.weight: [joint_hidden, encoder_hidden]
        let encoder_proj = linear(
            encoder_cfg.hidden_size,
            decoder_cfg.joint_hidden_size,
            vb.pp("enc"),
        )?;

        // NeMo: joint.pred.weight: [joint_hidden, predictor_hidden]
        let predictor_proj = linear(
            decoder_cfg.predictor_hidden_size,
            decoder_cfg.joint_hidden_size,
            vb.pp("pred"),
        )?;

        // RNN-T output: vocab_size + 1 (no duration logits)
        let num_outputs = decoder_cfg.vocab_size + 1;
        let output_linear = linear(
            decoder_cfg.joint_hidden_size,
            num_outputs,
            vb.pp("joint_net.2"),
        )?;

        Ok(Self {
            encoder_proj,
            predictor_proj,
            output_linear,
            vocab_size: num_outputs,
        })
    }

    /// Forward pass through joint network
    /// Returns token logits only (no duration logits unlike TDT)
    pub fn forward(
        &self,
        encoder_state: &Tensor,
        predictor_state: &Tensor,
    ) -> Result<Tensor> {
        // Project to joint space
        let enc = self.encoder_proj.forward(encoder_state)?;
        let pred = self.predictor_proj.forward(predictor_state)?;

        // Combine (additive) and apply activation
        let sum = (enc + pred)?;
        let joint = sum.relu()?;

        // Get token logits
        let output = self.output_linear.forward(&joint)?;

        Ok(output)
    }
}

/// RNN-T Decoder - full decoder with greedy decoding
#[derive(Debug, Clone)]
pub struct RnntDecoder {
    predictor: RnntPredictor,
    joint: RnntJointNetwork,
    blank_id: u32,
}

impl RnntDecoder {
    pub fn load(vb: VarBuilder, encoder_cfg: &EncoderConfig, decoder_cfg: &DecoderConfig) -> Result<Self> {
        // NeMo structure: decoder.prediction.*, joint.*
        let predictor = RnntPredictor::load(vb.pp("decoder").pp("prediction"), decoder_cfg)?;
        let joint = RnntJointNetwork::load(vb.pp("joint"), encoder_cfg, decoder_cfg)?;

        Ok(Self {
            predictor,
            joint,
            blank_id: decoder_cfg.blank_id,
        })
    }

    /// Greedy RNN-T decoding
    ///
    /// RNN-T decoding semantics:
    /// - At each frame, keep predicting tokens until blank is emitted
    /// - Only advance to the next frame when blank is predicted
    /// - This allows emitting multiple tokens per frame
    ///
    /// This is different from TDT which uses explicit duration prediction.
    pub fn decode_greedy(&self, encoder_output: &Tensor) -> Result<Vec<u32>> {
        let num_frames = encoder_output.dim(1)?;
        let device = encoder_output.device();

        let mut tokens = Vec::new();
        let mut t = 0;

        // Initialize LSTM state with zeros
        let mut lstm_state = self.predictor.zero_state(1)?;

        // Pass blank token through LSTM to get proper initial state
        let blank_token = Tensor::new(&[self.blank_id], device)?.unsqueeze(0)?;
        let (pred_output_init, lstm_state_init) = self.predictor.forward(&blank_token, &lstm_state)?;
        let mut pred_output = pred_output_init;
        lstm_state = lstm_state_init;

        // Maximum symbols per frame to prevent infinite loops
        const MAX_SYMBOLS_PER_FRAME: usize = 10;

        while t < num_frames {
            // Get encoder state at frame t
            let enc_state = encoder_output.narrow(1, t, 1)?;

            // Inner loop: keep emitting tokens until blank is predicted
            let mut symbols_emitted = 0;
            loop {
                // Joint network produces token distribution only
                let token_logits = self.joint.forward(&enc_state, &pred_output)?;

                // Get best token (greedy)
                let token = token_logits
                    .squeeze(0)?
                    .squeeze(0)?
                    .argmax(D::Minus1)?
                    .to_scalar::<u32>()?;

                if token == self.blank_id {
                    // Blank means "advance to next frame"
                    break;
                }

                // Emit non-blank token
                tokens.push(token);
                symbols_emitted += 1;

                // Update predictor state with the emitted token
                let pred_input = Tensor::new(&[token], device)?.unsqueeze(0)?;
                let (new_output, new_state) = self.predictor.forward(&pred_input, &lstm_state)?;
                pred_output = new_output;
                lstm_state = new_state;

                // Safety: prevent infinite loops
                if symbols_emitted >= MAX_SYMBOLS_PER_FRAME {
                    break;
                }
            }

            // Only advance frame after blank (or max symbols reached)
            t += 1;
        }

        Ok(tokens)
    }

    /// Beam search decoding for better accuracy
    pub fn decode_beam(
        &self,
        encoder_output: &Tensor,
        beam_width: usize,
    ) -> Result<Vec<u32>> {
        let num_frames = encoder_output.dim(1)?;
        let device = encoder_output.device();

        struct BeamState {
            tokens: Vec<u32>,
            lstm_state: PredictorState,
            pred_output: Tensor,
            score: f32,
            frame: usize,
        }

        impl Clone for BeamState {
            fn clone(&self) -> Self {
                Self {
                    tokens: self.tokens.clone(),
                    lstm_state: self.lstm_state.clone(),
                    pred_output: self.pred_output.clone(),
                    score: self.score,
                    frame: self.frame,
                }
            }
        }

        // Initialize LSTM state and pass blank token
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
        }];

        while beams.iter().any(|b| b.frame < num_frames) {
            let mut new_beams = Vec::new();

            for beam in &beams {
                if beam.frame >= num_frames {
                    new_beams.push(beam.clone());
                    continue;
                }

                let enc_state = encoder_output.narrow(1, beam.frame, 1)?;
                let token_logits = self.joint.forward(&enc_state, &beam.pred_output)?;

                let token_log_probs =
                    candle_nn::ops::log_softmax(&token_logits.squeeze(1)?, D::Minus1)?;

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
                    let total_score = beam.score + token_score;

                    let (new_tokens, new_lstm_state, new_pred_output, new_frame) = if token != self.blank_id {
                        // Emit token and update predictor, but STAY on same frame
                        let mut tokens = beam.tokens.clone();
                        tokens.push(token);
                        let pred_input = Tensor::new(&[token], device)?.unsqueeze(0)?;
                        let (pred_output, lstm_state) =
                            self.predictor.forward(&pred_input, &beam.lstm_state)?;
                        // Non-blank: stay on same frame to potentially emit more tokens
                        (tokens, lstm_state, pred_output, beam.frame)
                    } else {
                        // Blank - advance to next frame
                        (
                            beam.tokens.clone(),
                            beam.lstm_state.clone(),
                            beam.pred_output.clone(),
                            beam.frame + 1,  // Only blank advances the frame
                        )
                    };

                    new_beams.push(BeamState {
                        tokens: new_tokens,
                        lstm_state: new_lstm_state,
                        pred_output: new_pred_output,
                        score: total_score,
                        frame: new_frame,
                    });
                }
            }

            new_beams.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            new_beams.truncate(beam_width);
            beams = new_beams;
        }

        Ok(beams.first().map(|b| b.tokens.clone()).unwrap_or_default())
    }

    /// Greedy RNN-T decoding with rich results (timestamps, confidence)
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

        // Maximum symbols per frame to prevent infinite loops
        const MAX_SYMBOLS_PER_FRAME: usize = 10;

        while t < num_frames {
            // Get encoder state at frame t
            let enc_state = encoder_output.narrow(1, t, 1)?;

            // Inner loop: keep emitting tokens until blank is predicted
            let mut symbols_emitted = 0;
            loop {
                // Joint network produces token distribution only
                let token_logits = self.joint.forward(&enc_state, &pred_output)?;

                // Compute log softmax for confidence scores
                let token_log_probs =
                    candle_nn::ops::log_softmax(&token_logits.squeeze(0)?.squeeze(0)?, D::Minus1)?;

                // Get best token (greedy)
                let token = token_log_probs.argmax(D::Minus1)?.to_scalar::<u32>()?;

                // Get the log probability for this token
                let log_prob: f32 = token_log_probs.i(token as usize)?.to_scalar()?;

                if token == self.blank_id {
                    // Blank means "advance to next frame"
                    break;
                }

                // Record token with frame info and confidence
                // In RNNT, each token spans from its emission frame to the next frame
                result.tokens.push(TokenInfo {
                    token_id: token,
                    log_prob,
                    start_frame: t,
                    end_frame: t + 1, // RNNT advances 1 frame per blank
                });
                result.total_log_prob += log_prob;

                symbols_emitted += 1;

                // Update predictor state with the emitted token
                let pred_input = Tensor::new(&[token], device)?.unsqueeze(0)?;
                let (new_output, new_state) = self.predictor.forward(&pred_input, &lstm_state)?;
                pred_output = new_output;
                lstm_state = new_state;

                // Safety: prevent infinite loops
                if symbols_emitted >= MAX_SYMBOLS_PER_FRAME {
                    break;
                }
            }

            // Only advance frame after blank (or max symbols reached)
            t += 1;
        }

        Ok(result)
    }

    /// Beam search decoding with rich results (timestamps, confidence)
    pub fn decode_beam_with_info(
        &self,
        encoder_output: &Tensor,
        beam_width: usize,
    ) -> Result<DecodingResult> {
        let num_frames = encoder_output.dim(1)?;
        let device = encoder_output.device();

        struct BeamState {
            tokens: Vec<TokenInfo>,
            lstm_state: PredictorState,
            pred_output: Tensor,
            score: f32,
            frame: usize,
        }

        impl Clone for BeamState {
            fn clone(&self) -> Self {
                Self {
                    tokens: self.tokens.clone(),
                    lstm_state: self.lstm_state.clone(),
                    pred_output: self.pred_output.clone(),
                    score: self.score,
                    frame: self.frame,
                }
            }
        }

        // Initialize LSTM state and pass blank token
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
        }];

        while beams.iter().any(|b| b.frame < num_frames) {
            let mut new_beams = Vec::new();

            for beam in &beams {
                if beam.frame >= num_frames {
                    new_beams.push(beam.clone());
                    continue;
                }

                let enc_state = encoder_output.narrow(1, beam.frame, 1)?;
                let token_logits = self.joint.forward(&enc_state, &beam.pred_output)?;

                let token_log_probs =
                    candle_nn::ops::log_softmax(&token_logits.squeeze(1)?, D::Minus1)?;

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
                    let total_score = beam.score + token_score;

                    let (new_tokens, new_lstm_state, new_pred_output, new_frame) =
                        if token != self.blank_id {
                            // Emit token and update predictor, but STAY on same frame
                            let mut tokens = beam.tokens.clone();
                            tokens.push(TokenInfo {
                                token_id: token,
                                log_prob: token_score,
                                start_frame: beam.frame,
                                end_frame: beam.frame + 1,
                            });
                            let pred_input = Tensor::new(&[token], device)?.unsqueeze(0)?;
                            let (pred_output, lstm_state) =
                                self.predictor.forward(&pred_input, &beam.lstm_state)?;
                            // Non-blank: stay on same frame to potentially emit more tokens
                            (tokens, lstm_state, pred_output, beam.frame)
                        } else {
                            // Blank - advance to next frame
                            (
                                beam.tokens.clone(),
                                beam.lstm_state.clone(),
                                beam.pred_output.clone(),
                                beam.frame + 1, // Only blank advances the frame
                            )
                        };

                    new_beams.push(BeamState {
                        tokens: new_tokens,
                        lstm_state: new_lstm_state,
                        pred_output: new_pred_output,
                        score: total_score,
                        frame: new_frame,
                    });
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
}
