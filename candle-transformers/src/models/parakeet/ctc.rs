//! CTC (Connectionist Temporal Classification) Decoder for Parakeet
//!
//! CTC is a simpler decoder architecture compared to transducers (TDT/RNN-T).
//! It uses a single linear projection from encoder features to vocabulary logits,
//! then decodes by collapsing blanks and repeated tokens.
//!
//! NeMo uses Conv1d with kernel_size=1 which is mathematically equivalent to Linear.

use candle::{IndexOp, Module, Result, Tensor, D};
use candle_nn::{Linear, VarBuilder};

use super::{DecoderConfig, DecodingResult, EncoderConfig, TokenInfo};

/// CTC Decoder - simple frame-wise classification
///
/// NeMo structure:
/// - decoder.decoder_layers.0.weight: [vocab_size+1, encoder_hidden]
/// - decoder.decoder_layers.0.bias: [vocab_size+1]
///
/// This is equivalent to a Conv1d with kernel_size=1.
#[derive(Debug, Clone)]
pub struct CtcDecoder {
    output_proj: Linear,
    blank_id: u32,
    vocab_size: usize, // includes blank
}

impl CtcDecoder {
    pub fn load(vb: VarBuilder, encoder_cfg: &EncoderConfig, decoder_cfg: &DecoderConfig) -> Result<Self> {
        // NeMo: decoder.decoder_layers.0 - Conv1d(kernel_size=1)
        // Weight shape in NeMo: [vocab_size+1, encoder_hidden, 1] (Conv1d format)
        // We squeeze the last dimension to treat it as Linear
        let num_outputs = decoder_cfg.vocab_size + 1;
        let decoder_vb = vb.pp("decoder").pp("decoder_layers.0");

        // Load Conv1d weight and squeeze to Linear format
        let weight = decoder_vb.get((num_outputs, encoder_cfg.hidden_size, 1), "weight")?;
        let weight = weight.squeeze(2)?; // [num_outputs, hidden_size]
        let bias = decoder_vb.get(num_outputs, "bias")?;
        let output_proj = Linear::new(weight, Some(bias));

        Ok(Self {
            output_proj,
            blank_id: decoder_cfg.blank_id,
            vocab_size: num_outputs,
        })
    }

    /// Forward pass: project encoder output to vocabulary logits
    /// Input: [batch, time, encoder_hidden]
    /// Output: [batch, time, vocab_size+1]
    pub fn forward(&self, encoder_output: &Tensor) -> Result<Tensor> {
        self.output_proj.forward(encoder_output)
    }

    /// Greedy CTC decoding
    ///
    /// 1. Argmax at each frame to get predictions
    /// 2. Collapse blanks and repeated tokens
    ///
    /// Example: [a, a, blank, b, b, blank, c] -> [a, b, c]
    pub fn decode_greedy(&self, encoder_output: &Tensor) -> Result<Vec<u32>> {
        // Project to logits: [batch, time, vocab_size+1]
        let logits = self.forward(encoder_output)?;

        // Argmax per frame: [batch, time]
        let predictions = logits.argmax(D::Minus1)?;

        // Squeeze batch dimension (assuming batch=1)
        let predictions = predictions.squeeze(0)?;

        // Get predictions as vec
        let num_frames = predictions.dim(0)?;
        let predictions: Vec<u32> = predictions.to_vec1()?;

        // Collapse blanks and repeated tokens
        let mut tokens = Vec::new();
        let mut prev_token = self.blank_id;

        for t in 0..num_frames {
            let token = predictions[t];
            // Only emit if:
            // 1. Not a blank, AND
            // 2. Different from previous token (or previous was blank)
            if token != self.blank_id && token != prev_token {
                tokens.push(token);
            }
            prev_token = token;
        }

        Ok(tokens)
    }

    /// Beam search CTC decoding with prefix merging
    ///
    /// CTC beam search is different from transducer beam search:
    /// - Tracks prefixes (not individual tokens)
    /// - Each prefix has two probabilities: p_blank (ends in blank) and p_non_blank
    /// - Prefixes with same text but different blank patterns are merged
    pub fn decode_beam(
        &self,
        encoder_output: &Tensor,
        beam_width: usize,
    ) -> Result<Vec<u32>> {
        let logits = self.forward(encoder_output)?;
        let log_probs = candle_nn::ops::log_softmax(&logits, D::Minus1)?;

        // Squeeze batch dimension
        let log_probs = log_probs.squeeze(0)?;
        let num_frames = log_probs.dim(0)?;

        // Beam state: (prefix, p_blank, p_non_blank)
        // p_blank: probability of paths ending in blank
        // p_non_blank: probability of paths ending in non-blank
        struct BeamState {
            prefix: Vec<u32>,
            p_blank: f32,     // log prob of paths ending in blank
            p_non_blank: f32, // log prob of paths ending in non-blank
        }

        impl BeamState {
            fn total_prob(&self) -> f32 {
                // Log-sum-exp of p_blank and p_non_blank
                let max = self.p_blank.max(self.p_non_blank);
                if max == f32::NEG_INFINITY {
                    f32::NEG_INFINITY
                } else {
                    max + ((self.p_blank - max).exp() + (self.p_non_blank - max).exp()).ln()
                }
            }
        }

        // Initialize with empty prefix
        let mut beams = vec![BeamState {
            prefix: vec![],
            p_blank: 0.0,     // log(1) = 0
            p_non_blank: f32::NEG_INFINITY,
        }];

        for t in 0..num_frames {
            let frame_log_probs: Vec<f32> = log_probs.i(t)?.to_vec1()?;
            let blank_log_prob = frame_log_probs[self.blank_id as usize];

            // Use hashmap to merge beams with same prefix
            use std::collections::HashMap;
            let mut new_beams: HashMap<Vec<u32>, BeamState> = HashMap::new();

            for beam in &beams {
                let p_total = beam.total_prob();

                // Case 1: Extend with blank
                // New p_blank = old_p_total + p(blank)
                let new_p_blank = p_total + blank_log_prob;

                let entry = new_beams.entry(beam.prefix.clone()).or_insert(BeamState {
                    prefix: beam.prefix.clone(),
                    p_blank: f32::NEG_INFINITY,
                    p_non_blank: f32::NEG_INFINITY,
                });
                entry.p_blank = log_sum_exp(entry.p_blank, new_p_blank);

                // Case 2: Extend with each non-blank token
                for (c, &log_prob_c) in frame_log_probs.iter().enumerate() {
                    if c == self.blank_id as usize {
                        continue;
                    }
                    let c = c as u32;

                    let last_token = beam.prefix.last().copied();

                    if Some(c) == last_token {
                        // Same as last token: can only extend from blank
                        // (to emit the same character twice, need a blank between)
                        let new_p = beam.p_blank + log_prob_c;

                        let mut new_prefix = beam.prefix.clone();
                        new_prefix.push(c);

                        let entry = new_beams.entry(new_prefix.clone()).or_insert(BeamState {
                            prefix: new_prefix,
                            p_blank: f32::NEG_INFINITY,
                            p_non_blank: f32::NEG_INFINITY,
                        });
                        entry.p_non_blank = log_sum_exp(entry.p_non_blank, new_p);
                    } else {
                        // Different token: can extend from both blank and non-blank
                        let new_p = p_total + log_prob_c;

                        let mut new_prefix = beam.prefix.clone();
                        new_prefix.push(c);

                        let entry = new_beams.entry(new_prefix.clone()).or_insert(BeamState {
                            prefix: new_prefix,
                            p_blank: f32::NEG_INFINITY,
                            p_non_blank: f32::NEG_INFINITY,
                        });
                        entry.p_non_blank = log_sum_exp(entry.p_non_blank, new_p);
                    }
                }
            }

            // Convert to vec, sort by total prob, and keep top beam_width
            let mut beam_vec: Vec<BeamState> = new_beams.into_values().collect();
            beam_vec.sort_by(|a, b| b.total_prob().partial_cmp(&a.total_prob()).unwrap());
            beam_vec.truncate(beam_width);
            beams = beam_vec;
        }

        // Return best beam's prefix
        Ok(beams.first().map(|b| b.prefix.clone()).unwrap_or_default())
    }
}

/// Log-sum-exp of two log probabilities
fn log_sum_exp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        b
    } else if b == f32::NEG_INFINITY {
        a
    } else {
        let max = a.max(b);
        max + ((a - max).exp() + (b - max).exp()).ln()
    }
}

impl CtcDecoder {
    /// Greedy CTC decoding with rich results (timestamps, confidence)
    ///
    /// Similar to `decode_greedy` but tracks frame indices and log probabilities.
    /// For CTC, a token's span is from its first non-blank appearance to the
    /// frame before the next token (or blank that transitions to a different token).
    pub fn decode_greedy_with_info(&self, encoder_output: &Tensor) -> Result<DecodingResult> {
        // Project to logits: [batch, time, vocab_size+1]
        let logits = self.forward(encoder_output)?;
        let log_probs = candle_nn::ops::log_softmax(&logits, D::Minus1)?;

        // Squeeze batch dimension (assuming batch=1)
        let log_probs = log_probs.squeeze(0)?;
        let num_frames = log_probs.dim(0)?;

        let mut result = DecodingResult::new(num_frames);

        // Track current token span
        let mut current_token: Option<u32> = None;
        let mut token_start_frame = 0;
        let mut token_log_probs: Vec<f32> = Vec::new();

        for t in 0..num_frames {
            let frame_log_probs: Vec<f32> = log_probs.i(t)?.to_vec1()?;
            let token = frame_log_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            let log_prob = frame_log_probs[token as usize];

            if token == self.blank_id {
                // Blank frame - finalize any current token
                if let Some(tok) = current_token.take() {
                    // Average log prob over all frames where this token appeared
                    let avg_log_prob =
                        token_log_probs.iter().sum::<f32>() / token_log_probs.len() as f32;
                    result.tokens.push(TokenInfo {
                        token_id: tok,
                        log_prob: avg_log_prob,
                        start_frame: token_start_frame,
                        end_frame: t,
                    });
                    result.total_log_prob += avg_log_prob;
                    token_log_probs.clear();
                }
            } else if Some(token) == current_token {
                // Same token as before - extend the span
                token_log_probs.push(log_prob);
            } else {
                // Different token - finalize previous and start new
                if let Some(tok) = current_token.take() {
                    let avg_log_prob =
                        token_log_probs.iter().sum::<f32>() / token_log_probs.len() as f32;
                    result.tokens.push(TokenInfo {
                        token_id: tok,
                        log_prob: avg_log_prob,
                        start_frame: token_start_frame,
                        end_frame: t,
                    });
                    result.total_log_prob += avg_log_prob;
                    token_log_probs.clear();
                }
                // Start new token span
                current_token = Some(token);
                token_start_frame = t;
                token_log_probs.push(log_prob);
            }
        }

        // Finalize any remaining token
        if let Some(tok) = current_token {
            let avg_log_prob =
                token_log_probs.iter().sum::<f32>() / token_log_probs.len() as f32;
            result.tokens.push(TokenInfo {
                token_id: tok,
                log_prob: avg_log_prob,
                start_frame: token_start_frame,
                end_frame: num_frames,
            });
            result.total_log_prob += avg_log_prob;
        }

        Ok(result)
    }

    /// Beam search CTC decoding with rich results (timestamps, confidence)
    ///
    /// For beam search, timestamps are approximated based on when each token
    /// first appeared in the winning beam. Confidence is the total log probability.
    pub fn decode_beam_with_info(
        &self,
        encoder_output: &Tensor,
        beam_width: usize,
    ) -> Result<DecodingResult> {
        let logits = self.forward(encoder_output)?;
        let log_probs = candle_nn::ops::log_softmax(&logits, D::Minus1)?;

        // Squeeze batch dimension
        let log_probs = log_probs.squeeze(0)?;
        let num_frames = log_probs.dim(0)?;

        // Beam state with token info tracking
        struct BeamState {
            prefix: Vec<u32>,
            token_info: Vec<TokenInfo>,
            p_blank: f32,
            p_non_blank: f32,
            last_token_frame: usize,
        }

        impl BeamState {
            fn total_prob(&self) -> f32 {
                let max = self.p_blank.max(self.p_non_blank);
                if max == f32::NEG_INFINITY {
                    f32::NEG_INFINITY
                } else {
                    max + ((self.p_blank - max).exp() + (self.p_non_blank - max).exp()).ln()
                }
            }
        }

        // Initialize with empty prefix
        let mut beams = vec![BeamState {
            prefix: vec![],
            token_info: vec![],
            p_blank: 0.0,
            p_non_blank: f32::NEG_INFINITY,
            last_token_frame: 0,
        }];

        for t in 0..num_frames {
            let frame_log_probs: Vec<f32> = log_probs.i(t)?.to_vec1()?;
            let blank_log_prob = frame_log_probs[self.blank_id as usize];

            use std::collections::HashMap;
            let mut new_beams: HashMap<Vec<u32>, BeamState> = HashMap::new();

            for beam in &beams {
                let p_total = beam.total_prob();

                // Case 1: Extend with blank
                let new_p_blank = p_total + blank_log_prob;

                let entry = new_beams.entry(beam.prefix.clone()).or_insert(BeamState {
                    prefix: beam.prefix.clone(),
                    token_info: beam.token_info.clone(),
                    p_blank: f32::NEG_INFINITY,
                    p_non_blank: f32::NEG_INFINITY,
                    last_token_frame: beam.last_token_frame,
                });
                entry.p_blank = log_sum_exp(entry.p_blank, new_p_blank);

                // Case 2: Extend with each non-blank token
                for (c, &log_prob_c) in frame_log_probs.iter().enumerate() {
                    if c == self.blank_id as usize {
                        continue;
                    }
                    let c = c as u32;
                    let last_token = beam.prefix.last().copied();

                    if Some(c) == last_token {
                        let new_p = beam.p_blank + log_prob_c;

                        let mut new_prefix = beam.prefix.clone();
                        new_prefix.push(c);

                        // Add new token info
                        let mut new_token_info = beam.token_info.clone();
                        new_token_info.push(TokenInfo {
                            token_id: c,
                            log_prob: log_prob_c,
                            start_frame: t,
                            end_frame: t + 1,
                        });

                        let entry = new_beams.entry(new_prefix.clone()).or_insert(BeamState {
                            prefix: new_prefix,
                            token_info: new_token_info,
                            p_blank: f32::NEG_INFINITY,
                            p_non_blank: f32::NEG_INFINITY,
                            last_token_frame: t,
                        });
                        entry.p_non_blank = log_sum_exp(entry.p_non_blank, new_p);
                    } else {
                        let new_p = p_total + log_prob_c;

                        let mut new_prefix = beam.prefix.clone();
                        new_prefix.push(c);

                        // Add new token info
                        let mut new_token_info = beam.token_info.clone();
                        new_token_info.push(TokenInfo {
                            token_id: c,
                            log_prob: log_prob_c,
                            start_frame: t,
                            end_frame: t + 1,
                        });

                        let entry = new_beams.entry(new_prefix.clone()).or_insert(BeamState {
                            prefix: new_prefix,
                            token_info: new_token_info,
                            p_blank: f32::NEG_INFINITY,
                            p_non_blank: f32::NEG_INFINITY,
                            last_token_frame: t,
                        });
                        entry.p_non_blank = log_sum_exp(entry.p_non_blank, new_p);
                    }
                }
            }

            let mut beam_vec: Vec<BeamState> = new_beams.into_values().collect();
            beam_vec.sort_by(|a, b| b.total_prob().partial_cmp(&a.total_prob()).unwrap());
            beam_vec.truncate(beam_width);
            beams = beam_vec;
        }

        let best_beam = beams.first();
        Ok(DecodingResult {
            tokens: best_beam.map(|b| b.token_info.clone()).unwrap_or_default(),
            total_log_prob: best_beam.map(|b| b.total_prob()).unwrap_or(0.0),
            num_frames,
        })
    }
}
