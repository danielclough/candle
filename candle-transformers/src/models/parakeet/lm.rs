//! N-gram Language Model for ASR Decoding
//!
//! Provides N-gram language model support for beam search decoding.
//! Supports ARPA format files, which are the standard format for N-gram LMs.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// N-gram language model
///
/// Stores N-gram probabilities and backoff weights in log10 format (as in ARPA files).
/// Supports queries for scoring sequences and getting backoff probabilities.
#[derive(Debug, Clone)]
pub struct NgramLM {
    /// Maximum N-gram order (e.g., 3 for trigram)
    order: usize,
    /// Vocabulary mapping from word string to ID
    vocab: HashMap<String, u32>,
    /// Reverse vocabulary mapping from ID to word string
    id_to_word: Vec<String>,
    /// N-gram probabilities: key = sequence of token IDs, value = (log10_prob, log10_backoff)
    /// For the highest order, backoff is always 0.0
    ngrams: HashMap<Vec<u32>, (f32, f32)>,
    /// Unknown word ID (if present in vocab)
    unk_id: Option<u32>,
    /// Start-of-sentence token ID
    bos_id: Option<u32>,
    /// End-of-sentence token ID
    eos_id: Option<u32>,
}

impl NgramLM {
    /// Create a new empty N-gram LM
    pub fn new(order: usize) -> Self {
        Self {
            order,
            vocab: HashMap::new(),
            id_to_word: Vec::new(),
            ngrams: HashMap::new(),
            unk_id: None,
            bos_id: None,
            eos_id: None,
        }
    }

    /// Load an N-gram LM from ARPA format file
    ///
    /// ARPA format:
    /// ```text
    /// \data\
    /// ngram 1=<count>
    /// ngram 2=<count>
    /// ...
    ///
    /// \1-grams:
    /// <log10_prob> <word> [<log10_backoff>]
    /// ...
    ///
    /// \2-grams:
    /// <log10_prob> <word1> <word2> [<log10_backoff>]
    /// ...
    ///
    /// \end\
    /// ```
    pub fn load_arpa(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let mut lm = NgramLM::new(0);
        let mut current_order = 0;
        let mut in_data_section = false;
        let mut max_order = 0;

        while let Some(line) = lines.next() {
            let line = line?;
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            // Parse header
            if line == "\\data\\" {
                in_data_section = true;
                continue;
            }

            if in_data_section && line.starts_with("ngram ") {
                // Parse "ngram N=count"
                if let Some(order_str) = line.strip_prefix("ngram ") {
                    if let Some((order, _count)) = order_str.split_once('=') {
                        if let Ok(n) = order.parse::<usize>() {
                            max_order = max_order.max(n);
                        }
                    }
                }
                continue;
            }

            // Parse N-gram sections
            if line.starts_with('\\') && line.ends_with("-grams:") {
                let order_str = &line[1..line.len() - 7];
                if let Ok(n) = order_str.parse::<usize>() {
                    current_order = n;
                    in_data_section = false;
                }
                continue;
            }

            if line == "\\end\\" {
                break;
            }

            // Parse N-gram entries
            if current_order > 0 {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() < current_order + 1 {
                    continue;
                }

                // Parse log10 probability
                let log_prob = parts[0].parse::<f32>().unwrap_or(f32::NEG_INFINITY);

                // Parse words
                let words: Vec<&str> = parts[1..current_order + 1].to_vec();

                // Parse backoff (optional, only for non-highest order)
                let backoff = if parts.len() > current_order + 1 {
                    parts[current_order + 1].parse::<f32>().unwrap_or(0.0)
                } else {
                    0.0
                };

                // Convert words to IDs, adding to vocab if needed
                let mut ids = Vec::with_capacity(words.len());
                for word in words {
                    let word_string = word.to_string();
                    let id = if let Some(&id) = lm.vocab.get(&word_string) {
                        id
                    } else {
                        let id = lm.id_to_word.len() as u32;
                        lm.vocab.insert(word_string.clone(), id);
                        lm.id_to_word.push(word_string.clone());

                        // Track special tokens
                        if word == "<unk>" || word == "<UNK>" {
                            lm.unk_id = Some(id);
                        } else if word == "<s>" || word == "<BOS>" {
                            lm.bos_id = Some(id);
                        } else if word == "</s>" || word == "<EOS>" {
                            lm.eos_id = Some(id);
                        }

                        id
                    };
                    ids.push(id);
                }

                lm.ngrams.insert(ids, (log_prob, backoff));
            }
        }

        lm.order = max_order;
        Ok(lm)
    }

    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.id_to_word.len()
    }

    /// Get the order of the LM
    pub fn order(&self) -> usize {
        self.order
    }

    /// Look up a word's ID, returning UNK ID if not found
    pub fn word_to_id(&self, word: &str) -> Option<u32> {
        self.vocab.get(word).copied().or(self.unk_id)
    }

    /// Look up an ID's word
    pub fn id_to_word(&self, id: u32) -> Option<&str> {
        self.id_to_word.get(id as usize).map(|s| s.as_str())
    }

    /// Score the next word given context
    ///
    /// Uses backoff smoothing: if the full N-gram isn't found,
    /// backs off to (N-1)-gram with backoff weight.
    ///
    /// Returns log10 probability.
    pub fn score(&self, context: &[u32], next_word: u32) -> f32 {
        // Build the full N-gram
        let mut ngram: Vec<u32> = context
            .iter()
            .rev()
            .take(self.order - 1)
            .copied()
            .collect();
        ngram.reverse();
        ngram.push(next_word);

        // Try to find the N-gram, backing off if not found
        self.score_ngram(&ngram)
    }

    /// Score an N-gram with backoff
    fn score_ngram(&self, ngram: &[u32]) -> f32 {
        if ngram.is_empty() {
            return 0.0;
        }

        // Try to find this exact N-gram
        if let Some(&(log_prob, _)) = self.ngrams.get(ngram) {
            return log_prob;
        }

        // Back off: get backoff weight for context and score shorter N-gram
        if ngram.len() > 1 {
            let context = &ngram[..ngram.len() - 1];
            let backoff = self
                .ngrams
                .get(context)
                .map(|&(_, bo)| bo)
                .unwrap_or(0.0);

            let shorter = &ngram[1..];
            return backoff + self.score_ngram(shorter);
        }

        // Unigram not found - return log(1/vocab_size) as fallback
        -(self.vocab_size() as f32).log10()
    }

    /// Score a sequence of words
    ///
    /// Returns the total log10 probability.
    pub fn score_sequence(&self, words: &[u32]) -> f32 {
        let mut score = 0.0;
        for (i, &word) in words.iter().enumerate() {
            let context = &words[..i];
            score += self.score(context, word);
        }
        score
    }

    /// Convert log10 probability to natural log
    pub fn log10_to_ln(log10_prob: f32) -> f32 {
        log10_prob * std::f32::consts::LN_10
    }

    /// Check if this LM has a vocabulary for a given ASR vocabulary
    ///
    /// Returns the mapping from ASR token ID to LM word ID.
    pub fn create_asr_mapping(&self, asr_vocab: &[String]) -> Vec<Option<u32>> {
        asr_vocab
            .iter()
            .map(|word| {
                // Try different normalization strategies
                let word_lower = word.to_lowercase();
                let word_cleaned = word_lower
                    .trim_start_matches('▁')
                    .trim_start_matches("##");

                self.vocab
                    .get(&word_lower)
                    .or_else(|| self.vocab.get(word_cleaned))
                    .or_else(|| self.vocab.get(word))
                    .copied()
                    .or(self.unk_id)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_lm_basic() {
        let mut lm = NgramLM::new(2);

        // Manually add some N-grams
        lm.vocab.insert("hello".to_string(), 0);
        lm.vocab.insert("world".to_string(), 1);
        lm.id_to_word.push("hello".to_string());
        lm.id_to_word.push("world".to_string());

        // Unigrams
        lm.ngrams.insert(vec![0], (-1.0, -0.5)); // hello: log10(0.1), backoff -0.5
        lm.ngrams.insert(vec![1], (-1.5, 0.0)); // world: log10(~0.03)

        // Bigram
        lm.ngrams.insert(vec![0, 1], (-0.5, 0.0)); // hello world: log10(~0.3)

        lm.order = 2;

        // Test scoring
        let score = lm.score(&[0], 1); // P(world|hello)
        assert!((score - (-0.5)).abs() < 0.001);

        // Test backoff (context not found)
        let score = lm.score(&[1], 0); // P(hello|world) - should back off
        assert!(score < 0.0);
    }
}
