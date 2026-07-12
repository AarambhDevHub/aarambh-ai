use aarambh_ai_core::{AarambhError, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone)]
/// Candidate token probability shown for prediction views.
pub struct TokenCandidate {
    /// Candidate token id.
    pub token_id: u32,
    /// Candidate probability after filtering.
    pub probability: f32,
}

#[derive(Debug, Clone)]
/// Token sampling strategy.
pub enum Sampler {
    /// Always pick the highest-logit token.
    Greedy,
    /// Temperature sampling with optional top-k and top-p filters.
    TopKTopP {
        /// Sampling temperature.
        temperature: f32,
        /// Optional top-k candidate limit.
        top_k: Option<usize>,
        /// Optional nucleus sampling probability mass.
        top_p: Option<f32>,
        /// Random number generator used for sampling.
        rng: Box<StdRng>,
    },
}

impl Sampler {
    /// Create a greedy sampler.
    pub fn greedy() -> Self {
        Self::Greedy
    }

    /// Create a top-k/top-p sampler.
    pub fn top_k_top_p(
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
        seed: Option<u64>,
    ) -> Result<Self> {
        if temperature < 0.0 {
            return Err(AarambhError::Config(
                "temperature must be non-negative".into(),
            ));
        }
        if let Some(p) = top_p
            && !(0.0..=1.0).contains(&p)
        {
            return Err(AarambhError::Config("top_p must be in [0, 1]".into()));
        }
        let rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        Ok(Self::TopKTopP {
            temperature,
            top_k: top_k.filter(|k| *k > 0),
            top_p: top_p.filter(|p| *p > 0.0 && *p < 1.0),
            rng: Box::new(rng),
        })
    }

    /// Sample one token id from logits.
    pub fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        if logits.is_empty() {
            return Err(AarambhError::Shape("logits must be non-empty".into()));
        }

        match self {
            Self::Greedy => Ok(argmax(logits) as u32),
            Self::TopKTopP {
                temperature,
                top_k,
                top_p,
                rng,
            } => {
                if *temperature <= f32::EPSILON {
                    return Ok(argmax(logits) as u32);
                }
                let probs = filtered_probs(logits, *temperature, *top_k, *top_p)?;
                sample_from_probs(&probs, rng.as_mut())
            }
        }
    }

    /// Return true when sampling always selects the highest-logit token.
    pub fn is_deterministic(&self) -> bool {
        matches!(self, Self::Greedy)
            || matches!(
                self,
                Self::TopKTopP { temperature, .. } if *temperature <= f32::EPSILON
            )
    }

    /// Convert logits into the normalized distribution used by this sampler.
    ///
    /// Deterministic samplers return a one-hot distribution at the argmax.
    pub fn probabilities(&self, logits: &[f32]) -> Result<Vec<f32>> {
        if logits.is_empty() {
            return Err(AarambhError::Shape("logits must be non-empty".into()));
        }
        match self {
            Self::Greedy => Ok(one_hot_argmax(logits)),
            Self::TopKTopP {
                temperature,
                top_k,
                top_p,
                ..
            } if *temperature > f32::EPSILON => {
                filtered_probs(logits, *temperature, *top_k, *top_p)
            }
            Self::TopKTopP { .. } => Ok(one_hot_argmax(logits)),
        }
    }

    /// Convert logits into this sampler's distribution after restricting token ids.
    pub fn probabilities_allowed(&self, logits: &[f32], allowed: &[u32]) -> Result<Vec<f32>> {
        if allowed.is_empty() {
            return Err(AarambhError::Config(
                "constrained sampler received no allowed tokens".into(),
            ));
        }
        let mut masked = vec![f32::NEG_INFINITY; logits.len()];
        for &token_id in allowed {
            let index = token_id as usize;
            if index >= logits.len() {
                return Err(AarambhError::Shape(format!(
                    "allowed token {token_id} exceeds vocabulary size {}",
                    logits.len()
                )));
            }
            masked[index] = logits[index];
        }
        self.probabilities(&masked)
    }

    /// Sample from an already normalized probability distribution.
    pub fn sample_probabilities(&mut self, probabilities: &[f32]) -> Result<u32> {
        validate_probabilities(probabilities)?;
        let deterministic = self.is_deterministic();
        match self {
            Self::Greedy => Ok(argmax(probabilities) as u32),
            Self::TopKTopP { rng, .. } if !deterministic => {
                sample_from_probs(probabilities, rng.as_mut())
            }
            Self::TopKTopP { .. } => Ok(argmax(probabilities) as u32),
        }
    }

    /// Draw a uniform value in `[0, 1)` from this sampler's random stream.
    ///
    /// Deterministic samplers return zero and do not allocate an RNG.
    pub fn draw_uniform(&mut self) -> f32 {
        match self {
            Self::Greedy => 0.0,
            Self::TopKTopP { rng, .. } => rng.r#gen::<f32>(),
        }
    }

    /// Return the highest-probability candidate tokens for display.
    pub fn top_candidates(&self, logits: &[f32], n: usize) -> Result<Vec<TokenCandidate>> {
        if logits.is_empty() {
            return Err(AarambhError::Shape("logits must be non-empty".into()));
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        let probs = match self {
            Self::Greedy => softmax(logits, 1.0)?,
            Self::TopKTopP {
                temperature,
                top_k,
                top_p,
                ..
            } if *temperature > f32::EPSILON => {
                filtered_probs(logits, *temperature, *top_k, *top_p)?
            }
            Self::TopKTopP { .. } => softmax(logits, 1.0)?,
        };
        Self::top_candidates_from_probabilities(&probs, n)
    }

    /// Return the highest-probability candidates from a normalized distribution.
    pub fn top_candidates_from_probabilities(
        probabilities: &[f32],
        n: usize,
    ) -> Result<Vec<TokenCandidate>> {
        validate_probabilities(probabilities)?;
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut candidates = probabilities
            .iter()
            .copied()
            .enumerate()
            .map(|(token_id, probability)| TokenCandidate {
                token_id: token_id as u32,
                probability,
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|a, b| b.probability.total_cmp(&a.probability));
        candidates.truncate(n.min(candidates.len()));
        Ok(candidates)
    }
}

fn one_hot_argmax(values: &[f32]) -> Vec<f32> {
    let mut probabilities = vec![0.0; values.len()];
    probabilities[argmax(values)] = 1.0;
    probabilities
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn filtered_probs(
    logits: &[f32],
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
) -> Result<Vec<f32>> {
    let mut allowed = vec![true; logits.len()];

    if let Some(k) = top_k
        && k < logits.len()
    {
        let mut ranked = logits.iter().copied().enumerate().collect::<Vec<_>>();
        ranked.sort_by(|(_, a), (_, b)| b.total_cmp(a));
        for (idx, _) in ranked.into_iter().skip(k) {
            allowed[idx] = false;
        }
    }

    let masked_logits = logits
        .iter()
        .zip(allowed.iter())
        .map(|(logit, allowed)| {
            if *allowed {
                *logit / temperature
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect::<Vec<_>>();
    let mut probs = softmax(&masked_logits, 1.0)?;

    if let Some(p) = top_p {
        let mut ranked = probs.iter().copied().enumerate().collect::<Vec<_>>();
        ranked.sort_by(|(_, a), (_, b)| b.total_cmp(a));
        let mut keep = vec![false; probs.len()];
        let mut cumulative = 0.0f32;
        for (idx, probability) in ranked {
            keep[idx] = true;
            cumulative += probability;
            if cumulative >= p {
                break;
            }
        }
        for (idx, probability) in probs.iter_mut().enumerate() {
            if !keep[idx] {
                *probability = 0.0;
            }
        }
        renormalize(&mut probs)?;
    }

    Ok(probs)
}

fn softmax(logits: &[f32], temperature: f32) -> Result<Vec<f32>> {
    let max = logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .max_by(|a, b| a.total_cmp(b))
        .ok_or_else(|| AarambhError::Config("all logits are non-finite".into()))?;
    let mut probs = logits
        .iter()
        .map(|logit| {
            if logit.is_finite() {
                ((*logit / temperature) - max / temperature).exp()
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    renormalize(&mut probs)?;
    Ok(probs)
}

fn renormalize(probs: &mut [f32]) -> Result<()> {
    let sum = probs.iter().sum::<f32>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(AarambhError::Config(
            "sampling distribution has zero probability mass".into(),
        ));
    }
    for probability in probs {
        *probability /= sum;
    }
    Ok(())
}

fn validate_probabilities(probabilities: &[f32]) -> Result<()> {
    if probabilities.is_empty() {
        return Err(AarambhError::Shape(
            "probability distribution must be non-empty".into(),
        ));
    }
    if probabilities
        .iter()
        .any(|probability| !probability.is_finite() || *probability < 0.0)
    {
        return Err(AarambhError::Config(
            "probability distribution contains invalid values".into(),
        ));
    }
    let sum = probabilities.iter().sum::<f32>();
    if !sum.is_finite() || (sum - 1.0).abs() > 1e-4 {
        return Err(AarambhError::Config(format!(
            "probability distribution must sum to one, got {sum}"
        )));
    }
    Ok(())
}

fn sample_from_probs(probs: &[f32], rng: &mut StdRng) -> Result<u32> {
    let draw = rng.r#gen::<f32>();
    let mut cumulative = 0.0f32;
    for (idx, probability) in probs.iter().enumerate() {
        cumulative += *probability;
        if draw <= cumulative {
            return Ok(idx as u32);
        }
    }
    Ok((probs.len() - 1) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_is_deterministic() {
        let mut sampler = Sampler::greedy();
        let logits = [0.1, 4.0, 1.0];
        assert_eq!(sampler.sample(&logits).unwrap(), 1);
        assert_eq!(sampler.sample(&logits).unwrap(), 1);
    }

    #[test]
    fn top_k_excludes_lower_ranked_tokens() {
        let sampler = Sampler::top_k_top_p(1.0, Some(2), None, Some(42)).unwrap();
        let logits = [10.0, 9.0, 8.0, 7.0];
        let candidates = sampler.top_candidates(&logits, 4).unwrap();
        assert!(candidates.iter().any(|c| c.token_id == 0));
        assert!(candidates.iter().any(|c| c.token_id == 1));
        assert_eq!(
            candidates
                .iter()
                .find(|c| c.token_id == 2)
                .unwrap()
                .probability,
            0.0
        );
    }

    #[test]
    fn temperature_zero_returns_argmax() {
        let mut sampler = Sampler::top_k_top_p(0.0, Some(1), Some(0.1), Some(42)).unwrap();
        assert_eq!(sampler.sample(&[1.0, 2.0, 3.0]).unwrap(), 2);
    }

    #[test]
    fn top_candidates_are_sorted() {
        let sampler = Sampler::greedy();
        let candidates = sampler.top_candidates(&[1.0, 3.0, 2.0], 3).unwrap();
        assert_eq!(candidates[0].token_id, 1);
        assert!(
            candidates
                .windows(2)
                .all(|w| w[0].probability >= w[1].probability)
        );
    }

    #[test]
    fn probabilities_match_sampling_filters() {
        let sampler = Sampler::top_k_top_p(1.0, Some(2), None, Some(42)).unwrap();
        let probabilities = sampler.probabilities(&[10.0, 9.0, 8.0]).unwrap();
        assert_eq!(probabilities[2], 0.0);
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn deterministic_probabilities_are_one_hot() {
        let sampler = Sampler::greedy();
        assert_eq!(
            sampler.probabilities(&[1.0, 3.0, 2.0]).unwrap(),
            [0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn allowed_tokens_are_masked_before_top_k() {
        let sampler = Sampler::top_k_top_p(1.0, Some(1), None, Some(42)).unwrap();
        let probabilities = sampler
            .probabilities_allowed(&[100.0, 10.0, 9.0], &[1, 2])
            .unwrap();
        assert_eq!(probabilities, [0.0, 1.0, 0.0]);
    }
}
