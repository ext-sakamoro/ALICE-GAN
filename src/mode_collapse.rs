//! Mode collapse detection (`CollapseStats` / `detect_mode_collapse` / `cosine_similarity`).

use std::fmt;

// Mode collapse detection
// ---------------------------------------------------------------------------

/// Statistics for mode collapse detection.
#[derive(Debug, Clone)]
pub struct CollapseStats {
    /// Mean pairwise cosine similarity among generated samples.
    pub mean_similarity: f64,
    /// Standard deviation of the generated samples (averaged over dimensions).
    pub mean_std_dev: f64,
    /// Whether mode collapse is detected (similarity > threshold).
    pub collapsed: bool,
}

impl fmt::Display for CollapseStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CollapseStats(sim={:.4}, std={:.4}, collapsed={})",
            self.mean_similarity, self.mean_std_dev, self.collapsed
        )
    }
}

/// Detect mode collapse by examining a batch of generated samples.
///
/// `similarity_threshold`: above this, samples are too similar (collapse).
#[must_use]
pub fn detect_mode_collapse(samples: &[Vec<f64>], similarity_threshold: f64) -> CollapseStats {
    let n = samples.len();
    if n < 2 {
        return CollapseStats {
            mean_similarity: 0.0,
            mean_std_dev: 0.0,
            collapsed: false,
        };
    }

    // Mean pairwise cosine similarity
    let mut total_sim = 0.0;
    let mut count = 0u64;
    for i in 0..n {
        for j in (i + 1)..n {
            total_sim += cosine_similarity(&samples[i], &samples[j]);
            count += 1;
        }
    }
    let mean_similarity = if count > 0 {
        total_sim / count as f64
    } else {
        0.0
    };

    // Mean standard deviation across dimensions
    let dim = samples[0].len();
    let mut total_std = 0.0;
    for d in 0..dim {
        let mean_d: f64 = samples.iter().map(|s| s[d]).sum::<f64>() / n as f64;
        let var_d: f64 = samples.iter().map(|s| (s[d] - mean_d).powi(2)).sum::<f64>() / n as f64;
        total_std += var_d.sqrt();
    }
    let mean_std_dev = total_std / dim as f64;

    let collapsed = mean_similarity > similarity_threshold;

    CollapseStats {
        mean_similarity,
        mean_std_dev,
        collapsed,
    }
}

/// Cosine similarity between two vectors.
#[must_use]
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    dot / (na * nb)
}
