//! Utility functions (`l2_norm` / `normalize`).

// Utility functions
// ---------------------------------------------------------------------------

/// Compute the L2 norm of a vector.
#[must_use]
pub fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Normalize a vector to unit length.
#[must_use]
pub fn normalize(v: &[f64]) -> Vec<f64> {
    let n = l2_norm(v).max(1e-12);
    v.iter().map(|x| x / n).collect()
}

/// Compute element-wise mean of a batch of vectors.
#[must_use]
pub fn batch_mean(batch: &[Vec<f64>]) -> Vec<f64> {
    if batch.is_empty() {
        return Vec::new();
    }
    let dim = batch[0].len();
    let n = batch.len() as f64;
    let mut mean = vec![0.0; dim];
    for sample in batch {
        for (m, s) in mean.iter_mut().zip(sample.iter()) {
            *m += s;
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    mean
}

/// Compute element-wise variance of a batch of vectors.
#[must_use]
pub fn batch_variance(batch: &[Vec<f64>]) -> Vec<f64> {
    if batch.is_empty() {
        return Vec::new();
    }
    let mean = batch_mean(batch);
    let dim = batch[0].len();
    let n = batch.len() as f64;
    let mut var = vec![0.0; dim];
    for sample in batch {
        for (v, (s, m)) in var.iter_mut().zip(sample.iter().zip(mean.iter())) {
            *v += (s - m).powi(2);
        }
    }
    for v in &mut var {
        *v /= n;
    }
    var
}

// ===========================================================================
// Tests
// ===========================================================================
