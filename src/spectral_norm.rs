//! Spectral normalization.

use crate::network::Network;

// Spectral normalization
// ---------------------------------------------------------------------------

/// Apply spectral normalization to a weight matrix by estimating the largest
/// singular value via power iteration.
///
/// # Panics
///
/// Panics if `weights.len() != rows * cols`.
pub fn spectral_normalize(weights: &mut [f64], rows: usize, cols: usize, iterations: usize) {
    assert_eq!(weights.len(), rows * cols);
    if rows == 0 || cols == 0 {
        return;
    }

    // Initialize u as uniform
    let mut u = vec![1.0 / (rows as f64).sqrt(); rows];
    let mut v = vec![0.0; cols];

    for _ in 0..iterations {
        // v = W^T u, then normalize
        for j in 0..cols {
            let mut s = 0.0;
            for i in 0..rows {
                s += weights[i * cols + j] * u[i];
            }
            v[j] = s;
        }
        let v_norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        for x in &mut v {
            *x /= v_norm;
        }

        // u = W v, then normalize
        for i in 0..rows {
            let mut s = 0.0;
            for j in 0..cols {
                s += weights[i * cols + j] * v[j];
            }
            u[i] = s;
        }
        let u_norm = u.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        for x in &mut u {
            *x /= u_norm;
        }
    }

    // sigma = u^T W v
    let mut sigma = 0.0;
    for i in 0..rows {
        for j in 0..cols {
            sigma += u[i] * weights[i * cols + j] * v[j];
        }
    }
    let sigma = sigma.abs().max(1e-12);

    for w in weights.iter_mut() {
        *w /= sigma;
    }
}

/// Apply spectral normalization to all layers in a network.
pub fn spectral_normalize_network(network: &mut Network, iterations: usize) {
    for layer in &mut network.layers {
        spectral_normalize(
            &mut layer.weights,
            layer.out_size,
            layer.in_size,
            iterations,
        );
    }
}
