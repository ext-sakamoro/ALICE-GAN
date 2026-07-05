//! Latent space interpolation (`lerp` / `slerp` / `interpolate_latent`).

use crate::network::Network;

// Latent space interpolation
// ---------------------------------------------------------------------------

/// Linear interpolation between two latent vectors.
#[must_use]
pub fn lerp(z1: &[f64], z2: &[f64], t: f64) -> Vec<f64> {
    z1.iter()
        .zip(z2.iter())
        .map(|(a, b)| a * (1.0 - t) + b * t)
        .collect()
}

/// Spherical linear interpolation between two latent vectors.
#[must_use]
pub fn slerp(z1: &[f64], z2: &[f64], t: f64) -> Vec<f64> {
    let dot: f64 = z1.iter().zip(z2.iter()).map(|(a, b)| a * b).sum();
    let n1 = z1.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
    let n2 = z2.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
    let cos_omega = (dot / (n1 * n2)).clamp(-1.0, 1.0);
    let omega = cos_omega.acos();

    if omega.abs() < 1e-8 {
        return lerp(z1, z2, t);
    }

    let sin_omega = omega.sin();
    let w1 = ((1.0 - t) * omega).sin() / sin_omega;
    let w2 = (t * omega).sin() / sin_omega;

    z1.iter()
        .zip(z2.iter())
        .map(|(a, b)| w1 * a + w2 * b)
        .collect()
}

/// Generate `steps` interpolated points (including endpoints) between two
/// latent vectors, passed through the generator.
#[must_use]
pub fn interpolate_latent(
    generator: &Network,
    z1: &[f64],
    z2: &[f64],
    steps: usize,
    use_slerp: bool,
) -> Vec<Vec<f64>> {
    let mut results = Vec::with_capacity(steps);
    for i in 0..steps {
        let t = if steps <= 1 {
            0.0
        } else {
            i as f64 / (steps - 1) as f64
        };
        let z = if use_slerp {
            slerp(z1, z2, t)
        } else {
            lerp(z1, z2, t)
        };
        results.push(generator.predict(&z));
    }
    results
}
