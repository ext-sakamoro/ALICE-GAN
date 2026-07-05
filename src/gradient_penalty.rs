//! Gradient penalty (WGAN-GP).

use crate::network::Network;
use crate::rng::Rng;

// Gradient penalty (WGAN-GP)
// ---------------------------------------------------------------------------

/// Compute gradient penalty by finite differences on interpolated samples.
///
/// `real` and `fake` should be vectors of the same length.
/// Returns the penalty term: `(||grad|| - 1)^2`.
#[must_use]
pub fn gradient_penalty(discriminator: &Network, real: &[f64], fake: &[f64], rng: &mut Rng) -> f64 {
    let epsilon = rng.next_f64();
    let interpolated: Vec<f64> = real
        .iter()
        .zip(fake.iter())
        .map(|(r, f)| epsilon * r + (1.0 - epsilon) * f)
        .collect();

    let h = 1e-5;
    let base_out = discriminator.predict(&interpolated);
    let base_val = base_out[0];

    let mut grad_norm_sq = 0.0;
    for i in 0..interpolated.len() {
        let mut perturbed = interpolated.clone();
        perturbed[i] += h;
        let perturbed_out = discriminator.predict(&perturbed);
        let grad_i = (perturbed_out[0] - base_val) / h;
        grad_norm_sq += grad_i * grad_i;
    }

    let grad_norm = grad_norm_sq.sqrt();
    (grad_norm - 1.0).powi(2)
}
