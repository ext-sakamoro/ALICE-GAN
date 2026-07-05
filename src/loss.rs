//! Loss functions (BCE / Wasserstein / Hinge).

// Loss functions
// ---------------------------------------------------------------------------

/// GAN loss function variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossType {
    /// Binary Cross-Entropy.
    Bce,
    /// Wasserstein loss.
    Wasserstein,
    /// Hinge loss.
    Hinge,
}

/// Binary cross-entropy loss: `-[t*ln(p) + (1-t)*ln(1-p)]`.
#[must_use]
pub fn bce_loss(predictions: &[f64], targets: &[f64]) -> f64 {
    let eps = 1e-12;
    let n = predictions.len() as f64;
    let mut sum = 0.0;
    for (p, t) in predictions.iter().zip(targets.iter()) {
        let pc = p.clamp(eps, 1.0 - eps);
        sum += -((1.0 - t).mul_add((1.0 - pc).ln(), t * pc.ln()));
    }
    sum / n
}

/// Gradient of BCE loss w.r.t. predictions.
#[must_use]
pub fn bce_loss_grad(predictions: &[f64], targets: &[f64]) -> Vec<f64> {
    let eps = 1e-12;
    let n = predictions.len() as f64;
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| {
            let pc = p.clamp(eps, 1.0 - eps);
            (-t / pc + (1.0 - t) / (1.0 - pc)) / n
        })
        .collect()
}

/// Wasserstein loss: `mean(predictions * targets)` (negated for minimization).
#[must_use]
pub fn wasserstein_loss(predictions: &[f64], targets: &[f64]) -> f64 {
    let n = predictions.len() as f64;
    let sum: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| -p * t)
        .sum();
    sum / n
}

/// Gradient of Wasserstein loss w.r.t. predictions.
#[must_use]
pub fn wasserstein_loss_grad(predictions: &[f64], targets: &[f64]) -> Vec<f64> {
    let n = predictions.len() as f64;
    targets.iter().map(|t| -t / n).collect()
}

/// Hinge loss for discriminator: `mean(max(0, 1 - t*p))`.
#[must_use]
pub fn hinge_loss(predictions: &[f64], targets: &[f64]) -> f64 {
    let n = predictions.len() as f64;
    let sum: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (1.0 - t * p).max(0.0))
        .sum();
    sum / n
}

/// Gradient of hinge loss w.r.t. predictions.
#[must_use]
pub fn hinge_loss_grad(predictions: &[f64], targets: &[f64]) -> Vec<f64> {
    let n = predictions.len() as f64;
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| if 1.0 - t * p > 0.0 { -t / n } else { 0.0 })
        .collect()
}

/// Compute loss given a `LossType`.
#[must_use]
pub fn compute_loss(loss_type: LossType, predictions: &[f64], targets: &[f64]) -> f64 {
    match loss_type {
        LossType::Bce => bce_loss(predictions, targets),
        LossType::Wasserstein => wasserstein_loss(predictions, targets),
        LossType::Hinge => hinge_loss(predictions, targets),
    }
}

/// Compute loss gradient given a `LossType`.
#[must_use]
pub fn compute_loss_grad(loss_type: LossType, predictions: &[f64], targets: &[f64]) -> Vec<f64> {
    match loss_type {
        LossType::Bce => bce_loss_grad(predictions, targets),
        LossType::Wasserstein => wasserstein_loss_grad(predictions, targets),
        LossType::Hinge => hinge_loss_grad(predictions, targets),
    }
}
