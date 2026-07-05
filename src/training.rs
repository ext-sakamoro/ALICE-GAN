//! Training loop (`EpochRecord` / `TrainConfig` / `train`).

use crate::gan::Gan;
use crate::mode_collapse::{detect_mode_collapse, CollapseStats};
use crate::rng::Rng;

// Training loop
// ---------------------------------------------------------------------------

/// Record of a single training epoch.
#[derive(Debug, Clone)]
pub struct EpochRecord {
    /// Epoch number (0-based).
    pub epoch: usize,
    /// Discriminator loss.
    pub disc_loss: f64,
    /// Generator loss.
    pub gen_loss: f64,
    /// Mode collapse stats (if computed).
    pub collapse_stats: Option<CollapseStats>,
}

/// Configuration for the training loop.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Number of training epochs.
    pub epochs: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Number of discriminator steps per generator step.
    pub disc_steps: usize,
    /// If true, check for mode collapse every epoch.
    pub check_collapse: bool,
    /// Similarity threshold for mode collapse detection.
    pub collapse_threshold: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 16,
            disc_steps: 1,
            check_collapse: false,
            collapse_threshold: 0.95,
        }
    }
}

/// Run a full training loop.
///
/// `data_fn` provides real data batches: given a batch size and rng, return samples.
pub fn train(
    gan: &mut Gan,
    train_config: &TrainConfig,
    data_fn: &dyn Fn(usize, &mut Rng) -> Vec<Vec<f64>>,
    rng: &mut Rng,
) -> Vec<EpochRecord> {
    let mut history = Vec::with_capacity(train_config.epochs);

    for epoch in 0..train_config.epochs {
        let mut epoch_d_loss = 0.0;
        let mut epoch_g_loss = 0.0;

        let real_batch = data_fn(train_config.batch_size, rng);
        let (d_loss, g_loss) = gan.train_step(&real_batch, rng);
        epoch_d_loss += d_loss;
        epoch_g_loss += g_loss;

        let collapse_stats = if train_config.check_collapse {
            let samples = gan.generate_batch(16, rng);
            Some(detect_mode_collapse(
                &samples,
                train_config.collapse_threshold,
            ))
        } else {
            None
        };

        history.push(EpochRecord {
            epoch,
            disc_loss: epoch_d_loss,
            gen_loss: epoch_g_loss,
            collapse_stats,
        });
    }

    history
}
