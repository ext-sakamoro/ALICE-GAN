//! Convenience re-export (= `use alice_gan::prelude::*;`).

pub use crate::activation::{sigmoid, Activation};
pub use crate::dense_layer::DenseLayer;
pub use crate::gan::{Gan, GanConfig};
pub use crate::gradient_penalty::gradient_penalty;
pub use crate::latent_interp::{interpolate_latent, lerp, slerp};
pub use crate::loss::{
    bce_loss, bce_loss_grad, compute_loss, compute_loss_grad, hinge_loss, hinge_loss_grad,
    wasserstein_loss, wasserstein_loss_grad, LossType,
};
pub use crate::mode_collapse::{cosine_similarity, detect_mode_collapse, CollapseStats};
pub use crate::network::Network;
pub use crate::rng::Rng;
pub use crate::spectral_norm::{spectral_normalize, spectral_normalize_network};
pub use crate::training::{train, EpochRecord, TrainConfig};
pub use crate::util::{l2_norm, normalize};
