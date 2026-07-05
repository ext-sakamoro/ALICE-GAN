//! ALICE-GAN: Pure Rust Generative Adversarial Networks
//!
//! Provides generator/discriminator networks, multiple loss functions,
//! gradient penalty, spectral normalization, latent space interpolation,
//! mode collapse detection, and a training loop.

#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::module_name_repetitions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::wildcard_imports,
    clippy::doc_markdown,
    clippy::cast_lossless,
    clippy::suboptimal_flops,
    clippy::float_cmp
)]

pub mod activation;
pub mod dense_layer;
pub mod gan;
pub mod gradient_penalty;
pub mod latent_interp;
pub mod loss;
pub mod mode_collapse;
pub mod network;
pub mod prelude;
pub mod rng;
pub mod spectral_norm;
pub mod training;
pub mod util;

#[cfg(test)]
mod integration_tests;

// Backward-compat re-exports.
pub use crate::activation::*;
pub use crate::dense_layer::*;
pub use crate::gan::*;
pub use crate::gradient_penalty::*;
pub use crate::latent_interp::*;
pub use crate::loss::*;
pub use crate::mode_collapse::*;
pub use crate::network::*;
pub use crate::rng::*;
pub use crate::spectral_norm::*;
pub use crate::training::*;
pub use crate::util::*;
