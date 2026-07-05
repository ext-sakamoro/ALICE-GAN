//! `Gan` — Generator + Discriminator bundle (`GanConfig` / `Gan`).

use crate::activation::Activation;
use crate::loss::{compute_loss, compute_loss_grad, LossType};
use crate::network::Network;
use crate::rng::Rng;
use crate::spectral_norm::spectral_normalize_network;
use std::fmt;

// GAN — Generator + Discriminator bundle
// ---------------------------------------------------------------------------

/// Configuration for a GAN.
#[derive(Debug, Clone)]
pub struct GanConfig {
    /// Dimension of the latent noise vector.
    pub latent_dim: usize,
    /// Hidden layer sizes for the generator.
    pub gen_hidden: Vec<usize>,
    /// Hidden layer sizes for the discriminator.
    pub disc_hidden: Vec<usize>,
    /// Output dimension (data dimension).
    pub data_dim: usize,
    /// Loss function type.
    pub loss_type: LossType,
    /// Learning rate for the generator.
    pub gen_lr: f64,
    /// Learning rate for the discriminator.
    pub disc_lr: f64,
    /// Gradient penalty coefficient (0 to disable).
    pub gp_lambda: f64,
    /// Whether to apply spectral normalization to discriminator.
    pub spectral_norm: bool,
}

impl Default for GanConfig {
    fn default() -> Self {
        Self {
            latent_dim: 16,
            gen_hidden: vec![32, 32],
            disc_hidden: vec![32, 32],
            data_dim: 8,
            loss_type: LossType::Bce,
            gen_lr: 0.001,
            disc_lr: 0.001,
            gp_lambda: 0.0,
            spectral_norm: false,
        }
    }
}

/// A complete GAN consisting of a generator and discriminator.
#[derive(Debug, Clone)]
pub struct Gan {
    /// The generator network.
    pub generator: Network,
    /// The discriminator network.
    pub discriminator: Network,
    /// Configuration.
    pub config: GanConfig,
}

impl Gan {
    /// Create a new GAN from the given configuration.
    #[must_use]
    pub fn new(config: GanConfig, rng: &mut Rng) -> Self {
        let mut gen_specs: Vec<(usize, Activation)> = config
            .gen_hidden
            .iter()
            .map(|&s| (s, Activation::LeakyRelu))
            .collect();
        gen_specs.push((config.data_dim, Activation::Tanh));

        let mut disc_specs: Vec<(usize, Activation)> = config
            .disc_hidden
            .iter()
            .map(|&s| (s, Activation::LeakyRelu))
            .collect();
        let final_act = if config.loss_type == LossType::Bce {
            Activation::Sigmoid
        } else {
            Activation::Linear
        };
        disc_specs.push((1, final_act));

        let generator = Network::new(config.latent_dim, &gen_specs, rng);
        let discriminator = Network::new(config.data_dim, &disc_specs, rng);

        Self {
            generator,
            discriminator,
            config,
        }
    }

    /// Generate a sample from a random latent vector.
    #[must_use]
    pub fn generate(&self, rng: &mut Rng) -> Vec<f64> {
        let mut z = vec![0.0; self.config.latent_dim];
        rng.fill_normal(&mut z);
        self.generator.predict(&z)
    }

    /// Generate a batch of samples.
    #[must_use]
    pub fn generate_batch(&self, batch_size: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
        (0..batch_size).map(|_| self.generate(rng)).collect()
    }

    /// Train for one step on a batch of real data.
    /// Returns `(disc_loss, gen_loss)`.
    ///
    /// # Panics
    ///
    /// Panics if the network layers are empty.
    pub fn train_step(&mut self, real_batch: &[Vec<f64>], rng: &mut Rng) -> (f64, f64) {
        let batch_size = real_batch.len();

        // --- Train Discriminator ---
        let mut disc_loss_total = 0.0;

        // Collect discriminator gradients
        let mut disc_grads: Vec<(Vec<f64>, Vec<f64>)> = self
            .discriminator
            .layers
            .iter()
            .map(|l| (vec![0.0; l.out_size * l.in_size], vec![0.0; l.out_size]))
            .collect();

        for real_sample in real_batch {
            // Real sample
            let d_real_acts = self.discriminator.forward(real_sample);
            let d_real_out = d_real_acts.last().unwrap();
            let real_target = vec![1.0];
            disc_loss_total += compute_loss(self.config.loss_type, d_real_out, &real_target);
            let d_real_grad = compute_loss_grad(self.config.loss_type, d_real_out, &real_target);
            let real_grads = self.discriminator.backward(&d_real_acts, &d_real_grad);
            for (i, (gw, gb)) in real_grads.iter().enumerate() {
                for (a, b) in disc_grads[i].0.iter_mut().zip(gw.iter()) {
                    *a += *b;
                }
                for (a, b) in disc_grads[i].1.iter_mut().zip(gb.iter()) {
                    *a += *b;
                }
            }

            // Fake sample
            let fake = self.generate(rng);
            let d_fake_acts = self.discriminator.forward(&fake);
            let d_fake_out = d_fake_acts.last().unwrap();
            let fake_target = match self.config.loss_type {
                LossType::Bce => vec![0.0],
                LossType::Wasserstein | LossType::Hinge => vec![-1.0],
            };
            disc_loss_total += compute_loss(self.config.loss_type, d_fake_out, &fake_target);
            let d_fake_grad = compute_loss_grad(self.config.loss_type, d_fake_out, &fake_target);
            let fake_grads = self.discriminator.backward(&d_fake_acts, &d_fake_grad);
            for (i, (gw, gb)) in fake_grads.iter().enumerate() {
                for (a, b) in disc_grads[i].0.iter_mut().zip(gw.iter()) {
                    *a += *b;
                }
                for (a, b) in disc_grads[i].1.iter_mut().zip(gb.iter()) {
                    *a += *b;
                }
            }
        }

        // Average gradients
        let bs = batch_size as f64;
        for (gw, gb) in &mut disc_grads {
            for g in gw.iter_mut() {
                *g /= bs;
            }
            for g in gb.iter_mut() {
                *g /= bs;
            }
        }
        let disc_loss = disc_loss_total / (2.0 * bs);

        self.discriminator
            .sgd_update(&disc_grads, self.config.disc_lr);

        // Optional spectral normalization
        if self.config.spectral_norm {
            spectral_normalize_network(&mut self.discriminator, 3);
        }

        // --- Train Generator ---
        let mut gen_loss_total = 0.0;
        let mut gen_grads: Vec<(Vec<f64>, Vec<f64>)> = self
            .generator
            .layers
            .iter()
            .map(|l| (vec![0.0; l.out_size * l.in_size], vec![0.0; l.out_size]))
            .collect();

        for _ in 0..batch_size {
            let mut z = vec![0.0; self.config.latent_dim];
            rng.fill_normal(&mut z);
            let g_acts = self.generator.forward(&z);
            let fake_sample = g_acts.last().unwrap();

            let d_acts = self.discriminator.forward(fake_sample);
            let d_out = d_acts.last().unwrap();
            let gen_target = vec![1.0];
            gen_loss_total += compute_loss(self.config.loss_type, d_out, &gen_target);
            let d_grad = compute_loss_grad(self.config.loss_type, d_out, &gen_target);

            // Backprop through discriminator to get gradient w.r.t. fake sample
            let grad_fake;
            {
                let mut temp_grad = d_grad;
                for li in (0..self.discriminator.layers.len()).rev() {
                    let lay = &self.discriminator.layers[li];
                    let inp = &d_acts[li];
                    let out = &d_acts[li + 1];
                    let mut gi = vec![0.0; lay.in_size];
                    let mut gw = vec![0.0; lay.out_size * lay.in_size];
                    let mut gb = vec![0.0; lay.out_size];
                    lay.backward(inp, out, &temp_grad, &mut gi, &mut gw, &mut gb);
                    temp_grad = gi;
                }
                grad_fake = temp_grad;
            }

            // Backprop through generator
            let g_layer_grads = self.generator.backward(&g_acts, &grad_fake);
            for (i, (gw, gb)) in g_layer_grads.iter().enumerate() {
                for (a, b) in gen_grads[i].0.iter_mut().zip(gw.iter()) {
                    *a += *b;
                }
                for (a, b) in gen_grads[i].1.iter_mut().zip(gb.iter()) {
                    *a += *b;
                }
            }
        }

        for (gw, gb) in &mut gen_grads {
            for g in gw.iter_mut() {
                *g /= bs;
            }
            for g in gb.iter_mut() {
                *g /= bs;
            }
        }
        let gen_loss = gen_loss_total / bs;

        self.generator.sgd_update(&gen_grads, self.config.gen_lr);

        (disc_loss, gen_loss)
    }
}

impl fmt::Display for Gan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== GAN ===")?;
        writeln!(f, "Loss: {:?}", self.config.loss_type)?;
        writeln!(f, "Latent dim: {}", self.config.latent_dim)?;
        writeln!(f, "Data dim: {}", self.config.data_dim)?;
        writeln!(f, "--- Generator ---")?;
        write!(f, "{}", self.generator)?;
        writeln!(f, "--- Discriminator ---")?;
        write!(f, "{}", self.discriminator)?;
        Ok(())
    }
}
