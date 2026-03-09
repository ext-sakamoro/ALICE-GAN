#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::module_name_repetitions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

//! ALICE-GAN: Pure Rust Generative Adversarial Networks
//!
//! Provides generator/discriminator networks, multiple loss functions,
//! gradient penalty, spectral normalization, latent space interpolation,
//! mode collapse detection, and a training loop.

use std::fmt;

// ---------------------------------------------------------------------------
// Simple PRNG (xorshift64) — no external deps
// ---------------------------------------------------------------------------

/// A simple xorshift64 pseudo-random number generator.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Create a new RNG with the given seed.
    ///
    /// # Panics
    ///
    /// Panics if `seed` is zero.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        assert!(seed != 0, "seed must be non-zero");
        Self { state: seed }
    }

    /// Returns the next `u64` value.
    pub const fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a `f64` in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Returns a `f64` sampled from an approximate standard normal distribution
    /// using the Box-Muller transform.
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Fill a slice with values drawn from N(0, 1).
    pub fn fill_normal(&mut self, buf: &mut [f64]) {
        for v in buf.iter_mut() {
            *v = self.next_normal();
        }
    }

    /// Fill a slice with values in `[0, 1)`.
    pub fn fill_uniform(&mut self, buf: &mut [f64]) {
        for v in buf.iter_mut() {
            *v = self.next_f64();
        }
    }
}

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------

/// Activation function variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Rectified Linear Unit.
    Relu,
    /// Leaky `ReLU` with a fixed negative slope of 0.2.
    LeakyRelu,
    /// Hyperbolic tangent.
    Tanh,
    /// Sigmoid.
    Sigmoid,
    /// Identity (no activation).
    Linear,
}

impl Activation {
    /// Apply the activation element-wise in-place.
    pub fn apply(self, x: &mut [f64]) {
        match self {
            Self::Relu => {
                for v in x.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
            Self::LeakyRelu => {
                for v in x.iter_mut() {
                    if *v < 0.0 {
                        *v *= 0.2;
                    }
                }
            }
            Self::Tanh => {
                for v in x.iter_mut() {
                    *v = v.tanh();
                }
            }
            Self::Sigmoid => {
                for v in x.iter_mut() {
                    *v = 1.0 / (1.0 + (-*v).exp());
                }
            }
            Self::Linear => {}
        }
    }

    /// Derivative of the activation given the *output* value.
    pub fn derivative(self, output: &[f64], grad: &mut [f64]) {
        match self {
            Self::Relu => {
                for (g, o) in grad.iter_mut().zip(output.iter()) {
                    if *o <= 0.0 {
                        *g = 0.0;
                    }
                }
            }
            Self::LeakyRelu => {
                for (g, o) in grad.iter_mut().zip(output.iter()) {
                    if *o < 0.0 {
                        *g *= 0.2;
                    }
                }
            }
            Self::Tanh => {
                for (g, o) in grad.iter_mut().zip(output.iter()) {
                    *g *= 1.0 - o * o;
                }
            }
            Self::Sigmoid => {
                for (g, o) in grad.iter_mut().zip(output.iter()) {
                    *g *= o * (1.0 - o);
                }
            }
            Self::Linear => {}
        }
    }
}

/// Apply sigmoid to a single value.
#[must_use]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// Dense layer
// ---------------------------------------------------------------------------

/// A fully-connected (dense) layer.
#[derive(Debug, Clone)]
pub struct DenseLayer {
    /// Number of input neurons.
    pub in_size: usize,
    /// Number of output neurons.
    pub out_size: usize,
    /// Weight matrix stored row-major: `[out_size][in_size]`.
    pub weights: Vec<f64>,
    /// Bias vector of length `out_size`.
    pub biases: Vec<f64>,
    /// Activation function.
    pub activation: Activation,
}

impl DenseLayer {
    /// Create a new dense layer with Xavier-like initialization.
    #[must_use]
    pub fn new(in_size: usize, out_size: usize, activation: Activation, rng: &mut Rng) -> Self {
        let scale = (2.0 / (in_size + out_size) as f64).sqrt();
        let mut weights = vec![0.0; out_size * in_size];
        for w in &mut weights {
            *w = rng.next_normal() * scale;
        }
        let biases = vec![0.0; out_size];
        Self {
            in_size,
            out_size,
            weights,
            biases,
            activation,
        }
    }

    /// Forward pass: `output = activation(W * input + b)`.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != self.in_size` or `output.len() != self.out_size`.
    pub fn forward(&self, input: &[f64], output: &mut [f64]) {
        assert_eq!(input.len(), self.in_size);
        assert_eq!(output.len(), self.out_size);
        for (i, out_val) in output.iter_mut().enumerate() {
            let mut sum = self.biases[i];
            let row = i * self.in_size;
            for (j, inp_val) in input.iter().enumerate() {
                sum = self.weights[row + j].mul_add(*inp_val, sum);
            }
            *out_val = sum;
        }
        self.activation.apply(output);
    }

    /// Backward pass: computes `grad_input` and accumulates weight/bias gradients.
    pub fn backward(
        &self,
        input: &[f64],
        output: &[f64],
        grad_output: &[f64],
        grad_input: &mut [f64],
        grad_weights: &mut [f64],
        grad_biases: &mut [f64],
    ) {
        let mut local_grad = grad_output.to_vec();
        self.activation.derivative(output, &mut local_grad);

        // grad_biases
        for (gb, lg) in grad_biases.iter_mut().zip(local_grad.iter()) {
            *gb += *lg;
        }

        // grad_weights and grad_input
        for (i, &lg) in local_grad.iter().enumerate() {
            let row = i * self.in_size;
            for (j, &inp) in input.iter().enumerate() {
                grad_weights[row + j] = lg.mul_add(inp, grad_weights[row + j]);
                grad_input[j] = self.weights[row + j].mul_add(lg, grad_input[j]);
            }
        }
    }

    /// Total number of parameters.
    #[must_use]
    pub const fn param_count(&self) -> usize {
        self.out_size * self.in_size + self.out_size
    }

    /// Apply SGD update with the given learning rate.
    pub fn sgd_update(&mut self, grad_weights: &[f64], grad_biases: &[f64], lr: f64) {
        for (w, gw) in self.weights.iter_mut().zip(grad_weights.iter()) {
            *w -= lr * gw;
        }
        for (b, gb) in self.biases.iter_mut().zip(grad_biases.iter()) {
            *b -= lr * gb;
        }
    }
}

// ---------------------------------------------------------------------------
// Network — a stack of dense layers
// ---------------------------------------------------------------------------

/// A multi-layer feedforward network.
#[derive(Debug, Clone)]
pub struct Network {
    /// The layers of the network.
    pub layers: Vec<DenseLayer>,
}

impl Network {
    /// Create a network from a list of `(out_size, activation)` specs.
    /// The first element in `layer_specs` uses `in_size` as its input dimension.
    #[must_use]
    pub fn new(in_size: usize, layer_specs: &[(usize, Activation)], rng: &mut Rng) -> Self {
        let mut layers = Vec::with_capacity(layer_specs.len());
        let mut prev = in_size;
        for &(out, act) in layer_specs {
            layers.push(DenseLayer::new(prev, out, act, rng));
            prev = out;
        }
        Self { layers }
    }

    /// Output dimension of the network.
    #[must_use]
    pub fn output_size(&self) -> usize {
        self.layers.last().map_or(0, |l| l.out_size)
    }

    /// Input dimension of the network.
    #[must_use]
    pub fn input_size(&self) -> usize {
        self.layers.first().map_or(0, |l| l.in_size)
    }

    /// Forward pass through all layers, returning intermediate activations.
    ///
    /// # Panics
    ///
    /// Panics if any layer's input size doesn't match the previous layer's output size.
    #[must_use]
    pub fn forward(&self, input: &[f64]) -> Vec<Vec<f64>> {
        let mut activations = Vec::with_capacity(self.layers.len() + 1);
        activations.push(input.to_vec());
        for layer in &self.layers {
            let prev = activations.last().unwrap();
            let mut out = vec![0.0; layer.out_size];
            layer.forward(prev, &mut out);
            activations.push(out);
        }
        activations
    }

    /// Forward pass returning only the final output.
    #[must_use]
    pub fn predict(&self, input: &[f64]) -> Vec<f64> {
        let acts = self.forward(input);
        acts.into_iter().last().unwrap_or_default()
    }

    /// Total number of trainable parameters.
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.layers.iter().map(DenseLayer::param_count).sum()
    }

    /// Backward pass, returning weight and bias gradients for each layer.
    #[must_use]
    pub fn backward(
        &self,
        activations: &[Vec<f64>],
        loss_grad: &[f64],
    ) -> Vec<(Vec<f64>, Vec<f64>)> {
        let n = self.layers.len();
        let mut grads = Vec::with_capacity(n);
        let mut current_grad = loss_grad.to_vec();
        for i in (0..n).rev() {
            let layer = &self.layers[i];
            let input = &activations[i];
            let output = &activations[i + 1];
            let mut gw = vec![0.0; layer.out_size * layer.in_size];
            let mut gb = vec![0.0; layer.out_size];
            let mut gi = vec![0.0; layer.in_size];
            layer.backward(input, output, &current_grad, &mut gi, &mut gw, &mut gb);
            grads.push((gw, gb));
            current_grad = gi;
        }
        grads.reverse();
        grads
    }

    /// Apply SGD updates.
    pub fn sgd_update(&mut self, grads: &[(Vec<f64>, Vec<f64>)], lr: f64) {
        for (layer, (gw, gb)) in self.layers.iter_mut().zip(grads.iter()) {
            layer.sgd_update(gw, gb, lr);
        }
    }
}

impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Network({} layers, {} params)",
            self.layers.len(),
            self.param_count()
        )?;
        for (i, l) in self.layers.iter().enumerate() {
            writeln!(
                f,
                "  Layer {i}: {} -> {} ({:?})",
                l.in_size, l.out_size, l.activation
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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
#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> Rng {
        Rng::new(12345)
    }

    // --- RNG tests ---

    #[test]
    fn test_rng_deterministic() {
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn test_rng_range() {
        let mut rng = make_rng();
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_rng_normal_mean() {
        let mut rng = make_rng();
        let n = 10_000;
        let sum: f64 = (0..n).map(|_| rng.next_normal()).sum();
        let mean = sum / n as f64;
        assert!(mean.abs() < 0.1, "mean = {mean}");
    }

    #[test]
    fn test_fill_normal() {
        let mut rng = make_rng();
        let mut buf = vec![0.0; 100];
        rng.fill_normal(&mut buf);
        assert!(buf.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_fill_uniform() {
        let mut rng = make_rng();
        let mut buf = vec![0.0; 100];
        rng.fill_uniform(&mut buf);
        assert!(buf.iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    #[should_panic]
    fn test_rng_zero_seed_panics() {
        let _ = Rng::new(0);
    }

    // --- Activation tests ---

    #[test]
    fn test_relu() {
        let mut v = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        Activation::Relu.apply(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_leaky_relu() {
        let mut v = vec![-10.0, 0.0, 5.0];
        Activation::LeakyRelu.apply(&mut v);
        assert!((v[0] - (-2.0)).abs() < 1e-10);
        assert!((v[1]).abs() < 1e-10);
        assert!((v[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_activation() {
        let mut v = vec![0.0];
        Activation::Sigmoid.apply(&mut v);
        assert!((v[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tanh_activation() {
        let mut v = vec![0.0];
        Activation::Tanh.apply(&mut v);
        assert!(v[0].abs() < 1e-10);
    }

    #[test]
    fn test_linear_activation() {
        let mut v = vec![3.14, -2.71];
        let original = v.clone();
        Activation::Linear.apply(&mut v);
        assert_eq!(v, original);
    }

    #[test]
    fn test_relu_derivative() {
        let output = vec![0.0, 1.0, -0.5, 2.0];
        let mut grad = vec![1.0, 1.0, 1.0, 1.0];
        Activation::Relu.derivative(&output, &mut grad);
        assert_eq!(grad, vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_sigmoid_derivative() {
        let output = vec![0.5];
        let mut grad = vec![1.0];
        Activation::Sigmoid.derivative(&output, &mut grad);
        assert!((grad[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_tanh_derivative() {
        let output = vec![0.0];
        let mut grad = vec![1.0];
        Activation::Tanh.derivative(&output, &mut grad);
        assert!((grad[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_leaky_relu_derivative() {
        let output = vec![-1.0, 1.0];
        let mut grad = vec![1.0, 1.0];
        Activation::LeakyRelu.derivative(&output, &mut grad);
        assert!((grad[0] - 0.2).abs() < 1e-10);
        assert!((grad[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_derivative() {
        let output = vec![5.0];
        let mut grad = vec![3.0];
        Activation::Linear.derivative(&output, &mut grad);
        assert!((grad[0] - 3.0).abs() < 1e-10);
    }

    // --- Sigmoid function ---

    #[test]
    fn test_sigmoid_fn() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(100.0) > 0.999);
        assert!(sigmoid(-100.0) < 0.001);
    }

    // --- Dense layer tests ---

    #[test]
    fn test_dense_forward_shape() {
        let mut rng = make_rng();
        let layer = DenseLayer::new(4, 3, Activation::Relu, &mut rng);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 3];
        layer.forward(&input, &mut output);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_dense_param_count() {
        let mut rng = make_rng();
        let layer = DenseLayer::new(10, 5, Activation::Relu, &mut rng);
        assert_eq!(layer.param_count(), 55); // 10*5 + 5
    }

    #[test]
    fn test_dense_backward_shapes() {
        let mut rng = make_rng();
        let layer = DenseLayer::new(3, 2, Activation::Relu, &mut rng);
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 2];
        layer.forward(&input, &mut output);
        let grad_output = vec![1.0, 1.0];
        let mut gi = vec![0.0; 3];
        let mut gw = vec![0.0; 6];
        let mut gb = vec![0.0; 2];
        layer.backward(&input, &output, &grad_output, &mut gi, &mut gw, &mut gb);
        assert_eq!(gi.len(), 3);
        assert_eq!(gw.len(), 6);
        assert_eq!(gb.len(), 2);
    }

    #[test]
    fn test_dense_sgd_update() {
        let mut rng = make_rng();
        let mut layer = DenseLayer::new(2, 2, Activation::Linear, &mut rng);
        let old_w = layer.weights.clone();
        let gw = vec![1.0; 4];
        let gb = vec![1.0; 2];
        layer.sgd_update(&gw, &gb, 0.1);
        for (o, n) in old_w.iter().zip(layer.weights.iter()) {
            assert!((n - (o - 0.1)).abs() < 1e-10);
        }
    }

    // --- Network tests ---

    #[test]
    fn test_network_forward() {
        let mut rng = make_rng();
        let net = Network::new(
            4,
            &[(8, Activation::Relu), (2, Activation::Sigmoid)],
            &mut rng,
        );
        let input = vec![1.0, 0.0, -1.0, 0.5];
        let output = net.predict(&input);
        assert_eq!(output.len(), 2);
        for &v in &output {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_network_output_size() {
        let mut rng = make_rng();
        let net = Network::new(
            5,
            &[(10, Activation::Relu), (3, Activation::Tanh)],
            &mut rng,
        );
        assert_eq!(net.output_size(), 3);
        assert_eq!(net.input_size(), 5);
    }

    #[test]
    fn test_network_param_count() {
        let mut rng = make_rng();
        let net = Network::new(
            4,
            &[(8, Activation::Relu), (2, Activation::Linear)],
            &mut rng,
        );
        // layer1: 4*8+8=40, layer2: 8*2+2=18 => total 58
        assert_eq!(net.param_count(), 58);
    }

    #[test]
    fn test_network_activations_count() {
        let mut rng = make_rng();
        let net = Network::new(
            3,
            &[(5, Activation::Relu), (2, Activation::Linear)],
            &mut rng,
        );
        let acts = net.forward(&[1.0, 2.0, 3.0]);
        // input + 2 layers = 3 activation vectors
        assert_eq!(acts.len(), 3);
    }

    #[test]
    fn test_network_backward() {
        let mut rng = make_rng();
        let net = Network::new(
            3,
            &[(4, Activation::Relu), (1, Activation::Linear)],
            &mut rng,
        );
        let acts = net.forward(&[1.0, 2.0, 3.0]);
        let loss_grad = vec![1.0];
        let grads = net.backward(&acts, &loss_grad);
        assert_eq!(grads.len(), 2);
    }

    #[test]
    fn test_network_display() {
        let mut rng = make_rng();
        let net = Network::new(2, &[(4, Activation::Relu)], &mut rng);
        let s = format!("{net}");
        assert!(s.contains("Network"));
        assert!(s.contains("Layer 0"));
    }

    #[test]
    fn test_network_sgd_changes_weights() {
        let mut rng = make_rng();
        let mut net = Network::new(2, &[(3, Activation::Linear)], &mut rng);
        let old_w = net.layers[0].weights.clone();
        let acts = net.forward(&[1.0, 2.0]);
        let grads = net.backward(&acts, &[1.0, 1.0, 1.0]);
        net.sgd_update(&grads, 0.01);
        assert!(net.layers[0].weights != old_w);
    }

    // --- Loss function tests ---

    #[test]
    fn test_bce_loss_perfect() {
        let preds = vec![0.999, 0.001];
        let targets = vec![1.0, 0.0];
        let loss = bce_loss(&preds, &targets);
        assert!(loss < 0.01, "loss = {loss}");
    }

    #[test]
    fn test_bce_loss_worst() {
        let preds = vec![0.001, 0.999];
        let targets = vec![1.0, 0.0];
        let loss = bce_loss(&preds, &targets);
        assert!(loss > 5.0, "loss = {loss}");
    }

    #[test]
    fn test_bce_loss_grad_shape() {
        let preds = vec![0.5, 0.5];
        let targets = vec![1.0, 0.0];
        let grad = bce_loss_grad(&preds, &targets);
        assert_eq!(grad.len(), 2);
    }

    #[test]
    fn test_bce_loss_grad_direction() {
        let preds = vec![0.3];
        let targets = vec![1.0];
        let grad = bce_loss_grad(&preds, &targets);
        // grad should be negative (decrease loss by increasing pred)
        assert!(grad[0] < 0.0);
    }

    #[test]
    fn test_wasserstein_loss() {
        let preds = vec![1.0, -1.0];
        let targets = vec![1.0, -1.0];
        let loss = wasserstein_loss(&preds, &targets);
        // -(1*1 + (-1)*(-1))/2 = -1
        assert!((loss - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_wasserstein_loss_grad() {
        let preds = vec![0.5];
        let targets = vec![1.0];
        let grad = wasserstein_loss_grad(&preds, &targets);
        assert!((grad[0] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_hinge_loss_correct() {
        let preds = vec![2.0, -2.0];
        let targets = vec![1.0, -1.0];
        let loss = hinge_loss(&preds, &targets);
        // max(0, 1-2) + max(0, 1-2) = 0
        assert!(loss.abs() < 1e-10);
    }

    #[test]
    fn test_hinge_loss_wrong() {
        let preds = vec![-2.0];
        let targets = vec![1.0];
        let loss = hinge_loss(&preds, &targets);
        // max(0, 1-(-2)) = 3
        assert!((loss - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_hinge_loss_grad() {
        let preds = vec![0.5];
        let targets = vec![1.0];
        let grad = hinge_loss_grad(&preds, &targets);
        // 1 - 1*0.5 = 0.5 > 0, so grad = -1/1 = -1
        assert!((grad[0] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_hinge_loss_grad_zero_region() {
        let preds = vec![2.0];
        let targets = vec![1.0];
        let grad = hinge_loss_grad(&preds, &targets);
        assert!(grad[0].abs() < 1e-10);
    }

    #[test]
    fn test_compute_loss_bce() {
        let p = vec![0.5];
        let t = vec![1.0];
        let l1 = bce_loss(&p, &t);
        let l2 = compute_loss(LossType::Bce, &p, &t);
        assert!((l1 - l2).abs() < 1e-10);
    }

    #[test]
    fn test_compute_loss_wasserstein() {
        let p = vec![0.5];
        let t = vec![1.0];
        let l1 = wasserstein_loss(&p, &t);
        let l2 = compute_loss(LossType::Wasserstein, &p, &t);
        assert!((l1 - l2).abs() < 1e-10);
    }

    #[test]
    fn test_compute_loss_hinge() {
        let p = vec![0.5];
        let t = vec![1.0];
        let l1 = hinge_loss(&p, &t);
        let l2 = compute_loss(LossType::Hinge, &p, &t);
        assert!((l1 - l2).abs() < 1e-10);
    }

    #[test]
    fn test_compute_loss_grad_dispatch() {
        let p = vec![0.5];
        let t = vec![1.0];
        let g1 = bce_loss_grad(&p, &t);
        let g2 = compute_loss_grad(LossType::Bce, &p, &t);
        assert_eq!(g1, g2);
    }

    // --- Gradient penalty tests ---

    #[test]
    fn test_gradient_penalty_finite() {
        let mut rng = make_rng();
        let config = GanConfig {
            latent_dim: 4,
            data_dim: 4,
            gen_hidden: vec![8],
            disc_hidden: vec![8],
            ..GanConfig::default()
        };
        let gan = Gan::new(config, &mut rng);
        let real = vec![1.0, 0.5, -0.5, 0.0];
        let fake = vec![0.0, -0.5, 0.5, 1.0];
        let gp = gradient_penalty(&gan.discriminator, &real, &fake, &mut rng);
        assert!(gp.is_finite(), "gp = {gp}");
        assert!(gp >= 0.0);
    }

    #[test]
    fn test_gradient_penalty_nonnegative() {
        let mut rng = make_rng();
        let net = Network::new(
            2,
            &[(4, Activation::Relu), (1, Activation::Linear)],
            &mut rng,
        );
        let real = vec![1.0, 1.0];
        let fake = vec![-1.0, -1.0];
        let gp = gradient_penalty(&net, &real, &fake, &mut rng);
        assert!(gp >= 0.0);
    }

    // --- Spectral normalization tests ---

    #[test]
    fn test_spectral_normalize_reduces_norm() {
        let mut weights = vec![10.0, 0.0, 0.0, 10.0];
        spectral_normalize(&mut weights, 2, 2, 10);
        let max_w = weights.iter().map(|w| w.abs()).fold(0.0_f64, f64::max);
        assert!(max_w <= 1.0 + 1e-6, "max_w = {max_w}");
    }

    #[test]
    fn test_spectral_normalize_identity() {
        // Identity matrix has singular value 1, should remain ~unchanged
        let mut weights = vec![1.0, 0.0, 0.0, 1.0];
        spectral_normalize(&mut weights, 2, 2, 20);
        assert!((weights[0] - 1.0).abs() < 0.1);
        assert!((weights[3] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_spectral_normalize_empty() {
        let mut weights: Vec<f64> = Vec::new();
        spectral_normalize(&mut weights, 0, 0, 5);
        assert!(weights.is_empty());
    }

    #[test]
    fn test_spectral_normalize_network_runs() {
        let mut rng = make_rng();
        let mut net = Network::new(
            4,
            &[(8, Activation::Relu), (1, Activation::Linear)],
            &mut rng,
        );
        spectral_normalize_network(&mut net, 5);
        // Just check it doesn't panic and weights are finite
        for layer in &net.layers {
            assert!(layer.weights.iter().all(|w| w.is_finite()));
        }
    }

    // --- Latent interpolation tests ---

    #[test]
    fn test_lerp_endpoints() {
        let z1 = vec![0.0, 0.0];
        let z2 = vec![1.0, 1.0];
        let r0 = lerp(&z1, &z2, 0.0);
        let r1 = lerp(&z1, &z2, 1.0);
        assert_eq!(r0, z1);
        assert_eq!(r1, z2);
    }

    #[test]
    fn test_lerp_midpoint() {
        let z1 = vec![0.0, 0.0];
        let z2 = vec![2.0, 4.0];
        let mid = lerp(&z1, &z2, 0.5);
        assert!((mid[0] - 1.0).abs() < 1e-10);
        assert!((mid[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_slerp_endpoints() {
        let z1 = vec![1.0, 0.0];
        let z2 = vec![0.0, 1.0];
        let r0 = slerp(&z1, &z2, 0.0);
        let r1 = slerp(&z1, &z2, 1.0);
        assert!((r0[0] - 1.0).abs() < 1e-6);
        assert!((r0[1]).abs() < 1e-6);
        assert!((r1[0]).abs() < 1e-6);
        assert!((r1[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_slerp_maintains_norm() {
        let z1 = vec![1.0, 0.0, 0.0];
        let z2 = vec![0.0, 1.0, 0.0];
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            let interp = slerp(&z1, &z2, t);
            let norm = l2_norm(&interp);
            assert!((norm - 1.0).abs() < 1e-6, "t={t}, norm={norm}");
        }
    }

    #[test]
    fn test_slerp_collinear_fallback() {
        let z1 = vec![1.0, 0.0];
        let z2 = vec![2.0, 0.0]; // same direction
        let mid = slerp(&z1, &z2, 0.5);
        // Should fallback to lerp
        assert!((mid[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_latent_count() {
        let mut rng = make_rng();
        let gen = Network::new(4, &[(8, Activation::Tanh)], &mut rng);
        let z1 = vec![1.0, 0.0, 0.0, 0.0];
        let z2 = vec![0.0, 0.0, 0.0, 1.0];
        let results = interpolate_latent(&gen, &z1, &z2, 5, false);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_interpolate_latent_slerp() {
        let mut rng = make_rng();
        let gen = Network::new(4, &[(8, Activation::Tanh)], &mut rng);
        let z1 = vec![1.0, 0.0, 0.0, 0.0];
        let z2 = vec![0.0, 1.0, 0.0, 0.0];
        let results = interpolate_latent(&gen, &z1, &z2, 3, true);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_interpolate_single_step() {
        let mut rng = make_rng();
        let gen = Network::new(2, &[(4, Activation::Tanh)], &mut rng);
        let z1 = vec![1.0, 0.0];
        let z2 = vec![0.0, 1.0];
        let results = interpolate_latent(&gen, &z1, &z2, 1, false);
        assert_eq!(results.len(), 1);
    }

    // --- Mode collapse detection tests ---

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_detect_collapse_identical_samples() {
        let samples = vec![vec![1.0, 2.0]; 10];
        let stats = detect_mode_collapse(&samples, 0.95);
        assert!(stats.collapsed);
        assert!((stats.mean_similarity - 1.0).abs() < 1e-10);
        assert!(stats.mean_std_dev.abs() < 1e-10);
    }

    #[test]
    fn test_detect_collapse_diverse_samples() {
        let mut rng = make_rng();
        let samples: Vec<Vec<f64>> = (0..20)
            .map(|_| {
                let mut v = vec![0.0; 8];
                rng.fill_normal(&mut v);
                v
            })
            .collect();
        let stats = detect_mode_collapse(&samples, 0.95);
        assert!(!stats.collapsed);
    }

    #[test]
    fn test_detect_collapse_single_sample() {
        let samples = vec![vec![1.0, 2.0]];
        let stats = detect_mode_collapse(&samples, 0.95);
        assert!(!stats.collapsed);
    }

    #[test]
    fn test_detect_collapse_empty() {
        let samples: Vec<Vec<f64>> = Vec::new();
        let stats = detect_mode_collapse(&samples, 0.95);
        assert!(!stats.collapsed);
    }

    #[test]
    fn test_collapse_stats_display() {
        let stats = CollapseStats {
            mean_similarity: 0.5,
            mean_std_dev: 1.0,
            collapsed: false,
        };
        let s = format!("{stats}");
        assert!(s.contains("0.5000"));
    }

    // --- GAN tests ---

    #[test]
    fn test_gan_creation() {
        let mut rng = make_rng();
        let config = GanConfig::default();
        let gan = Gan::new(config, &mut rng);
        assert_eq!(gan.generator.input_size(), 16);
        assert_eq!(gan.generator.output_size(), 8);
        assert_eq!(gan.discriminator.input_size(), 8);
        assert_eq!(gan.discriminator.output_size(), 1);
    }

    #[test]
    fn test_gan_generate() {
        let mut rng = make_rng();
        let config = GanConfig::default();
        let gan = Gan::new(config, &mut rng);
        let sample = gan.generate(&mut rng);
        assert_eq!(sample.len(), 8);
        // Tanh output
        for &v in &sample {
            assert!((-1.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_gan_generate_batch() {
        let mut rng = make_rng();
        let config = GanConfig::default();
        let gan = Gan::new(config, &mut rng);
        let batch = gan.generate_batch(10, &mut rng);
        assert_eq!(batch.len(), 10);
        assert_eq!(batch[0].len(), 8);
    }

    #[test]
    fn test_gan_train_step_bce() {
        let mut rng = make_rng();
        let config = GanConfig {
            latent_dim: 4,
            data_dim: 4,
            gen_hidden: vec![8],
            disc_hidden: vec![8],
            loss_type: LossType::Bce,
            ..GanConfig::default()
        };
        let mut gan = Gan::new(config, &mut rng);
        let real_batch: Vec<Vec<f64>> = (0..4)
            .map(|_| {
                let mut v = vec![0.0; 4];
                rng.fill_uniform(&mut v);
                v
            })
            .collect();
        let (d_loss, g_loss) = gan.train_step(&real_batch, &mut rng);
        assert!(d_loss.is_finite());
        assert!(g_loss.is_finite());
    }

    #[test]
    fn test_gan_train_step_wasserstein() {
        let mut rng = make_rng();
        let config = GanConfig {
            latent_dim: 4,
            data_dim: 4,
            gen_hidden: vec![8],
            disc_hidden: vec![8],
            loss_type: LossType::Wasserstein,
            ..GanConfig::default()
        };
        let mut gan = Gan::new(config, &mut rng);
        let real_batch: Vec<Vec<f64>> = (0..4)
            .map(|_| {
                let mut v = vec![0.0; 4];
                rng.fill_normal(&mut v);
                v
            })
            .collect();
        let (d_loss, g_loss) = gan.train_step(&real_batch, &mut rng);
        assert!(d_loss.is_finite());
        assert!(g_loss.is_finite());
    }

    #[test]
    fn test_gan_train_step_hinge() {
        let mut rng = make_rng();
        let config = GanConfig {
            latent_dim: 4,
            data_dim: 4,
            gen_hidden: vec![8],
            disc_hidden: vec![8],
            loss_type: LossType::Hinge,
            ..GanConfig::default()
        };
        let mut gan = Gan::new(config, &mut rng);
        let real_batch: Vec<Vec<f64>> = (0..4)
            .map(|_| {
                let mut v = vec![0.0; 4];
                rng.fill_normal(&mut v);
                v
            })
            .collect();
        let (d_loss, g_loss) = gan.train_step(&real_batch, &mut rng);
        assert!(d_loss.is_finite());
        assert!(g_loss.is_finite());
    }

    #[test]
    fn test_gan_with_spectral_norm() {
        let mut rng = make_rng();
        let config = GanConfig {
            latent_dim: 4,
            data_dim: 4,
            gen_hidden: vec![8],
            disc_hidden: vec![8],
            spectral_norm: true,
            ..GanConfig::default()
        };
        let mut gan = Gan::new(config, &mut rng);
        let real_batch: Vec<Vec<f64>> = (0..4)
            .map(|_| {
                let mut v = vec![0.0; 4];
                rng.fill_uniform(&mut v);
                v
            })
            .collect();
        let (d, g) = gan.train_step(&real_batch, &mut rng);
        assert!(d.is_finite());
        assert!(g.is_finite());
    }

    #[test]
    fn test_gan_display() {
        let mut rng = make_rng();
        let config = GanConfig::default();
        let gan = Gan::new(config, &mut rng);
        let s = format!("{gan}");
        assert!(s.contains("GAN"));
        assert!(s.contains("Generator"));
        assert!(s.contains("Discriminator"));
    }

    #[test]
    fn test_gan_default_config() {
        let config = GanConfig::default();
        assert_eq!(config.latent_dim, 16);
        assert_eq!(config.data_dim, 8);
        assert_eq!(config.loss_type, LossType::Bce);
    }

    // --- Training loop tests ---

    #[test]
    fn test_train_runs() {
        let mut rng = make_rng();
        let config = GanConfig {
            latent_dim: 4,
            data_dim: 4,
            gen_hidden: vec![8],
            disc_hidden: vec![8],
            ..GanConfig::default()
        };
        let mut gan = Gan::new(config, &mut rng);
        let train_config = TrainConfig {
            epochs: 3,
            batch_size: 4,
            ..TrainConfig::default()
        };
        let data_fn = |batch_size: usize, rng: &mut Rng| -> Vec<Vec<f64>> {
            (0..batch_size)
                .map(|_| {
                    let mut v = vec![0.0; 4];
                    rng.fill_uniform(&mut v);
                    v
                })
                .collect()
        };
        let history = train(&mut gan, &train_config, &data_fn, &mut rng);
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn test_train_with_collapse_check() {
        let mut rng = make_rng();
        let config = GanConfig {
            latent_dim: 4,
            data_dim: 4,
            gen_hidden: vec![8],
            disc_hidden: vec![8],
            ..GanConfig::default()
        };
        let mut gan = Gan::new(config, &mut rng);
        let train_config = TrainConfig {
            epochs: 2,
            batch_size: 4,
            check_collapse: true,
            ..TrainConfig::default()
        };
        let data_fn = |batch_size: usize, rng: &mut Rng| -> Vec<Vec<f64>> {
            (0..batch_size)
                .map(|_| {
                    let mut v = vec![0.0; 4];
                    rng.fill_uniform(&mut v);
                    v
                })
                .collect()
        };
        let history = train(&mut gan, &train_config, &data_fn, &mut rng);
        assert!(history[0].collapse_stats.is_some());
    }

    #[test]
    fn test_train_config_default() {
        let tc = TrainConfig::default();
        assert_eq!(tc.epochs, 100);
        assert_eq!(tc.batch_size, 16);
        assert_eq!(tc.disc_steps, 1);
        assert!(!tc.check_collapse);
    }

    #[test]
    fn test_epoch_record_fields() {
        let record = EpochRecord {
            epoch: 0,
            disc_loss: 0.5,
            gen_loss: 0.7,
            collapse_stats: None,
        };
        assert_eq!(record.epoch, 0);
        assert!((record.disc_loss - 0.5).abs() < 1e-10);
    }

    // --- Utility function tests ---

    #[test]
    fn test_l2_norm() {
        let v = vec![3.0, 4.0];
        assert!((l2_norm(&v) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_l2_norm_zero() {
        let v = vec![0.0, 0.0];
        assert!(l2_norm(&v).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let n = normalize(&v);
        assert!((l2_norm(&n) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_zero() {
        let v = vec![0.0, 0.0];
        let n = normalize(&v);
        // Should not panic, norm is clamped
        assert!(n.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_batch_mean() {
        let batch = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let mean = batch_mean(&batch);
        assert!((mean[0] - 2.0).abs() < 1e-10);
        assert!((mean[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_batch_mean_empty() {
        let batch: Vec<Vec<f64>> = Vec::new();
        let mean = batch_mean(&batch);
        assert!(mean.is_empty());
    }

    #[test]
    fn test_batch_variance() {
        let batch = vec![vec![1.0], vec![3.0]];
        let var = batch_variance(&batch);
        // mean=2, var = ((1-2)^2 + (3-2)^2)/2 = 1
        assert!((var[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_batch_variance_empty() {
        let batch: Vec<Vec<f64>> = Vec::new();
        let var = batch_variance(&batch);
        assert!(var.is_empty());
    }

    #[test]
    fn test_batch_variance_constant() {
        let batch = vec![vec![5.0, 5.0]; 10];
        let var = batch_variance(&batch);
        assert!(var[0].abs() < 1e-10);
        assert!(var[1].abs() < 1e-10);
    }

    // --- Integration tests ---

    #[test]
    fn test_full_pipeline_bce() {
        let mut rng = make_rng();
        let config = GanConfig {
            latent_dim: 4,
            data_dim: 2,
            gen_hidden: vec![8],
            disc_hidden: vec![8],
            loss_type: LossType::Bce,
            gen_lr: 0.01,
            disc_lr: 0.01,
            gp_lambda: 0.0,
            spectral_norm: false,
        };
        let mut gan = Gan::new(config, &mut rng);
        for _ in 0..5 {
            let real: Vec<Vec<f64>> = (0..4)
                .map(|_| vec![rng.next_f64() * 0.5, rng.next_f64() * 0.5])
                .collect();
            let (d, g) = gan.train_step(&real, &mut rng);
            assert!(d.is_finite());
            assert!(g.is_finite());
        }
        let samples = gan.generate_batch(10, &mut rng);
        let stats = detect_mode_collapse(&samples, 0.99);
        assert!(stats.mean_similarity.is_finite());
    }

    #[test]
    fn test_full_pipeline_wasserstein_gp() {
        let mut rng = make_rng();
        let config = GanConfig {
            latent_dim: 4,
            data_dim: 2,
            gen_hidden: vec![8],
            disc_hidden: vec![8],
            loss_type: LossType::Wasserstein,
            gen_lr: 0.001,
            disc_lr: 0.001,
            gp_lambda: 10.0,
            spectral_norm: false,
        };
        let mut gan = Gan::new(config, &mut rng);
        let real: Vec<Vec<f64>> = (0..4)
            .map(|_| vec![rng.next_f64(), rng.next_f64()])
            .collect();
        let (d, g) = gan.train_step(&real, &mut rng);
        assert!(d.is_finite());
        assert!(g.is_finite());
    }

    #[test]
    fn test_full_pipeline_hinge_spectral() {
        let mut rng = make_rng();
        let config = GanConfig {
            latent_dim: 4,
            data_dim: 2,
            gen_hidden: vec![8],
            disc_hidden: vec![8],
            loss_type: LossType::Hinge,
            gen_lr: 0.001,
            disc_lr: 0.001,
            gp_lambda: 0.0,
            spectral_norm: true,
        };
        let mut gan = Gan::new(config, &mut rng);
        let real: Vec<Vec<f64>> = (0..4)
            .map(|_| vec![rng.next_f64(), rng.next_f64()])
            .collect();
        let (d, g) = gan.train_step(&real, &mut rng);
        assert!(d.is_finite());
        assert!(g.is_finite());
    }

    #[test]
    fn test_interpolation_through_generator() {
        let mut rng = make_rng();
        let config = GanConfig {
            latent_dim: 4,
            data_dim: 2,
            gen_hidden: vec![8],
            disc_hidden: vec![8],
            ..GanConfig::default()
        };
        let gan = Gan::new(config, &mut rng);
        let z1 = vec![1.0, 0.0, 0.0, 0.0];
        let z2 = vec![0.0, 0.0, 0.0, 1.0];
        let interp = interpolate_latent(&gan.generator, &z1, &z2, 5, true);
        assert_eq!(interp.len(), 5);
        for sample in &interp {
            assert_eq!(sample.len(), 2);
        }
    }

    #[test]
    fn test_gradient_penalty_in_training() {
        let mut rng = make_rng();
        let config = GanConfig {
            latent_dim: 4,
            data_dim: 4,
            gen_hidden: vec![8],
            disc_hidden: vec![8],
            gp_lambda: 10.0,
            ..GanConfig::default()
        };
        let gan = Gan::new(config, &mut rng);
        let real = vec![1.0, 0.5, 0.0, -0.5];
        let fake = gan.generate(&mut rng);
        let gp = gradient_penalty(&gan.discriminator, &real, &fake, &mut rng);
        assert!(gp.is_finite());
    }

    #[test]
    fn test_loss_type_eq() {
        assert_eq!(LossType::Bce, LossType::Bce);
        assert_ne!(LossType::Bce, LossType::Wasserstein);
        assert_ne!(LossType::Wasserstein, LossType::Hinge);
    }

    #[test]
    fn test_activation_eq() {
        assert_eq!(Activation::Relu, Activation::Relu);
        assert_ne!(Activation::Relu, Activation::Tanh);
    }

    #[test]
    fn test_gan_clone() {
        let mut rng = make_rng();
        let config = GanConfig::default();
        let gan = Gan::new(config, &mut rng);
        let gan2 = gan.clone();
        assert_eq!(gan.generator.param_count(), gan2.generator.param_count());
    }

    #[test]
    fn test_network_clone() {
        let mut rng = make_rng();
        let net = Network::new(3, &[(5, Activation::Relu)], &mut rng);
        let net2 = net.clone();
        assert_eq!(net.layers[0].weights, net2.layers[0].weights);
    }

    #[test]
    fn test_rng_clone() {
        let mut r1 = Rng::new(42);
        let mut r2 = r1.clone();
        assert_eq!(r1.next_u64(), r2.next_u64());
    }

    #[test]
    fn test_rng_different_seeds() {
        let mut r1 = Rng::new(1);
        let mut r2 = Rng::new(2);
        assert_ne!(r1.next_u64(), r2.next_u64());
    }

    #[test]
    fn test_dense_relu_clips_negative() {
        let mut rng = make_rng();
        let mut layer = DenseLayer::new(2, 2, Activation::Relu, &mut rng);
        // Set weights to produce negative outputs
        layer.weights = vec![-10.0, 0.0, 0.0, -10.0];
        layer.biases = vec![-1.0, -1.0];
        let input = vec![1.0, 1.0];
        let mut output = vec![0.0; 2];
        layer.forward(&input, &mut output);
        assert!(output[0] >= 0.0);
        assert!(output[1] >= 0.0);
    }

    #[test]
    fn test_bce_loss_symmetric() {
        let loss1 = bce_loss(&[0.9], &[1.0]);
        let loss2 = bce_loss(&[0.1], &[0.0]);
        assert!((loss1 - loss2).abs() < 1e-10);
    }

    #[test]
    fn test_wasserstein_loss_antisymmetric() {
        let l1 = wasserstein_loss(&[1.0], &[1.0]);
        let l2 = wasserstein_loss(&[1.0], &[-1.0]);
        assert!((l1 + l2).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_normalize_rectangular() {
        let mut weights = vec![5.0, 0.0, 0.0, 0.0, 5.0, 0.0];
        spectral_normalize(&mut weights, 2, 3, 10);
        assert!(weights.iter().all(|w| w.is_finite()));
    }

    #[test]
    fn test_lerp_quarter() {
        let z1 = vec![0.0];
        let z2 = vec![4.0];
        let r = lerp(&z1, &z2, 0.25);
        assert!((r[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_batch_mean_single() {
        let batch = vec![vec![3.0, 7.0]];
        let mean = batch_mean(&batch);
        assert!((mean[0] - 3.0).abs() < 1e-10);
        assert!((mean[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_network_empty_layers() {
        let mut rng = make_rng();
        let net = Network::new(5, &[], &mut rng);
        assert_eq!(net.output_size(), 0);
        assert_eq!(net.input_size(), 0);
        assert_eq!(net.param_count(), 0);
    }
}
