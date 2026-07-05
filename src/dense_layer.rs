//! `DenseLayer` — dense layer.

use crate::activation::Activation;
use crate::rng::Rng;

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
