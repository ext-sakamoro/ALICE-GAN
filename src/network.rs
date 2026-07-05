//! `Network` — a stack of dense layers.

use std::fmt;

use crate::activation::Activation;
use crate::dense_layer::DenseLayer;
use crate::rng::Rng;

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
