//! Activation functions (`Activation` / `sigmoid`).

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
