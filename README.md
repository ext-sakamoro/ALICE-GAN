**English** | [日本語](README_JP.md)

# ALICE-GAN

Pure Rust Generative Adversarial Networks framework for the ALICE ecosystem. Provides generator/discriminator networks, multiple loss functions, and training utilities with zero external dependencies.

## Overview

| Item | Value |
|------|-------|
| **Crate** | `alice-gan` |
| **Version** | 1.0.0 |
| **License** | MIT OR Apache-2.0 |
| **Edition** | 2021 |

## Features

- **Generator / Discriminator** — Configurable multi-layer networks with dense layers
- **Activation Functions** — ReLU, LeakyReLU, Tanh, Sigmoid, Linear
- **Loss Functions** — Binary cross-entropy, Wasserstein, and hinge loss
- **Gradient Penalty** — WGAN-GP style gradient penalty for training stability
- **Spectral Normalization** — Weight normalization for discriminator regularization
- **Latent Space Interpolation** — Linear and spherical interpolation between latent vectors
- **Mode Collapse Detection** — Monitor generator output diversity during training
- **Built-in PRNG** — Xorshift64 RNG with normal/uniform sampling (no external deps)

## Architecture

```
alice-gan (lib.rs — single-file crate)
├── Rng                          # Xorshift64 PRNG
├── Activation                   # Activation functions
├── DenseLayer / Network         # Neural network layers
├── Generator / Discriminator    # GAN components
├── LossFunction                 # BCE, Wasserstein, Hinge
├── GradientPenalty              # WGAN-GP regularization
└── GanTrainer                   # Training loop orchestrator
```

## Quick Start

```rust
use alice_gan::{Generator, Discriminator, GanTrainer, Rng};

let mut rng = Rng::new(42);
let gen = Generator::new(&mut rng, 64, &[128, 256], 784);
let disc = Discriminator::new(&mut rng, 784, &[256, 128]);
let mut trainer = GanTrainer::new(gen, disc, 0.0002);
trainer.train_step(&mut rng, 32);
```

## Build

```bash
cargo build
cargo test
cargo clippy -- -W clippy::all
```

## License

MIT OR Apache-2.0 -- see [LICENSE](LICENSE) for details.
