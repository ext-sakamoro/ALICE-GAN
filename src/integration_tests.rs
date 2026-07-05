//! Integration tests spanning multiple modules.

#![allow(
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::explicit_iter_loop,
    clippy::bool_to_int_with_if,
    clippy::approx_constant,
    clippy::cast_lossless,
    clippy::redundant_clone,
    clippy::format_collect,
    clippy::similar_names,
    clippy::needless_collect,
    clippy::iter_cloned_collect,
    clippy::suboptimal_flops,
    clippy::should_panic_without_expect
)]

use crate::activation::*;
use crate::dense_layer::*;
use crate::gan::*;
use crate::gradient_penalty::*;
use crate::latent_interp::*;
use crate::loss::*;
use crate::mode_collapse::*;
use crate::network::*;
use crate::rng::*;
use crate::spectral_norm::*;
use crate::training::*;
use crate::util::*;

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
