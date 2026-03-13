[English](README.md) | **日本語**

# ALICE-GAN

ALICEエコシステムの純Rust敵対的生成ネットワーク (GAN) フレームワーク。Generator/Discriminatorネットワーク、複数の損失関数、学習ユーティリティを外部依存なしで提供。

## 概要

| 項目 | 値 |
|------|-----|
| **クレート名** | `alice-gan` |
| **バージョン** | 1.0.0 |
| **ライセンス** | MIT OR Apache-2.0 |
| **エディション** | 2021 |

## 機能

- **Generator / Discriminator** — 全結合層による設定可能な多層ネットワーク
- **活性化関数** — ReLU、LeakyReLU、Tanh、Sigmoid、Linear
- **損失関数** — バイナリ交差エントロピー、Wasserstein、ヒンジ損失
- **勾配ペナルティ** — WGAN-GP方式の学習安定化
- **スペクトル正規化** — Discriminator正則化のための重み正規化
- **潜在空間補間** — 潜在ベクトル間の線形・球面補間
- **モード崩壊検出** — 学習中のGenerator出力多様性の監視
- **内蔵PRNG** — 正規/一様サンプリング対応のXorshift64乱数生成器

## アーキテクチャ

```
alice-gan (lib.rs — 単一ファイルクレート)
├── Rng                          # Xorshift64 乱数生成器
├── Activation                   # 活性化関数
├── DenseLayer / Network         # ニューラルネットワーク層
├── Generator / Discriminator    # GANコンポーネント
├── LossFunction                 # BCE / Wasserstein / Hinge
├── GradientPenalty              # WGAN-GP 正則化
└── GanTrainer                   # 学習ループオーケストレーター
```

## クイックスタート

```rust
use alice_gan::{Generator, Discriminator, GanTrainer, Rng};

let mut rng = Rng::new(42);
let gen = Generator::new(&mut rng, 64, &[128, 256], 784);
let disc = Discriminator::new(&mut rng, 784, &[256, 128]);
let mut trainer = GanTrainer::new(gen, disc, 0.0002);
trainer.train_step(&mut rng, 32);
```

## ビルド

```bash
cargo build
cargo test
cargo clippy -- -W clippy::all
```

## ライセンス

MIT OR Apache-2.0 — 詳細は [LICENSE](LICENSE) を参照。
