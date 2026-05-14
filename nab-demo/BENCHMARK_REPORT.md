# NAB Anomaly Detection Benchmark Report: SSM+Latent (Rust)

**Last updated**: 2026-05-14 (Phase 4 Multi-Scale Architecture Implementation)

## 1. Architecture Changes (Phase 4)

### Multi-Scale SSM Stack (`MultiScaleSsmBlock`)
従来の単層SSM → 3層スタックに進化：

| Layer | Timescale | `a_re` initialization | Purpose |
|:---|:---|:---|:---|
| **Layer 0 (Fast)** | 短周期 | [-1.0, -0.3] | 急激なスパイク・外れ値検知 |
| **Layer 1 (Medium)** | 日周期 | [-0.3, -0.05] | 日次パターン・時間帯変動 |
| **Layer 2 (Slow)** | 週周期+ | [-0.05, -0.005] | 週次/月次の季節性変化 |

各層はresidual connection + RMSNormで接続され、異なる時定数の情報を同時に処理。

### Feature Engineering
入力次元: 7D → **11D** に拡張
- `[value, z_short, z_long, diff, diff_long, hour_sin, hour_cos, dow_sin, dow_cos, rolling_mean, rolling_std]`
- 短期(48step) + 長期(336step) の2重ローリング統計
- MAD-based robust normalization

### Training Improvements
- Warmup + Cosine Decay の学習率スケジュール
- MultiScaleLatentPredictor (Burn Module derive でクリーンな実装)
- Early stopping + 適応エポック数

### Scoring Engine (Multi-Horizon)
5つの誤差信号を統合:
| Signal | Weight | Purpose |
|:---|:---:|:---|
| 1-step prediction error | 0.35 | 急峻な異常 |
| 5-step deviation | 0.25 | 日次パターン逸脱 |
| 10-step deviation | 0.15 | 長周期トレンド変化 |
| Reconstruction error | 0.15 | オートエンコーダ品質 |
| Latent space error | 0.10 | 構造的変化 |

Adaptive power transform: z-scoreの歪度に応じてpercentileのpowerを調整（0.15〜0.50）

## 2. Expected Improvements

| Component | Phase 3 (前回) | Phase 4 (今回) |
|:---|:---|:---|
| SSM layers | 1 (d_model=64) | 3 (d_model=128) |
| Feature dims | 7 | 11 |
| Rolling windows | 1 (48 steps) | 2 (48 + 336 steps) |
| Normalization | Min-max | MAD robust |
| LR schedule | Cosine oscillation | Warmup + Cosine decay |
| Scoring signals | 3 | 5 (multi-horizon) |
| Power transform | Fixed (0.3) | Adaptive (skew-based) |

## 3. Running

```bash
# Generate anomaly scores (requires NAB data in nab-demo/data/)
cargo run -p nab-demo --features "ssm-latent-model/wgpu,ssm-latent-model/ndarray,ssm-latent-model/train"

# Evaluate with NAB scoring pipeline
cd nab-demo && python evaluate_nab.py
```

Detector name: `ssm_latent_multiscale`
