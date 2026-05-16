# NAB Anomaly Detection Benchmark Report: Pure Mamba Predictor

**Last updated**: 2026-05-16 (JEPA removed — Pure Mamba predictor)

## 1. Architecture: Pure Mamba (No JEPA)

### Why This Rewrite

The previous JEPA-based approach had fundamental conflicts with anomaly detection:

| JEPA component | Why removed |
|:---|:---|
| Latent-space prediction | Anomalies become "predictable" in latent space — error signal vanishes |
| VICReg stability loss | Forces all representations (normal + anomalous) into same distribution |
| Curvature loss | Penalizes the very "trajectory bends" that constitute anomalies |
| Reconstruction loss | Distracts model from prediction quality; anomalies still reconstructable |
| Encoder/Decoder | Unnecessary indirection; ~10K wasted params |
| MultiScaleSsmBlock | 280K params on 1K-20K point series = extreme overfitting |
| 75% train ratio | Anomaly windows leak into training data |

### New Architecture

```
Observation[8D] → Linear(input_proj) → SsmBlock(1層) → Linear(output_proj) → Prediction[8D]
```

- **Single SSM block** with complex rotation (`d_model=32, d_state=8, expand=2, heads=2`)
- **~9K parameters** (vs ~280K previously) — appropriate for 1K–20K point series
- **Pure MSE loss**: `(pred[t] - target[t+1])²`
- **Direct observation space prediction** — no latent space, no encoder/decoder
- **Training on first 15% only** — anomaly windows never seen during training
- **Streaming inference** — `forward_step()` with persistent hidden state

### Feature Engineering

8D input: `[value, diff, z_short, z_long, hour_sin, hour_cos, dow_sin, dow_cos]`

| Feature | Description |
|:---|:---|
| `value` | MAD-normalized original value |
| `diff` | First difference |
| `z_short` | Rolling z-score (48-step window) |
| `z_long` | Rolling z-score (336-step window = ~1 week at 5min) |
| `hour_sin/cos` | Hour-of-day cyclic encoding |
| `dow_sin/cos` | Day-of-week cyclic encoding |

### Scoring Pipeline

```
1. Streaming inference → weighted prediction error
2. MAD calibration on probationary period
3. Contamination-guarded EWMA z-score normalization (guard_z=3.5)
4. Percentile rank + power transform (0.25) → anomaly score ∈ [0, 1]
```

### Key Design Decisions

1. **Small model, normal-only training**: Prevents "learn to ignore anomalies" behavior
2. **MAD normalization**: Robust to outlier contamination in raw values
3. **Dual-horizon z-scores**: Captures both spike anomalies (short) and trend shifts (long)
4. **Contamination guard (z < 3.5)**: Prevents anomalies from corrupting EWMA baseline
5. **Power 0.25 transform**: Sharper contrast at high end for NAB threshold optimizer

## 2. Running

```bash
# Generate anomaly scores
cargo run -p nab-demo --features "ssm-latent-model/wgpu,ssm-latent-model/ndarray,ssm-latent-model/train"

# Evaluate with NAB scoring pipeline
cd nab-demo && python evaluate_nab.py
```

Detector name: `ssm_latent_multiscale`
