# NAB Anomaly Detection Benchmark Report: SSM+Latent (Rust)

**Last updated**: 2026-05-12 (after Phase 1+2 fixes)

## ⚠️ Previous Report Accuracy Issue

The previous version of this report claimed Standard=37.95, but the official NAB scoring
(`thresholds.json`) recorded Standard=-27.95. The discrepancy was caused by a broken
scoring pipeline that produced near-zero anomaly scores via a `sigmoid(z-4)` transform,
making NAB's threshold optimization ineffective.

## 1. Official NAB Scores (Before Fix)

| Profile | Score | SOTA (ARTime) | Gap |
| :--- | :--- | :--- | :--- |
| **Standard** | **-27.95** | 57.66 | 85.6 |
| **Reward Low FP** | **-64.90** | 35.16 | 100.1 |
| **Reward Low FN** | **-68.27** | 47.66 | 115.9 |

## 2. Root Cause Analysis

Five layers of compounding issues were identified:

### Layer 5 (Critical): Scoring Transform
`sigmoid(z_score - 4.0)` compressed 99%+ of scores to ≈0, making NAB threshold optimization
pick θ≈0.999648 which catches only chunk-boundary spikes and EMA transients — all false positives.

### Layer 4 (Critical): EMA Initialization
`moving_mean = 0.0` and `moving_var = 0.05` created artificial spikes at the start of each
time series, and no contamination prevention allowed anomalies to pollute the baseline.

### Layer 3 (Medium): Chunk Boundary State Reset
Processing in 5000-step chunks with `forward()` (parallel scan) resets SSM hidden state at each
boundary, creating periodic false spikes every 5000 steps.

### Layer 2 (Medium): Undertrained Model
Only 15% of data for 50 epochs with LR=1e-3 and unbalanced loss weights
(stability=1.0, curvature=0.5, recon=2.0).

### Layer 1 (Lower): Architecture
Single-layer SSM with d_model=64, action_dim=1 (unused zero vector).

## 3. Phase 1+2 Fixes Applied

### Scoring Pipeline Fix
- **MAD (Median Absolute Deviation)** for robust baseline calibration
- **EWMA with contamination prevention**: only update baseline on z < 3.0
- **|z| score** instead of signed z-score (detect anomalies in both directions)
- **sqrt transform** instead of sigmoid (preserves discriminability)
- **Min-max normalization** to [0, 1] for NAB compatibility
- **Probationary period zeroing** (scores in first 15% set to 0)

### Inference Pipeline Fix
- Increased chunk size from 5000 to 10000 (fewer boundary resets)
- Reduced loss weight imbalance: stability=0.1, curvature=0.05, recon=5.0
- Increased training data from 15% to 50%
- Increased epochs from 50 to 100
- Lowered learning rate from 1e-3 to 5e-4
- Balanced recon/latent error weights (0.5/0.7 vs old 0.7/0.3)

## 4. Remaining Roadmap

### Phase 3: Streaming Inference with State Continuity
Use `forward_step()` for O(1) per-step inference with SSM state carried across
chunk boundaries, eliminating all boundary artifacts.

### Phase 4: Multi-Scale Architecture
Stack 2-3 SSM layers with different temporal scales to capture both rapid spikes
and slow seasonal patterns.

### Phase 5: Feature Engineering
Add rolling window statistics (mean, std, slope) as auxiliary input channels
to give the model richer temporal context beyond raw normalized value.

### Phase 6: Threshold Optimization
Implement per-file threshold optimization using NAB's sweeper to find optimal
thresholds for each profile, rather than relying on the generic min-max approach.

---

*See `ANALYSIS_REPORT.md` for the full technical analysis.*