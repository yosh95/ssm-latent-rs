# SSM Latent World Model — Mamba-3 × JEPA

[![CI](https://github.com/yosh95/ssm-latent-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/yosh95/ssm-latent-rs/actions/workflows/ci.yml)
[![Security Audit](https://github.com/yosh95/ssm-latent-rs/actions/workflows/security-audit.yml/badge.svg)](https://github.com/yosh95/ssm-latent-rs/actions/workflows/security-audit.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.87%2B-blue.svg)](https://www.rust-lang.org)

A Rust ([Burn](https://burn.dev/)) implementation of **Mamba-3** (Lahoti et al., ICLR 2026) integrated with **Joint-Embedding Predictive Architecture (JEPA)** for latent world modeling.

![Circle Demo](images/circle_demo.gif)

### 🧬 Mamba-3: Three Core Innovations (all implemented)

| Innovation | Implementation | Mamba-3 Ref |
|---|---|---|
| **Exponential-Trapezoidal Discretization** | `λ_t`-gated 3-term recurrence: `h_t = α_t·h_{t-1} + β_t·B_{t-1}·x_{t-1} + γ_t·B_t·x_t` | Prop. 1, Eq. 5 |
| **Complex-Valued SSM** (data-dependent RoPE) | `a_re + i·a_im` with per-head rotation via `theta_proj` | Prop. 3–4 |
| **MIMO** (Multi-Input Multi-Output) | `mimo_rank` parameter; matmul state updates | §3.3 |
| **BCNorm** (QK normalization) | RMSNorm on B/C projections before bias addition | §3.4 |
| **B/C Biases** | Head-specific learnable biases; with exp-trap, makes short conv optional | §3.4, §4.2 |

---

### 🎮 Ball Catch Game (WASM Demo)
*Learning physics through observation.* This demo showcases the model's ability to approximate object trajectories and react in real-time within a browser environment.

![Ball Catch Demo](images/ball_catch.gif)

---

## 🚀 Key Characteristics

- **Mamba-3 SSM Core**: Exponential-trapezoidal discretization, complex-valued state transitions (data-dependent RoPE), MIMO formulation, and BCNorm — all implemented in pure Rust/Burn. Short convolutions are **disabled by default** as exp-trap + B/C biases make them redundant (Mamba-3 §4.2).
- **Latent-Space Prediction (JEPA)**: Following the JEPA philosophy, the model predicts future states in a learned embedding space. This approach focuses on capturing essential dynamics rather than predicting every pixel, which helps in maintaining stability.
- **Collapse Prevention (LeJEPA / SIGReg)**: Implements *Sketched Isotropic Gaussian Regularization* (Balestriero & LeCun, 2025) — a **provably optimal** distribution-matching objective that constrains embeddings to an isotropic Gaussian.  SIGReg replaces the heuristics (stop-gradient, teacher-student, EMA schedule) with a single tunable hyperparameter, exactly matching the LeJEPA blueprint.  A lightweight moment-matching fallback (`stability_loss`) is also available for resource-constrained settings.
- **Multi-Scale SSM Stack**: Stacked SSM layers with different timescale initializations (fast/medium/slow) for capturing patterns across multiple temporal resolutions.
- **Trajectory Regularization**: Incorporates *Temporal Straightening* (Wang, Bounou, Zhou et al., 2026) to encourage locally linear, predictable latent trajectories, aiding long-term planning.
- **Cross-Platform Implementation**: Built with [Burn](https://burn.dev/), enabling the same model logic to run across different backends, including WGPU for browser-based WASM execution.

## 🕹 Demos & Usage

### 1. WebAssembly Demos (In-Browser)
These experiments run locally in your browser, performing both training and inference.

![WASM Metronome Demo](images/wasm_demo.gif)

- **Ball Catch Game**: A simple physics environment where the agent learns to intercept a ball.
- **Metronome**: A task focused on synchronizing internal state with external periodic signals.

**How to Run:**
1. Install [Trunk](https://trunkrs.dev/): `cargo install trunk`
2. Navigate to the desired demo (e.g., `cd game-playing-wasm`).
3. Start the local server: `trunk serve --release`

### 2. Log Anomaly Detection
This demo showcases semantic anomaly detection in system logs. It combines **SentenceTransformer** embeddings with the **Latent SSM** to identify deviations from learned temporal patterns.
- **Hybrid Adaptive Thresholding**: Implements a robust anomaly detection engine using Median Absolute Deviation (MAD) for calibration and Exponential Weighted Moving Average (EWMA) online tracking.
- **Contamination Prevention**: Only normal observations update the threshold, ensuring the model remains resilient to persistent anomalies.
- **Pre-warmed EWMA**: The EWMA is initialized with all calibration scores during construction, so the adaptive threshold is fully effective from the very first inference sample — no warmup delay.

```bash
cargo run -p log-anomaly-demo --release
```
![Log Anomaly Demo](images/log_demo.gif)

### 3. Native Latent Visualization
A CLI-based visualization of the model's "imagination" process.
```bash
cargo run --release
```

---

## 🔬 Other Experiments

### Deterministic AI Agent
A high-performance agent designed for industrial (OT) environments, focusing on reliability and determinacy.
- **Features**: Neural intent classification, Out-of-Distribution (OOD) detection via class centroids, and hybrid (Neural + Exact Match) Named Entity Recognition (NER).
- **Safety**: Designed to reject inputs falling outside the training distribution, ensuring predictable behavior in sensitive environments.
- **More Info**: See the [Agent README](deterministic-ai-agent-demo/README.md).

```bash
cargo run -p deterministic-ai-agent-demo --release
```

## 🧪 Technical Notes

- **Stability**: Uses random projections as a lightweight regularizer to prevent latent representation collapse. Two variants are provided:
  - `sigreg_loss`: Full **SIGReg** (LeJEPA) — characteristic-function matching against N(0,I), provably optimal and heuristics-free.
  - `stability_loss`: Moment-matching fallback (VICReg-style) for environments where the CF loop overhead is undesirable.
- **Complexity**: The implementation balances $O(L \log L)$ training complexity with $O(1)$ state updates during deployment.

### Running Tests

The project includes comprehensive tests covering core functionality, equivalence verification, and edge cases:

```bash
# Run all tests (including extended tests)
cargo test --all-targets --all-features

# Run specific test suites
cargo test --test core_tests          # Stability loss, curvature loss, save/load
cargo test --test equivalence_test    # Parallel scan ≡ sequential step equivalence
cargo test --test consistency_test     # Gradient computability
cargo test --test multimodal_tests   # Multimodal forward shape verification
cargo test --test extended_tests      # Edge cases, MIMO rank > 1, step(), vision, conv equivalence
```

#### Test Coverage

| Category | Tests | Description |
|---|---|---|
| **Equivalence** | Parallel vs. Sequential | Verifies `forward()` ≡ `forward_step()` loop |
| **Equivalence** | MIMO Rank 2 | Same equivalence test with `mimo_rank=2` |
| **Equivalence** | Conv1d enabled | Parallel/sequential equivalence with causal convolution |
| **Edge Cases** | `curvature_loss(seq_len < 3)` | Returns 0.0 for insufficient sequence length |
| **Edge Cases** | Constant velocity trajectory | Verifies curvature loss ≈ 0 for straight paths |
| **Step** | `LatentPredictor::step()` | Shape verification with/without conv |
| **Step** | Multi-step consistency | Finite outputs, evolving hidden state |
| **Vision** | Encoder/Decoder shapes | Round-trip shape preservation |
| **Vision** | Multimodal loss | Loss is finite and non-negative |
| **Gradient** | Conv1d gradients | Verifies conv weights receive gradients |
| **Gradient** | SSM parameters | Verifies `a_re`, `a_im`, `dt_proj`, `out_proj` gradients |
| **SIGReg** | Collapse prevention | Collapsed embeddings → higher loss than normal ones |
| **LeJEPA** | Combined loss | `lejepa_loss` is finite and non-negative |

## 📚 References

- Balestriero, R., & LeCun, Y. (2025). **LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics**. *arXiv:2511.08544*.
- Lahoti, A., et al. (2026). **Mamba-3: Improved Sequence Modeling using State Space Principles**.
- Maes, L., et al. (2026). **LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels**.
- Wang, Y., Bounou, O., Zhou, G., Balestriero, R., Rudner, T.G., LeCun, Y., & Ren, M. (2026). **Temporal Straightening for Latent Planning**.

## 📄 License
MIT License
