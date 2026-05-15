# Beyond Backpropagation: Scaling LLMs with Biologically Plausible Learning

> **Date**: 2026-05-16
> **Project**: `ssm-latent-rs`
> **Context**: Exploring alternatives to backpropagation for world models combining SSM (Mamba-style) + JEPA (Joint Embedding Predictive Architecture)

---

## 1. Motivation

Transformer-based LLMs face fundamental limits:

| Problem | Detail |
|---|---|
| **Scaling wall** | "The era of pre-training is ending" — Ilya Sutskever (NeurIPS 2024). Internet data is exhaustible like fossil fuel. |
| **Energy inefficiency** | Human brain: ~20W. GPT-4 training: tens of megawatts. 6+ orders of magnitude gap. |
| **Distillation is palliative** | Small models inherit intelligence from large ones via distillation — no fundamental breakthrough. |
| **Weight transport problem** | Backpropagation requires symmetric forward/backward paths, biologically implausible and hardware-inefficient. |

**Analogy**: Steam engines → Oil → Nuclear power. Each transition changed the **principle**, not just efficiency. AI needs the same.

---

## 2. Architecture: SSM + JEPA + Forward-Forward

### 2.1 Components

```
obs(t) → [FF Encoder] → z(t) ──[detach]──┐
                                           ├→ [Fusion] → u(t) → [SSM] → ẑ(t+1)
act(t) → [Action Enc] ───────────────────┘         ↑ 1-step gradient only
                                                     (h_{t-1} detached)

ẑ(t+1) → [FF Decoder] → ôbs(t+1)
          ↑ FF goodness + local MSE
```

| Component | Training | Gradient Flow |
|---|---|---|
| **FF Encoder** (2→16→d_model) | Forward-Forward goodness | Per-layer only (detach between layers) |
| **FF Decoder** (d_model→16→2) | Forward-Forward goodness + local MSE | Per-layer only |
| **Multi-Scale SSM** (d_model-dim) | 1-step truncated RTRL | Within single step only (h_{t-1} detached) |
| **Action Encoder + Fusion** | 1-step BP | Local (few params) |

### 2.2 Forward-Forward Algorithm (Hinton, 2022)

Each layer learns via local "goodness":

- **Positive data** (real circle points) → high `mean(activations²)` → **goodness↑**
- **Negative data** (random points) → low `mean(activations²)` → **goodness↓**
- Each layer optimizes `ReLU(threshold - goodness(pos))² + goodness(neg)²`
- Gradients do **NOT** flow between layers (`.detach()` between layers)

### 2.3 Truncated RTRL (Real-Time Recurrent Learning)

SSM training without BPTT:

- At each timestep `t`: `h_{t-1}`, `bx_{t-1}`, `conv_state` are **detached**
- Loss: `MSE(ẑ(t), z(t))` — gradient only through `f(h_detach, x_t, θ)`
- **O(1) memory**: no computation graph grows with sequence length
- **O(1) gradient per step**: no BPTT

---

## 3. Experiments

### 3.1 Setup

- **Task**: Circle World — predict a point moving on a unit circle
- **Observation**: 2D (x, y) with ±0.02 noise
- **Action**: 2D tangential velocity
- **Sequence**: 32 steps, batch size 4, 80-200 epochs

### 3.2 Results Summary

| Approach | Encode Quality | Imagination (SSM roll-out) | BPTT | Cross-BP |
|---|---|---|---|---|
| **Original BP** (SSM+JEPA) | ✅ Excellent | ✅ Full circle | 100% | 100% |
| **FF Autoencoder only** | ✅ Perfect | ❌ Impossible (static) | 0% | 0% |
| **FF-SSM Hybrid** (BP SSM) | △ Blurry | ❌ Reverse rotation | ~30% | ~30% |
| **RTRL v1** (Zero BPTT) | ✅ Good tracking | ✅ Follows circle | **0%** | **0%** |
| **RTRL v2** (velocity + episode BP) | ❌ 1st quadrant loop | ❌ Broken | ~5% (accidental) | ~5% |
| **RTRL v3** (velocity + detach'd prev) | ❌ 3rd quadrant stall | ❌ Collapse | 0% | 0% |

### 3.3 Key Findings

**RTRL v1 works.** With zero BPTT and zero cross-component BP, the SSM learns to predict latent dynamics and the FF encoder/decoder learn to represent the circle. Slight tracking offset is expected — the model learns a latent representation, not pixel-perfect reconstruction.

**Velocity smoothness penalty always fails.** Three attempts, three failures:

1. **v2 (episode-level backward)**: `prev_z` without detach → accidentally reintroduced BPTT → FF goodness signal destroyed
2. **v3 (detach'd prev_z)**: One-directional "gravity" pulling each `z_t` toward `z_{t-1}` → all latents collapse to a single point
3. The fundamental tension: **FF goodness wants distinct representations per point; velocity penalty wants all representations close together.** These objectives are directly opposed.

---

## 4. Architectural Decisions

### 4.1 Why Component Isolation Matters

```
Original BP:
  SSM error → decoder error → encoder error
  (single global gradient through all components)

RTRL v1:
  SSM error ⊥ encoder (detach)
  Decoder error ⊥ encoder (detach)
  Encoder FF loss ⊥ decoder (detach)
  (three independent gradient graphs)
```

The isolation means each component optimizes its **own** objective:

- **Encoder**: represent the circle's shape (FF goodness)
- **SSM**: predict latent dynamics (1-step RTRL)
- **Decoder**: reconstruct from latent (FF goodness + MSE)

This is closer to how biological systems learn — specialized circuits with local objectives, not global error minimization.

### 4.2 The Remaining Gap

The encoder learns representations that are **good for reconstruction** but not necessarily **good for prediction**. In the original BP model, SSM prediction error flows back to shape the encoder's representation. RTRL breaks this feedback loop.

**This is the "Nuclear Power" problem**: How to train the encoder to produce prediction-friendly representations **without** backpropagating prediction error through time.

---

## 5. Future Work

### 5.1 Predictive Coding

Predictive coding (Rao & Ballard, 1999; Friston, 2005) is the most promising path to "Nuclear Power":

```
Each layer predicts the layer below:
  L_n predicts L_{n-1}
  Prediction error = actual - predicted
  Weight update = prediction_error × input  (Hebbian-like, local)

For temporal sequences:
  L_n(t) predicts L_n(t+1)
  Prediction error updates both predictor AND encoder
  All updates are LOCAL — no weight transport
```

Key advantage: prediction error naturally shapes the encoder to produce **predictable** representations, without BPTT.

### 5.2 Equilibrium Propagation

Scellier & Bengio (2017): A learning algorithm where the network settles to an equilibrium in two phases (free and clamped). Gradients are computed from the difference in equilibrium states — no explicit backpropagation. Particularly compatible with energy-based models and JEPA.

### 5.3 Forward-Forward with Temporal Labels

Hinton's original FF paper uses "positive = real data, negative = fake data." For temporal data, redefine:

- **Positive**: `(z(t), z(t+1))` — temporally consecutive latents
- **Negative**: `(z(t), z_rand)` — random pair

The encoder learns to make consecutive latents "agree" (high goodness) while random pairs disagree. This injects temporal structure into FF without BPTT.

### 5.4 Hardware Co-Design

- **Neuromorphic chips** (Intel Loihi, IBM NorthPole): In-memory compute, spiking neurons
- **Analog computing**: Memristor crossbars for O(1) vector-matrix multiply
- **Photonic computing**: Light-based matrix operations at femtojoule energy

RTRL's per-step locality makes it naturally suited for neuromorphic hardware — no need to buffer entire sequences.

---

## 6. File Structure

```
src/
├── lib.rs                    # Module registry
├── ssm.rs                    # SSM block, MultiScaleSsmBlock, SsmConfig
├── latent.rs                 # Original JEPA predictor (BP), stability/curvature loss
├── forward_forward.rs        # L2 normalisation, goodness, FF loss functions
├── ff_model.rs               # FfLayer, FfEncoder (stackable FF layers)
├── ff_hybrid.rs              # FfSsmWorldModel (FF enc/dec + BP SSM) — failed hybrid
├── rtrl.rs                   # ssm_step_detached, RtrlAccumulator
└── rtrl_world.rs             # RtrlWorldModel (reusable: FF enc + RTRL SSM + FF dec)

circle-world-demo/src/
├── main.rs                   # Original BP demo (SSM+JEPA, full BPTT)
├── ff_main.rs                # FF autoencoder only (static, no dynamics)
├── ff_hybrid_main.rs         # FF-SSM hybrid demo (failed)
└── rtrl_main.rs              # RTRL demo v1 (zero BPTT) — BEST RESULT
```

### 6.1 Running the Demos

```bash
# Original BP (baseline)
cargo run -p ssm-latent-model --release --bin circle-world-demo \
  --features ndarray,autodiff --no-default-features

# FF autoencoder only
cargo run -p ssm-latent-model --release --bin circle-world-ff \
  --features ndarray,autodiff --no-default-features

# RTRL (zero BPTT, best result)
cargo run -p ssm-latent-model --release --bin circle-world-rtrl \
  --features ndarray,autodiff --no-default-features
```

### 6.2 Using RtrlWorldModel in Other Demos

```rust
use ssm_latent_model::rtrl_world::{RtrlConfig, RtrlStepLossArgs, RtrlWorldModel};
use ssm_latent_model::ssm::SsmConfig;

let config = RtrlConfig {
    ssm: SsmConfig::new(64, 16, 2, 4, 1),
    n_ssm_layers: 2,
    input_dim: 2,
    action_dim: 2,
    encoder_sizes: vec![2, 16, 64],
    decoder_sizes: vec![64, 16, 2],
};
let model = RtrlWorldModel::new(&config, &device);
let ssm_state = model.initial_ssm_state(1, &device);

for t in 0..seq_len {
    let output = model.training_step(obs, neg_obs, action, &ssm_state, 3.0);
    let ssm_loss = if t + 1 < seq_len {
        Some(MSE(output.pred_z, model.encode(next_obs).detach()))
    } else { None };
    let loss = model.compute_step_loss(&output, &obs, ssm_loss, &loss_args);
    loss.backward();
    optimizer.step();
    ssm_state = output.next_ssm_state;
}
```

---

## 7. Conclusion

### What We Achieved

1. **Zero-BPTT world model** that learns to track and predict a dynamic system
2. **Forward-Forward encoder/decoder** with per-layer local learning
3. **Reusable library** (`rtrl_world.rs`) for plugging into other demos
4. **Clear failure modes documented** — velocity penalty, hybrid architectures

### What Remains

The "Nuclear Power" breakthrough — making the encoder produce prediction-friendly representations without BPTT — is still open. Predictive Coding is the most promising candidate.

### The Steam Engine Analogy, Updated

```
Steam Engine  = Transformer + BPTT (works, scales, but fundamentally limited)
Oil           = RTRL (BPTT-free, proven viable at small scale)
Nuclear Power = Predictive Coding / Equilibrium Propagation (theoretical, not yet demonstrated)
```

We have found Oil. It burns cleaner than coal. But the reactor is still on the drawing board.
