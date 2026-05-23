# Research Standing & Future Directions

*2026-05-21 — compiled from the JEPA landscape review and subsequent LeJEPA/SIGReg integration.*

---

## 1. JEPA Research Landscape (2025–2026)

### 1.1 Theoretical Breakthrough: LeJEPA (Nov 2025)

**Balestriero & LeCun** (`arXiv:2511.08544`) established the theoretical foundation
that JEPA had been missing:

- **Isotropic Gaussian is the provably optimal embedding distribution** for
  minimizing downstream prediction risk — for both linear probes (OLS, ridge
  regression) and nonlinear probes (k-NN, kernel methods).
- **SIGReg** (Sketched Isotropic Gaussian Regularization): a characteristic-function
  matching objective that enforces N(0,I) embeddings via random 1D projections.
  ~50 lines of code, linear complexity, **single hyperparameter**, no
  stop-gradient / teacher-student / EMA schedule.
- ViT-H/14 reaches 79% top-1 on ImageNet-1K linear probe. On domain-specific
  datasets (Galaxy10, Food101), in-domain LeJEPA pretraining beats DINOv2/v3
  transfer learning.

This is arguably LeCun's last paper at Meta FAIR before leaving to found AMI Labs.

### 1.2 World Models: LeWorldModel (Mar 2026)

**Maes, Le Lidec, Scieur, LeCun, Balestriero** (`arXiv:2603.19312`):

- First JEPA that trains stably **end-to-end from pixels** with only 2 loss terms
  (prediction + Gaussian regularizer).
- ~15M parameters, trains on a single GPU in hours.
- Plans up to 48× faster than foundation-model-based world models.
- Demonstrates surprise detection (physically implausible events) and plays
  Super Mario Bros.

### 1.3 Video & Robotics

| Model | Date | Key Contribution |
|---|---|---|
| V-JEPA 2 | Jun 2025 | 1M hours internet video + robot trajectories → understanding, prediction, planning |
| V-JEPA 2-AC | Jun 2025 | Action-conditioned predictor with block-causal attention + rollout loss |
| V-JEPA 2.1 | Mar 2026 | Dense temporally-consistent features for spatial grounding |
| Causal-JEPA | Feb 2026 | Object-level masking → counterfactual intervention reasoning |

### 1.4 Multimodal: VL-JEPA (Dec 2025, ICLR 2026)

**Chen, Shukor, Moutakanni, … LeCun, Fung** (`arXiv:2512.10942`):

- Non-autoregressive VLM: predicts **continuous text embeddings** instead of tokens.
- 50% fewer parameters than equivalent autoregressive VLM, better average
  performance across 8 video classification + 8 retrieval benchmarks.
- Selective decoding reduces operations by ~2.85×.

### 1.5 Language: LLM-JEPA (Sep 2025)

**Huang, LeCun, Balestriero** (`arXiv:2509.14252`):

- Adds JEPA embedding-space prediction loss to standard LLM training.
- Consistent improvements on NL-RX, GSM8K, Spider across Llama-3.2-1B,
  R1-Distill-Qwen-1.5B.

### 1.6 Domain-Specific JEPAs (2025)

| Domain | Model | Key Result |
|---|---|---|
| Audio | Audio-JEPA | Matches wav2vec 2.0 with ~1/5 the data |
| Time Series | TS-JEPA | SOTA on UCR classification + long-term forecasting |
| Graphs/Polymers | Graph-JEPA / Polymer-JEPA | Strong few-shot transfer across chemical spaces |
| Trajectories | T-JEPA / HiT-JEPA | Robust to downsampling and spatial distortion |
| Robotics | ACT-JEPA | +40% world model understanding over baselines |
| Geospatial | GeoJEPA | Best harmonic mean across multimodal geo benchmarks |
| Autonomous Driving | AD-L-JEPA | SOTA 3D object detection + scene understanding |

### 1.7 Theoretical Analyses

- **Collapse theorems**: sufficient target diversity + non-trivial context-target
  mapping → global minimizers are non-collapsed (Huang, Jan 2026).
- **Koopman invariants**: JEPA loss structure naturally recovers dynamical regime
  indicators (Ruiz-Morales et al., Nov 2025).
- **Implicit bias in deep linear nets**: JEPA prioritizes high-influence features
  vs. MAE's variance preference (Littwin et al., Apple, ICML 2026).
- **Variational JEPA (VJEPA)**: predictor outputs distributions; formal guarantees
  for POMDP optimal control (Huang, Jan 2026).

---

## 2. Our Implementation: Positioning

### 2.1 Architecture

```
Mamba SSM (complex rotation, multi-scale) × JEPA (latent prediction)
```

### 2.2 Strengths vs. the Field

| Axis | Literature | This Repo |
|---|---|---|
| **Backbone** | Transformer (all JEPA papers) | **Mamba SSM** — unique in the JEPA space |
| **Collapse prevention** | LeJEPA SIGReg (Nov 2025) | **SIGReg implemented** (May 2026) |
| **Batch-size=1 stability** | VICReg/SIGReg need batch stats | **`stability_loss_running`** with EMA |
| **Closed-loop imagination** | LeWM: d_model → obs → d_model round-trip | **`step_imagine_h`**: SSM hidden state → d_model, bypasses bottleneck |
| **Scheduled sampling for h-loop** | Not addressed in LeWM | **`forward_with_h_sampling`** (Bernoulli) |
| **Multi-scale dynamics** | Not in any JEPA paper | **3-layer SSM** with fast/medium/slow decay |
| **Edge inference** | GPU clusters | **WASM / WGPU / CPU** via Burn |
| **Temporal Straightening** | Wang et al. 2026 | **Implemented** (`curvature_loss` + dt-aware variant) |

### 2.3 Gaps vs. the Field

| Gap | Severity | Action |
|---|---|---|
| No empirical collapse analysis | Medium | Run `sigreg_loss` vs `stability_loss` comparison on varied distributions |
| No comparison with LeWM | Medium | Benchmark on Circle World or similar control task |
| No object-level masking (C-JEPA) | Low | Requires segmentation front-end; out of current scope |
| No diffusion/flow-matching head (D-JEPA) | Low | Pure prediction focus; generation is a separate concern |
| No multimodal vision-language (VL-JEPA) | Low | `multimodal.rs` handles vision+sensor+action, not text |

### 2.4 Known Limitation: JEPA × Anomaly Detection

**Empirically confirmed: JEPA (latent-space prediction) is ill-suited for anomaly detection.**

The mechanism is structurally opposed to the task:

1. **JEPA's core objective** is to learn a latent representation that discards
   information irrelevant for prediction — it is designed to *ignore* non-essential
   perturbations.
2. **Anomaly detection** requires *sensitivity* to any deviation from the learned
   steady-state distribution — precisely the information JEPA suppresses.
3. **Multi-scale SSM** compounds this: the exponential decay of the state space
   (`a_re < 0`) further smooths out transient spikes, and JEPA's encoder filters
   them before they reach the SSM.

**Consequence**: Mamba+JEPA excels at world modeling (Circle World phase-locked
prediction, LeWorldModel-style tasks) but is **structurally weak at detecting
point anomalies** in time series. For anomaly detection, use the pure Mamba
predictor in [`predictor.rs`](../src/predictor.rs) which operates directly in
observation space with no latent bottleneck — see Section 4.2.

This finding supersedes the previously planned "NAB anomaly detection with SIGReg"
experiment (removed below).

---

## 3. What We Changed (May 2026)

### 3.1 `sigreg_loss(z, w, freqs)`

Characteristic-function matching against the standard normal distribution:

1. Project embeddings `z` onto `M` random 1D directions (reuses existing
   `stability_projections` matrix).
2. Standardize each projection to zero mean, unit variance.
3. Evaluate empirical CF at `K` frequency points (default: `[0.5, 1.0, 1.5, 2.0]`).
4. Penalize |φ_emp(t) − exp(−t²/2)|² summed over directions and frequencies.

Complexity: O(N·d·M + N·M·K), linear in all dimensions.

### 3.2 `lejepa_loss(z, pred_z, projections, sigreg_weight, freqs)`

```
L = MSE(pred_z[t], z[t+1]) + γ · SIGReg(z)
```

Single tunable hyperparameter γ (`sigreg_weight`). No stop-gradient on the
target encoder needed — the predictive loss naturally provides the asymmetry.

### 3.3 `stability_loss` retained

The existing VICReg-style moment matching (mean→0, var→1 on random projections)
is kept as a lightweight fallback. It can be seen as a second-order approximation
to SIGReg (matching only the first two moments rather than the full distribution).

### 3.4 Config

```toml
[train]
sigreg_weight = 1.0
sigreg_freqs = [0.5, 1.0, 1.5, 2.0]
```

### 3.5 Tests

- `test_sigreg_loss_prevents_collapse`: collapsed (all-zero) embeddings produce
  strictly higher SIGReg loss than N(0,1) embeddings.
- `test_lejepa_loss_finite`: combined loss is finite and non-negative.

---

## 4. Future Work

### 4.1 Immediate (next weeks)

- [ ] **Empirical collapse analysis**: compare `stability_loss` vs `sigreg_loss`
  on deliberately difficult distributions (anisotropic, low-rank, degenerate).
  Plot RankMe, LiDAR, or effective rank over training.
- [ ] **Circle World benchmark**: train both moment-matching and SIGReg variants
  on the existing circle task; compare convergence speed, latent geometry,
  downstream prediction accuracy.
- [ ] **LeWorldModel comparison**: reproduce a LeWM-scale task (e.g. 2D control)
  with our SSM backbone and compare parameter efficiency.

### 4.2 Short-term (1–2 months)

- [ ] **TechRxiv preprint**: "Mamba-JEPA: Efficient World Models with State Space
  Dynamics and Provable Collapse Prevention." Core claims:
  1. First SSM-based JEPA world model.
  2. SIGReg integration with multi-scale Mamba dynamics.
  3. h-based closed-loop imagination (no observation-space bottleneck).
  4. WASM/edge deployment capability.
- [ ] **Ablation study**: multi-scale vs single-scale SSM, SIGReg vs stability_loss,
  h-loop vs d_model-loop, scheduled sampling vs teacher forcing.
- [ ] **NAB anomaly detection with pure Mamba predictor**: benchmark
  [`MambaPredictor`](../src/predictor.rs) (observation-space, no JEPA) on NAB
  datasets. This validates the structural finding that JEPA's latent filtering
  is antithetical to point anomaly detection (see §2.4).

### 4.3 Medium-term (3–6 months)

- [ ] **Causal-JEPA integration**: object-level masking for counterfactual reasoning.
  Requires a segmentation or object-discovery front-end.
- [ ] **Mamba-3 features**: Lahoti et al. (2026) propose improvements to the SSM
  formulation — evaluate whether the multi-scale design already captures the
  benefits.
- [ ] **Scaling study**: how does the Mamba SSM backbone scale compared to
  Transformer JEPA? Parameter count vs. sequence length trade-offs.
- [ ] **Real robot demo**: port to a simple real-world control task using the
  action-conditioned predictor.

### 4.4 Long-term

- [ ] **Multimodal JEPA**: extend `multimodal.rs` to handle text via the existing
  ONNX embedding pipeline.
- [ ] **Hierarchical JEPA (H-JEPA)**: short-term detail predictor + long-term
  planning predictor, as described in LeCun's architecture vision.
- [ ] **Energy-Based Model formulation**: reframe the predictive loss as an EBM
  energy, enabling compositionality and reasoning via gradient-based inference.

---

## 5. Key References

| Paper | arxiv | Date |
|---|---|---|
| LeJEPA: Provable and Scalable SSL Without the Heuristics | `2511.08544` | Nov 2025 |
| LeWorldModel: Stable End-to-End JEPA from Pixels | `2603.19312` | Mar 2026 |
| V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning | `2506.09985` | Jun 2025 |
| VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language | `2512.10942` | Dec 2025 |
| LLM-JEPA: Large Language Models Meet JEPA | `2509.14252` | Sep 2025 |
| Causal-JEPA: Learning World Models through Object-Level Latent Interventions | `2602.11389` | Feb 2026 |
| Temporal Straightening for Latent Planning | Wang et al. | 2026 |
| Mamba-3: Improved Sequence Modeling using State Space Principles | Lahoti et al. | 2026 |
| A Path Towards Autonomous Machine Intelligence | LeCun | 2022 |

---

## 6. Repository Map

```
src/
├── ssm.rs         # Mamba SSM: complex rotation, parallel scan, multi-scale
├── latent.rs      # JEPA predictor, stability_loss, sigreg_loss, lejepa_loss,
│                  #   curvature_loss, MultiScaleLatentPredictor
├── predictor.rs   # Pure Mamba predictor (no encoder/decoder), h-imagination
├── multimodal.rs  # Vision + sensor + action fusion predictor
├── preprocess.rs  # ONNX text embeddings, projection normalization
├── lib.rs         # Public API
├── main.rs        # Native latent visualization CLI
└── error.rs       # Error types

tests/
├── core_tests.rs       # stability, SIGReg, LeJEPA, curvature, save/load, config
├── equivalence_test.rs # parallel scan ≡ sequential step
├── consistency_test.rs # gradient computability
├── extended_tests.rs   # edge cases, MIMO rank>1, step(), vision, conv
└── multimodal_tests.rs # multimodal forward shape verification

demos/
├── circle-world-demo/          # 2D latent world model visualization
├── game-playing-wasm/          # Ball catch + metronome (browser)
└── tiny-stories-jepa-demo/     # JEPA on TinyStories text
```
