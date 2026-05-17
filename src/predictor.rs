//! Pure Mamba time-series predictor for anomaly detection.
//!
//! Unlike the JEPA-based [`crate::latent`] module, this predictor operates
//! directly in observation space with no encoder/decoder, no VICReg stability
//! loss, no curvature loss, and no "predict everything in latent space" objective.
//!
//! Two variants are provided:
//! - [`MambaPredictor`]: single SSM block, lightweight (~15K params)
//! - [`MultiScaleMambaPredictor`]: stacked multi-scale SSM blocks, captures
//!   patterns across fast/medium/slow timescales simultaneously

use crate::ssm::{MultiScaleSsmBlock, MultiScaleSsmConfig, MultiScaleState, SsmBlock, SsmConfig};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// Mutable state for single-layer autoregressive streaming inference.
#[derive(Clone)]
pub struct PredictorState<B: Backend> {
    /// SSM hidden state: `[batch, n_heads, d_state, d_head_mimo]`
    pub h: Tensor<B, 4>,
    /// Previous B·x contribution (lambda-gated recurrence)
    pub prev_bx: Option<Tensor<B, 4>>,
    /// Causal convolution state: `[batch, d_inner, kernel_size - 1]`
    pub conv_state: Option<Tensor<B, 3>>,
}

/// Mutable state for multi-scale autoregressive streaming inference.
#[derive(Clone)]
pub struct MultiScalePredictorState<B: Backend> {
    /// Multi-scale SSM states
    pub ssms: MultiScaleState<B>,
}

// ─── Single-layer MambaPredictor ────────────────────────────────────────

/// Pure Mamba predictor: observation → SSM → next-observation.
///
/// Stripped of all JEPA machinery. A single SSM block with small parameter
/// count (~20K–50K depending on config), suitable for univariate time series
/// with thousands of points.
#[derive(Module, Debug)]
pub struct MambaPredictor<B: Backend> {
    /// Input projection: `input_dim` → `d_model`
    pub input_proj: Linear<B>,
    /// Core selective state space block
    pub ssm: SsmBlock<B>,
    /// Output projection: `d_model` → `output_dim`
    pub output_proj: Linear<B>,
    /// Model dimension
    pub d_model: usize,
    /// Inner SSM dimension
    pub d_inner: usize,
    /// Number of heads
    pub n_heads: usize,
    /// State dimension
    pub d_state: usize,
    /// MIMO rank
    pub mimo_rank: usize,
    /// Whether conv1d is enabled
    pub use_conv: bool,
    /// Conv kernel size
    pub conv_kernel: usize,
}

impl<B: Backend> MambaPredictor<B> {
    /// Create a new predictor.
    pub fn new(
        config: &SsmConfig,
        input_dim: usize,
        output_dim: usize,
        device: &B::Device,
    ) -> Self {
        let input_proj = LinearConfig::new(input_dim, config.d_model).init(device);
        let ssm = SsmBlock::new(config, device);
        let output_proj = LinearConfig::new(config.d_model, output_dim).init(device);

        Self {
            input_proj,
            ssm,
            output_proj,
            d_model: config.d_model,
            d_inner: config.d_model * config.expand,
            n_heads: config.n_heads,
            d_state: config.d_state,
            mimo_rank: config.mimo_rank,
            use_conv: config.use_conv,
            conv_kernel: config.conv_kernel,
        }
    }

    /// Full-sequence forward pass for training.
    ///
    /// Returns predictions for the *next* step at each position.
    /// `predictions[t]` is the model's estimate of what `x[t+1]` will be.
    /// The last position's prediction is a copy of its input (no ground truth).
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // [batch, seq_len, input_dim] → [batch, seq_len, d_model]
        let u = self.input_proj.forward(x);
        // SSM dynamics → [batch, seq_len, d_model]
        let y = self.ssm.forward(u);
        // → [batch, seq_len, output_dim]
        self.output_proj.forward(y)
    }

    /// Autoregressive step for streaming inference.
    ///
    /// Returns `(prediction, next_state)` where `prediction` is the model's
    /// estimate of the *next* observation.
    pub fn step(
        &self,
        x: Tensor<B, 2>,
        state: PredictorState<B>,
    ) -> (Tensor<B, 2>, PredictorState<B>) {
        // [batch, input_dim] → [batch, d_model]
        let u = self.input_proj.forward(x);

        let (y, next_h, current_bx, next_conv_state) =
            self.ssm
                .forward_step(u, state.h, state.prev_bx, state.conv_state);

        // [batch, d_model] → [batch, output_dim]
        let pred = self.output_proj.forward(y);

        (
            pred,
            PredictorState {
                h: next_h,
                prev_bx: Some(current_bx),
                conv_state: next_conv_state,
            },
        )
    }

    /// Create a zero-initialized state for streaming inference.
    pub fn zero_state(&self, batch: usize, device: &B::Device) -> PredictorState<B> {
        let d_head = self.d_inner / self.n_heads;
        let d_head_mimo = d_head / self.mimo_rank;

        let h = Tensor::zeros([batch, self.n_heads, self.d_state, d_head_mimo], device);
        let conv_state = if self.use_conv {
            Some(Tensor::zeros(
                [batch, self.d_inner, self.conv_kernel - 1],
                device,
            ))
        } else {
            None
        };

        PredictorState {
            h,
            prev_bx: None,
            conv_state,
        }
    }

    /// Simple MSE loss for next-step prediction.
    ///
    /// `predictions[t]` is compared against `targets[t+1]`.
    pub fn loss(&self, predictions: Tensor<B, 3>, targets: Tensor<B, 3>) -> Tensor<B, 1> {
        let [batch, seq_len, _] = predictions.dims();
        // predictions[0..T-1] vs targets[1..T]
        let pred_slice = predictions.slice([0..batch, 0..seq_len - 1]);
        let target_slice = targets.slice([0..batch, 1..seq_len]);
        (target_slice - pred_slice).powf_scalar(2.0).mean()
    }
}

// ─── Multi-Scale MambaPredictor ─────────────────────────────────────────

/// Multi-scale Mamba predictor: observation → multi-scale SSM stack → next-observation.
///
/// Stacks multiple SSM layers with different timescale initializations:
/// - **Layer 0 (fast)**: rapid decay `a_re ∈ [-1.0, -0.3]` — spike/outlier detection
/// - **Layer 1 (medium)**: moderate decay `a_re ∈ [-0.3, -0.05]` — daily patterns
/// - **Layer 2+ (slow)**: slow decay `a_re ∈ [-0.05, -0.005]` — weekly/monthly seasonality
///
/// Each layer has residual connections + RMSNorm. No encoder/decoder, no JEPA.
/// The SSM hidden state *is* the world model.
///
/// ## Forward paths
///
/// All paths share the same observation and fusion projections:
///
/// - **Observation-only** (`forward` / `step`): fuses `obs_proj(x)` with a
///   zero-action vector through `imagine_fusion`. Training and inference produce
///   consistent representations because the same weights are always used.
///
/// - **Action-conditioned** (`forward_with_action` / `step_imagine`): fuses
///   `obs_proj(obs)` with `action_proj(action)` through `imagine_fusion`.
///   Use this when ground-truth or planned actions are available.
#[derive(Module, Debug)]
pub struct MultiScaleMambaPredictor<B: Backend> {
    /// Multi-scale stacked SSM blocks (different timescales per layer)
    pub ssms: MultiScaleSsmBlock<B>,
    /// Output projection: `d_model` → `output_dim`
    pub output_proj: Linear<B>,
    /// Fuses encoded observation and action: `2 * d_model` → `d_model`
    pub imagine_fusion: Linear<B>,
    /// Observation projection: `obs_dim` → `d_model` (shared across all paths)
    pub obs_proj: Linear<B>,
    /// Action projection for imagination: `action_dim` → `d_model`
    pub action_proj: Linear<B>,
    /// Projects flattened SSM hidden state h → d_model for closed-loop imagination.
    /// This replaces the obs_proj(output_proj(y)) round-trip, avoiding the
    /// information bottleneck of passing through low-dimensional observation space.
    /// h is [batch, n_heads, d_state, d_head_mimo] — far richer than d_model alone.
    pub h_proj: Linear<B>,
    /// Model dimension
    pub d_model: usize,
    /// Inner SSM dimension
    pub d_inner: usize,
    /// Number of heads
    pub n_heads: usize,
    /// State dimension
    pub d_state: usize,
    /// MIMO rank
    pub mimo_rank: usize,
    /// Number of stacked layers
    pub n_layers: usize,
    /// Whether conv1d is enabled (first layer only)
    pub use_conv: bool,
    /// Conv kernel size
    pub conv_kernel: usize,
    /// Flattened h dimension for h_proj input
    pub h_flat_dim: usize,
}

impl<B: Backend> MultiScaleMambaPredictor<B> {
    /// Create a new multi-scale predictor.
    ///
    /// `input_dim` and `output_dim` are the observation and prediction
    /// dimensions respectively. Both paths (`forward` and `forward_with_action`)
    /// share the same `obs_proj` weights.
    pub fn new(
        config: &MultiScaleSsmConfig,
        input_dim: usize,
        output_dim: usize,
        device: &B::Device,
    ) -> Self {
        let ssms = MultiScaleSsmBlock::new(config, device);
        let output_proj = LinearConfig::new(config.d_model, output_dim).init(device);
        // fuses obs_proj(obs) and action_proj(action): 2*d_model → d_model
        let imagine_fusion = LinearConfig::new(config.d_model * 2, config.d_model).init(device);
        // shared observation encoder used by all forward paths
        let obs_proj = LinearConfig::new(input_dim, config.d_model).init(device);
        let action_proj = LinearConfig::new(output_dim, config.d_model).init(device);

        // h_proj: flatten h [n_heads * d_state * d_head_mimo] → d_model
        // h is the last layer's SSM hidden state — far richer than d_model
        let d_inner = config.d_model * config.expand;
        let d_head = d_inner / config.n_heads;
        let d_head_mimo = d_head / config.mimo_rank;
        let h_flat_dim = config.n_heads * config.d_state * d_head_mimo;
        let h_proj = LinearConfig::new(h_flat_dim, config.d_model).init(device);

        Self {
            ssms,
            output_proj,
            imagine_fusion,
            action_proj,
            obs_proj,
            h_proj,
            d_model: config.d_model,
            d_inner,
            n_heads: config.n_heads,
            d_state: config.d_state,
            mimo_rank: config.mimo_rank,
            n_layers: config.n_layers,
            use_conv: config.use_conv,
            conv_kernel: config.conv_kernel,
            h_flat_dim,
        }
    }

    /// Full-sequence forward pass for training (observation-only, no action conditioning).
    ///
    /// Routes through `obs_proj → imagine_fusion(obs, zero_action) → SSM`, sharing
    /// the same learned projections as [`Self::forward_with_action`].
    /// `predictions[t]` is the model's estimate of `x[t+1]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _] = x.dims();
        let obs_enc = self.obs_proj.forward(x);
        let zero_action = Tensor::zeros([batch, seq_len, self.d_model], &obs_enc.device());
        let u = self
            .imagine_fusion
            .forward(Tensor::cat(vec![obs_enc, zero_action], 2));
        let y = self.ssms.forward(u);
        self.output_proj.forward(y)
    }

    /// Autoregressive step for streaming inference (observation-only).
    ///
    /// Equivalent to [`Self::forward`] but processes one timestep at a time.
    pub fn step(
        &self,
        x: Tensor<B, 2>,
        state: MultiScalePredictorState<B>,
    ) -> (Tensor<B, 2>, MultiScalePredictorState<B>) {
        let [batch, _] = x.dims();
        let obs_enc = self.obs_proj.forward(x);
        let zero_action = Tensor::zeros([batch, self.d_model], &obs_enc.device());
        let u = self
            .imagine_fusion
            .forward(Tensor::cat(vec![obs_enc, zero_action], 1));
        let (y, next_ssms) = self.ssms.forward_step(u, &state.ssms);
        let pred = self.output_proj.forward(y);
        (pred, MultiScalePredictorState { ssms: next_ssms })
    }

    /// Create a zero-initialized state for streaming inference.
    pub fn zero_state(&self, batch: usize, device: &B::Device) -> MultiScalePredictorState<B> {
        MultiScalePredictorState {
            ssms: self.ssms.zero_state(batch, device),
        }
    }

    /// Simple MSE loss for next-step prediction.
    ///
    /// `predictions[t]` is compared against `targets[t+1]`.
    pub fn loss(&self, predictions: Tensor<B, 3>, targets: Tensor<B, 3>) -> Tensor<B, 1> {
        let [batch, seq_len, _] = predictions.dims();
        let pred_slice = predictions.slice([0..batch, 0..seq_len - 1]);
        let target_slice = targets.slice([0..batch, 1..seq_len]);
        (target_slice - pred_slice).powf_scalar(2.0).mean()
    }

    /// Imagination step: predict next observation from SSM output `y` (d_model loop).
    ///
    /// This is the **original** d_model-space closed loop. It decodes y_prev to
    /// observation space, re-encodes to match training distribution, then fuses
    /// with action and feeds the SSM. The obs_proj(output_proj(y)) round-trip
    /// prevents exposure bias at the cost of an information bottleneck through
    /// low-dimensional observation space.
    ///
    /// For a richer closed loop that avoids this bottleneck, see [`step_imagine_h`].
    ///
    /// Returns `(prediction, y_next, next_state)`.
    pub fn step_imagine(
        &self,
        y_prev: Tensor<B, 2>, // [batch, d_model] — previous SSM output
        action: Tensor<B, 2>, // [batch, action_dim]
        state: MultiScalePredictorState<B>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, MultiScalePredictorState<B>) {
        // Decode SSM output to observation space, then re-encode to match
        // the training distribution. During training, imagine_fusion receives
        // [obs_proj(obs), action_proj(action)], not [ssm_output, action_proj(action)].
        // Closing the loop directly in d_model space causes the SSM to drift
        // because it has never seen its own output as input.
        let y_obs = self.output_proj.forward(y_prev);
        let y_enc = self.obs_proj.forward(y_obs.clone());

        let a = self.action_proj.forward(action); // [batch, d_model]
        let u_concat = Tensor::cat(vec![y_enc, a], 1); // [batch, 2*d_model]
        let u = self.imagine_fusion.forward(u_concat); // [batch, d_model]

        let (y_next, next_ssms) = self.ssms.forward_step(u, &state.ssms);

        let pred = y_obs;

        (pred, y_next, MultiScalePredictorState { ssms: next_ssms })
    }

    /// Imagination step using SSM hidden state `h` directly — no observation-space round-trip.
    ///
    /// Unlike [`step_imagine`], this closes the loop entirely within the SSM's internal
    /// representation. The last layer's hidden state `h` (256 dims in default config)
    /// is flattened and projected to `d_model` via `h_proj`, fused with the encoded
    /// action, and fed back into the SSM. This eliminates the information bottleneck
    /// of `output_proj` (d_model → 2) followed by `obs_proj` (2 → d_model).
    ///
    /// For this to work without distribution shift, the model must be trained with
    /// [`forward_with_h_sampling`] so it learns to process h-derived inputs during
    /// training, not just observation-derived inputs.
    ///
    /// Returns `(prediction, y_next, next_state)` where:
    /// - `prediction` is the predicted next observation (output_dim)
    /// - `y_next` is the SSM output (d_model), preserved for compatibility with
    ///   callers that track y_prev
    /// - `next_state` is the updated multi-scale SSM state (containing the new h)
    pub fn step_imagine_h(
        &self,
        state: MultiScalePredictorState<B>,
        action: Tensor<B, 2>, // [batch, action_dim]
    ) -> (Tensor<B, 2>, Tensor<B, 2>, MultiScalePredictorState<B>) {
        // Extract the last layer's hidden state h (the richest summary of history)
        let h_last = state.ssms.h[self.n_layers - 1].clone();
        let [batch, n_heads, d_state, d_head_mimo] = h_last.dims();

        // Flatten h → [batch, h_flat_dim] and project to d_model
        let h_flat = h_last.reshape([batch, n_heads * d_state * d_head_mimo]);
        let h_enc = self.h_proj.forward(h_flat); // [batch, d_model]

        // Fuse with action (same pathway as forward_with_h_sampling training)
        let a = self.action_proj.forward(action); // [batch, d_model]
        let u_concat = Tensor::cat(vec![h_enc, a], 1); // [batch, 2*d_model]
        let u = self.imagine_fusion.forward(u_concat); // [batch, d_model]

        let (y_next, next_ssms) = self.ssms.forward_step(u, &state.ssms);
        let pred = self.output_proj.forward(y_next.clone());

        (pred, y_next, MultiScalePredictorState { ssms: next_ssms })
    }

    /// Full-sequence forward with h-derived input sampling for training.
    ///
    /// This is the training counterpart to [`step_imagine_h`]. At each timestep t,
    /// with probability `h_sampling_prob`, the SSM input is derived from the
    /// previous step's hidden state `h_{t-1}` (via `h_proj`) instead of from the
    /// ground-truth observation `obs_t` (via `obs_proj`).
    ///
    /// This scheduled sampling strategy ensures the SSM learns to process its own
    /// h-derived inputs, eliminating the exposure bias that would otherwise cause
    /// drift when [`step_imagine_h`] is used for open-loop imagination.
    ///
    /// When `h_sampling_prob = 0.0`, this is equivalent to [`forward_with_action`].
    /// Typical values: 0.2–0.5, annealing upward during training.
    ///
    /// # Note
    /// The hidden state from the SSM parallel scan is *not* the same as the
    /// sequential step-by-step h. For exact matching we would need sequential
    /// training. Here we approximate by using a separate sequential pass to
    /// collect per-timestep h states, which is more expensive but correct.
    /// For large-scale training, consider alternating between full forward
    /// and this method every N epochs, or using a teacher-forcing ratio schedule.
    pub fn forward_with_h_sampling(
        &self,
        observations: Tensor<B, 3>, // [batch, seq_len, obs_dim]
        actions: Tensor<B, 3>,      // [batch, seq_len, action_dim]
        h_sampling_prob: f64,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _] = observations.dims();

        if h_sampling_prob <= 0.0 {
            return self.forward_with_action(observations, actions);
        }

        // We need per-step h states. Run a sequential forward pass through the
        // SSM to collect them, while also doing scheduled sampling.
        let mut state = self.zero_state(batch, &observations.device());
        let mut predictions = Vec::with_capacity(seq_len);

        // For sampling: use a simple deterministic threshold based on position
        // (avoids needing RNG in tensor code; in production, use a proper RNG)
        let use_h_at = |t: usize| -> bool {
            if t == 0 {
                return false; // first step always uses observation
            }
            // Deterministic pattern that approximates the target probability
            // e.g. prob=0.3 → use h every 3rd step after the first
            if h_sampling_prob >= 1.0 {
                return true;
            }
            let period = (1.0 / h_sampling_prob).round() as usize;
            t % period == 0
        };

        for t in 0..seq_len {
            let obs_dim = observations.dims()[2];
            let obs_t = observations.clone().slice([0..batch, t..t + 1]).reshape([batch, obs_dim]);
            let act_t = actions.clone().slice([0..batch, t..t + 1]).reshape([batch, actions.dims()[2]]);

            let obs_enc = self.obs_proj.forward(obs_t); // [batch, d_model]

            // Decide whether to use h-derived input or observation-derived input
            let input_enc = if use_h_at(t) {
                // Use hidden state from previous step
                let h_last = state.ssms.h[self.n_layers - 1].clone();
                let [b, nh, ds, dhm] = h_last.dims();
                let h_flat = h_last.reshape([b, nh * ds * dhm]);
                self.h_proj.forward(h_flat)
            } else {
                obs_enc
            };

            let a = self.action_proj.forward(act_t);
            let u_concat = Tensor::cat(vec![input_enc, a], 1); // [batch, 2*d_model]
            let u = self.imagine_fusion.forward(u_concat); // [batch, d_model]

            let (y_t, next_ssms) = self.ssms.forward_step(u, &state.ssms);
            let pred_t = self.output_proj.forward(y_t); // [batch, output_dim]
            predictions.push(pred_t.unsqueeze_dim::<3>(1)); // [batch, 1, output_dim]

            state = MultiScalePredictorState { ssms: next_ssms };
        }

        Tensor::cat(predictions, 1) // [batch, seq_len, output_dim]
    }

    /// Full-sequence forward with separate observations and actions.
    ///
    /// Training variant: encode each timestep's observation and action
    /// independently, fuse, then run SSM to predict the next observation.
    /// This matches the JEPA training pattern but with observation output.
    pub fn forward_with_action(
        &self,
        observations: Tensor<B, 3>, // [batch, seq_len, obs_dim]
        actions: Tensor<B, 3>,      // [batch, seq_len, action_dim]
    ) -> Tensor<B, 3> {
        let obs_enc = self.obs_proj.forward(observations); // [B, T, d_model]
        let act_enc = self.action_proj.forward(actions); // [B, T, d_model]
        let u_concat = Tensor::cat(vec![obs_enc, act_enc], 2); // [B, T, 2*d_model]
        let u = self.imagine_fusion.forward(u_concat); // [B, T, d_model]
        let y = self.ssms.forward(u);
        self.output_proj.forward(y)
    }
}
