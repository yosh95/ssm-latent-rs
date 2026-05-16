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

use crate::ssm::{
    MultiScaleSsmBlock, MultiScaleSsmConfig, MultiScaleState, SsmBlock, SsmConfig,
};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

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

impl<B: Backend> MultiScalePredictorState<B> {
    pub fn zero(
        batch: usize,
        n_layers: usize,
        n_heads: usize,
        d_state: usize,
        d_head_mimo: usize,
        use_conv: bool,
        d_inner: usize,
        conv_kernel: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            ssms: MultiScaleState::zeros(
                batch,
                n_layers,
                n_heads,
                d_state,
                d_head_mimo,
                use_conv,
                d_inner,
                conv_kernel,
                device,
            ),
        }
    }
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
#[derive(Module, Debug)]
pub struct MultiScaleMambaPredictor<B: Backend> {
    /// Input projection: `input_dim` → `d_model`
    pub input_proj: Linear<B>,
    /// Multi-scale stacked SSM blocks (different timescales per layer)
    pub ssms: MultiScaleSsmBlock<B>,
    /// Output projection: `d_model` → `output_dim`
    pub output_proj: Linear<B>,
    /// Fusion for imagination mode: concat(y_t, action_encoded) → d_model
    pub imagine_fusion: Linear<B>,
    /// Observation projection (separate from combined input_proj): obs_dim → d_model
    pub obs_proj: Linear<B>,
    /// Action projection for imagination: action_dim → d_model
    pub action_proj: Linear<B>,
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
}

impl<B: Backend> MultiScaleMambaPredictor<B> {
    /// Create a new multi-scale predictor.
    pub fn new(
        config: &MultiScaleSsmConfig,
        input_dim: usize,
        output_dim: usize,
        device: &B::Device,
    ) -> Self {
        let input_proj = LinearConfig::new(input_dim, config.d_model).init(device);
        let ssms = MultiScaleSsmBlock::new(config, device);
        let output_proj = LinearConfig::new(config.d_model, output_dim).init(device);
        // imagination: concat(ssm_output[d_model], action_encoded[d_model]) → d_model
        let imagine_fusion = LinearConfig::new(config.d_model * 2, config.d_model).init(device);
        // action encoder for imagination mode (shared dim)
        let action_proj = LinearConfig::new(output_dim, config.d_model).init(device);
        // observation projection for separate obs/action path: obs_dim → d_model
        let obs_proj = LinearConfig::new(output_dim, config.d_model).init(device);

        Self {
            input_proj,
            ssms,
            output_proj,
            imagine_fusion,
            action_proj,
            obs_proj,
            d_model: config.d_model,
            d_inner: config.d_model * config.expand,
            n_heads: config.n_heads,
            d_state: config.d_state,
            mimo_rank: config.mimo_rank,
            n_layers: config.n_layers,
            use_conv: config.use_conv,
            conv_kernel: config.conv_kernel,
        }
    }

    /// Full-sequence forward pass for training.
    ///
    /// `predictions[t]` is the model's estimate of `x[t+1]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let u = self.input_proj.forward(x);
        let y = self.ssms.forward(u);
        self.output_proj.forward(y)
    }

    /// Autoregressive step for streaming inference.
    pub fn step(
        &self,
        x: Tensor<B, 2>,
        state: MultiScalePredictorState<B>,
    ) -> (Tensor<B, 2>, MultiScalePredictorState<B>) {
        let u = self.input_proj.forward(x);
        let (y, next_ssms) = self.ssms.forward_step(u, &state.ssms);

        let pred = self.output_proj.forward(y);

        (
            pred,
            MultiScalePredictorState { ssms: next_ssms },
        )
    }

    /// Create a zero-initialized state for streaming inference.
    pub fn zero_state(
        &self,
        batch: usize,
        device: &B::Device,
    ) -> MultiScalePredictorState<B> {
        let d_head = self.d_inner / self.n_heads;
        let d_head_mimo = d_head / self.mimo_rank;

        MultiScalePredictorState::zero(
            batch,
            self.n_layers,
            self.n_heads,
            self.d_state,
            d_head_mimo,
            self.use_conv,
            self.d_inner,
            self.conv_kernel,
            device,
        )
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

    /// Imagination step: predict next observation from SSM internal state, without
    /// external observations. The loop is closed in d_model space (like JEPA's
    /// latent loop), avoiding the input_proj/output_proj round-trip distortion.
    ///
    /// Takes the previous SSM output `y_prev` (d_model), fuses it with the
    /// encoded action, runs the SSM step, and projects to observation space.
    ///
    /// Returns `(prediction, y_next, next_state)` where:
    /// - `prediction` is the predicted next observation (output_dim)
    /// - `y_next` is the SSM output to feed back as `y_prev` on the next call
    /// - `next_state` is the updated multi-scale SSM state
    pub fn step_imagine(
        &self,
        y_prev: Tensor<B, 2>,       // [batch, d_model] — previous SSM output
        action: Tensor<B, 2>,       // [batch, action_dim]
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
        let u = self.imagine_fusion.forward(u_concat);  // [batch, d_model]

        let (y_next, next_ssms) = self.ssms.forward_step(u, &state.ssms);

        // During training, output_proj(SSM_output[t-1]) ≈ obs[t].
        // y_obs = output_proj(y_prev) is the prediction for the *current* step.
        // y_next will become y_prev on the next call (after this step's action
        // has been applied), so output_proj(y_next) would be the *next* step.
        let pred = y_obs;

        (pred, y_next, MultiScalePredictorState { ssms: next_ssms })
    }

    /// Full-sequence forward with separate observations and actions.
    ///
    /// Training variant: encode each timestep's observation and action
    /// independently, fuse, then run SSM to predict the next observation.
    /// This matches the JEPA training pattern but with observation output.
    pub fn forward_with_action(
        &self,
        observations: Tensor<B, 3>,  // [batch, seq_len, obs_dim]
        actions: Tensor<B, 3>,       // [batch, seq_len, action_dim]
    ) -> Tensor<B, 3> {
        let obs_enc = self.obs_proj.forward(observations); // [B, T, d_model]
        let act_enc = self.action_proj.forward(actions);     // [B, T, d_model]
        let u_concat = Tensor::cat(vec![obs_enc, act_enc], 2); // [B, T, 2*d_model]
        let u = self.imagine_fusion.forward(u_concat);          // [B, T, d_model]
        let y = self.ssms.forward(u);
        self.output_proj.forward(y)
    }
}
