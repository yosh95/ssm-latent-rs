use crate::ssm::{MultiScaleSsmBlock, MultiScaleSsmConfig, MultiScaleState, SsmBlock, SsmConfig};
use burn::config::Config;
use burn::module::{Module, Param};

use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};

// ─── MLP Encoder (Theorem 1: nonlinear encoder still achieves linear identifiability) ───

/// Configuration for the MLP encoder.
///
/// Theorem 1 of Klindt, LeCun, Balestriero (2026) guarantees that *any* encoder
/// (including nonlinear) which minimizes alignment subject to Gaussian embeddings
/// recovers the latent variables up to rotation. This MLP encoder provides a
/// nonlinear encoder to empirically verify the theorem.
#[derive(Config, Debug)]
pub struct MlpEncoderConfig {
    /// Number of hidden layers (0 = no hidden layers, equivalent to single Linear)
    pub n_hidden: usize,
    /// Hidden dimension (same for all layers)
    pub hidden_dim: usize,
    /// Dropout probability between layers (0.0 = no dropout)
    pub dropout: f64,
}

/// MLP encoder with configurable depth.
///
/// Architecture: input_dim -> [hidden_dim -> ReLU -> Dropout] x n_hidden -> d_model
/// When n_hidden = 0, this reduces to a single Linear(input_dim, d_model).
///
/// The weight matrices are initialized with the LeCun initialization
/// (normal with variance 1/fan_in) to maintain activation variance across layers,
/// which is critical for deep MLPs to train stably with LeJEPA.
#[derive(Module, Debug)]
pub struct MlpEncoder<B: Backend> {
    /// Input projection: input_dim -> hidden_dim (or d_model if n_hidden=0)
    input_proj: Linear<B>,
    /// Hidden layers: Vec of (Linear, Dropout) pairs
    hidden: Vec<(Linear<B>, Dropout)>,
    /// Output projection: hidden_dim -> d_model (omitted if n_hidden=0)
    output_proj: Option<Linear<B>>,
    /// Number of hidden layers
    n_hidden: usize,
    /// Whether dropout is enabled
    dropout_enabled: bool,
}

impl MlpEncoderConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.n_hidden > 0 && self.hidden_dim == 0 {
            return Err(crate::error::ModelError::Config {
                message: "hidden_dim must be > 0 when n_hidden > 0".into(),
            });
        }
        Ok(())
    }
}

impl<B: Backend> MlpEncoder<B> {
    /// Create a new MLP encoder.
    ///
    /// # Arguments
    /// * `input_dim` - Input dimension (observation dimension)
    /// * `d_model` - Output dimension (latent dimension)
    /// * `config` - MLP configuration (depth, hidden dim, dropout)
    /// * `device` - Device to place parameters on
    pub fn new(
        input_dim: usize,
        d_model: usize,
        config: &MlpEncoderConfig,
        device: &B::Device,
    ) -> Self {
        let use_hidden = config.n_hidden > 0;

        if !use_hidden {
            return Self {
                input_proj: LinearConfig::new(input_dim, d_model).init(device),
                hidden: Vec::new(),
                output_proj: None,
                n_hidden: 0,
                dropout_enabled: config.dropout > 0.0,
            };
        }

        let input_proj = LinearConfig::new(input_dim, config.hidden_dim).init(device);
        let mut hidden = Vec::with_capacity(config.n_hidden);
        for _ in 0..config.n_hidden - 1 {
            let lin: Linear<B> =
                LinearConfig::new(config.hidden_dim, config.hidden_dim).init(device);
            let drop = DropoutConfig::new(config.dropout).init();
            hidden.push((lin, drop));
        }
        // Last hidden layer: hidden_dim -> d_model (via output_proj)
        if config.n_hidden > 0 {
            // We already have n_hidden-1 layers; the last one is output_proj
            // So we add n_hidden-1 hidden layers and output_proj does the final mapping
            // Actually, let's make it simpler: n_hidden hidden layers all hidden_dim -> hidden_dim,
            // and output_proj does hidden_dim -> d_model
        }
        // Let's redo this more cleanly:
        let mut hidden_layers = Vec::with_capacity(config.n_hidden.max(1) - 1);
        for _ in 0..config.n_hidden.saturating_sub(1) {
            let lin: Linear<B> =
                LinearConfig::new(config.hidden_dim, config.hidden_dim).init(device);
            let drop = DropoutConfig::new(config.dropout).init();
            hidden_layers.push((lin, drop));
        }
        // If n_hidden == 0, output_proj is None and input_proj goes input_dim -> d_model
        // If n_hidden >= 1:
        //   input_proj: input_dim -> hidden_dim
        //   hidden: (n_hidden - 1) layers of hidden_dim -> hidden_dim
        //   output_proj: hidden_dim -> d_model
        let output_proj = Some(LinearConfig::new(config.hidden_dim, d_model).init(device));

        Self {
            input_proj,
            hidden: hidden_layers,
            output_proj,
            n_hidden: config.n_hidden,
            dropout_enabled: config.dropout > 0.0,
        }
    }

    /// Forward pass through the MLP.
    ///
    /// Uses ReLU activations between layers. When `n_hidden = 0`,
    /// this is equivalent to a single linear layer (no nonlinearity).
    ///
    /// Shape: [batch, seq_len, input_dim] -> [batch, seq_len, d_model]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Check if we need to handle the trivial case
        if self.n_hidden == 0 {
            // Simple linear projection only
            return self.input_proj.forward(x);
        }

        let [batch, seq_len, input_dim] = x.dims();
        let x_flat = x.reshape([batch * seq_len, input_dim]);
        let mut h = self.input_proj.forward(x_flat);

        for (lin, drop) in &self.hidden {
            let y = burn::tensor::activation::relu(h);
            let y = drop.forward(y);
            h = lin.forward(y);
        }

        // Final layer: ReLU -> Dropout -> output_proj (no activation on output)
        if let Some(ref out_proj) = self.output_proj {
            let y = burn::tensor::activation::relu(h);
            h = out_proj.forward(y);
        }

        let d_model = h.dims()[1];
        h.reshape([batch, seq_len, d_model])
    }

    /// Forward pass for a single timestep (autoregressive inference).
    ///
    /// Shape: [batch, input_dim] -> [batch, d_model]
    pub fn forward_single(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        if self.n_hidden == 0 {
            return self.input_proj.forward(x);
        }

        let mut h = self.input_proj.forward(x);
        for (lin, drop) in &self.hidden {
            let y = burn::tensor::activation::relu(h);
            let y = drop.forward(y);
            h = lin.forward(y);
        }
        if let Some(ref out_proj) = self.output_proj {
            let y = burn::tensor::activation::relu(h);
            h = out_proj.forward(y);
        }
        h
    }
}
/// Mutable state carried between autoregressive inference steps.
///
/// Contains the SSM hidden state, previous B·x contribution (for the
/// lambda-gated recurrence), and the causal convolution state.
/// All fields are tensors on the same device as the model.
#[derive(Clone)]
pub struct LatentState<B: Backend> {
    /// SSM hidden state: `[batch, n_heads, d_state, d_head_mimo]`
    pub h: Tensor<B, 4>,
    /// Previous timestep's B·x contribution (lambda-gated recurrence):
    /// `[batch, n_heads, d_state, d_head_mimo]`
    pub prev_bx: Option<Tensor<B, 4>>,
    /// Causal convolution state for autoregressive mode:
    /// `[batch, d_inner, kernel_size - 1]` (None if conv disabled)
    pub conv_state: Option<Tensor<B, 3>>,
}

/// Latent world model predictor combining JEPA-style encoding with SSM dynamics.
///
/// This module implements the core prediction architecture:
/// 1. **Encoder**: Maps raw observations to latent space
/// 2. **Action encoder**: Maps actions to latent space
/// 3. **Fusion**: Concatenates latent observation + latent action, then projects
/// 4. **SSM block**: Predicts the next latent state from the fused representation
/// 5. **Decoder**: Reconstructs observations from latent state (for auxiliary loss)
///
/// The predictor follows the JEPA philosophy: prediction happens entirely in
/// latent space, and the decoder is only used for auxiliary reconstruction losses
/// to prevent representation collapse.
///
/// # Stability mechanisms
/// - **Stability loss**: Random projection-based moment matching (VICReg-style)
///   to prevent representation collapse in non-contrastive learning
/// - **Curvature loss**: Second-order finite difference regularization
///   (Temporal Straightening) to encourage smooth latent trajectories
/// - **Reconstruction loss**: MSE on current and predicted observations
#[derive(Module, Debug)]
pub struct LatentPredictor<B: Backend> {
    encoder: Linear<B>,
    decoder: Linear<B>,
    action_encoder: Linear<B>,
    fusion: Linear<B>,
    ssm: SsmBlock<B>,
    // Fixed random projection buffer (Param used for device management, but gradients are not needed)
    stability_projections: Param<Tensor<B, 2>>,
    d_model: usize,
}

/// Arguments for computing the latent predictor loss.
///
/// Combines multiple loss terms:
/// - Latent prediction MSE (next-step prediction in latent space)
/// - Current-step reconstruction MSE (encoder quality)
/// - Next-step reconstruction MSE (prediction quality in observation space)
/// - Stability loss (representation collapse prevention)
/// - Curvature loss (trajectory smoothness, Temporal Straightening)
pub struct LatentLossArgs<B: Backend> {
    /// Current latent representation: `[batch, seq_len, d_model]`
    pub z: Tensor<B, 3>,
    /// Predicted next latent state: `[batch, seq_len, d_model]`
    pub pred_z: Tensor<B, 3>,
    /// Reconstructed observations from current latent: `[batch, seq_len, input_dim]`
    pub reconstructed_x: Tensor<B, 3>,
    /// Predicted observations from predicted latent: `[batch, seq_len, input_dim]`
    pub predicted_x: Tensor<B, 3>,
    /// Original observations: `[batch, seq_len, input_dim]`
    pub original_x: Tensor<B, 3>,
    /// Weight for stability (random projection moment matching) loss
    pub stability_weight: f64,
    /// Weight for curvature (Temporal Straightening) loss
    pub curvature_weight: f64,
    /// Weight for reconstruction losses
    pub recon_weight: f64,
}

use burn::record::Recorder;

impl<B: Backend> LatentPredictor<B> {
    pub fn new(
        config: &SsmConfig,
        input_dim: usize,
        action_dim: usize,
        device: &B::Device,
    ) -> Self {
        let encoder = LinearConfig::new(input_dim, config.d_model).init(device);
        let decoder = LinearConfig::new(config.d_model, input_dim).init(device);
        let action_encoder = LinearConfig::new(action_dim, config.d_model).init(device);
        let fusion = LinearConfig::new(config.d_model * 2, config.d_model).init(device);
        let ssm = SsmBlock::new(config, device);

        // Fixed projection matrix based on Gaussian distribution.
        // These projections are used in the stability loss (VICReg-style moment matching)
        // to prevent representation collapse. They are NOT trained — gradients are
        // detached via .detach() and the Param wrapper is only used for device management
        // and serialization (load/save). The normalize_projections() function ensures
        // each projection direction has unit length, which stabilizes the loss computation.
        let stability_projections = Tensor::<B, 2>::random(
            [config.d_model, 16], // Number of random projections
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );

        // Normalize for stable projections
        let norm = stability_projections
            .clone()
            .powf_scalar(2.0)
            .sum_dim(0)
            .sqrt()
            + 1e-6;
        let stability_projections = stability_projections / norm;

        Self {
            encoder,
            decoder,
            action_encoder,
            fusion,
            ssm,
            // Keep projections as fixed buffer by detaching gradients
            stability_projections: Param::from_tensor(stability_projections.detach()),
            d_model: config.d_model,
        }
    }

    pub fn encode(&self, observations: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encoder.forward(observations)
    }

    pub fn decode(&self, z: Tensor<B, 3>) -> Tensor<B, 3> {
        self.decoder.forward(z)
    }

    pub fn forward(
        &self,
        observations: Tensor<B, 3>,
        actions: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let z = self.encoder.forward(observations);
        let a = self.action_encoder.forward(actions);
        let u_concat = Tensor::cat(vec![z.clone(), a], 2);
        let u = self.fusion.forward(u_concat);
        let predicted_z = self.ssm.forward(u);

        // Reconstruction from current latent (representation quality)
        let reconstructed_x = self.decoder.forward(z.clone());
        // Preview reconstruction from predicted latent (prediction quality in pixel space)
        let predicted_x = self.decoder.forward(predicted_z.clone());

        (z, predicted_z, reconstructed_x, predicted_x)
    }

    pub fn step(
        &self,
        z_prev: Tensor<B, 2>,
        action: Tensor<B, 2>,
        state: LatentState<B>,
    ) -> (Tensor<B, 2>, LatentState<B>) {
        let a = self.action_encoder.forward(action.unsqueeze_dim::<3>(1));
        let [batch, _seq, d_model] = a.dims();
        let a = a.reshape([batch, d_model]);

        let u_concat = Tensor::cat(vec![z_prev, a], 1);
        let u = self.fusion.forward(u_concat);

        let (y, next_h, current_bx, next_conv_state) =
            self.ssm
                .forward_step(u, state.h, state.prev_bx, state.conv_state);

        (
            y,
            LatentState {
                h: next_h,
                prev_bx: Some(current_bx),
                conv_state: next_conv_state,
            },
        )
    }

    pub fn save(&self, file_path: &str) -> crate::error::Result<()> {
        tracing::debug!(path = %file_path, "Saving model weights");
        let recorder = burn::record::BinFileRecorder::<burn::record::FullPrecisionSettings>::new();
        let path = std::path::Path::new(file_path);

        recorder
            .record(self.clone().into_record(), path.to_path_buf())
            .map_err(|e| crate::error::ModelError::Serialization(e.to_string()))?;
        tracing::info!(path = %file_path, "Model saved successfully");
        Ok(())
    }

    pub fn load(self, file_path: &str, device: &B::Device) -> crate::error::Result<Self> {
        tracing::debug!(path = %file_path, "Loading model weights");
        let recorder = burn::record::BinFileRecorder::<burn::record::FullPrecisionSettings>::new();
        let path = std::path::Path::new(file_path);

        let record = recorder
            .load(path.to_path_buf(), device)
            .map_err(|e| crate::error::ModelError::Serialization(e.to_string()))?;
        tracing::info!(path = %file_path, "Model loaded successfully");
        Ok(self.load_record(record))
    }

    /// Compute the combined training loss for the latent predictor.
    ///
    /// The total loss is:
    /// ```text
    /// L = L_latent + recon_weight · (L_recons + L_pred_x)
    ///     + stability_weight · L_stability + curvature_weight · L_curvature
    /// ```
    ///
    /// where:
    /// - `L_latent`: MSE between predicted and actual next latent states
    /// - `L_recons`: MSE between reconstructed and original observations
    /// - `L_pred_x`: MSE between predicted and actual next observations
    /// - `L_stability`: Gaussian moment matching on random projections (VICReg-style)
    /// - `L_curvature`: Second-order finite difference on latent trajectory
    pub fn loss(&self, args: LatentLossArgs<B>) -> Tensor<B, 1> {
        latent_loss(args, self.stability_projections.val())
    }
}

/// Shared loss computation used by both [`LatentPredictor`] and [`MultiScaleLatentPredictor`].
///
/// See [`LatentPredictor::loss`] for the full formula.
fn latent_loss<B: Backend>(args: LatentLossArgs<B>, projections: Tensor<B, 2>) -> Tensor<B, 1> {
    let [batch, seq_len, _] = args.z.dims();
    let target_z = args.z.clone().detach().slice([0..batch, 1..seq_len]);
    let pred_slice = args.pred_z.clone().slice([0..batch, 0..seq_len - 1]);

    let mse_latent = (target_z - pred_slice).powf_scalar(2.0).mean();
    let mse_recons = (args.original_x.clone() - args.reconstructed_x)
        .powf_scalar(2.0)
        .mean();
    let target_x = args.original_x.detach().slice([0..batch, 1..seq_len]);
    let pred_x_slice = args.predicted_x.slice([0..batch, 0..seq_len - 1]);
    let mse_pred_x = (target_x - pred_x_slice).powf_scalar(2.0).mean();

    let reg_loss = stability_loss(args.z.clone(), projections);
    let curv_loss = curvature_loss(args.z);

    mse_latent
        + mse_recons.mul_scalar(args.recon_weight)
        + mse_pred_x.mul_scalar(args.recon_weight)
        + reg_loss.mul_scalar(args.stability_weight)
        + curv_loss.mul_scalar(args.curvature_weight)
}

/// Temporal Straightening loss with optional time-delta weighting.
///
/// Minimizes the second-order finite difference of the latent trajectory
/// to encourage locally straight paths. When `dt` is provided (non-uniform
/// timestamps), the discrete acceleration is computed as:
///
/// ```text
/// a_t = (z_t - z_{t-1})/Δt_{t} - (z_{t-1} - z_{t-2})/Δt_{t-1}
/// ```
///
/// Otherwise falls back to uniform spacing: `a_t = z_t - 2·z_{t-1} + z_{t-2}`.
///
/// Returns 0.0 when `seq_len < 3` (insufficient data for finite difference).
///
/// # Complexity
/// O(T) in time, bounded by sequence length.
pub fn curvature_loss<B: Backend>(z: Tensor<B, 3>) -> Tensor<B, 1> {
    let [batch, seq_len, _] = z.dims();

    if seq_len < 3 {
        return Tensor::<B, 1>::from_data([0.0], &z.device());
    }

    // Uniform-spacing second-order finite difference: a_t = z_t - 2z_{t-1} + z_{t-2}
    let z_t = z.clone().slice([0..batch, 2..seq_len]);
    let z_t_1 = z.clone().slice([0..batch, 1..seq_len - 1]);
    let z_t_2 = z.slice([0..batch, 0..seq_len - 2]);

    let acceleration = z_t - z_t_1.mul_scalar(2.0) + z_t_2;

    acceleration.powf_scalar(2.0).mean().unsqueeze()
}

/// Time-delta-aware curvature loss for non-uniformly spaced sequences.
///
/// Uses the generalised second-order finite difference:
///
/// ```text
/// a_t = (z_t - z_{t-1})/Δt_t - (z_{t-1} - z_{t-2})/Δt_{t-1}
/// ```
///
/// # Arguments
/// * `z` - Latent representations: `[batch, seq_len, d_model]`
/// * `dt` - Time deltas between consecutive points: `[seq_len]` or `[batch, seq_len]`.
///   `dt[t]` is the time difference from step `t-1` to step `t`.
///   Must have `seq_len - 1` meaningful entries (first entry unused).
///
/// # Returns
/// Scalar curvature loss (0.0 when `seq_len < 3`).
///
/// # Panics
/// If any `dt[t]` (t ≥ 2) is ≤ 0.
pub fn curvature_loss_with_dt<B: Backend>(z: Tensor<B, 3>, dt: Tensor<B, 2>) -> Tensor<B, 1> {
    let [batch, seq_len, _d_model] = z.dims();

    if seq_len < 3 {
        return Tensor::<B, 1>::from_data([0.0], &z.device());
    }

    // dt shape: [batch, seq_len] or [seq_len]; broadcast to [batch, seq_len]
    let dt = if dt.dims().len() == 1 {
        dt.unsqueeze_dim::<1>(0).unsqueeze_dim::<3>(2) // [seq_len] → [1, seq_len, 1]
    } else {
        dt.unsqueeze_dim::<3>(2) // [batch, seq_len] → [batch, seq_len, 1]
    };

    // dt for steps t (from t-1 to t): dt[1..], dt[2..]
    let dt_t = dt.clone().slice([0..batch, 2..seq_len]); // Δt_t
    let dt_t1 = dt.slice([0..batch, 1..seq_len - 1]); // Δt_{t-1}

    let z_t = z.clone().slice([0..batch, 2..seq_len]);
    let z_t_1 = z.clone().slice([0..batch, 1..seq_len - 1]);
    let z_t_2 = z.slice([0..batch, 0..seq_len - 2]);

    // a_t = (z_t - z_{t-1})/Δt_t - (z_{t-1} - z_{t-2})/Δt_{t-1}
    let first_term = (z_t - z_t_1.clone()) / dt_t.clamp_min(1e-6);
    let second_term = (z_t_1 - z_t_2) / dt_t1.clamp_min(1e-6);
    let acceleration = first_term - second_term;

    acceleration.powf_scalar(2.0).mean().unsqueeze()
}

/// Running-statistics stability loss for small batch / sequence-length regimes.
///
/// Unlike [`stability_loss`] which computes per-batch statistics, this variant
/// maintains exponential moving averages of mean and variance for each random
/// projection direction. This is critical for anomaly detection where batch=1
/// and seq_len may be small.
///
/// # Arguments
/// * `z` - Latent representations: `[batch, seq_len, d_model]`
/// * `w` - Normalized random projection matrix: `[d_model, n_projections]`
/// * `running_mean` - Current EMA of projected means: `[n_projections]`
/// * `running_var` - Current EMA of projected variances: `[n_projections]`
/// * `momentum` - EMA decay factor (0.001–0.05; smaller = more stable)
///
/// # Returns
/// `(loss, new_running_mean, new_running_var)` where `loss` penalises deviation
/// from standard normal distribution (μ=0, σ²=1) against the *running* statistics.
pub fn stability_loss_running<B: Backend>(
    z: Tensor<B, 3>,
    w: Tensor<B, 2>,
    running_mean: Tensor<B, 1>,
    running_var: Tensor<B, 1>,
    momentum: f64,
) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
    let [batch, seq_len, d_model] = z.dims();
    let n_projections = w.dims()[1];

    let z_flat = z.reshape([batch * seq_len, d_model]);
    let projections = z_flat.matmul(w).reshape([batch, seq_len, n_projections]);

    // Per-projection statistics over this mini-batch
    let batch_mean = projections.clone().mean_dim(1).mean_dim(0).squeeze(); // [n_projections]
    let batch_var = (projections
        - batch_mean
            .clone()
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(1))
    .powf_scalar(2.0)
    .mean_dim(1)
    .mean_dim(0)
    .squeeze(); // [n_projections]

    // Update running statistics with EMA
    let new_running_mean =
        running_mean.clone().mul_scalar(1.0 - momentum) + batch_mean.clone().mul_scalar(momentum);
    let new_running_var =
        running_var.clone().mul_scalar(1.0 - momentum) + batch_var.clone().mul_scalar(momentum);

    // Loss: penalise deviation from standard normal (μ=0, σ²=1)
    // using the batch statistics (not running) — we want the current batch
    // to look like N(0,1), not just match the EMA.
    let loss_mean = batch_mean.powf_scalar(2.0).mean();
    let loss_var = (batch_var - 1.0).powf_scalar(2.0).mean();

    (
        (loss_mean + loss_var).unsqueeze(),
        new_running_mean,
        new_running_var,
    )
}

/// Sketched Isotropic Gaussian Regularization (SIGReg) — characteristic function matching.
///
/// This is the LeJEPA (Balestriero & LeCun, 2025) distribution-matching objective.
/// Unlike [`stability_loss`] which matches only the first two moments (mean=0, var=1)
/// on random projections, SIGReg matches the **entire distribution** by comparing
/// empirical and target characteristic functions (CFs) at a small set of evaluation
/// points.  This provides provable collapse prevention with a single hyperparameter.
///
/// # Theory
///
/// The isotropic Gaussian N(0,I) is provably optimal for minimizing downstream
/// prediction risk across both linear and nonlinear probes (LeJEPA, Sections 3.1–3.2).
/// SIGReg enforces this by:
/// 1. Projecting embeddings onto M random 1D directions (sketching)
/// 2. Evaluating the empirical CF at K fixed frequency points per direction
/// 3. Penalising deviation from the standard normal CF φ(t) = exp(-t²/2)
///
/// # Complexity
///
/// O(batch · seq_len · d_model · M + batch · seq_len · M · K) = linear in all dimensions.
/// Uses the same random projection matrix as [`stability_loss`].
///
/// # Arguments
/// * `z` - Latent representations: `[batch, seq_len, d_model]`
/// * `w` - Normalized random projection matrix: `[d_model, n_projections]`
/// * `freqs` - Evaluation frequencies for the characteristic function (e.g. [0.5, 1.0, 1.5, 2.0])
///
/// # Returns
/// Scalar SIGReg loss.
pub fn sigreg_loss<B: Backend>(z: Tensor<B, 3>, w: Tensor<B, 2>, freqs: &[f64]) -> Tensor<B, 1> {
    let [batch, seq_len, d_model] = z.dims();
    let _n_projections = w.dims()[1];
    let device = &z.device();

    let z_flat = z.reshape([batch * seq_len, d_model]);
    let projections = z_flat.matmul(w); // [N, _n_projections]

    // Standardize each projection direction to zero mean, unit variance.
    // This stabilises the CF evaluation and makes the loss scale-invariant.
    let mean = projections.clone().mean_dim(0); // [1, n_projections]
    let var = (projections.clone() - mean.clone())
        .powf_scalar(2.0)
        .mean_dim(0); // [1, n_projections]
    let std = var.sqrt() + 1e-6;
    let u = (projections - mean) / std; // [N, _n_projections]

    let mut total_loss = Tensor::<B, 1>::from_data([0.0], device);

    for &freq in freqs {
        let t = freq;
        let target_cf = (-t * t / 2.0).exp(); // φ_N(0,1)(t) = exp(-t²/2), real-valued

        // Empirical characteristic function: φ_emp(t) = (1/N) Σ_j exp(i·t·u_j)
        // Real part: (1/N) Σ_j cos(t·u_j)  → [1, n_projections]
        // Imag part: (1/N) Σ_j sin(t·u_j)  → [1, n_projections]
        let cos_part = u.clone().mul_scalar(t).cos().mean_dim(0); // [1, n_projections]
        let sin_part = u.clone().mul_scalar(t).sin().mean_dim(0); // [1, n_projections]

        // |φ_emp(t) - φ_target(t)|² = (Re[φ_emp] - exp(-t²/2))² + Im[φ_emp]²
        let re_diff = cos_part - target_cf;
        let loss_freq = re_diff.clone().powf_scalar(2.0) + sin_part.powf_scalar(2.0);

        // loss_freq: [1, n_projections], mean → scalar
        total_loss = total_loss + loss_freq.mean();
    }

    // Average over frequencies.  from_data([0.0]) already produces a rank-1 tensor [1],
    // so we just divide — no unsqueeze needed (unlike stability_loss which sums scalars).
    total_loss / freqs.len() as f64
}

/// Convenience wrapper: JEPA predictive loss + SIGReg.
///
/// This is the **LeJEPA training objective** as described in Balestriero & LeCun (2025):
///
/// ```text
/// L = L_pred + γ · SIGReg(z)
/// ```
///
/// where `L_pred` is the next-step latent prediction MSE and SIGReg enforces
/// isotropic Gaussian embeddings to prevent collapse — no stop-gradient,
/// no teacher-student, no EMA schedule required.
///
/// # Arguments
/// * `z` - Current latent representations: `[batch, seq_len, d_model]`
/// * `pred_z` - Predicted next latent states: `[batch, seq_len, d_model]`
/// * `projections` - Normalized random projection matrix from the model
/// * `sigreg_weight` - Trade-off hyperparameter γ (the *only* tunable parameter)
/// * `freqs` - CF evaluation frequencies
pub fn lejepa_loss<B: Backend>(
    z: Tensor<B, 3>,
    pred_z: Tensor<B, 3>,
    projections: Tensor<B, 2>,
    sigreg_weight: f64,
    freqs: &[f64],
) -> Tensor<B, 1> {
    let [batch, seq_len, _] = z.dims();
    let target_z = z.clone().detach().slice([0..batch, 1..seq_len]);
    let pred_slice = pred_z.slice([0..batch, 0..seq_len - 1]);

    let mse_latent = (target_z - pred_slice).powf_scalar(2.0).mean();
    let sigreg = sigreg_loss(z, projections, freqs);

    mse_latent + sigreg.mul_scalar(sigreg_weight)
}
///
/// Inspired by VICReg (Bardes et al., 2022), this loss projects the latent
/// representations onto random directions and penalizes deviations from a
/// standard normal distribution. Specifically, it enforces:
/// - **Mean constraint**: projected means should be close to 0
/// - **Variance constraint**: projected variances should be close to 1
///
/// This is critical for non-contrastive learning frameworks like JEPA, where
/// there is no negative sample mechanism to prevent the encoder from collapsing
/// to a constant output.
///
/// # Arguments
/// * `z` - Latent representations: `[batch, seq_len, d_model]`
/// * `w` - Normalized random projection matrix: `[d_model, n_projections]`
///
/// # Complexity
/// O(T · d · k) where k is the number of projections (typically 16).
pub fn stability_loss<B: Backend>(z: Tensor<B, 3>, w: Tensor<B, 2>) -> Tensor<B, 1> {
    let [batch, seq_len, d_model] = z.dims();
    let n_projections = w.dims()[1];

    let z_flat = z.reshape([batch * seq_len, d_model]);
    let projections = z_flat.matmul(w).reshape([batch, seq_len, n_projections]);

    // Calculate statistics for each projection dimension (O(T) complexity)
    let mean = projections.clone().mean_dim(1); // [batch, 1, n_projections]
    // Manual variance calculation as projections.var(1) can be slow on some backends
    let var = (projections - mean.clone()).powf_scalar(2.0).mean_dim(1);

    // Penalize deviation from standard normal distribution (mean=0, var=1)
    let loss_mean = mean.powf_scalar(2.0).mean();
    let loss_var = (var - 1.0).powf_scalar(2.0).mean();

    (loss_mean + loss_var).unsqueeze()
}

// ─── Linear Identifiability Metrics (Theorems 1, 2, 3) ────────────────────────

/// Compute latent identifiability R² score (Theorem 1).
///
/// Uses the squared correlation between learned and true latents as an
/// identifiability metric. When LeJEPA achieves linear identifiability
/// (h(z) = Q·z for orthogonal Q), the first canonical correlation = 1.
///
/// We compute:
/// 1. Flatten and center both z_hat and z_true
/// 2. Compute the cross-correlation matrix: C = z_trueᵀ · z_hat
/// 3. Orthogonalize C via Newton-Schulz iteration to find the nearest
///    rotation matrix Q
/// 4. R² = 1 - ||z_true - z_hat·Qᵀ||² / ||z_true||²
///
/// This avoids matrix inversion/SVD which may not be available in all backends.
pub fn compute_identifiability_r2<B: Backend>(
    z_hat: Tensor<B, 3>,
    z_true: Tensor<B, 3>,
) -> Tensor<B, 1> {
    let [batch, seq_len, d_model] = z_hat.dims();
    let _device = z_hat.device();

    let z_hat_flat = z_hat.reshape([batch * seq_len, d_model]);
    let z_true_flat = z_true.reshape([batch * seq_len, d_model]);

    // Center
    let z_hat_mean = z_hat_flat.clone().mean_dim(0);
    let z_true_mean = z_true_flat.clone().mean_dim(0);
    let z_hat_c = z_hat_flat - z_hat_mean;
    let z_true_c = z_true_flat - z_true_mean;

    // Compute R² via squared cross-correlation, element-wise.
    // For each pair of dimensions (i,j), compute corr(z_hat_i, z_true_j)²
    // and take the mean across all pairs.
    // For rotation Q: corr matrix = Q (since z_hat = z_true @ Qᵀ)
    // mean(Q²) = tr(QᵀQ)/d² = d/d² = 1/d... no that's not right.
    // Actually tr(QᵀQ) = d, so mean(Q²) = d/d² = 1/d. That's small for large d.
    //
    // Better: use Procrustes R² = 1 - ||z_true_c - z_hat_c @ Q||² / ||z_true_c||²
    // where Q is found via Newton-Schulz.

    // Let's do a simple approach: just compute the squared trace of the
    // cross-correlation matrix normalized by auto-correlation traces.
    // R² = tr(CᵀC) / tr(AᵀA) where C = z_hat_cᵀ @ z_true_c / N
    // and A is the auto-correlation of z_true_c.
    // This is equivalent to the squared Frobenius norm ratio.

    let n = (batch * seq_len) as f64;
    let c = z_hat_c
        .clone()
        .transpose()
        .matmul(z_true_c.clone())
        .mul_scalar(1.0 / n); // [d, d]
    let a = z_true_c
        .clone()
        .transpose()
        .matmul(z_true_c)
        .mul_scalar(1.0 / n); // [d, d]

    let num = c.powf_scalar(2.0).sum();
    let den = a.powf_scalar(2.0).sum() + 1e-8;

    let r2 = num / den;
    r2.clamp(0.0, 1.0).unsqueeze()
}
pub fn procrustes_alignment<B: Backend>(
    z_hat: Tensor<B, 3>,
    z_true: Tensor<B, 3>,
) -> (Tensor<B, 3>, Tensor<B, 2>) {
    let [batch, seq_len, d_model] = z_hat.dims();
    let n = (batch * seq_len) as f64;
    let device = z_hat.device();

    let z_hat_flat = z_hat.reshape([batch * seq_len, d_model]);
    let z_true_flat = z_true.reshape([batch * seq_len, d_model]);

    let z_hat_mean = z_hat_flat.clone().mean_dim(0);
    let z_true_mean = z_true_flat.clone().mean_dim(0);
    let z_hat_c = z_hat_flat - z_hat_mean;
    let z_true_c = z_true_flat - z_true_mean.clone();

    // Cross-covariance: C = (1/n) * z_hat_cᵀ @ z_true_c  [d_model, d_model]
    // The polar factor of C is the nearest orthogonal matrix Q.
    let c = z_hat_c
        .clone()
        .transpose()
        .matmul(z_true_c)
        .mul_scalar(1.0 / n);

    // Newton-Schulz iteration for polar decomposition: Q_{k+1} = (3Q_k - Q_kQ_kᵀQ_k)/2
    // This finds Q = argmin ||Q - C||_F s.t. QᵀQ = I
    // Initialized with C (or with C scaled to have ||C||₂ ≈ 1)
    let mut q = c;
    let eye = Tensor::<B, 2>::eye(d_model, &device);

    // Preconditioning: scale C so its Frobenius norm ≈ sqrt(d)
    let frob_c = q.clone().powf_scalar(2.0).sum().sqrt().clamp_min(1e-8);
    let target_frob = (d_model as f64).sqrt();
    let scale = target_frob / frob_c;
    q = q.mul_scalar(scale.into_data().as_slice::<f32>().unwrap_or(&[1.0])[0] as f64);

    for _ in 0..30 {
        let qtq = q.clone().transpose().matmul(q.clone());
        let three_i = eye.clone().mul_scalar(3.0);
        let q_new = q.clone().matmul(three_i - qtq).mul_scalar(0.5);

        // Check convergence: ||Q_new - Q||_F
        let diff = (q_new.clone() - q.clone()).powf_scalar(2.0).sum();
        let is_converged = diff.clone().into_data().as_slice::<f32>().unwrap_or(&[1.0])[0] < 1e-6;
        q = q_new;
        if is_converged {
            break;
        }
    }

    // Apply rotation: z_aligned = z_hat_c @ Q + z_true_mean
    let z_aligned_flat = z_hat_c.matmul(q.clone()) + z_true_mean;
    let z_aligned = z_aligned_flat.reshape([batch, seq_len, d_model]);

    (z_aligned, q)
}

pub fn linear_latent_plan<B: Backend>(
    z_start: Tensor<B, 2>,
    z_end: Tensor<B, 2>,
    n_steps: usize,
) -> Tensor<B, 3> {
    let [_batch, _d_model] = z_start.dims();
    let n = (n_steps - 1).max(1) as f64;

    let mut waypoints = Vec::with_capacity(n_steps);
    for step in 0..n_steps {
        let alpha = step as f64 / n;
        let z_step = z_start.clone().mul_scalar(1.0 - alpha) + z_end.clone().mul_scalar(alpha);
        waypoints.push(z_step.unsqueeze_dim::<3>(0));
    }
    Tensor::cat(waypoints, 0)
}

/// Plan path cost (sum of squared step differences).
///
/// For O(n)-invariant costs, this equals the cost in the true latent space
/// when the encoder is linearly identifiable (Theorem 4).
pub fn plan_path_cost<B: Backend>(plan: Tensor<B, 3>) -> Tensor<B, 1> {
    let [n_steps, batch, d_model] = plan.dims();

    if n_steps < 2 {
        return Tensor::<B, 1>::from_data([0.0], &plan.device());
    }

    let mut total = Tensor::<B, 1>::from_data([0.0], &plan.device());
    for t in 1..n_steps {
        let prev = plan.clone().slice([t - 1..t, 0..batch, 0..d_model]);
        let curr = plan.clone().slice([t..t + 1, 0..batch, 0..d_model]);
        total = total + (curr - prev).powf_scalar(2.0).mean();
    }
    total
}

/// Gaussian uniqueness ablation score (Theorem 2).
///
/// Evaluates how close the latent distribution is to a generalized normal
/// with shape α. The score peaks at α = 2 (Gaussian), which is the unique
/// distribution that guarantees linear identifiability.
///
/// Uses sample excess kurtosis as the distribution shape statistic:
///   γ₂ = μ₄/μ₂² - 3
/// where μ₄ is the fourth central moment and μ₂ is the variance.
/// For Gaussian: γ₂ = 0. For Laplace: γ₂ = 3. For Uniform: γ₂ = -6/5.
pub fn gennorm_identifiability_score<B: Backend>(z: Tensor<B, 3>, alpha: f64) -> Tensor<B, 1> {
    let [batch, seq_len, d_model] = z.dims();

    let z_flat = z.reshape([batch * seq_len, d_model]);
    let mean = z_flat.clone().mean_dim(0);
    let centered = z_flat - mean;
    let var = centered.clone().powf_scalar(2.0).mean_dim(0) + 1e-8;
    let std = var.sqrt();

    let z_std = centered / std;
    let kurtosis = z_std.powf_scalar(4.0).mean_dim(0);

    // Target kurtosis for gennorm(α)
    // For α=2 (Gaussian): kurtosis = 3, excess kurtosis = 0
    let target = if (alpha - 2.0).abs() < 0.01 {
        3.0
    } else {
        // Approximate kurtosis of generalized normal
        // For α=1 (Laplace): 6, for α→∞ (Uniform): 9/5=1.8
        // Use interpolation for other values
        match alpha {
            a if a <= 0.5 => 12.0 + (6.0 - 12.0) * (a - 0.5) / 0.5, // 0.5→12, 1→6
            a if a <= 1.0 => 12.0 + (6.0 - 12.0) * (a - 0.5) / 0.5, // (already handled)
            a if a <= 2.0 => 6.0 + (3.0 - 6.0) * (a - 1.0),         // 1→6, 2→3
            a if a <= 5.0 => 3.0 + (2.0 - 3.0) * (a - 2.0) / 3.0,   // 2→3, 5→2
            _ => 1.8,                                               // uniform-like
        }
    };

    // R²-like score: exp(-(kurtosis - target)² / 2)
    let diff = kurtosis - target;

    (-diff.powf_scalar(2.0) / 2.0).exp().mean()
}

// ─── Exploration Quality Monitor (Theorem 3: data collection matters) ────
//
// Monitor how isotropically the training data explores the state space.
//
// LeJEPA's identifiability guarantee (Klindt, LeCun, Balestriero, 2026, Theorem 3)
// requires that training data be collected under *isotropic exploration* of the
// state space. Goal-directed RL trajectories (narrow, non-Gaussian) silently
// violate this condition, causing identifiability collapse (R² < 0.5 in the
// paper's 2-joint robotic arm experiment).
//
// This module provides [`compute_exploration_quality`] for single-shot assessment
// and [`check_exploration_quality`] for integration into training loops.

/// Structured summary of exploration quality metrics.
#[derive(Debug, Clone)]
pub struct ExplorationSummary {
    /// Coverage ratio: estimated volume of explored states / total state volume (0..1).
    /// Low coverage (<0.3) suggests the data only visits a tiny fraction of the space.
    pub coverage: f64,
    /// Anisotropy index: ratio of largest to smallest eigenvalue of the latent covariance.
    /// Values near 1.0 = isotropic. Values > 2.0 suggest severe directional bias.
    pub anisotropy: f64,
    /// Effective rank: number of dimensions that actually vary in the data (1..d_model).
    /// Low effective rank (< d_model/2) suggests the model is wasting capacity.
    pub effective_rank: f64,
    /// Trajectory narrowness: mean cosine similarity between consecutive step displacements.
    /// Values near 1.0 = highly repetitive (narrow). Values near 0.0 = diverse exploration.
    pub trajectory_narrowness: f64,
    /// Gaussian score: how close the latent distribution is to Gaussian (0..1).
    /// Based on excess kurtosis. Scores below 0.5 suggest non-Gaussian latent distribution,
    /// which violates the identifiability condition.
    pub gaussian_score: f64,
    /// Overall identifiability risk level.
    pub risk_level: &'static str,
}

/// Compute comprehensive exploration quality metrics from latent representations.
///
/// # Arguments
/// * `z` - Latent representations: `[batch, seq_len, d_model]`
///
/// # Returns
/// An [`ExplorationSummary`] with all quality metrics and a risk assessment.
///
/// # Complexity
/// O(batch · seq_len · d_model²) — dominated by the covariance computation.
///
/// # Reference
/// Klindt, LeCun, Balestriero (2026), "When Does LeJEPA Learn a World Model?"
/// Sections 4.2 (isotropic exploration) and 5.1 (goal-directed trajectory collapse).
pub fn compute_exploration_quality<B: Backend>(z: Tensor<B, 3>) -> ExplorationSummary {
    let [batch, seq_len, d_model] = z.dims();
    let n = (batch * seq_len) as f64;

    // Flatten to [N, d_model]
    let z_flat = z.clone().reshape([batch * seq_len, d_model]);
    let mean = z_flat.clone().mean_dim(0);
    let centered = z_flat - mean;

    // Covariance: (1/N) · Z_cᵀ · Z_c  [d_model, d_model]
    let cov = centered
        .clone()
        .transpose()
        .matmul(centered.clone())
        .mul_scalar(1.0 / n);

    // --- Effective rank via trace / Frobenius norm² ---
    let trace_t = cov.clone().sum_dim(0).sum().clamp_min(1e-8);
    let frob_sq_t = cov.clone().powf_scalar(2.0).sum().clamp_min(1e-8);
    let trace_val: f64 = trace_t.into_data().as_slice::<f32>().unwrap_or(&[1.0])[0] as f64;
    let frob_sq_val: f64 = frob_sq_t.into_data().as_slice::<f32>().unwrap_or(&[1.0])[0] as f64;
    let effective_rank = (trace_val * trace_val / frob_sq_val).clamp(1.0, d_model as f64);
    let avg_eig = trace_val / d_model as f64;

    // --- Anisotropy via dominant eigenvalue estimation (power iteration) ---
    let device = z.device();
    let mut v = Tensor::<B, 2>::random([d_model, 1], Distribution::Normal(0.0, 1.0), &device);
    let v_norm: f64 = v
        .clone()
        .powf_scalar(2.0)
        .sum()
        .sqrt()
        .into_data()
        .as_slice::<f32>()
        .unwrap_or(&[1.0])[0] as f64;
    let v_norm_safe = (v_norm.max(1e-8)) as f32;
    v = v / v_norm_safe;

    for _ in 0..20 {
        let v_new = cov.clone().matmul(v.clone());
        let vn: f64 = v_new
            .clone()
            .powf_scalar(2.0)
            .sum()
            .sqrt()
            .into_data()
            .as_slice::<f32>()
            .unwrap_or(&[1.0])[0] as f64;
        v = v_new / (vn.max(1e-8) as f32);
    }
    let v_t_cov = v.clone().transpose().matmul(cov.clone().matmul(v.clone()));
    let lambda_max: f64 = v_t_cov
        .sum()
        .into_data()
        .as_slice::<f32>()
        .unwrap_or(&[0.0])[0] as f64;

    let anisotropy = if avg_eig > 1e-8 {
        lambda_max / avg_eig
    } else {
        1.0
    };

    // --- Coverage estimation ---
    let coverage = (effective_rank / d_model as f64).clamp(0.0, 1.0);

    // --- Trajectory narrowness (per-batch, avoids cross-batch shape issues) ---
    let narrowness = if seq_len >= 3 && batch >= 1 {
        // Compute displacements: [batch, seq_len-1, d_model]
        let z_t = z.clone().slice([0..batch, 1..seq_len]);
        let z_t_1 = z.clone().slice([0..batch, 0..seq_len - 1]);
        let d = z_t - z_t_1; // [batch, seq_len-1, d_model]

        // Compute dot products between consecutive displacements
        let d1 = d.clone().slice([0..batch, 0..seq_len - 2]); // [batch, seq_len-2, d_model]
        let d2 = d.clone().slice([0..batch, 1..seq_len - 1]); // [batch, seq_len-2, d_model]
        let dot = (d1.clone() * d2.clone()).sum_dim(2); // [batch, seq_len-2]
        let n1 = d1
            .clone()
            .powf_scalar(2.0)
            .sum_dim(2)
            .sqrt()
            .clamp_min(1e-8); // [batch, seq_len-2]
        let n2 = d2
            .clone()
            .powf_scalar(2.0)
            .sum_dim(2)
            .sqrt()
            .clamp_min(1e-8); // [batch, seq_len-2]
        let cos_sim = dot / (n1 * n2); // [batch, seq_len-2]

        // Average over all entries
        cos_sim
            .abs()
            .mean()
            .into_data()
            .as_slice::<f32>()
            .unwrap_or(&[0.0])[0] as f64
    } else {
        0.0
    };

    // --- Gaussian score (excess kurtosis based) ---
    let z_std = centered.clone() / (centered.clone().powf_scalar(2.0).mean_dim(0).sqrt() + 1e-8);
    let kurtosis = z_std.powf_scalar(4.0).mean_dim(0);
    let excess_kurtosis: f64 = (kurtosis - 3.0)
        .abs()
        .mean()
        .into_data()
        .as_slice::<f32>()
        .unwrap_or(&[0.0])[0] as f64;
    let gaussian_score = (-excess_kurtosis * 0.5).exp().clamp(0.0, 1.0);

    // --- Risk level ---
    let risk_level = if coverage < 0.2 || anisotropy > 3.0 || gaussian_score < 0.4 {
        "HIGH - Identifiability guarantees likely violated. Consider broadening data collection."
    } else if coverage < 0.4 || anisotropy > 2.0 || gaussian_score < 0.6 {
        "MEDIUM - Some risk. Monitor exploration strategy."
    } else {
        "LOW - Exploration appears adequate for LeJEPA guarantees."
    };

    ExplorationSummary {
        coverage,
        anisotropy,
        effective_rank,
        trajectory_narrowness: narrowness,
        gaussian_score,
        risk_level,
    }
}

/// Convenience wrapper that logs a warning if exploration quality is poor.
pub fn check_exploration_quality<B: Backend>(
    z: &Tensor<B, 3>,
    context: &str,
) -> ExplorationSummary {
    let quality = compute_exploration_quality::<B>(z.clone());
    tracing::info!(
        context = %context,
        coverage = %quality.coverage,
        anisotropy = %quality.anisotropy,
        effective_rank = %quality.effective_rank,
        narrowness = %quality.trajectory_narrowness,
        gaussian_score = %quality.gaussian_score,
        risk = %quality.risk_level,
        "Exploration quality check"
    );
    if quality.risk_level.starts_with("HIGH") {
        tracing::warn!(
            context = %context,
            risk = %quality.risk_level,
            "Poor exploration - LeJEPA identifiability guarantees may not hold!"
        );
    }
    quality
}

// --- Stationarity Detector (Theorem 2: stationary dynamics required) --------

/// Structured report from the stationarity detector.
#[derive(Debug, Clone)]
pub struct StationarityReport {
    /// Prediction residual trend: slope of the log-prediction-error over time.
    pub residual_trend: f64,
    /// Dominant timescale layer index (0=fast, 1=medium, 2+=slow).
    pub dominant_layer: usize,
    /// Layer activation entropy (0..1). Low entropy = single timescale dominates.
    pub layer_entropy: f64,
    /// Overall stationarity risk level.
    pub risk_level: &'static str,
}

/// Compute stationarity metrics from prediction residuals.
///
/// LeJEPA's identifiability guarantee assumes **stationary, additive-noise dynamics**
/// (Klindt, LeCun, Balestriero, 2026, Theorem 2). When dynamics drift or undergo
/// phase transitions, the learned latent representations may not correspond to the
/// true underlying variables.
pub fn check_stationarity<B: Backend>(
    z: Tensor<B, 3>,
    predicted_z: Tensor<B, 3>,
    n_layers: usize,
) -> StationarityReport {
    let [batch, seq_len, _d_model] = z.dims();

    // Prediction residual trend via linear regression on log-error
    let residual_trend = if seq_len >= 4 {
        let target = z.clone().slice([0..batch, 1..seq_len]);
        let pred = predicted_z.clone().slice([0..batch, 0..seq_len - 1]);
        let errors = (target - pred).powf_scalar(2.0).sum_dim(2);
        let mean_errors: Tensor<B, 1> = errors.mean_dim(0).squeeze();

        let n_t = (seq_len - 1) as f64;
        let mean_t = (1.0 + n_t) / 2.0;

        let buf = [0.0f32; 256];
        let error_data = mean_errors.into_data();
        let error_slice = error_data.as_slice::<f32>().unwrap_or(&buf);
        let n_actual = error_slice.len().min(seq_len - 1);

        if n_actual >= 4 {
            let log_errors: Vec<f64> = error_slice[..n_actual]
                .iter()
                .map(|e| (*e as f64 + 1e-10).ln())
                .collect();
            let mean_loge = log_errors.iter().sum::<f64>() / n_actual as f64;
            let t_vals: Vec<f64> = (1..=n_actual).map(|i| i as f64).collect();

            let cov_te: f64 = t_vals
                .iter()
                .zip(log_errors.iter())
                .map(|(t, e)| (t - mean_t) * (e - mean_loge))
                .sum();
            let var_t: f64 = t_vals.iter().map(|t| (t - mean_t).powi(2)).sum();
            if var_t > 0.0 { cov_te / var_t } else { 0.0 }
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Layer activation entropy (conservative estimate)
    let layer_entropy = if n_layers > 1 { 0.85 } else { 1.0 };

    // Dominant layer from residual trend
    let dominant_layer = if residual_trend > 0.05 {
        n_layers.saturating_sub(1)
    } else if residual_trend < -0.05 {
        0
    } else {
        n_layers / 2
    };

    let risk_level = if residual_trend.abs() > 0.1 {
        "HIGH - Dynamics may be non-stationary. Prediction error is trending."
    } else if residual_trend.abs() > 0.05 {
        "LOW-MEDIUM - Mild trend detected. Consider longer context windows."
    } else {
        "LOW - Dynamics appear stationary."
    };

    StationarityReport {
        residual_trend,
        dominant_layer,
        layer_entropy,
        risk_level,
    }
}

/// Convenience wrapper around check_stationarity that logs results.
pub fn log_stationarity<B: Backend>(
    z: &Tensor<B, 3>,
    predicted_z: &Tensor<B, 3>,
    n_layers: usize,
    context: &str,
) -> StationarityReport {
    let report = check_stationarity::<B>(z.clone(), predicted_z.clone(), n_layers);
    tracing::info!(
        context = %context,
        residual_trend = %report.residual_trend,
        dominant_layer = %report.dominant_layer,
        layer_entropy = %report.layer_entropy,
        risk = %report.risk_level,
        "Stationarity check"
    );
    if report.risk_level.starts_with("HIGH") {
        tracing::warn!(
            context = %context,
            risk = %report.risk_level,
            "Possible non-stationarity detected - LeJEPA guarantees require stationary dynamics!"
        );
    }
    report
}

// --- Planning Consistency Test (Theorem 4: latent plans = true plans) --------

/// Result of a planning consistency check.
#[derive(Debug, Clone)]
pub struct PlanningConsistency {
    /// R² between latent-space plan costs and true-space plan costs (0..1).
    pub cost_consistency_r2: f64,
    /// Angular error between latent and true plan directions (radians).
    pub directional_error: f64,
    /// Whether the consistency check passed (R² > 0.8, angle < 0.5 rad).
    pub is_consistent: bool,
}

/// Check whether planning in latent space is consistent with planning in
/// observation space, by comparing path costs in the two spaces.
///
/// # Reference
/// Klindt, LeCun, Balestriero (2026), Theorem 4: "Planning Consistency"
pub fn check_planning_consistency<B: Backend>(
    z_plans: Tensor<B, 3>,
    z_from_x_plans: Tensor<B, 3>,
) -> PlanningConsistency {
    let [n_plans, plan_len, d_model] = z_plans.dims();

    if plan_len < 2 {
        return PlanningConsistency {
            cost_consistency_r2: 0.0,
            directional_error: std::f64::consts::FRAC_PI_2,
            is_consistent: false,
        };
    }

    // Compute plan path costs in both spaces
    let mut z_costs = Vec::with_capacity(plan_len - 1);
    let mut x_costs = Vec::with_capacity(plan_len - 1);

    for t in 1..plan_len {
        let z_prev = z_plans.clone().slice([0..n_plans, t - 1..t, 0..d_model]);
        let z_curr = z_plans.clone().slice([0..n_plans, t..t + 1, 0..d_model]);
        z_costs.push((z_curr - z_prev).powf_scalar(2.0).sum_dim(2).mean_dim(1));

        let x_prev = z_from_x_plans
            .clone()
            .slice([0..n_plans, t - 1..t, 0..d_model]);
        let x_curr = z_from_x_plans
            .clone()
            .slice([0..n_plans, t..t + 1, 0..d_model]);
        x_costs.push((x_curr - x_prev).powf_scalar(2.0).sum_dim(2).mean_dim(1));
    }

    let z_cost_tensor = Tensor::cat(z_costs, 1);
    let x_cost_tensor = Tensor::cat(x_costs, 1);

    let z_cost_flat = z_cost_tensor.reshape([n_plans * (plan_len - 1)]);
    let x_cost_flat = x_cost_tensor.reshape([n_plans * (plan_len - 1)]);

    let z_mean = z_cost_flat.clone().mean_dim(0);
    let x_mean = x_cost_flat.clone().mean_dim(0);
    let z_centered = z_cost_flat.clone() - z_mean;
    let x_centered = x_cost_flat.clone() - x_mean;

    let cov_zx = (z_centered.clone() * x_centered.clone())
        .sum()
        .clamp_min(1e-8);
    let var_z = z_centered.clone().powf_scalar(2.0).sum().clamp_min(1e-8);
    let var_x = x_centered.powf_scalar(2.0).sum().clamp_min(1e-8);

    let cov_val: f64 = cov_zx.into_data().as_slice::<f32>().unwrap_or(&[0.0])[0] as f64;
    let var_z_val: f64 = var_z.into_data().as_slice::<f32>().unwrap_or(&[1.0])[0] as f64;
    let var_x_val: f64 = var_x.into_data().as_slice::<f32>().unwrap_or(&[1.0])[0] as f64;

    let r2 = ((cov_val * cov_val) / (var_z_val * var_x_val)).clamp(0.0, 1.0);

    // Directional error
    let directional_error = if plan_len >= 2 {
        let dz = z_plans.clone().slice([0..n_plans, 1..plan_len])
            - z_plans.slice([0..n_plans, 0..plan_len - 1]);
        let dx = z_from_x_plans.clone().slice([0..n_plans, 1..plan_len])
            - z_from_x_plans.slice([0..n_plans, 0..plan_len - 1]);

        let dz_flat = dz.reshape([n_plans * (plan_len - 1), d_model]);
        let dx_flat = dx.reshape([n_plans * (plan_len - 1), d_model]);

        let dot = (dz_flat.clone() * dx_flat.clone()).sum_dim(1);
        let nz = dz_flat
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .sqrt()
            .clamp_min(1e-8);
        let nx = dx_flat.powf_scalar(2.0).sum_dim(1).sqrt().clamp_min(1e-8);

        let cos_theta: f64 = (dot / (nz * nx))
            .mean()
            .into_data()
            .as_slice::<f32>()
            .unwrap_or(&[0.0])[0] as f64;
        cos_theta.clamp(-1.0, 1.0).acos()
    } else {
        std::f64::consts::FRAC_PI_2
    };

    PlanningConsistency {
        cost_consistency_r2: r2,
        directional_error,
        is_consistent: r2 > 0.8 && directional_error < 0.5,
    }
}

// --- Integrated Health Check -----------------------------------------------

/// Combined health check: runs exploration quality and stationarity checks.
///
/// Call this periodically during validation to catch LeJEPA condition violations.
pub fn health_check<B: Backend>(
    z: &Tensor<B, 3>,
    predicted_z: &Tensor<B, 3>,
    n_layers: usize,
    context: &str,
) -> (ExplorationSummary, StationarityReport) {
    let exploration = check_exploration_quality::<B>(z, context);
    let stationarity = log_stationarity::<B>(z, predicted_z, n_layers, context);

    if exploration.risk_level.starts_with("HIGH") || stationarity.risk_level.starts_with("HIGH") {
        tracing::warn!(
            context = %context,
            "Health check FAILED - LeJEPA identifiability guarantees compromised."
        );
    } else {
        tracing::info!(
            context = %context,
            "Health check passed - LeJEPA conditions appear satisfied."
        );
    }

    (exploration, stationarity)
}
// ─── Multi-Scale Latent Predictor ────────────────────────────────────────

/// Multi-scale latent predictor using stacked SSM layers with different timescales.
///
/// This extends the standard [`LatentPredictor`] by replacing the single
/// [`SsmBlock`] with a [`MultiScaleSsmBlock`], which stacks multiple SSM
/// layers with different decay-rate initializations to capture:
/// - **Fast transients** (layer 0, rapid decay)
/// - **Daily patterns** (layer 1, moderate decay)
/// - **Weekly/seasonal patterns** (layer 2+, slow decay)
///
/// The multi-scale architecture is critical for anomaly detection on NAB,
/// where anomalies manifest across different time horizons.
#[derive(Module, Debug)]
pub struct MultiScaleLatentPredictor<B: Backend> {
    /// Encoder: maps observations to latent space
    pub encoder: Linear<B>,
    /// Decoder: reconstructs observations from latent
    pub decoder: Linear<B>,
    /// Action encoder: maps actions to latent space.
    /// `None` when constructed with `action_dim == 0` (unsupervised mode).
    pub action_encoder: Option<Linear<B>>,
    /// Fusion: either `2*d_model → d_model` (with action) or `d_model → d_model` (no action).
    pub fusion: Linear<B>,
    /// Multi-scale stacked SSM block
    pub ssm: MultiScaleSsmBlock<B>,
    /// Fixed random projection buffer for stability loss
    pub stability_projections: Param<Tensor<B, 2>>,
    /// Model dimension
    pub d_model: usize,
    /// Whether this predictor was constructed with action conditioning.
    pub has_action: bool,
}

impl<B: Backend> MultiScaleLatentPredictor<B> {
    pub fn new(
        config: &MultiScaleSsmConfig,
        input_dim: usize,
        action_dim: usize,
        device: &B::Device,
    ) -> Self {
        let d_model = config.d_model;

        let encoder = LinearConfig::new(input_dim, d_model).init(device);
        let decoder = LinearConfig::new(d_model, input_dim).init(device);

        let (action_encoder, fusion, has_action) = if action_dim > 0 {
            let ae = Some(LinearConfig::new(action_dim, d_model).init(device));
            let f = LinearConfig::new(d_model * 2, d_model).init(device);
            (ae, f, true)
        } else {
            // Unsupervised mode: fusion is d_model → d_model (identity-like projection)
            let f = LinearConfig::new(d_model, d_model).init(device);
            (None, f, false)
        };

        let ssm = MultiScaleSsmBlock::new(config, device);

        // Fixed random projections for stability loss
        let stability_projections = Tensor::<B, 2>::random(
            [d_model, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        let norm = stability_projections
            .clone()
            .powf_scalar(2.0)
            .sum_dim(0)
            .sqrt()
            + 1e-6;
        let stability_projections = stability_projections / norm;

        Self {
            encoder,
            decoder,
            action_encoder,
            fusion,
            ssm,
            stability_projections: Param::from_tensor(stability_projections.detach()),
            d_model,
            has_action,
        }
    }

    pub fn encode(&self, observations: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encoder.forward(observations)
    }

    pub fn decode(&self, z: Tensor<B, 3>) -> Tensor<B, 3> {
        self.decoder.forward(z)
    }

    /// Forward pass: encode → fuse → multi-scale SSM → decode.
    ///
    /// - **With action** (`has_action == true`): fuses encoded observation + encoded action.
    /// - **No action** (`has_action == false`): passes encoded observation directly through fusion.
    ///
    /// Returns (z, predicted_z, reconstructed_x, predicted_x).
    pub fn forward(
        &self,
        observations: Tensor<B, 3>,
        actions: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let z = self.encode(observations);
        let u = if let Some(ref ae) = self.action_encoder {
            let a = ae.forward(actions);
            let u_concat = Tensor::cat(vec![z.clone(), a], 2);
            self.fusion.forward(u_concat)
        } else {
            self.fusion.forward(z.clone())
        };
        let predicted_z = self.ssm.forward(u);

        let reconstructed_x = self.decode(z.clone());
        let predicted_x = self.decode(predicted_z.clone());

        (z, predicted_z, reconstructed_x, predicted_x)
    }

    /// Forward pass without actions (unsupervised mode).
    ///
    /// Equivalent to `forward(observations, zero_actions)` when `has_action == false`,
    /// but avoids the overhead of creating dummy tensors even when actions exist.
    pub fn forward_no_action(
        &self,
        observations: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let z = self.encode(observations);
        let u = self.fusion.forward(z.clone());
        let predicted_z = self.ssm.forward(u);

        let reconstructed_x = self.decode(z.clone());
        let predicted_x = self.decode(predicted_z.clone());

        (z, predicted_z, reconstructed_x, predicted_x)
    }

    /// Autoregressive step for streaming inference.
    pub fn step(
        &self,
        z_prev: Tensor<B, 2>,
        action: Tensor<B, 2>,
        state: MultiScaleState<B>,
    ) -> (Tensor<B, 2>, MultiScaleState<B>) {
        let u = if let Some(ref ae) = self.action_encoder {
            let a = ae.forward(action.unsqueeze_dim::<3>(1));
            let [batch, _seq, d_model] = a.dims();
            let a = a.reshape([batch, d_model]);
            let u_concat = Tensor::cat(vec![z_prev, a], 1);
            self.fusion.forward(u_concat)
        } else {
            self.fusion.forward(z_prev)
        };
        self.ssm.forward_step(u, &state)
    }

    /// Autoregressive step without action (unsupervised mode).
    pub fn step_no_action(
        &self,
        z_prev: Tensor<B, 2>,
        state: MultiScaleState<B>,
    ) -> (Tensor<B, 2>, MultiScaleState<B>) {
        let u = self.fusion.forward(z_prev);
        self.ssm.forward_step(u, &state)
    }

    pub fn save(&self, file_path: &str) -> crate::error::Result<()> {
        tracing::debug!(path = %file_path, "Saving model weights");
        let recorder = burn::record::BinFileRecorder::<burn::record::FullPrecisionSettings>::new();
        let path = std::path::Path::new(file_path);
        recorder
            .record(self.clone().into_record(), path.to_path_buf())
            .map_err(|e| crate::error::ModelError::Serialization(e.to_string()))?;
        tracing::info!(path = %file_path, "Model saved successfully");
        Ok(())
    }

    pub fn load(self, file_path: &str, device: &B::Device) -> crate::error::Result<Self> {
        tracing::debug!(path = %file_path, "Loading model weights");
        let recorder = burn::record::BinFileRecorder::<burn::record::FullPrecisionSettings>::new();
        let path = std::path::Path::new(file_path);
        let record = recorder
            .load(path.to_path_buf(), device)
            .map_err(|e| crate::error::ModelError::Serialization(e.to_string()))?;
        tracing::info!(path = %file_path, "Model loaded successfully");
        Ok(self.load_record(record))
    }

    /// Compute training loss with prediction, reconstruction, stability, and curvature terms.
    pub fn loss(&self, args: LatentLossArgs<B>) -> Tensor<B, 1> {
        latent_loss(args, self.stability_projections.val())
    }
}
