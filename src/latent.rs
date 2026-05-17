use crate::ssm::{MultiScaleSsmBlock, MultiScaleSsmConfig, MultiScaleState, SsmBlock, SsmConfig};
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

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

/// Temporal Straightening loss: minimizes the second-order finite difference
/// of the latent trajectory to encourage locally straight paths.
///
/// Based on Wang et al. (2026) "Temporal Straightening for Latent Planning",
/// this loss encourages the latent representation to evolve smoothly over time,
/// which improves the accuracy of long-horizon planning.
///
/// The second-order finite difference (discrete acceleration) is:
/// ```text
/// a_t = z_t - 2·z_{t-1} + z_{t-2}
/// ```
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

    // Direct second-order finite difference: a_t = z_t - 2z_{t-1} + z_{t-2}
    // This reduces the number of intermediate tensors and slice operations.
    let z_t = z.clone().slice([0..batch, 2..seq_len]);
    let z_t_1 = z.clone().slice([0..batch, 1..seq_len - 1]);
    let z_t_2 = z.slice([0..batch, 0..seq_len - 2]);

    let acceleration = z_t - z_t_1.mul_scalar(2.0) + z_t_2;

    acceleration.powf_scalar(2.0).mean().unsqueeze()
}

/// Gaussian-based O(T) stability loss to prevent representation collapse in JEPA.
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
    /// Action encoder: maps actions to latent space (can receive zero actions for unsupervised mode)
    pub action_encoder: Linear<B>,
    /// Fusion: concatenates latent observation + latent action
    pub fusion: Linear<B>,
    /// Multi-scale stacked SSM block
    pub ssm: MultiScaleSsmBlock<B>,
    /// Fixed random projection buffer for stability loss
    pub stability_projections: Param<Tensor<B, 2>>,
    /// Model dimension
    pub d_model: usize,
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
        let action_encoder = LinearConfig::new(action_dim, d_model).init(device);
        let fusion = LinearConfig::new(d_model * 2, d_model).init(device);
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
        }
    }

    pub fn encode(&self, observations: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encoder.forward(observations)
    }

    pub fn decode(&self, z: Tensor<B, 3>) -> Tensor<B, 3> {
        self.decoder.forward(z)
    }

    /// Forward pass: encode → fuse with actions → multi-scale SSM → decode.
    ///
    /// Returns (z, predicted_z, reconstructed_x, predicted_x).
    pub fn forward(
        &self,
        observations: Tensor<B, 3>,
        actions: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let z = self.encode(observations);
        let a = self.action_encoder.forward(actions);
        let u_concat = Tensor::cat(vec![z.clone(), a], 2);
        let u = self.fusion.forward(u_concat);
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
        let a = self.action_encoder.forward(action.unsqueeze_dim::<3>(1));
        let [batch, _seq, d_model] = a.dims();
        let a = a.reshape([batch, d_model]);

        let u_concat = Tensor::cat(vec![z_prev, a], 1);
        let u = self.fusion.forward(u_concat);

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
