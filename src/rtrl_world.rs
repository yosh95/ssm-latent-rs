/// RTRL World Model — Zero-BPTT SSM with Forward-Forward Encoder/Decoder.
///
/// This module provides a reusable world model combining:
/// - **Forward-Forward encoder**: observation → latent (per-layer local goodness, no BP)
/// - **Multi-scale SSM**: latent dynamics via 1-step truncated RTRL (no BPTT)
/// - **Forward-Forward decoder**: latent → observation (per-layer local goodness, no BP)
/// - **Action encoder + Fusion**: small BP-trained components (few params)
///
/// # Key Properties
///
/// - **Zero BPTT**: All recurrent state is detached between steps. Gradients only
///   flow within a single timestep through `f(h_detach, x_t, θ)`.
/// - **Zero cross-component BP**: Encoder output is detached before SSM input.
///   SSM prediction loss never backpropagates into the encoder.
/// - **Zero inter-layer BP**: Each encoder/decoder layer trains independently
///   via Forward-Forward goodness.
/// - **O(1) memory**: No computation graph grows with sequence length.
///
/// # Architecture
///
/// ```text
/// obs(t) → [FF Encoder] → z(t) ──[detach]──┐
///                                            ├→ [Fusion] → u(t) → [SSM] → ẑ(t+1)
/// act(t) → [Action Enc] ────────────────────┘         ↑ 1-step gradient only
///                                                      (h_{t-1} detached)
///
/// ẑ(t+1) → [FF Decoder] → ôbs(t+1)
///           ↑ FF goodness + local MSE
/// ```
///
/// # Usage
///
/// This is the training loop pattern (per-step backward, zero BPTT):
///
/// ```ignore
/// let mut ssm_state = model.initial_ssm_state(batch_size, device);
///
/// for t in 0..seq_len {
///     let obs_t = /* current observation */;
///     let action_t = /* current action */;
///     let neg_obs = /* random negative observation for FF */;
///
///     // Forward-Forward encoder pass
///     let (z_t, enc_losses) = model.encode_with_loss(obs_t, neg_obs, threshold);
///
///     // Forward-Forward decoder + reconstruction
///     let (reconstructed, dec_losses) =
///         model.decode_with_loss(z_t.clone().detach(), obs_t, threshold);
///
///     // RTRL SSM step (h_{t-1} detached internally)
///     let (pred_z, new_state) =
///         model.ssm_step_detached(z_t.detach(), action_t, &ssm_state);
///
///     // SSM prediction loss (target: next observation's latent)
///     let ssm_loss = if t + 1 < seq_len {
///         let next_z = model.encode(next_obs).detach();
///         Some(MSE(pred_z, next_z))
///     } else { None };
///
///     // Combine and backward (all gradients local)
///     let loss = enc_losses + dec_losses + recon_loss + ssm_loss;
///     loss.backward();
///     optimizer.step();
///
///     ssm_state = new_state;
/// }
/// ```
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use crate::ff_model::FfEncoder;
use crate::rtrl::ssm_step_detached;
use crate::ssm::{MultiScaleSsmBlock, MultiScaleSsmConfig, MultiScaleState, SsmConfig};

/// RTRL World Model: FF Encoder + Multi-Scale SSM + FF Decoder.
///
/// All dimensions are configurable via `RtrlConfig`. The encoder/decoder
/// use Forward-Forward training (per-layer local goodness), and the SSM
/// uses 1-step truncated RTRL (no BPTT).
#[derive(Module, Debug)]
pub struct RtrlWorldModel<B: Backend> {
    /// Forward-Forward encoder: observation → latent
    pub encoder: FfEncoder<B>,
    /// Forward-Forward decoder: latent → observation
    pub decoder: FfEncoder<B>,
    /// Action encoder: action → action_latent
    pub action_encoder: Linear<B>,
    /// Fusion: [z ⊕ action_latent] → d_model
    pub fusion: Linear<B>,
    /// Multi-scale SSM for latent dynamics (trained via 1-step RTRL)
    pub ssm: MultiScaleSsmBlock<B>,
    /// Fixed random projection buffer for stability loss
    pub stability_projections: Param<Tensor<B, 2>>,
    /// Model dimension
    pub d_model: usize,
    /// Number of SSM layers
    pub n_ssm_layers: usize,
    /// Observation input dimension
    pub input_dim: usize,
    /// Action dimension
    pub action_dim: usize,
}

/// Configuration for the RTRL World Model.
pub struct RtrlConfig {
    /// SSM base configuration
    pub ssm: SsmConfig,
    /// Number of multi-scale SSM layers
    pub n_ssm_layers: usize,
    /// Observation input dimension
    pub input_dim: usize,
    /// Action dimension
    pub action_dim: usize,
    /// Encoder layer sizes (first = input_dim, last = d_model)
    pub encoder_sizes: Vec<usize>,
    /// Decoder layer sizes (first = d_model, last = input_dim)
    pub decoder_sizes: Vec<usize>,
}

/// Arguments for computing per-step FF + RTRL loss.
pub struct RtrlStepLossArgs {
    /// Threshold for FF goodness loss
    pub ff_threshold: f64,
    /// Weight for FF encoder loss
    pub ff_enc_weight: f64,
    /// Weight for FF decoder loss
    pub ff_dec_weight: f64,
    /// Weight for decoder reconstruction MSE
    pub recon_weight: f64,
    /// Weight for SSM latent prediction MSE
    pub ssm_weight: f64,
}

/// Output of a single RTRL encode+decode step.
///
/// Contains all intermediate tensors needed for loss computation.
pub struct RtrlStepOutput<B: Backend> {
    /// Encoded latent z(t)
    pub z: Tensor<B, 2>,
    /// Per-layer encoder activations (for FF loss)
    pub pos_enc_acts: Vec<Tensor<B, 2>>,
    /// Per-layer encoder activations on negative data (for FF loss)
    pub neg_enc_acts: Vec<Tensor<B, 2>>,
    /// Reconstructed observation ôbs(t)
    pub reconstructed: Tensor<B, 2>,
    /// Per-layer decoder activations (for FF loss)
    pub pos_dec_acts: Vec<Tensor<B, 2>>,
    /// Per-layer decoder activations on random latent (for FF loss)
    pub neg_dec_acts: Vec<Tensor<B, 2>>,
    /// SSM-predicted next latent ẑ(t+1)
    pub pred_z: Tensor<B, 2>,
    /// Updated SSM state
    pub next_ssm_state: MultiScaleState<B>,
}

impl<B: Backend> RtrlWorldModel<B> {
    /// Create a new RTRL world model.
    pub fn new(config: &RtrlConfig, device: &B::Device) -> Self {
        let d_model = config.ssm.d_model;
        let d_state = config.ssm.d_state;
        let expand = config.ssm.expand;
        let n_heads = config.ssm.n_heads;
        let mimo_rank = config.ssm.mimo_rank;
        let conv_kernel = config.ssm.conv_kernel;

        let encoder = FfEncoder::new(&config.encoder_sizes, device);
        let decoder = FfEncoder::new(&config.decoder_sizes, device);
        let action_encoder = LinearConfig::new(config.action_dim, d_model).init(device);
        let fusion = LinearConfig::new(d_model * 2, d_model).init(device);

        let mscale_config = MultiScaleSsmConfig::new(d_model, d_state, expand, n_heads, mimo_rank)
            .with_n_layers(config.n_ssm_layers)
            .with_use_conv(true)
            .with_conv_kernel(conv_kernel);
        let ssm = MultiScaleSsmBlock::new(&mscale_config, device);

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
            n_ssm_layers: config.n_ssm_layers,
            input_dim: config.input_dim,
            action_dim: config.action_dim,
        }
    }

    /// Create initial zero SSM state for a new episode.
    pub fn initial_ssm_state(&self, batch_size: usize, device: &B::Device) -> MultiScaleState<B> {
        let d_inner = self.d_model * self.ssm.layers[0].d_inner / self.d_model;
        let n_heads = self.ssm.n_heads;
        let d_state = self.ssm.d_state;
        let d_head_mimo = (d_inner / n_heads) / self.ssm.layers[0].mimo_rank;
        let conv_kernel = 4usize; // matches default

        MultiScaleState::<B>::zeros(
            batch_size,
            self.n_ssm_layers,
            n_heads,
            d_state,
            d_head_mimo,
            true,
            d_inner,
            conv_kernel,
            device,
        )
    }

    /// Encode observation → latent (inference only, no loss).
    pub fn encode(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        self.encoder.encode(obs)
    }

    /// Decode latent → observation (inference only, no loss).
    pub fn decode(&self, z: Tensor<B, 2>) -> Tensor<B, 2> {
        self.decoder.encode(z)
    }

    /// Full training step: encode, decode, SSM predict — all with per-component loss.
    ///
    /// This is the core RTRL step. It:
    /// 1. Passes positive and negative data through the FF encoder → z(t), FF losses
    /// 2. Passes z(t) (detached) through FF decoder → reconstruction, FF losses
    /// 3. Passes z(t) (detached) + action through SSM (with detach'd h_{t-1}) → ẑ(t+1)
    ///
    /// Returns `RtrlStepOutput` containing all tensors needed for loss computation.
    /// The caller computes `MSE(pred_z, target_next_z)` externally.
    pub fn training_step(
        &self,
        obs: Tensor<B, 2>,
        neg_obs: Tensor<B, 2>,
        action: Tensor<B, 2>,
        ssm_state: &MultiScaleState<B>,
        _ff_threshold: f64,
    ) -> RtrlStepOutput<B> {
        // ─── FF Encoder ───
        let pos_enc_acts = self.encoder.positive_pass(obs);
        let z = pos_enc_acts.last().unwrap().clone();
        let neg_enc_acts = self.encoder.negative_pass(neg_obs);

        // ─── FF Decoder ───
        let pos_dec_acts = self.decoder.positive_pass(z.clone().detach());
        let reconstructed = pos_dec_acts.last().unwrap().clone();
        let random_latent = Tensor::<B, 2>::random(
            z.dims(),
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &z.device(),
        );
        let neg_dec_acts = self.decoder.negative_pass(random_latent);

        // ─── SSM RTRL step ───
        let a_enc = self.action_encoder.forward(action);
        let u_concat = Tensor::cat(vec![z.clone().detach(), a_enc], 1);
        let u = self.fusion.forward(u_concat);
        let (pred_z, next_ssm_state) = ssm_step_detached(&self.ssm, u, ssm_state);

        RtrlStepOutput {
            z,
            pos_enc_acts,
            neg_enc_acts,
            reconstructed,
            pos_dec_acts,
            neg_dec_acts,
            pred_z,
            next_ssm_state,
        }
    }

    /// Compute combined per-step loss from a `RtrlStepOutput`.
    ///
    /// Includes optional SSM prediction loss (set `ssm_loss` to `None` for the
    /// last timestep in a sequence, where no next target exists).
    pub fn compute_step_loss(
        &self,
        output: &RtrlStepOutput<B>,
        original_obs: &Tensor<B, 2>,
        ssm_loss: Option<Tensor<B, 1>>,
        args: &RtrlStepLossArgs,
    ) -> Tensor<B, 1> {
        // FF Encoder loss
        let enc_losses = self.encoder.compute_layer_losses(
            &output.pos_enc_acts,
            &output.neg_enc_acts,
            args.ff_threshold,
        );
        let enc_loss: Tensor<B, 1> = enc_losses.into_iter().fold(
            Tensor::<B, 1>::from_data([0.0f32], &original_obs.device()),
            |acc, l| acc + l,
        );

        // FF Decoder loss
        let dec_losses = self.decoder.compute_layer_losses(
            &output.pos_dec_acts,
            &output.neg_dec_acts,
            args.ff_threshold,
        );
        let dec_loss: Tensor<B, 1> = dec_losses.into_iter().fold(
            Tensor::<B, 1>::from_data([0.0f32], &original_obs.device()),
            |acc, l| acc + l,
        );

        // Reconstruction loss
        let recon_loss = (output.reconstructed.clone() - original_obs.clone())
            .powf_scalar(2.0)
            .mean()
            .unsqueeze();

        let mut total = enc_loss.mul_scalar(args.ff_enc_weight)
            + dec_loss.mul_scalar(args.ff_dec_weight)
            + recon_loss.mul_scalar(args.recon_weight);

        if let Some(ssl) = ssm_loss {
            total = total + ssl.mul_scalar(args.ssm_weight);
        }

        total
    }

    /// Autoregressive inference step.
    ///
    /// Takes current latent, action, and SSM state; returns predicted next
    /// latent and updated SSM state. No gradient tracking needed.
    pub fn inference_step(
        &self,
        z_prev: Tensor<B, 2>,
        action: Tensor<B, 2>,
        ssm_state: &MultiScaleState<B>,
    ) -> (Tensor<B, 2>, MultiScaleState<B>) {
        let a_enc = self.action_encoder.forward(action);
        let u_concat = Tensor::cat(vec![z_prev, a_enc], 1);
        let u = self.fusion.forward(u_concat);
        ssm_step_detached(&self.ssm, u, ssm_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn test_rtrl_world_model_builds() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = RtrlConfig {
            ssm: SsmConfig::new(32, 8, 2, 2, 1),
            n_ssm_layers: 1,
            input_dim: 3,
            action_dim: 2,
            encoder_sizes: vec![3, 16, 32],
            decoder_sizes: vec![32, 16, 3],
        };
        let model = RtrlWorldModel::<NdArray>::new(&config, &device);
        assert_eq!(model.d_model, 32);
        assert_eq!(model.input_dim, 3);
        assert_eq!(model.action_dim, 2);
    }

    #[test]
    fn test_rtrl_encode_decode() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = RtrlConfig {
            ssm: SsmConfig::new(32, 8, 2, 2, 1),
            n_ssm_layers: 1,
            input_dim: 3,
            action_dim: 2,
            encoder_sizes: vec![3, 16, 32],
            decoder_sizes: vec![32, 16, 3],
        };
        let model = RtrlWorldModel::<NdArray>::new(&config, &device);

        let obs = Tensor::<NdArray, 2>::from_data(
            burn::tensor::TensorData::new(vec![1.0f32, 0.0, 0.0], [1, 3]),
            &device,
        );
        let z = model.encode(obs);
        assert_eq!(z.dims(), [1, 32]);

        let decoded = model.decode(z);
        assert_eq!(decoded.dims(), [1, 3]);
    }

    #[test]
    fn test_rtrl_training_step() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = RtrlConfig {
            ssm: SsmConfig::new(32, 8, 2, 2, 1),
            n_ssm_layers: 1,
            input_dim: 2,
            action_dim: 2,
            encoder_sizes: vec![2, 16, 32],
            decoder_sizes: vec![32, 16, 2],
        };
        let model = RtrlWorldModel::<NdArray>::new(&config, &device);

        let obs = Tensor::<NdArray, 2>::from_data(
            burn::tensor::TensorData::new(vec![1.0f32, 0.0], [1, 2]),
            &device,
        );
        let neg_obs = Tensor::<NdArray, 2>::from_data(
            burn::tensor::TensorData::new(vec![0.5f32, -0.8], [1, 2]),
            &device,
        );
        let action = Tensor::<NdArray, 2>::from_data(
            burn::tensor::TensorData::new(vec![0.1f32, -0.05], [1, 2]),
            &device,
        );
        let state = model.initial_ssm_state(1, &device);

        let output = model.training_step(obs.clone(), neg_obs, action, &state, 2.0);
        assert_eq!(output.z.dims(), [1, 32]);
        assert_eq!(output.reconstructed.dims(), [1, 2]);
        assert_eq!(output.pred_z.dims(), [1, 32]);
    }
}
