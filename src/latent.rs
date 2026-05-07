use crate::ssm::{SsmBlock, SsmConfig};
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

#[derive(Clone)]
pub struct LatentState<B: Backend> {
    pub h: Tensor<B, 4>,
    pub prev_bx: Option<Tensor<B, 4>>,
    pub conv_state: Option<Tensor<B, 3>>,
}

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

pub struct LatentLossArgs<B: Backend> {
    pub z: Tensor<B, 3>,
    pub pred_z: Tensor<B, 3>,
    pub reconstructed_x: Tensor<B, 3>,
    pub predicted_x: Tensor<B, 3>,
    pub original_x: Tensor<B, 3>,
    pub stability_weight: f64,
    pub curvature_weight: f64,
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

        // Fixed projection matrix based on Gaussian distribution
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

    pub fn save(&self, file_path: &str) -> Result<(), std::io::Error> {
        let recorder = burn::record::BinFileRecorder::<burn::record::FullPrecisionSettings>::new();
        let path = std::path::Path::new(file_path);

        recorder
            .record(self.clone().into_record(), path.to_path_buf())
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        Ok(())
    }

    pub fn load(self, file_path: &str, device: &B::Device) -> Result<Self, std::io::Error> {
        let recorder = burn::record::BinFileRecorder::<burn::record::FullPrecisionSettings>::new();
        let path = std::path::Path::new(file_path);

        let record = recorder
            .load(path.to_path_buf(), device)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        Ok(self.load_record(record))
    }

    pub fn loss(&self, args: LatentLossArgs<B>) -> Tensor<B, 1> {
        let [batch, seq_len, _] = args.z.dims();
        let target_z = args.z.clone().detach().slice([0..batch, 1..seq_len]);
        let pred_slice = args.pred_z.clone().slice([0..batch, 0..seq_len - 1]);

        let mse_latent = (target_z - pred_slice).powf_scalar(2.0).mean();

        // Reconstruction from current
        let mse_recons = (args.original_x.clone() - args.reconstructed_x)
            .powf_scalar(2.0)
            .mean();
        // Reconstruction from predicted (next obs)
        let target_x = args.original_x.detach().slice([0..batch, 1..seq_len]);
        let pred_x_slice = args.predicted_x.slice([0..batch, 0..seq_len - 1]);
        let mse_pred_x = (target_x - pred_x_slice).powf_scalar(2.0).mean();

        // Efficient O(T) stability loss via Gaussian moment matching
        let reg_loss = stability_loss(args.z.clone(), self.stability_projections.val());

        // Temporal Straightening: Reduce curvature to improve planning stability
        let curv_loss = curvature_loss(args.z);

        mse_latent
            + mse_recons.mul_scalar(args.recon_weight)
            + mse_pred_x.mul_scalar(args.recon_weight)
            + reg_loss.mul_scalar(args.stability_weight)
            + curv_loss.mul_scalar(args.curvature_weight)
    }
}

/// Temporal Straightening loss: minimizes the second-order finite difference
/// of the latent trajectory to encourage locally straight paths.
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

/// Gaussian-based O(T) stability loss to prevent representation collapse in JEPA
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
