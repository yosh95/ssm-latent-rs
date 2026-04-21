use crate::ssm::{SsmBlock, SsmConfig};
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Clone)]
pub struct LatentState<B: Backend> {
    pub h: Tensor<B, 4>,
    pub prev_bx: Option<Tensor<B, 4>>,
    pub conv_state: Option<Tensor<B, 3>>,
}

#[derive(Module, Debug)]
pub struct LatentPredictor<B: Backend> {
    encoder: Linear<B>,
    action_encoder: Linear<B>,
    fusion: Linear<B>,
    ssm: SsmBlock<B>,
    // Fixed random projection buffer (Param used for device management, but gradients are not needed)
    stability_projections: Param<Tensor<B, 2>>,
    d_model: usize,
}

impl<B: Backend> LatentPredictor<B> {
    pub fn new(
        config: &SsmConfig,
        input_dim: usize,
        action_dim: usize,
        device: &B::Device,
    ) -> Self {
        let encoder = LinearConfig::new(input_dim, config.d_model).init(device);
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

    pub fn forward(
        &self,
        observations: Tensor<B, 3>,
        actions: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let z = self.encoder.forward(observations);
        let a = self.action_encoder.forward(actions);
        let u_concat = Tensor::cat(vec![z.clone(), a], 2);
        let u = self.fusion.forward(u_concat);
        let predicted_z = self.ssm.forward(u);
        (z, predicted_z)
    }

    pub fn step(
        &self,
        z_prev: Tensor<B, 2>,
        action: Tensor<B, 2>,
        state: LatentState<B>,
    ) -> (Tensor<B, 2>, LatentState<B>) {
        let a = self
            .action_encoder
            .forward(action.unsqueeze_dim::<3>(1))
            .squeeze::<2>(1);
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

    pub fn loss(
        &self,
        z: Tensor<B, 3>,
        pred_z: Tensor<B, 3>,
        stability_weight: f64,
    ) -> Tensor<B, 1> {
        let [batch, seq_len, d_model] = z.dims();
        let target_z = z.clone().detach().slice([0..batch, 1..seq_len, 0..d_model]);
        let pred_slice = pred_z.slice([0..batch, 0..(seq_len - 1), 0..d_model]);

        let mse_loss = (target_z - pred_slice).powf_scalar(2.0).mean();

        // Efficient O(T) stability loss via Gaussian moment matching
        let reg_loss = stability_loss(z, self.stability_projections.val());

        mse_loss + reg_loss.mul_scalar(stability_weight)
    }
}

/// Gaussian-based O(T) stability loss to prevent representation collapse in JEPA
pub fn stability_loss<B: Backend>(z: Tensor<B, 3>, w: Tensor<B, 2>) -> Tensor<B, 1> {
    let [batch, seq_len, d_model] = z.dims();
    let n_projections = w.dims()[1];

    let z_flat = z.reshape([batch * seq_len, d_model]);
    let projections = z_flat.matmul(w).reshape([batch, seq_len, n_projections]);

    // Calculate statistics for each projection dimension (O(T) complexity)
    let mean = projections.clone().mean_dim(1); // [batch, 1, n_projections]
    let var = projections.var(1); // [batch, 1, n_projections]

    // Penalize deviation from standard normal distribution (mean=0, var=1)
    let loss_mean = mean.powf_scalar(2.0).mean();
    let loss_var = (var - 1.0).powf_scalar(2.0).mean();

    (loss_mean + loss_var).unsqueeze()
}
