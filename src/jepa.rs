use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use crate::model::{MambaBlock, MambaConfig};

/// JEPA (Joint-Embedding Predictive Architecture) World Model
/// 
/// This implementation uses a Mamba state-space model as the predictor
/// and incorporates SIGReg (Sketched Isotropic Gaussian Regularization)
/// to prevent representation collapse, as described in arXiv:2603.19312.
#[derive(Module, Debug)]
pub struct JepaWorldModel<B: Backend> {
    pub encoder: Linear<B>,
    pub action_encoder: Linear<B>,
    pub mamba: MambaBlock<B>,
    pub d_model: usize,
    pub d_state: usize,
}

impl<B: Backend> JepaWorldModel<B> {
    pub fn new(config: &MambaConfig, input_dim: usize, action_dim: usize, device: &B::Device) -> Self {
        let d_model = config.d_model;
        let encoder = LinearConfig::new(input_dim, d_model).init(device);
        let action_encoder = LinearConfig::new(action_dim, d_model).init(device);
        let mamba = MambaBlock::new(config, device);

        Self {
            encoder,
            action_encoder,
            mamba,
            d_model,
            d_state: config.d_state,
        }
    }

    /// Predicts next latent states from observations and actions.
    /// Returns (latents, predicted_latents).
    pub fn forward(
        &self,
        observations: Tensor<B, 3>, // [batch, seq_len, input_dim]
        actions: Tensor<B, 3>,      // [batch, seq_len, action_dim]
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, seq_len, _] = observations.dims();
        let device = &observations.device();
        
        // 1. Encode observations into latents z
        let z = self.encoder.forward(observations); // [batch, seq_len, d_model]
        
        // 2. Encode actions
        let a = self.action_encoder.forward(actions); // [batch, seq_len, d_model]
        
        // 3. Fusion of context and action (simple addition or MLP)
        // Here we use addition as a basic fusion for the Mamba input.
        let u = z.clone() + a; 
        
        // 4. Mamba selective scan for world dynamics
        // We generate dynamic parameters for the SSM.
        // In a full implementation, these would be learnable projections.
        let delta = Tensor::ones([batch, seq_len, self.d_model], device).mul_scalar(0.1);
        let b_re = Tensor::ones([batch, seq_len, self.d_state], device).mul_scalar(0.1);
        let b_im = Tensor::zeros([batch, seq_len, self.d_state], device);
        let c_re = Tensor::ones([batch, seq_len, self.d_state], device).mul_scalar(0.1);
        let c_im = Tensor::zeros([batch, seq_len, self.d_state], device);

        let (predicted_z, _h_re, _h_im) = self.mamba.selective_scan(u, delta, b_re, b_im, c_re, c_im);
        
        // In JEPA, the output predicted_z[t] aims to predict z[t+1]
        (z, predicted_z)
    }

    /// Calculates the JEPA loss: Prediction Loss (MSE) + SIGReg Regularization.
    pub fn loss(
        &self,
        z: Tensor<B, 3>,
        predicted_z: Tensor<B, 3>,
        sigreg_weight: f64,
    ) -> Tensor<B, 1> {
        let [_batch, seq_len, _d_model] = z.dims();
        
        // Prediction Loss: MSE between predicted z_t and actual z_{t+1}
        let target_z = z.clone().slice([0.._batch, 1..seq_len]);
        let pred_slice = predicted_z.slice([0.._batch, 0..(seq_len - 1)]);
        let mse_loss = (target_z - pred_slice).powf_scalar(2.0).mean();
        
        // Anti-collapse Loss: SIGReg
        let reg_loss = sigreg_loss(z, 8); // Using 8 random projections
        
        mse_loss + reg_loss.mul_scalar(sigreg_weight)
    }
}

/// SIGReg: Sketched Isotropic Gaussian Regularizer
/// Implements the normality test based on the Cramér-Wold Theorem
/// to prevent representation collapse in JEPA.
pub fn sigreg_loss<B: Backend>(z: Tensor<B, 3>, n_projections: usize) -> Tensor<B, 1> {
    let [batch, seq_len, d_model] = z.dims();
    let _n = (batch * seq_len) as f32;
    let device = &z.device();

    // Flatten to [N, d_model]
    let z_flat = z.reshape([batch * seq_len, d_model]);

    // Generate M random projection directions (unit vectors)
    let w = Tensor::<B, 2>::random(
        [d_model, n_projections],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    let w_norm = w.clone().powf_scalar(2.0).sum_dim(0).sqrt() + 1e-6;
    let w = w / w_norm;

    // Project: [N, d_model] @ [d_model, M] -> [N, M]
    let projections = z_flat.matmul(w);

    // Center and scale projections to N(0, 1) per direction
    let mean = projections.clone().mean_dim(0);
    let var = (projections.clone() - mean.clone()).powf_scalar(2.0).mean_dim(0) + 1e-6;
    let x = (projections - mean) / var.sqrt();

    // Epps-Pulley normality test statistic T
    // T = mean(exp(-0.5*(xi-xj)^2)) - 2*sqrt(2)*mean(exp(-0.25*xi^2)) + 1/sqrt(3)
    let mut total_t = Tensor::zeros([1], device);

    for m in 0..n_projections {
        let xm = x.clone().slice([0..(batch * seq_len), m..m + 1]).squeeze::<1>(1);
        
        // Pairwise difference term
        let xi = xm.clone().unsqueeze_dim::<2>(1); // [N, 1]
        let xj = xm.clone().unsqueeze_dim::<2>(0); // [1, N]
        let term1 = (xi - xj).powf_scalar(2.0).mul_scalar(-0.5).exp().mean();

        // Gaussian kernel term
        let term2 = xm.powf_scalar(2.0).mul_scalar(-0.25).exp().mean().mul_scalar(2.0 * 2.0f32.sqrt());

        let tm = term1 - term2 + (1.0 / 3.0f32.sqrt());
        // Use abs() or pow2() to ensure it's a positive loss if T is a distance
        total_t = total_t + tm.powf_scalar(2.0).unsqueeze();
    }

    total_t / (n_projections as f32)
}
