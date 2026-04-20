use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use crate::mamba::{MambaBlock, MambaConfig, ComplexTensor};

#[derive(Clone)]
pub struct JepaState<B: Backend> {
    pub h_re: Tensor<B, 3>,
    pub h_im: Tensor<B, 3>,
}

#[derive(Module, Debug)]
pub struct JepaWorldModel<B: Backend> {
    pub encoder: Linear<B>,
    pub action_encoder: Linear<B>,
    pub mamba: MambaBlock<B>,
    pub d_model: usize,
}

impl<B: Backend> JepaWorldModel<B> {
    pub fn new(config: &MambaConfig, input_dim: usize, action_dim: usize, device: &B::Device) -> Self {
        let encoder = LinearConfig::new(input_dim, config.d_model).init(device);
        let action_encoder = LinearConfig::new(action_dim, config.d_model).init(device);
        let mamba = MambaBlock::new(config, device);

        Self {
            encoder,
            action_encoder,
            mamba,
            d_model: config.d_model,
        }
    }

    /// Parallel forward for training
    pub fn forward(
        &self,
        observations: Tensor<B, 3>,
        actions: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let z = self.encoder.forward(observations);
        let a = self.action_encoder.forward(actions);
        
        let u = z.clone() + a; 
        let predicted_z = self.mamba.forward(u);
        
        (z, predicted_z)
    }

    /// Sequential step for open-loop imagination
    pub fn step(
        &self,
        z_prev: Tensor<B, 2>,
        action: Tensor<B, 2>,
        state: JepaState<B>,
    ) -> (Tensor<B, 2>, JepaState<B>) {
        let a = self.action_encoder.forward(action.unsqueeze::<3>());
        let u = z_prev.unsqueeze::<3>() + a;
        let u_sq = u.squeeze::<2>(0);

        let (y, next_h) = self.mamba.forward_step(
            u_sq,
            ComplexTensor { re: state.h_re, im: state.h_im }
        );

        (y, JepaState { h_re: next_h.re, h_im: next_h.im })
    }

    pub fn loss(&self, z: Tensor<B, 3>, pred_z: Tensor<B, 3>, sigreg_weight: f64) -> Tensor<B, 1> {
        let [batch, seq_len, _] = z.dims();
        let target_z = z.clone().slice([0..batch, 1..seq_len]);
        let pred_slice = pred_z.slice([0..batch, 0..(seq_len - 1)]);
        
        let mse_loss = (target_z - pred_slice).powf_scalar(2.0).mean();
        let reg_loss = sigreg_loss(z, 8);
        
        mse_loss + reg_loss.mul_scalar(sigreg_weight)
    }
}

pub fn sigreg_loss<B: Backend>(z: Tensor<B, 3>, n_projections: usize) -> Tensor<B, 1> {
    let [batch, seq_len, d_model] = z.dims();
    let device = &z.device();
    let z_flat = z.reshape([batch * seq_len, d_model]);

    let w = Tensor::<B, 2>::random([d_model, n_projections], burn::tensor::Distribution::Normal(0.0, 1.0), device);
    let w = w.clone() / (w.powf_scalar(2.0).sum_dim(0).sqrt() + 1e-6);
    let projections = z_flat.matmul(w);

    let mean = projections.clone().mean_dim(0);
    let var = (projections.clone() - mean.clone()).powf_scalar(2.0).mean_dim(0) + 1e-6;
    let x = (projections - mean) / var.sqrt();

    let mut total_t = Tensor::zeros([1], device);
    let _n = (batch * seq_len) as f32;

    for m in 0..n_projections {
        let xm = x.clone().slice([0..(batch * seq_len), m..m + 1]).squeeze::<1>(1);
        
        // Term 1: exp(-0.5 * (xi - xj)^2)
        // To avoid O(N^2), we use the kernel trick: mean(k(xi, xj))
        // For simple demos, we keep N small.
        let xi = xm.clone().unsqueeze_dim::<2>(1);
        let xj = xm.clone().unsqueeze_dim::<2>(0);
        let term1 = (xi - xj).powf_scalar(2.0).mul_scalar(-0.5).exp().mean();

        let term2 = xm.powf_scalar(2.0).mul_scalar(-0.25).exp().mean().mul_scalar(2.0 * 2.0f32.sqrt());
        let tm = term1 - term2 + (1.0 / 3.0f32.sqrt());
        total_t = total_t + tm.powf_scalar(2.0).unsqueeze();
    }

    total_t / (n_projections as f32)
}
