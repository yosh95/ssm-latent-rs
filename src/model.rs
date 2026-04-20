use burn::config::Config;
use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

#[derive(Config, Debug)]
pub struct MambaConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub expand: usize,
}

pub struct ComplexTensor<B: Backend, const D: usize> {
    pub re: Tensor<B, D>,
    pub im: Tensor<B, D>,
}

#[derive(Module, Debug)]
pub struct MambaBlock<B: Backend> {
    pub a_re: Param<Tensor<B, 2>>,
    pub a_im: Param<Tensor<B, 2>>,
    pub d: Param<Tensor<B, 1>>,
    pub in_proj_weight: Param<Tensor<B, 2>>,
    pub out_proj_weight: Param<Tensor<B, 2>>,
    pub d_state: usize,
}

impl<B: Backend> MambaBlock<B> {
    pub fn new(config: &MambaConfig, device: &B::Device) -> Self {
        let d_inner = config.d_model * config.expand;
        let d_state = config.d_state;

        let a_re = Tensor::from_data(
            TensorData::new(vec![-1.0f32; d_inner * d_state], [d_inner, d_state]),
            device,
        );
        let a_im = Tensor::zeros([d_inner, d_state], device);
        let d = Tensor::ones([d_inner], device);
        let in_proj_weight = Tensor::zeros([d_inner * 2, config.d_model], device);
        let out_proj_weight = Tensor::zeros([config.d_model, d_inner], device);

        Self {
            a_re: Param::from_tensor(a_re),
            a_im: Param::from_tensor(a_im),
            d: Param::from_tensor(d),
            in_proj_weight: Param::from_tensor(in_proj_weight),
            out_proj_weight: Param::from_tensor(out_proj_weight),
            d_state,
        }
    }

    /// Associative Scan (O(log L)) for training. Returns (output, h_re, h_im) for all time steps.
    pub fn selective_scan(
        &self,
        u: Tensor<B, 3>,     // [batch, seq_len, d_inner]
        delta: Tensor<B, 3>, // [batch, seq_len, d_inner]
        b_re: Tensor<B, 3>,  // [batch, seq_len, d_state]
        b_im: Tensor<B, 3>,
        c_re: Tensor<B, 3>,
        c_im: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>, Tensor<B, 4>) {
        let [_batch, _seq_len, _d_inner] = u.dims();

        // 1. Discretization
        let dt = delta.unsqueeze_dim::<4>(3); // [batch, seq, d_inner, 1]
        let a_re = self.a_re.val().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
        let a_im = self.a_im.val().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);

        let da_abs = (dt.clone() * a_re).exp();
        let da_angle = dt.clone() * a_im;
        let alpha_re = da_abs.clone() * da_angle.clone().cos(); // [batch, seq, d_inner, d_state]
        let alpha_im = da_abs * da_angle.sin();

        let u_u = u.clone().unsqueeze_dim::<4>(3); // [batch, seq, d_inner, 1]
        let beta_re = (dt.clone() * b_re.unsqueeze_dim::<4>(2)) * u_u.clone(); // [batch, seq, d_inner, d_state]
        let beta_im = (dt * b_im.unsqueeze_dim::<4>(2)) * u_u;

        // 2. Parallel Scan
        let (h_re, h_im) = self.parallel_scan(alpha_re, alpha_im, beta_re, beta_im);

        // 3. Output calculation
        let cr = c_re.unsqueeze_dim::<4>(2); // [batch, seq, 1, d_state]
        let ci = c_im.unsqueeze_dim::<4>(2);

        let out_re = (h_re.clone() * cr).sum_dim(3).squeeze::<3>(3)
            - (h_im.clone() * ci).sum_dim(3).squeeze::<3>(3);

        let y = out_re + self.d.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0) * u;

        (y, h_re, h_im)
    }

    /// O(1) step for inference/JEPA. Returns (output, next_h).
    pub fn step(
        &self,
        u: Tensor<B, 2>,
        delta: Tensor<B, 2>,
        b: ComplexTensor<B, 2>,
        c: ComplexTensor<B, 2>,
        prev_h: ComplexTensor<B, 3>,
    ) -> (Tensor<B, 2>, ComplexTensor<B, 3>) {
        let dt_u = delta.unsqueeze_dim::<3>(2); // [batch, d_inner, 1]

        let a_re = self.a_re.val().unsqueeze_dim::<3>(0);
        let a_im = self.a_im.val().unsqueeze_dim::<3>(0);

        let da_abs = (dt_u.clone() * a_re).exp();
        let da_angle = dt_u.clone() * a_im;
        let da_re = da_abs.clone() * da_angle.clone().cos();
        let da_im = da_abs * da_angle.sin();

        let dt_b_re = dt_u.clone() * b.re.unsqueeze_dim::<3>(1);
        let dt_b_im = dt_u * b.im.unsqueeze_dim::<3>(1);
        let ut_u = u.clone().unsqueeze_dim::<3>(2);

        let next_h_re = (da_re.clone() * prev_h.re.clone() - da_im.clone() * prev_h.im.clone())
            + (dt_b_re * ut_u.clone());
        let next_h_im = (da_re * prev_h.im + da_im * prev_h.re) + (dt_b_im * ut_u);

        let cr_u = c.re.unsqueeze_dim::<3>(1);
        let ci_u = c.im.unsqueeze_dim::<3>(1);

        let out_re = (next_h_re.clone() * cr_u).sum_dim(2).squeeze::<2>(2)
            - (next_h_im.clone() * ci_u).sum_dim(2).squeeze::<2>(2);

        let y = out_re + self.d.val().unsqueeze_dim::<2>(0) * u;

        (
            y,
            ComplexTensor {
                re: next_h_re,
                im: next_h_im,
            },
        )
    }

    fn parallel_scan(
        &self,
        mut alpha_re: Tensor<B, 4>,
        mut alpha_im: Tensor<B, 4>,
        mut beta_re: Tensor<B, 4>,
        mut beta_im: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq_len, _d_inner, _d_state] = alpha_re.dims();
        let mut offset = 1;

        while offset < seq_len {
            let num_elements = seq_len - offset;

            // Left side (i-offset)
            let a_re_l = alpha_re.clone().slice([0..batch, 0..num_elements]);
            let a_im_l = alpha_im.clone().slice([0..batch, 0..num_elements]);
            let b_re_l = beta_re.clone().slice([0..batch, 0..num_elements]);
            let b_im_l = beta_im.clone().slice([0..batch, 0..num_elements]);

            // Right side (i)
            let a_re_r = alpha_re.clone().slice([0..batch, offset..seq_len]);
            let a_im_r = alpha_im.clone().slice([0..batch, offset..seq_len]);
            let b_re_r = beta_re.clone().slice([0..batch, offset..seq_len]);
            let b_im_r = beta_im.clone().slice([0..batch, offset..seq_len]);

            // Binary operation (a_r, b_r) ∘ (a_l, b_l)
            let new_alpha_re = a_re_r.clone() * a_re_l.clone() - a_im_r.clone() * a_im_l.clone();
            let new_alpha_im = a_re_r.clone() * a_im_l.clone() + a_im_r.clone() * a_re_l.clone();

            let new_beta_re =
                (a_re_r.clone() * b_re_l.clone() - a_im_r.clone() * b_im_l.clone()) + b_re_r;
            let new_beta_im = (a_re_r * b_im_l + a_im_r * b_re_l) + b_im_r;

            // Update slices
            alpha_re = Tensor::cat(vec![alpha_re.slice([0..batch, 0..offset]), new_alpha_re], 1);
            alpha_im = Tensor::cat(vec![alpha_im.slice([0..batch, 0..offset]), new_alpha_im], 1);
            beta_re = Tensor::cat(vec![beta_re.slice([0..batch, 0..offset]), new_beta_re], 1);
            beta_im = Tensor::cat(vec![beta_im.slice([0..batch, 0..offset]), new_beta_im], 1);

            offset *= 2;
        }
        (beta_re, beta_im)
    }
}
