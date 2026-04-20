use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, Distribution};

#[derive(Config, Debug)]
pub struct MambaConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub expand: usize,
}

#[derive(Clone)]
pub struct ComplexTensor<B: Backend, const D: usize> {
    pub re: Tensor<B, D>,
    pub im: Tensor<B, D>,
}

#[derive(Module, Debug)]
pub struct MambaBlock<B: Backend> {
    pub in_proj: Linear<B>,
    pub out_proj: Linear<B>,
    pub dt_proj: Linear<B>,
    pub b_proj: Linear<B>,
    pub c_proj: Linear<B>,
    pub a_re: Param<Tensor<B, 2>>,
    pub a_im: Param<Tensor<B, 2>>,
    pub d: Param<Tensor<B, 1>>,
    pub d_inner: usize,
    pub d_state: usize,
}

impl<B: Backend> MambaBlock<B> {
    pub fn new(config: &MambaConfig, device: &B::Device) -> Self {
        let d_inner = config.d_model * config.expand;
        let d_state = config.d_state;

        // Projections
        let in_proj = LinearConfig::new(config.d_model, d_inner * 2).init(device);
        let out_proj = LinearConfig::new(d_inner, config.d_model).init(device);
        
        // SSM Parameter Projections (Dynamic)
        let dt_proj = LinearConfig::new(d_inner, d_inner).init(device);
        let b_proj = LinearConfig::new(d_inner, d_state).init(device);
        let c_proj = LinearConfig::new(d_inner, d_state).init(device);

        // A is the system matrix, usually initialized with special patterns
        let a_re = Tensor::random([d_inner, d_state], Distribution::Uniform(-1.0, -0.1), device);
        let a_im = Tensor::random([d_inner, d_state], Distribution::Default, device);
        
        let d = Tensor::ones([d_inner], device);

        Self {
            in_proj,
            out_proj,
            dt_proj,
            b_proj,
            c_proj,
            a_re: Param::from_tensor(a_re),
            a_im: Param::from_tensor(a_im),
            d: Param::from_tensor(d),
            d_inner,
            d_state,
        }
    }

    /// Forward pass through Mamba block
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = &x.device();

        // 1. Input Projection & Split
        let projected = self.in_proj.forward(x);
        let mut chunks = projected.chunk(2, 2);
        let u = chunks.remove(0); // Branch for SSM
        let gate = chunks.remove(0); // Branch for Gating

        // 2. Generate dynamic parameters (Selection Mechanism)
        // delta = softplus(dt_proj(u))
        let delta = burn::tensor::activation::softplus(self.dt_proj.forward(u.clone()), 1.0);
        
        // B and C can be complex or real. Here we simplify to real input for B,C 
        // and treat them as complex with 0 imaginary part or just real.
        let b_raw = self.b_proj.forward(u.clone());
        let c_raw = self.c_proj.forward(u.clone());
        
        // 3. Selective Scan
        let (y_ssm, _, _) = self.selective_scan(
            u.clone(), 
            delta, 
            b_raw.clone(), 
            Tensor::zeros(b_raw.dims(), device), // b_im
            c_raw.clone(), 
            Tensor::zeros(c_raw.dims(), device)  // c_im
        );

        // 4. Gating and Output Projection
        let y = y_ssm * burn::tensor::activation::silu(gate);
        self.out_proj.forward(y)
    }

    pub fn forward_step(
        &self,
        x: Tensor<B, 2>,
        prev_h: ComplexTensor<B, 3>,
    ) -> (Tensor<B, 2>, ComplexTensor<B, 3>) {
        let device = &x.device();
        
        // 1. Input Projection
        let projected = self.in_proj.forward(x);
        let mut chunks = projected.chunk(2, 1);
        let u = chunks.remove(0);
        let gate = chunks.remove(0);

        // 2. Selection Mechanism
        let delta = burn::tensor::activation::softplus(self.dt_proj.forward(u.clone()), 1.0);
        let b_raw = self.b_proj.forward(u.clone());
        let c_raw = self.c_proj.forward(u.clone());

        // 3. SSM Step
        let (y_ssm, next_h) = self.step(
            u,
            delta,
            ComplexTensor { re: b_raw.clone(), im: Tensor::zeros_like(&b_raw) },
            ComplexTensor { re: c_raw.clone(), im: Tensor::zeros_like(&c_raw) },
            prev_h
        );

        // 4. Gating & Output
        let y = y_ssm * burn::tensor::activation::silu(gate);
        let out = self.out_proj.forward(y);

        (out, next_h)
    }

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

        // Discretization
        let dt = delta.unsqueeze_dim::<4>(3); 
        let a_re = self.a_re.val().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
        let a_im = self.a_im.val().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);

        let da_abs = (dt.clone() * a_re).exp();
        let da_angle = dt.clone() * a_im;
        let alpha_re = da_abs.clone() * da_angle.clone().cos();
        let alpha_im = da_abs * da_angle.sin();

        let u_u = u.clone().unsqueeze_dim::<4>(3);
        let beta_re = (dt.clone() * b_re.unsqueeze_dim::<4>(2)) * u_u.clone();
        let beta_im = (dt * b_im.unsqueeze_dim::<4>(2)) * u_u;

        // Correct Parallel Scan (Associative Scan)
        let (h_re, h_im) = self.parallel_scan(alpha_re, alpha_im, beta_re, beta_im);

        let cr = c_re.unsqueeze_dim::<4>(2);
        let ci = c_im.unsqueeze_dim::<4>(2);

        let out_re = (h_re.clone() * cr).sum_dim(3).squeeze::<3>(3)
            - (h_im.clone() * ci).sum_dim(3).squeeze::<3>(3);

        let y = out_re + self.d.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0) * u;

        (y, h_re, h_im)
    }

    pub fn step(
        &self,
        u: Tensor<B, 2>,
        delta: Tensor<B, 2>,
        b: ComplexTensor<B, 2>,
        c: ComplexTensor<B, 2>,
        prev_h: ComplexTensor<B, 3>,
    ) -> (Tensor<B, 2>, ComplexTensor<B, 3>) {
        let dt_u = delta.unsqueeze_dim::<3>(2);

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

        (y, ComplexTensor { re: next_h_re, im: next_h_im })
    }

    fn parallel_scan(
        &self,
        alpha_re: Tensor<B, 4>,
        alpha_im: Tensor<B, 4>,
        beta_re: Tensor<B, 4>,
        beta_im: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, seq_len, _d_inner, _d_state] = alpha_re.dims();
        let mut out_alpha_re = alpha_re;
        let mut out_alpha_im = alpha_im;
        let mut out_beta_re = beta_re;
        let mut out_beta_im = beta_im;

        let mut offset = 1;
        while offset < seq_len {
            let left_indices = 0..(seq_len - offset);
            let right_indices = offset..seq_len;

            let a_re_l = out_alpha_re.clone().slice([0..batch, left_indices.clone()]);
            let a_im_l = out_alpha_im.clone().slice([0..batch, left_indices.clone()]);
            let b_re_l = out_beta_re.clone().slice([0..batch, left_indices.clone()]);
            let b_im_l = out_beta_im.clone().slice([0..batch, left_indices.clone()]);

            let a_re_r = out_alpha_re.clone().slice([0..batch, right_indices.clone()]);
            let a_im_r = out_alpha_im.clone().slice([0..batch, right_indices.clone()]);
            let b_re_r = out_beta_re.clone().slice([0..batch, right_indices.clone()]);
            let b_im_r = out_beta_im.clone().slice([0..batch, right_indices.clone()]);

            // Complex associative scan: (a_r, b_r) ∘ (a_l, b_l) = (a_r * a_l, a_r * b_l + b_r)
            let res_alpha_re = a_re_r.clone() * a_re_l.clone() - a_im_r.clone() * a_im_l.clone();
            let res_alpha_im = a_re_r.clone() * a_im_l + a_im_r.clone() * a_re_l;
            let res_beta_re = (a_re_r.clone() * b_re_l.clone() - a_im_r.clone() * b_im_l.clone()) + b_re_r;
            let res_beta_im = (a_re_r * b_im_l + a_im_r * b_re_l) + b_im_r;

            // Update only the right parts for Hillis-Steele scan
            out_alpha_re = Tensor::cat(vec![out_alpha_re.slice([0..batch, 0..offset]), res_alpha_re], 1);
            out_alpha_im = Tensor::cat(vec![out_alpha_im.slice([0..batch, 0..offset]), res_alpha_im], 1);
            out_beta_re = Tensor::cat(vec![out_beta_re.slice([0..batch, 0..offset]), res_beta_re], 1);
            out_beta_im = Tensor::cat(vec![out_beta_im.slice([0..batch, 0..offset]), res_beta_im], 1);

            offset *= 2;
        }
        (out_beta_re, out_beta_im)
    }
}
