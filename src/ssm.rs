use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};
use core::f32::consts::PI;

#[derive(Config, Debug)]
pub struct SsmConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub expand: usize,
    pub n_heads: usize,
    pub mimo_rank: usize,
    #[config(default = true)]
    pub use_conv: bool,
    #[config(default = 4)]
    pub conv_kernel: usize,
}

#[derive(Module, Debug)]
pub struct SsmBlock<B: Backend> {
    pub in_proj: Linear<B>,
    pub conv1d: Option<Conv1d<B>>,
    pub out_proj: Linear<B>,
    pub dt_proj: Linear<B>,
    pub lambda_proj: Linear<B>,
    pub theta_proj: Linear<B>,
    pub b_proj: Linear<B>,
    pub c_proj: Linear<B>,
    pub b_bias: Param<Tensor<B, 3>>,
    pub c_bias: Param<Tensor<B, 3>>,
    pub a_re: Param<Tensor<B, 2>>,
    pub a_im: Param<Tensor<B, 2>>,
    pub d: Param<Tensor<B, 1>>,
    pub norm: RmsNorm<B>,
    pub d_inner: usize,
    pub d_state: usize,
    pub n_heads: usize,
    pub mimo_rank: usize,
}

impl<B: Backend> SsmBlock<B> {
    pub fn new(config: &SsmConfig, device: &B::Device) -> Self {
        let d_inner = config.d_model * config.expand;
        let d_state = config.d_state;
        let n_heads = config.n_heads;
        let mimo_rank = config.mimo_rank;

        assert!(
            d_inner.is_multiple_of(n_heads),
            "d_inner must be divisible by n_heads"
        );
        assert!(
            (d_inner / n_heads).is_multiple_of(mimo_rank),
            "d_head must be divisible by mimo_rank"
        );
        assert!(
            d_state.is_multiple_of(2),
            "d_state must be even for complex rotation"
        );

        let in_proj = LinearConfig::new(config.d_model, d_inner * 2).init(device);
        let out_proj = LinearConfig::new(d_inner, config.d_model).init(device);

        let conv1d = if config.use_conv {
            Some(
                Conv1dConfig::new(d_inner, d_inner, config.conv_kernel)
                    .with_groups(d_inner)
                    .with_padding(burn::nn::PaddingConfig1d::Valid)
                    .init(device),
            )
        } else {
            None
        };

        let dt_proj = LinearConfig::new(d_inner, n_heads).init(device);
        let lambda_proj = LinearConfig::new(d_inner, n_heads).init(device);
        let theta_proj = LinearConfig::new(d_inner, n_heads * (d_state / 2)).init(device);

        let b_proj = LinearConfig::new(d_inner, n_heads * mimo_rank * d_state).init(device);
        let c_proj = LinearConfig::new(d_inner, n_heads * mimo_rank * d_state).init(device);

        let b_bias = Tensor::zeros([n_heads, mimo_rank, d_state], device);
        let c_bias = Tensor::zeros([n_heads, mimo_rank, d_state], device);

        let a_re = Tensor::random(
            [n_heads, d_state],
            Distribution::Uniform(-1.0, -0.1),
            device,
        );
        let a_im = Tensor::random(
            [n_heads, d_state],
            Distribution::Uniform(0.0, PI as f64 * 2.0),
            device,
        );

        let d = Tensor::ones([d_inner], device);
        let norm = RmsNormConfig::new(d_inner).init(device);

        Self {
            in_proj,
            conv1d,
            out_proj,
            dt_proj,
            lambda_proj,
            theta_proj,
            b_proj,
            c_proj,
            b_bias: Param::from_tensor(b_bias),
            c_bias: Param::from_tensor(c_bias),
            a_re: Param::from_tensor(a_re),
            a_im: Param::from_tensor(a_im),
            d: Param::from_tensor(d),
            norm,
            d_inner,
            d_state,
            n_heads,
            mimo_rank,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, _, _] = x.dims();
        let projected = self.in_proj.forward(x);
        let mut chunks = projected.chunk(2, 2);
        let u_orig = chunks.remove(0);
        let evo_gate = chunks.remove(0);

        let mut u = u_orig;
        if let Some(conv) = &self.conv1d {
            let kernel_size = conv.weight.dims()[2];
            u = u.swap_dims(1, 2);
            u = Tensor::cat(
                vec![
                    Tensor::zeros([batch, self.d_inner, kernel_size - 1], &u.device()),
                    u,
                ],
                2,
            );
            u = conv.forward(u);
            u = u.swap_dims(1, 2);
        }

        let u_silu = burn::tensor::activation::silu(u);
        let delta = burn::tensor::activation::softplus(self.dt_proj.forward(u_silu.clone()), 1.0);
        let lambda = burn::tensor::activation::sigmoid(self.lambda_proj.forward(u_silu.clone()));
        let theta = self.theta_proj.forward(u_silu.clone());
        let b = self.b_proj.forward(u_silu.clone());
        let c = self.c_proj.forward(u_silu.clone());

        let y_ssm = self.selective_scan(u_silu.clone(), delta, lambda, theta, b, c);
        let y_res = y_ssm + u_silu * self.d.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);
        let y = self.norm.forward(y_res);
        let y = y * burn::tensor::activation::silu(evo_gate);
        self.out_proj.forward(y)
    }

    /// Selective scan using complex state-space dynamics and parallel scan
    pub fn selective_scan(
        &self,
        u: Tensor<B, 3>,
        delta: Tensor<B, 3>,
        lambda: Tensor<B, 3>,
        theta: Tensor<B, 3>,
        b: Tensor<B, 3>,
        c: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _] = u.dims();
        let (n_heads, d_state, mimo_rank) = (self.n_heads, self.d_state, self.mimo_rank);
        let d_head = self.d_inner / n_heads;

        let u_heads = u.reshape([batch, seq_len, n_heads, d_head]);

        let b_bias_expanded = self
            .b_bias
            .val()
            .unsqueeze_dim::<4>(0)
            .unsqueeze_dim::<5>(0);
        let b = b.reshape([batch, seq_len, n_heads, mimo_rank, d_state]) + b_bias_expanded;

        let c_bias_expanded = self
            .c_bias
            .val()
            .unsqueeze_dim::<4>(0)
            .unsqueeze_dim::<5>(0);
        let c = c.reshape([batch, seq_len, n_heads, mimo_rank, d_state]) + c_bias_expanded;

        let dt = delta.clone().unsqueeze_dim::<4>(3);
        let da_re =
            (self.a_re.val().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0) * dt.clone()).exp();
        let da_im = self.a_im.val().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0) * dt;

        let theta_rot = theta.reshape([batch, seq_len, n_heads, d_state / 2]);
        let angle = da_im.slice([0..batch, 0..seq_len, 0..n_heads, 0..d_state / 2]) + theta_rot;
        let cos = angle.clone().cos().unsqueeze_dim::<5>(4);
        let sin = angle.sin().unsqueeze_dim::<5>(4);

        let da_re_half = da_re
            .clone()
            .slice([0..batch, 0..seq_len, 0..n_heads, 0..d_state / 2])
            .unsqueeze_dim::<5>(4);
        let a00 = da_re_half.clone() * cos.clone();
        let a01 = -(da_re_half.clone() * sin.clone());
        let a10 = da_re_half.clone() * sin;
        let a11 = da_re_half * cos;

        let u_mimo = u_heads.reshape([batch, seq_len, n_heads, mimo_rank, d_head / mimo_rank]);
        let bx = b.swap_dims(3, 4).matmul(u_mimo);

        let gamma = (delta.clone() * lambda.clone())
            .unsqueeze_dim::<4>(3)
            .unsqueeze_dim::<5>(4);
        let beta = (delta * (Tensor::ones_like(&lambda) - lambda))
            .unsqueeze_dim::<4>(3)
            .unsqueeze_dim::<5>(4)
            * da_re.unsqueeze_dim::<5>(4);

        let d_head_mimo = d_head / mimo_rank;
        let mut bx_prev = bx.clone().slice([
            0..batch,
            0..seq_len - 1,
            0..n_heads,
            0..d_state,
            0..d_head_mimo,
        ]);
        bx_prev = Tensor::cat(
            vec![
                Tensor::zeros([batch, 1, n_heads, d_state, d_head_mimo], &bx.device()),
                bx_prev,
            ],
            1,
        );
        let w = gamma * bx + beta * bx_prev;

        let w0 = w.clone().slice([
            0..batch,
            0..seq_len,
            0..n_heads,
            0..d_state / 2,
            0..d_head_mimo,
        ]);
        let w1 = w.slice([
            0..batch,
            0..seq_len,
            0..n_heads,
            d_state / 2..d_state,
            0..d_head_mimo,
        ]);

        let (h_re, h_im) = self.parallel_scan(a00, a01, a10, a11, w0, w1);
        let h = Tensor::cat(vec![h_re, h_im], 3);

        c.matmul(h).reshape([batch, seq_len, self.d_inner])
    }

    /// Parallel prefix scan for O(log T) complexity
    fn parallel_scan(
        &self,
        mut a00: Tensor<B, 5>,
        mut a01: Tensor<B, 5>,
        mut a10: Tensor<B, 5>,
        mut a11: Tensor<B, 5>,
        mut w0: Tensor<B, 5>,
        mut w1: Tensor<B, 5>,
    ) -> (Tensor<B, 5>, Tensor<B, 5>) {
        let [batch, seq_len, n_heads, dim4, a_dim5] = a00.dims();
        let [_b, _s, _n, _d4, w_dim5] = w0.dims();
        let mut offset = 1;
        while offset < seq_len {
            let left_range = 0..(seq_len - offset);
            let right_range = offset..seq_len;

            let r00 = a00.clone().slice([
                0..batch,
                right_range.clone(),
                0..n_heads,
                0..dim4,
                0..a_dim5,
            ]);
            let r01 = a01.clone().slice([
                0..batch,
                right_range.clone(),
                0..n_heads,
                0..dim4,
                0..a_dim5,
            ]);
            let r10 = a10.clone().slice([
                0..batch,
                right_range.clone(),
                0..n_heads,
                0..dim4,
                0..a_dim5,
            ]);
            let r11 = a11.clone().slice([
                0..batch,
                right_range.clone(),
                0..n_heads,
                0..dim4,
                0..a_dim5,
            ]);

            let l00 =
                a00.clone()
                    .slice([0..batch, left_range.clone(), 0..n_heads, 0..dim4, 0..a_dim5]);
            let l01 =
                a01.clone()
                    .slice([0..batch, left_range.clone(), 0..n_heads, 0..dim4, 0..a_dim5]);
            let l10 =
                a10.clone()
                    .slice([0..batch, left_range.clone(), 0..n_heads, 0..dim4, 0..a_dim5]);
            let l11 =
                a11.clone()
                    .slice([0..batch, left_range.clone(), 0..n_heads, 0..dim4, 0..a_dim5]);

            let rw0 = w0.clone().slice([
                0..batch,
                right_range.clone(),
                0..n_heads,
                0..dim4,
                0..w_dim5,
            ]);
            let rw1 = w1.clone().slice([
                0..batch,
                right_range.clone(),
                0..n_heads,
                0..dim4,
                0..w_dim5,
            ]);
            let lw0 =
                w0.clone()
                    .slice([0..batch, left_range.clone(), 0..n_heads, 0..dim4, 0..w_dim5]);
            let lw1 =
                w1.clone()
                    .slice([0..batch, left_range.clone(), 0..n_heads, 0..dim4, 0..w_dim5]);

            let n00 = r00.clone() * l00.clone() + r01.clone() * l10.clone();
            let n01 = r00.clone() * l01.clone() + r01.clone() * l11.clone();
            let n10 = r10.clone() * l00.clone() + r11.clone() * l10.clone();
            let n11 = r10.clone() * l01.clone() + r11.clone() * l11.clone();
            let nw0 = r00 * lw0.clone() + r01 * lw1.clone() + rw0;
            let nw1 = r10 * lw0 + r11 * lw1 + rw1;

            a00 = Tensor::cat(
                vec![
                    a00.slice([0..batch, 0..offset, 0..n_heads, 0..dim4, 0..a_dim5]),
                    n00,
                ],
                1,
            );
            a01 = Tensor::cat(
                vec![
                    a01.slice([0..batch, 0..offset, 0..n_heads, 0..dim4, 0..a_dim5]),
                    n01,
                ],
                1,
            );
            a10 = Tensor::cat(
                vec![
                    a10.slice([0..batch, 0..offset, 0..n_heads, 0..dim4, 0..a_dim5]),
                    n10,
                ],
                1,
            );
            a11 = Tensor::cat(
                vec![
                    a11.slice([0..batch, 0..offset, 0..n_heads, 0..dim4, 0..a_dim5]),
                    n11,
                ],
                1,
            );
            w0 = Tensor::cat(
                vec![
                    w0.slice([0..batch, 0..offset, 0..n_heads, 0..dim4, 0..w_dim5]),
                    nw0,
                ],
                1,
            );
            w1 = Tensor::cat(
                vec![
                    w1.slice([0..batch, 0..offset, 0..n_heads, 0..dim4, 0..w_dim5]),
                    nw1,
                ],
                1,
            );
            offset *= 2;
        }
        (w0, w1)
    }

    /// Sequential forward step for autoregressive inference
    pub fn forward_step(
        &self,
        x: Tensor<B, 2>,
        prev_h: Tensor<B, 4>,
        prev_bx: Option<Tensor<B, 4>>,
        conv_state: Option<Tensor<B, 3>>,
    ) -> (
        Tensor<B, 2>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Option<Tensor<B, 3>>,
    ) {
        let projected = self.in_proj.forward(x);
        let mut chunks = projected.chunk(2, 1);
        let u_orig = chunks.remove(0);
        let evo_gate = chunks.remove(0);

        let (u_conv, next_conv_state) = if let Some(conv) = &self.conv1d {
            let [batch, d_inner] = u_orig.dims();
            let kernel_size = conv.weight.dims()[2];
            let current_conv_state = conv_state.unwrap_or_else(|| {
                Tensor::zeros([batch, d_inner, kernel_size - 1], &u_orig.device())
            });
            let x_conv = Tensor::cat(vec![current_conv_state, u_orig.unsqueeze_dim::<3>(2)], 2);
            (
                conv.forward(x_conv.clone()).squeeze::<2>(2),
                Some(x_conv.slice([0..batch, 0..d_inner, 1..kernel_size])),
            )
        } else {
            (u_orig, None)
        };

        let u_silu = burn::tensor::activation::silu(u_conv);
        let delta = burn::tensor::activation::softplus(self.dt_proj.forward(u_silu.clone()), 1.0);
        let lambda = burn::tensor::activation::sigmoid(self.lambda_proj.forward(u_silu.clone()));
        let theta = self.theta_proj.forward(u_silu.clone());
        let [batch, _] = u_silu.dims();
        let (n_heads, d_state, mimo_rank) = (self.n_heads, self.d_state, self.mimo_rank);
        let d_head = self.d_inner / n_heads;

        let dt_t = delta;
        let la_t = lambda;
        let dt_u = dt_t.clone().unsqueeze_dim::<3>(2);

        let da_re = (self.a_re.val().unsqueeze_dim::<3>(0) * dt_u.clone()).exp();
        let da_im = self.a_im.val().unsqueeze_dim::<3>(0) * dt_u;

        let theta_rot = theta.reshape([batch, n_heads, d_state / 2]);
        let angle = da_im.slice([0..batch, 0..n_heads, 0..d_state / 2]) + theta_rot;
        let h_rot = self.rotate_state(prev_h, angle);

        let u_mimo = u_silu
            .clone()
            .reshape([batch, n_heads, mimo_rank, d_head / mimo_rank]);
        let current_bx = (self
            .b_proj
            .forward(u_silu.clone())
            .reshape([batch, n_heads, mimo_rank, d_state])
            + self.b_bias.val().unsqueeze_dim::<4>(0))
        .swap_dims(2, 3)
        .matmul(u_mimo);

        let gamma_t = (dt_t.clone() * la_t.clone())
            .unsqueeze_dim::<3>(2)
            .unsqueeze_dim::<4>(3);
        let beta_t = (dt_t * (Tensor::ones_like(&la_t) - la_t))
            .unsqueeze_dim::<3>(2)
            .unsqueeze_dim::<4>(3)
            * da_re.clone().unsqueeze_dim::<4>(3);

        let da_re_half = da_re.clone().slice([0..batch, 0..n_heads, 0..d_state / 2]);
        let da_re_for_h = Tensor::cat(vec![da_re_half.clone(), da_re_half], 2);

        let h_next = da_re_for_h.unsqueeze_dim::<4>(3) * h_rot
            + gamma_t * current_bx.clone()
            + beta_t * prev_bx.unwrap_or_else(|| Tensor::zeros_like(&current_bx));
        let c_rs = self
            .c_proj
            .forward(u_silu.clone())
            .reshape([batch, n_heads, mimo_rank, d_state])
            + self.c_bias.val().unsqueeze_dim::<4>(0);
        let y_ssm = c_rs.matmul(h_next.clone()).reshape([batch, self.d_inner]);
        let y_res = y_ssm + u_silu * self.d.val().unsqueeze_dim::<2>(0);
        let y = self.norm.forward(y_res) * burn::tensor::activation::silu(evo_gate);
        (
            self.out_proj.forward(y),
            h_next,
            current_bx,
            next_conv_state,
        )
    }

    /// Complex rotation in state space
    fn rotate_state(&self, h: Tensor<B, 4>, angle: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch, n_heads, d_state, d_head_mimo] = h.dims();
        let cos = angle.clone().cos().unsqueeze_dim::<4>(3);
        let sin = angle.sin().unsqueeze_dim::<4>(3);
        let h_re = h
            .clone()
            .slice([0..batch, 0..n_heads, 0..d_state / 2, 0..d_head_mimo]);
        let h_im = h.slice([0..batch, 0..n_heads, d_state / 2..d_state, 0..d_head_mimo]);
        Tensor::cat(
            vec![
                h_re.clone() * cos.clone() - h_im.clone() * sin.clone(),
                h_re * sin + h_im * cos,
            ],
            2,
        )
    }
}
