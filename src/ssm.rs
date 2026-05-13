use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};
use core::f32::consts::PI;

/// Configuration for the SSM (State Space Model) block.
///
/// This implements a Mamba-style selective state space model with complex rotation
/// dynamics. The architecture follows the design principles of:
/// - **Mamba** (Gu & Dao, 2023): Selective scan with input-dependent parameters
/// - **Mamba-3** (Lahoti et al., 2026): Complex rotation in state space with MIMO rank
///
/// # Key parameters
/// - `d_model`: Model dimension (input/output)
/// - `d_state`: State space dimension (must be even for complex rotation)
/// - `expand`: Inner dimension multiplier; `d_inner = d_model * expand`
/// - `n_heads`: Number of attention heads; `d_inner` must be divisible by `n_heads`
/// - `mimo_rank`: Multi-input multi-output rank; each head outputs `mimo_rank` values
/// - `use_conv`: Whether to apply causal 1D convolution for local context
/// - `conv_kernel`: Kernel size for the causal convolution
///
/// # Constraints
/// - `d_inner = d_model * expand` must be divisible by `n_heads`
/// - `d_head = d_inner / n_heads` must be divisible by `mimo_rank`
/// - `d_state` must be even (for complex rotation decomposition)
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

impl SsmConfig {
    /// Validate the configuration parameters.
    ///
    /// Returns `Ok(())` if all constraints are satisfied, or an error
    /// describing the first validation failure encountered.
    ///
    /// # Constraints
    /// - `d_inner = d_model * expand` must be divisible by `n_heads`
    /// - `d_head = d_inner / n_heads` must be divisible by `mimo_rank`
    /// - `d_state` must be even (for complex rotation decomposition)
    /// - `mimo_rank` must be greater than 0
    /// - `conv_kernel` must be greater than 0 when `use_conv` is true
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.mimo_rank == 0 {
            return Err(crate::error::ModelError::Config {
                message: "mimo_rank must be greater than 0".into(),
            });
        }
        if self.n_heads == 0 {
            return Err(crate::error::ModelError::Config {
                message: "n_heads must be greater than 0".into(),
            });
        }
        if self.d_model == 0 {
            return Err(crate::error::ModelError::Config {
                message: "d_model must be greater than 0".into(),
            });
        }
        if self.d_state == 0 {
            return Err(crate::error::ModelError::Config {
                message: "d_state must be greater than 0".into(),
            });
        }
        if !self.d_state.is_multiple_of(2) {
            return Err(crate::error::ModelError::Config {
                message: format!(
                    "d_state must be even for complex rotation (got {})",
                    self.d_state
                ),
            });
        }

        let d_inner = self.d_model * self.expand;
        if !d_inner.is_multiple_of(self.n_heads) {
            return Err(crate::error::ModelError::Config {
                message: format!(
                    "d_inner ({}) must be divisible by n_heads ({})",
                    d_inner, self.n_heads
                ),
            });
        }

        let d_head = d_inner / self.n_heads;
        if !d_head.is_multiple_of(self.mimo_rank) {
            return Err(crate::error::ModelError::Config {
                message: format!(
                    "d_head ({}) must be divisible by mimo_rank ({})",
                    d_head, self.mimo_rank
                ),
            });
        }

        if self.use_conv && self.conv_kernel == 0 {
            return Err(crate::error::ModelError::Config {
                message: "conv_kernel must be greater than 0 when use_conv is true".into(),
            });
        }

        Ok(())
    }
}

/// A selective state space model block with complex rotation dynamics.
///
/// This block implements the core SSM computation following Mamba-style architecture:
/// 1. Input projection splits into two branches: the main branch (u) and an evolution gate (evo_gate)
/// 2. Causal 1D convolution captures local context (optional)
/// 3. SiLU activation produces the input to the selective scan
/// 4. Input-dependent parameters (delta, lambda, theta, B, C) are computed via projections
/// 5. Selective scan applies complex rotation state-space dynamics
/// 6. Residual connection with learnable skip (D parameter)
/// 7. RMSNorm and gated output projection (SwiGLU-style gating)
///
/// # Parallel vs Sequential modes
/// - [`forward()`](Self::forward): Parallel scan for O(L log L) training
/// - [`forward_step()`](Self::forward_step): Sequential step for O(1) autoregressive inference
///
/// These two modes are mathematically equivalent, verified by the equivalence test.
#[derive(Module, Debug)]
pub struct SsmBlock<B: Backend> {
    /// Input projection: maps `d_model` → `2 * d_inner` (split into u and evo_gate)
    pub in_proj: Linear<B>,
    /// Causal 1D convolution for local context aggregation (optional)
    pub conv1d: Option<Conv1d<B>>,
    /// Output projection: maps `d_inner` → `d_model`
    pub out_proj: Linear<B>,
    /// Projects SiLU-activated input to step size (delta) per head: `d_inner` → `n_heads`
    pub dt_proj: Linear<B>,
    /// Projects SiLU-activated input to decay gate (lambda) per head: `d_inner` → `n_heads`
    pub lambda_proj: Linear<B>,
    /// Projects SiLU-activated input to rotation angles (theta) per head: `d_inner` → `n_heads * (d_state/2)`
    pub theta_proj: Linear<B>,
    /// Projects SiLU-activated input to B matrix input: `d_inner` → `n_heads * mimo_rank * d_state`
    pub b_proj: Linear<B>,
    /// Projects SiLU-activated input to C matrix output: `d_inner` → `n_heads * mimo_rank * d_state`
    pub c_proj: Linear<B>,
    /// Bias term for B input projection: shape `[n_heads, mimo_rank, d_state]`
    pub b_bias: Param<Tensor<B, 3>>,
    /// Bias term for C output projection: shape `[n_heads, mimo_rank, d_state]`
    pub c_bias: Param<Tensor<B, 3>>,
    /// Real part of diagonal state transition matrix A: shape `[n_heads, d_state]`
    /// Initialized uniformly in [-1.0, -0.1] to ensure damping (exponential decay)
    pub a_re: Param<Tensor<B, 2>>,
    /// Imaginary part of diagonal state transition matrix A: shape `[n_heads, d_state]`
    /// Initialized uniformly in [0, 2π] to enable complex rotation dynamics
    pub a_im: Param<Tensor<B, 2>>,
    /// Skip connection parameter D: shape `[d_inner]`
    /// Enables direct input-to-output pathway (D-skip in SSM formulation)
    pub d: Param<Tensor<B, 1>>,
    /// RMS normalization applied after SSM output + residual
    pub norm: RmsNorm<B>,
    /// Inner dimension: `d_model * expand`
    pub d_inner: usize,
    /// State space dimension
    pub d_state: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Multi-input multi-output rank
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
        assert!(mimo_rank > 0, "mimo_rank must be greater than 0");
        let d_head = d_inner / n_heads;
        assert!(
            d_head.is_multiple_of(mimo_rank),
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

        let dt_proj = LinearConfig::new(d_inner, n_heads)
            .with_bias(true)
            .init(device);
        // Initialize dt_proj bias to a small negative value so that initial delta is small
        let dt_bias = Tensor::full([n_heads], -3.1, device);
        let dt_proj = Linear {
            weight: dt_proj.weight,
            bias: Some(Param::from_tensor(dt_bias)),
        };

        let lambda_proj = LinearConfig::new(d_inner, n_heads).init(device);
        let theta_proj = LinearConfig::new(d_inner, n_heads * (d_state / 2)).init(device);

        let b_proj = LinearConfig::new(d_inner, n_heads * mimo_rank * d_state).init(device);
        let c_proj = LinearConfig::new(d_inner, n_heads * mimo_rank * d_state).init(device);

        let b_bias = Tensor::zeros([n_heads, mimo_rank, d_state], device);
        let c_bias = Tensor::zeros([n_heads, mimo_rank, d_state], device);

        // State transition matrix initialization:
        // - a_re (real part): initialized in [-1.0, -0.1] to ensure damping.
        //   Values near -1.0 produce rapid decay (stable but potentially too fast),
        //   while values near -0.1 allow slower exponential decay.
        //   During training, softplus(dt) * sigmoid(lambda) scaling means the
        //   effective decay rate is modulated by the input-dependent step size and gate.
        // - a_im (imaginary part): initialized in [0, 2π] to enable oscillatory dynamics.
        //   This allows the model to capture periodic and quasi-periodic patterns
        //   in the state space, which is essential for modeling cyclical phenomena.
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

    /// Forward pass using parallel scan for O(L log L) training complexity.
    ///
    /// Processes the entire sequence at once using a parallel prefix scan algorithm,
    /// enabling efficient gradient computation during backpropagation.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape `[batch, seq_len, d_model]`
    ///
    /// # Returns
    /// Output tensor of shape `[batch, seq_len, d_model]`
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

    /// Selective scan using complex state-space dynamics and parallel scan.
    ///
    /// Implements the core SSM recurrence with input-dependent parameters:
    ///
    /// ```text
    /// h_t = (λ_t · Δ_t) · B_t · u_t + (1 - λ_t) · Δ_t · h_{t-1} · A_t + β_t
    /// y_t = C_t · h_t
    /// ```
    ///
    /// where:
    /// - `A_t = exp(a_re · Δ_t) · R(θ_t)` is the complex rotation state transition
    /// - `Δ_t` (delta) is the input-dependent step size
    /// - `λ_t` (lambda) is the input-dependent exponential gate controlling
    ///   the balance between new information and recurrent state
    /// - `θ_t` (theta) is the input-dependent rotation angle
    /// - `β_t` incorporates the previous timestep's input with decay
    ///
    /// The computation uses a parallel prefix scan for O(L log L) complexity.
    ///
    /// # Arguments
    /// * `u` - SiLU-activated input: `[batch, seq_len, d_inner]`
    /// * `delta` - Step size per head: `[batch, seq_len, n_heads]`
    /// * `lambda` - Decay gate per head: `[batch, seq_len, n_heads]`
    /// * `theta` - Rotation angles per head: `[batch, seq_len, n_heads * (d_state/2)]`
    /// * `b` - Input-dependent B matrix: `[batch, seq_len, n_heads * mimo_rank * d_state]`
    /// * `c` - Input-dependent C matrix: `[batch, seq_len, n_heads * mimo_rank * d_state]`
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
        let d_head_mimo = d_head / mimo_rank;

        let u_mimo = u.reshape([batch, seq_len, n_heads, mimo_rank, d_head_mimo]);

        let b = b.reshape([batch, seq_len, n_heads, mimo_rank, d_state])
            + self
                .b_bias
                .val()
                .unsqueeze_dim::<4>(0)
                .unsqueeze_dim::<5>(0);

        let c = c.reshape([batch, seq_len, n_heads, mimo_rank, d_state])
            + self
                .c_bias
                .val()
                .unsqueeze_dim::<4>(0)
                .unsqueeze_dim::<5>(0);

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

        // Efficient batched matmul by flattening batch, seq, head
        let b_flat = b
            .swap_dims(3, 4)
            .reshape([batch * seq_len * n_heads, d_state, mimo_rank]);
        let u_flat = u_mimo.reshape([batch * seq_len * n_heads, mimo_rank, d_head_mimo]);
        let bx = b_flat
            .matmul(u_flat)
            .reshape([batch, seq_len, n_heads, d_state, d_head_mimo]);

        let delta_lambda = (delta.clone() * lambda.clone())
            .unsqueeze_dim::<4>(3)
            .unsqueeze_dim::<5>(4);
        let delta_not_lambda = (delta * (Tensor::ones_like(&lambda) - lambda))
            .unsqueeze_dim::<4>(3)
            .unsqueeze_dim::<5>(4);
        let beta = delta_not_lambda * da_re.unsqueeze_dim::<5>(4);

        let mut bx_prev = bx.clone().slice([0..batch, 0..seq_len - 1]);
        bx_prev = Tensor::cat(
            vec![
                Tensor::zeros([batch, 1, n_heads, d_state, d_head_mimo], &bx.device()),
                bx_prev,
            ],
            1,
        );
        let w = delta_lambda * bx + beta * bx_prev;

        let w0 = w
            .clone()
            .slice([0..batch, 0..seq_len, 0..n_heads, 0..d_state / 2]);
        let w1 = w.slice([0..batch, 0..seq_len, 0..n_heads, d_state / 2..d_state]);

        let (h_re, h_im) = self.parallel_scan(a00, a01, a10, a11, w0, w1);
        let h = Tensor::cat(vec![h_re, h_im], 3);

        let c_flat = c.reshape([batch * seq_len * n_heads, mimo_rank, d_state]);
        let h_flat = h.reshape([batch * seq_len * n_heads, d_state, d_head_mimo]);
        c_flat
            .matmul(h_flat)
            .reshape([batch, seq_len, self.d_inner])
    }

    /// Parallel prefix scan for O(log T) complexity.
    ///
    /// Implements the Blelloch-style parallel scan for associative operations
    /// on the 2×2 complex rotation matrices. Each step composes pairs of
    /// (state transition, contribution) tuples:
    ///
    /// ```text
    /// A_new = A_right · A_left
    /// w_new = A_right · w_left + w_right
    /// ```
    ///
    /// The scan operates on the real and imaginary parts of the complex
    /// rotation separately, with the transition matrix represented as:
    ///
    /// ```text
    /// A = | a00  a01 |   = | Re·cos  -Re·sin |
    ///     | a10  a11 |     | Re·sin   Re·cos |
    /// ```
    ///
    /// where `Re = exp(a_re · Δ)` and the angle comes from `(a_im · Δ + θ)`.
    fn parallel_scan(
        &self,
        mut a00: Tensor<B, 5>,
        mut a01: Tensor<B, 5>,
        mut a10: Tensor<B, 5>,
        mut a11: Tensor<B, 5>,
        mut w0: Tensor<B, 5>,
        mut w1: Tensor<B, 5>,
    ) -> (Tensor<B, 5>, Tensor<B, 5>) {
        let [batch, seq_len, _n_heads, _dim4, _dim5] = a00.dims();
        let mut offset = 1;

        while offset < seq_len {
            let num_pairs = seq_len - offset;

            // Current states at position i (right)
            let r_range = offset..seq_len;
            let l_range = 0..num_pairs;

            let r00 = a00.clone().slice([0..batch, r_range.clone()]);
            let r01 = a01.clone().slice([0..batch, r_range.clone()]);
            let r10 = a10.clone().slice([0..batch, r_range.clone()]);
            let r11 = a11.clone().slice([0..batch, r_range.clone()]);
            let rw0 = w0.clone().slice([0..batch, r_range.clone()]);
            let rw1 = w1.clone().slice([0..batch, r_range.clone()]);

            // States at position i-offset (left)
            let l00 = a00.clone().slice([0..batch, l_range.clone()]);
            let l01 = a01.clone().slice([0..batch, l_range.clone()]);
            let l10 = a10.clone().slice([0..batch, l_range.clone()]);
            let l11 = a11.clone().slice([0..batch, l_range.clone()]);
            let lw0 = w0.clone().slice([0..batch, l_range.clone()]);
            let lw1 = w1.clone().slice([0..batch, l_range.clone()]);

            // Compose matrix multiplications: (R_a * L_a) and (R_a * L_w + R_w)
            let n00 = r00.clone() * l00.clone() + r01.clone() * l10.clone();
            let n01 = r00.clone() * l01.clone() + r01.clone() * l11.clone();
            let n10 = r10.clone() * l00.clone() + r11.clone() * l10.clone();
            let n11 = r10.clone() * l01.clone() + r11.clone() * l11.clone();
            let nw0 = r00 * lw0.clone() + r01 * lw1.clone() + rw0;
            let nw1 = r10 * lw0 + r11 * lw1 + rw1;

            // In-place update of ranges to avoid heavy concatenation of the whole tensor at each step
            // Burn tensors are immutable, but we update the binding.
            // We only replace the part that was updated (from 'offset' onwards)
            a00 = Tensor::cat(vec![a00.slice([0..batch, 0..offset]), n00], 1);
            a01 = Tensor::cat(vec![a01.slice([0..batch, 0..offset]), n01], 1);
            a10 = Tensor::cat(vec![a10.slice([0..batch, 0..offset]), n10], 1);
            a11 = Tensor::cat(vec![a11.slice([0..batch, 0..offset]), n11], 1);
            w0 = Tensor::cat(vec![w0.slice([0..batch, 0..offset]), nw0], 1);
            w1 = Tensor::cat(vec![w1.slice([0..batch, 0..offset]), nw1], 1);

            offset *= 2;
        }
        (w0, w1)
    }

    /// Sequential forward step for autoregressive inference.
    ///
    /// Processes a single timestep with O(1) state updates, suitable for
    /// real-time/streaming inference. Maintains an explicit hidden state
    /// (`h`), previous input contribution (`prev_bx`), and optional
    /// convolution state for causal conv1d.
    ///
    /// This method is mathematically equivalent to [`forward()`](Self::forward)
    /// but processes one step at a time, making it ideal for:
    /// - Autoregressive generation
    /// - Real-time inference on streaming data
    /// - Step-by-step world model prediction in reinforcement learning
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape `[batch, d_model]` (single timestep)
    /// * `prev_h` - Previous hidden state of shape `[batch, n_heads, d_state, d_head_mimo]`
    /// * `prev_bx` - Previous B·x contribution (optional, `None` for first step)
    /// * `conv_state` - Previous convolution state (optional, `None` if no conv or first step)
    ///
    /// # Returns
    /// A tuple of:
    /// - Output tensor of shape `[batch, d_model]`
    /// - Updated hidden state
    /// - Current B·x contribution (to pass to next step as `prev_bx`)
    /// - Updated conv state (if conv is enabled)
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
            let u_conv = conv.forward(x_conv.clone());
            let [batch, d_inner, _seq] = u_conv.dims();
            (
                u_conv.reshape([batch, d_inner]),
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

    /// Apply complex rotation to the hidden state.
    ///
    /// The state `h` is split into real and imaginary halves along the
    /// state dimension. The rotation is applied as:
    ///
    /// ```text
    /// h_re' = h_re · cos(angle) - h_im · sin(angle)
    /// h_im' = h_re · sin(angle) + h_im · cos(angle)
    /// ```
    ///
    /// This enables oscillatory dynamics in the state space, which is
    /// crucial for modeling periodic and quasi-periodic phenomena.
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
