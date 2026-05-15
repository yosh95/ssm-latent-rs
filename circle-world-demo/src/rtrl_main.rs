#![recursion_limit = "256"]
/// Circle World Demo — Full RTRL (Real-Time Recurrent Learning) variant.
///
/// **Zero backpropagation through time. Zero cross-component backprop.**
///
/// Architecture:
///   Encoder: 2 → 16 → 64   (Forward-Forward, per-layer goodness)
///   SSM:     64-dim latent dynamics (1-step truncated RTRL, no BPTT)
///   Decoder: 64 → 16 → 2   (Forward-Forward, per-layer goodness)
///   ActionEnc + Fusion:    (1-step BP, small)
///
/// Gradient flow:
///   - Encoder: FF goodness(pos, neg) + local recon MSE — no BP between layers
///   - SSM:     At each timestep t, h_{t-1} is detached. Loss = MSE(ẑ(t), z(t)).
///     Gradient only flows through f(h_{t-1}^detach, x_t, θ) → ∂loss/∂θ.
///     NO gradient through ∂h_t/∂h_{t-1}.
///   - Decoder: FF goodness(pos, neg) + local recon MSE — no BP between layers
///
/// Key insight: This is equivalent to "truncated BPTT with horizon 1" +
/// "Forward-Forward encoders". Every parameter update is purely local.
use burn::module::AutodiffModule;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::Tensor;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use ssm_latent_model::ff_model::FfEncoder;
use ssm_latent_model::rtrl::RtrlAccumulator;
use ssm_latent_model::ssm::{MultiScaleSsmBlock, MultiScaleSsmConfig, MultiScaleState, SsmConfig};

// --- Backend Selection ---
#[cfg(feature = "wgpu")]
type MyBackend = burn::backend::Wgpu;
#[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
type MyBackend = burn::backend::NdArray;
#[cfg(all(not(feature = "wgpu"), not(feature = "ndarray")))]
type MyBackend = burn::backend::NdArray;

type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

// --- Full RTRL World Model ---

/// Complete world model with:
/// - FF encoder: obs → latent
/// - Multi-scale SSM: latent dynamics (RTRL, no BPTT)
/// - FF decoder: latent → obs
#[derive(Debug, burn::module::Module)]
struct RtrlWorldModel<B: burn::tensor::backend::Backend> {
    encoder: FfEncoder<B>,
    decoder: FfEncoder<B>,
    action_encoder: Linear<B>,
    fusion: Linear<B>,
    ssm: MultiScaleSsmBlock<B>,
    d_model: usize,
    n_layers: usize,
}

impl<B: burn::tensor::backend::Backend> RtrlWorldModel<B> {
    fn new(config: &SsmConfig, n_ssm_layers: usize, device: &B::Device) -> Self {
        let d_model = config.d_model;
        let encoder = FfEncoder::new(&[2, 16, d_model], device);
        let decoder = FfEncoder::new(&[d_model, 16, 2], device);
        let action_encoder = LinearConfig::new(2, d_model).init(device);
        let fusion = LinearConfig::new(d_model * 2, d_model).init(device);

        let mscale_config = MultiScaleSsmConfig::new(
            d_model,
            config.d_state,
            config.expand,
            config.n_heads,
            config.mimo_rank,
        )
        .with_n_layers(n_ssm_layers)
        .with_use_conv(true)
        .with_conv_kernel(config.conv_kernel);

        let ssm = MultiScaleSsmBlock::new(&mscale_config, device);

        Self {
            encoder,
            decoder,
            action_encoder,
            fusion,
            ssm,
            d_model,
            n_layers: n_ssm_layers,
        }
    }

    /// Encode observation to latent (FF, no BP between layers).
    fn encode(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        self.encoder.encode(obs)
    }

    /// Decode latent to observation (FF, no BP between layers).
    fn decode(&self, z: Tensor<B, 2>) -> Tensor<B, 2> {
        self.decoder.encode(z)
    }
}

// --- Visualization ---

fn draw_frame(
    title: &str,
    step: usize,
    left_point: Option<([f32; 2], char, &str)>,
    right_point: Option<([f32; 2], char, &str)>,
) {
    let width = 30;
    let height = 15;
    let mut grids = vec![vec![vec![' '; width]; height]; 2];
    let points = [left_point, right_point];

    for (g_idx, pt_opt) in points.iter().enumerate() {
        if let Some(([x, y], marker, _)) = pt_opt {
            let grid = &mut grids[g_idx];
            grid[height / 2].fill('-');
            for row in grid.iter_mut() {
                row[width / 2] = '|';
            }
            grid[height / 2][width / 2] = '+';
            let gx = (((x + 2.0) / 4.0) * (width as f32 - 1.0)) as i32;
            let gy = (((2.0 - y) / 4.0) * (height as f32 - 1.0)) as i32;
            if gx >= 0 && gx < width as i32 && gy >= 0 && gy < height as i32 {
                grid[gy as usize][gx as usize] = *marker;
            }
        }
    }

    println!("\x1B[H\x1B[2J");
    println!("=== {} (Step: {}) ===", title, step);
    if right_point.is_some() {
        let label_left = left_point.as_ref().map(|p| p.2).unwrap_or("");
        let label_right = right_point.as_ref().map(|p| p.2).unwrap_or("");
        println!(
            "  {:^width$}    {:^width$}",
            label_left,
            label_right,
            width = width
        );
        for (row_left, row_right) in grids[0].iter().zip(grids[1].iter()) {
            let s_left: String = row_left.iter().collect();
            let s_right: String = row_right.iter().collect();
            println!("  {}    {}", s_left, s_right);
        }
    } else if let Some((_, _, label)) = left_point {
        println!("  {:^width$}", label, width = width);
        for row in &grids[0] {
            let s: String = row.iter().collect();
            println!("  {}", s);
        }
    }
}

fn main() {
    // --- Config ---
    let config = SsmConfig::new(64, 16, 2, 4, 1);
    let n_ssm_layers = 2;
    let epochs = 80;
    let batch_size = 4;
    let seq_len = 32;
    let learning_rate = 1e-3;
    let ff_threshold = 3.0;

    #[cfg(feature = "wgpu")]
    let device = burn::backend::wgpu::WgpuDevice::default();
    #[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
    let device = burn::backend::ndarray::NdArrayDevice::default();
    #[cfg(all(not(feature = "wgpu"), not(feature = "ndarray")))]
    let device = burn::backend::ndarray::NdArrayDevice::default();

    let mut rng = StdRng::seed_from_u64(42);

    println!("===================================================================");
    println!("  🔮 Circle World — Full RTRL (Zero BPTT + Forward-Forward)");
    println!("===================================================================");
    println!();
    println!("Architecture:");
    println!("  Encoder:  2 → 16 → 64  (Forward-Forward, per-layer goodness)");
    println!(
        "  SSM:      {}-layer, 64-dim  (1-step RTRL, NO BPTT)",
        n_ssm_layers
    );
    println!("  Decoder:  64 → 16 → 2  (Forward-Forward, per-layer goodness)");
    println!("  ActionEnc + Fusion:    (1-step BP, small)");
    println!();
    println!("Gradient flow: ALL LOCAL — no BPTT, no cross-component BP");
    println!("  SSM: h(t-1) detached -- gradient only through f(h_detach, x, theta)");
    println!("  Encoder: FF goodness + recon MSE -- per-layer only");
    println!("  Decoder: FF goodness + recon MSE -- per-layer only");
    println!();

    // --- Build model ---
    let mut model = RtrlWorldModel::<MyAutodiffBackend>::new(&config, n_ssm_layers, &device);
    let mut optimizer =
        AdamConfig::new().init::<MyAutodiffBackend, RtrlWorldModel<MyAutodiffBackend>>();

    // --- Observe the circle ---
    std::thread::sleep(std::time::Duration::from_millis(500));
    println!("[Part 1: Observation]");
    for t in 1..=20 {
        let angle = (t as f32) * 0.3;
        draw_frame(
            "Observing the circle",
            t,
            Some(([angle.cos(), angle.sin()], 'X', "External World")),
            None,
        );
        std::thread::sleep(std::time::Duration::from_millis(80));
    }

    println!("\n[Part 2: RTRL Training (zero BPTT)]");

    // Show untrained state
    let model_valid = model.valid();
    run_inference(
        &model_valid,
        &config,
        n_ssm_layers,
        &device,
        "Epoch 0 (Untrained)",
    );
    std::thread::sleep(std::time::Duration::from_millis(1200));

    let d_model = config.d_model;
    let d_inner = config.d_model * config.expand;
    let d_head = d_inner / config.n_heads;
    let d_head_mimo = d_head / config.mimo_rank;
    let conv_kernel = config.conv_kernel;

    // --- Training Loop ---
    for epoch in 1..=epochs {
        let mut accum = RtrlAccumulator::new();

        for b in 0..batch_size {
            // Generate one episode of circle data
            let phase_shift = (b as f32) * 1.25;

            // Reset SSM state for this episode
            let mut ssm_state = MultiScaleState::<MyAutodiffBackend>::zeros(
                1, // single batch for step-wise processing
                n_ssm_layers,
                config.n_heads,
                config.d_state,
                d_head_mimo,
                true,
                d_inner,
                conv_kernel,
                &device,
            );

            for t in 0..seq_len {
                let time = (t as f32) * 0.3;
                let angle = time + phase_shift;
                let noise: f32 = rng.random_range(-0.02..0.02);

                // Current observation
                let obs = Tensor::<MyAutodiffBackend, 2>::from_data(
                    burn::tensor::TensorData::new(
                        vec![angle.cos() + noise, angle.sin() + noise],
                        [1, 2],
                    ),
                    &device,
                );

                // Action (tangential velocity)
                let act_noise: f32 = rng.random_range(-0.005..0.005);
                let action = Tensor::<MyAutodiffBackend, 2>::from_data(
                    burn::tensor::TensorData::new(
                        vec![
                            -0.1 * angle.sin() + act_noise,
                            0.1 * angle.cos() + act_noise,
                        ],
                        [1, 2],
                    ),
                    &device,
                );

                // Generate negative observation for FF encoder
                let neg_x: f32 = rng.random_range(-2.0..2.0);
                let neg_y: f32 = rng.random_range(-2.0..2.0);
                let neg_obs = Tensor::<MyAutodiffBackend, 2>::from_data(
                    burn::tensor::TensorData::new(vec![neg_x, neg_y], [1, 2]),
                    &device,
                );

                // ─── FF Encoder: encode current observation ───
                let pos_act = model.encoder.positive_pass(obs.clone());
                let z_t = pos_act.last().unwrap().clone(); // latent, detach'd from prev layers

                let neg_act = model.encoder.negative_pass(neg_obs.clone());

                // ─── FF Decoder: decode latent → observation ───
                let pos_dec_act = model.decoder.positive_pass(z_t.clone().detach());
                let reconstructed = pos_dec_act.last().unwrap().clone();

                // Random latent as negative for decoder
                let rand_z = Tensor::<MyAutodiffBackend, 2>::random(
                    [1, d_model],
                    burn::tensor::Distribution::Normal(0.0, 1.0),
                    &device,
                );
                let neg_dec_act = model.decoder.negative_pass(rand_z);

                // ─── SSM RTRL step: z(t) + action → ẑ(t+1) ───
                let a_enc = model.action_encoder.forward(action);
                let u_concat = Tensor::cat(vec![z_t.clone().detach(), a_enc], 1);
                let u = model.fusion.forward(u_concat);

                // RTRL: step through SSM with PREVIOUS state detached
                let (pred_z, new_ssm_state) =
                    ssm_latent_model::rtrl::ssm_step_detached(&model.ssm, u, &ssm_state);

                // Compute target: encode NEXT observation (if available)
                let mut ssm_loss_opt: Option<Tensor<MyAutodiffBackend, 1>> = None;
                if t < seq_len - 1 {
                    let next_time = ((t + 1) as f32) * 0.3;
                    let next_angle = next_time + phase_shift;
                    let next_noise: f32 = rng.random_range(-0.02..0.02);
                    let next_obs = Tensor::<MyAutodiffBackend, 2>::from_data(
                        burn::tensor::TensorData::new(
                            vec![next_angle.cos() + next_noise, next_angle.sin() + next_noise],
                            [1, 2],
                        ),
                        &device,
                    );
                    // Detach encoder output for target (SSM doesn't flow into encoder)
                    let next_z = model.encode(next_obs).detach();
                    let ssm_err = (pred_z.clone() - next_z)
                        .powf_scalar(2.0)
                        .mean()
                        .unsqueeze();
                    ssm_loss_opt = Some(ssm_err);
                }

                // ─── Compute all losses ───
                // FF Encoder loss
                let enc_losses =
                    model
                        .encoder
                        .compute_layer_losses(&pos_act, &neg_act, ff_threshold);
                let enc_loss: Tensor<MyAutodiffBackend, 1> = enc_losses.into_iter().fold(
                    Tensor::<MyAutodiffBackend, 1>::from_data([0.0f32], &device),
                    |acc, l| acc + l,
                );

                // FF Decoder loss
                let dec_losses =
                    model
                        .decoder
                        .compute_layer_losses(&pos_dec_act, &neg_dec_act, ff_threshold);
                let dec_loss: Tensor<MyAutodiffBackend, 1> = dec_losses.into_iter().fold(
                    Tensor::<MyAutodiffBackend, 1>::from_data([0.0f32], &device),
                    |acc, l| acc + l,
                );

                // Reconstruction loss (decoder only, detached latent)
                let recon_loss = (reconstructed - obs).powf_scalar(2.0).mean().unsqueeze();

                // Combine and backward
                let mut total = enc_loss.mul_scalar(0.5)
                    + dec_loss.mul_scalar(0.5)
                    + recon_loss.mul_scalar(1.0);

                if let Some(ssm_l) = ssm_loss_opt {
                    total = total + ssm_l.mul_scalar(2.0);
                }

                let loss_val: f32 = total.clone().into_data().as_slice::<f32>().unwrap()[0];
                accum.record(loss_val);

                let grads = total.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optimizer.step(learning_rate, model, grads);

                // Advance SSM state
                ssm_state = new_ssm_state;
            }
        }

        if epoch % 20 == 0 || epoch == 1 {
            println!(
                "Epoch {:4}: Avg Loss = {:.6} ({} steps)",
                epoch,
                accum.avg_loss(),
                accum.step_count,
            );

            let model_valid = model.valid();
            run_inference(
                &model_valid,
                &config,
                n_ssm_layers,
                &device,
                &format!("Training (Epoch {})", epoch),
            );
            std::thread::sleep(std::time::Duration::from_millis(400));
        }
    }

    // --- Part 3: Imagination (Pure SSM roll-out with RTRL model) ---
    println!("\n[Part 3: Pure Imagination — SSM Roll-out (Zero BPTT)]");
    println!("The Explorer closes its eyes and imagines the future.");
    std::thread::sleep(std::time::Duration::from_millis(1500));

    let model_valid = model.valid();
    run_imagination(
        &model_valid,
        &config,
        n_ssm_layers,
        &device,
        &format!("RTRL Model (Epoch {})", epochs),
    );
    std::thread::sleep(std::time::Duration::from_millis(1000));

    println!();
    println!("===================================================================");
    println!("  RTRL Achievement Summary");
    println!("===================================================================");
    println!("  Encoder:       Forward-Forward → NO BP between layers");
    println!("  Decoder:       Forward-Forward → NO BP between layers");
    println!("  SSM:           1-step RTRL -- NO BPTT (h_{{t-1}} detached)");
    println!("  Cross-comp:    All latent inputs detached → NO cross-BP");
    println!("  ----------------------------------------------");
    println!("  Total BPTT:    ZERO");
    println!("  Cross-comp BP: ZERO");
    println!("  Per-step BP:   Only within single SSM step");
    println!("===================================================================");
}

/// Inference: encode→decode for quality check.
fn run_inference<B: burn::tensor::backend::Backend>(
    model: &RtrlWorldModel<B>,
    _config: &SsmConfig,
    _n_ssm_layers: usize,
    device: &B::Device,
    title: &str,
) {
    println!("\n--- {} ---", title);
    for t in 1..=20 {
        let angle = (t as f32) * 0.3;
        let real_x = angle.cos();
        let real_y = angle.sin();

        let obs = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![real_x, real_y], [1, 2]),
            device,
        );
        let z = model.encode(obs);
        let decoded = model.decode(z);
        let pos = decoded.into_data();
        let pos_slice = pos.as_slice::<f32>().unwrap();

        draw_frame(
            title,
            t as usize,
            Some(([real_x, real_y], 'X', "Ground Truth")),
            Some(([pos_slice[0], pos_slice[1]], 'O', "RTRL Model")),
        );
        std::thread::sleep(std::time::Duration::from_millis(150));
    }
}

/// Pure imagination: SSM autoregressive roll-out.
fn run_imagination<B: burn::tensor::backend::Backend>(
    model: &RtrlWorldModel<B>,
    config: &SsmConfig,
    n_ssm_layers: usize,
    device: &B::Device,
    title: &str,
) {
    println!("\n--- {} ---", title);

    let d_inner = config.d_model * config.expand;
    let d_head = d_inner / config.n_heads;
    let d_head_mimo = d_head / config.mimo_rank;

    // Initialize from (1.0, 0.0)
    let initial_obs = Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(vec![1.0f32, 0.0], [1, 2]),
        device,
    );
    let mut current_z = model.encode(initial_obs);

    let mut ssm_state = MultiScaleState::<B>::zeros(
        1,
        n_ssm_layers,
        config.n_heads,
        config.d_state,
        d_head_mimo,
        true,
        d_inner,
        config.conv_kernel,
        device,
    );

    for t in 1..=30 {
        let angle = (t as f32) * 0.3;
        let real_x = angle.cos();
        let real_y = angle.sin();

        let action = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![-0.1 * angle.sin(), 0.1 * angle.cos()], [1, 2]),
            device,
        );

        let a_enc = model.action_encoder.forward(action);
        let u = model
            .fusion
            .forward(Tensor::cat(vec![current_z.clone(), a_enc], 1));

        let (pred_z, new_state) =
            ssm_latent_model::rtrl::ssm_step_detached(&model.ssm, u, &ssm_state);

        let decoded = model.decode(pred_z.clone());
        let pos = decoded.into_data();
        let pos_slice = pos.as_slice::<f32>().unwrap();

        draw_frame(
            title,
            t as usize,
            Some(([real_x, real_y], 'X', "Ground Truth")),
            Some(([pos_slice[0], pos_slice[1]], 'O', "Imagination")),
        );
        std::thread::sleep(std::time::Duration::from_millis(120));

        current_z = pred_z;
        ssm_state = new_state;
    }
}
