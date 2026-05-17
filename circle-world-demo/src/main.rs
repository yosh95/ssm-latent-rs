#![recursion_limit = "256"]
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::Tensor;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use ssm_latent_model::latent::{LatentLossArgs, MultiScaleLatentPredictor};
use ssm_latent_model::ssm::MultiScaleSsmConfig;
use std::thread::sleep;
use std::time::Duration;

// --- Backend Selection based on features ---
#[cfg(feature = "wgpu")]
type MyBackend = burn::backend::Wgpu;
#[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
type MyBackend = burn::backend::NdArray;
#[cfg(all(not(feature = "wgpu"), not(feature = "ndarray")))]
type MyBackend = burn::backend::NdArray;

type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

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
    #[cfg(feature = "wgpu")]
    let device = burn::backend::wgpu::WgpuDevice::default();
    #[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
    let device = burn::backend::ndarray::NdArrayDevice::default();
    #[cfg(all(not(feature = "wgpu"), not(feature = "ndarray")))]
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let mut rng = StdRng::seed_from_u64(42);

    let backend_name = {
        #[cfg(feature = "wgpu")]
        {
            "Wgpu (GPU)"
        }
        #[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
        {
            "NdArray (CPU)"
        }
        #[cfg(all(not(feature = "wgpu"), not(feature = "ndarray")))]
        {
            "NdArray (CPU)"
        }
    };

    println!("==========================================================");
    println!("     📖 The Chronicles of the Digital Explorer");
    println!("     (Mamba + JEPA — Latent-space prediction)");
    println!("     Backend: {}", backend_name);
    println!("==========================================================");
    sleep(Duration::from_millis(800));

    println!("\n[Part 1: The Encounter]");
    println!("The Explorer observes (x, y) dancing in circles...");
    sleep(Duration::from_millis(1500));

    for t in 1..=20 {
        let angle = (t as f32) * 0.3;
        draw_frame(
            "Observation of the Phenomenon",
            t,
            Some(([angle.cos(), angle.sin()], 'X', "External World")),
            None,
        );
        sleep(Duration::from_millis(100));
    }

    // ── Model config ──
    let obs_dim = 2; // (x, y)
    let action_dim = 2; // (ax, ay) — velocity tangent to circle
    let seq_len = 32;
    let batch_size = 4;
    let epochs = 150;
    let learning_rate = 1e-3;

    // JEPA loss weights
    let recon_weight = 1.0; // reconstruction weight (encoder quality)
    let stability_weight = 0.01; // VICReg — prevents latent collapse
    let curvature_weight = 0.005; // Temporal Straightening — smooth trajectories

    let ssm_config = MultiScaleSsmConfig::new(32, 8, 2, 2, 1)
        .with_n_layers(3)
        .with_use_conv(true)
        .with_conv_kernel(4);

    println!(
        "Config: d_model={}, d_state={}, expand={}, heads={}, layers={}",
        ssm_config.d_model,
        ssm_config.d_state,
        ssm_config.expand,
        ssm_config.n_heads,
        ssm_config.n_layers,
    );
    println!(
        "JEPA: recon_w={}, stability_w={}, curvature_w={}",
        recon_weight, stability_weight, curvature_weight,
    );

    let mut explorer = MultiScaleLatentPredictor::<MyAutodiffBackend>::new(
        &ssm_config,
        obs_dim,
        action_dim,
        &device,
    );
    let mut brain_optimizer =
        AdamConfig::new().init::<MyAutodiffBackend, MultiScaleLatentPredictor<MyAutodiffBackend>>();

    // --- Part 2: Dreaming ---
    println!("\n[Part 2: Dreaming]");
    println!("The Explorer dreams. Training in latent space (JEPA-style).");
    sleep(Duration::from_millis(1000));

    // Initial state
    println!("\n--- Initial Mental Map (Before Training) ---");
    run_demo(&explorer.valid(), &device, "Epoch 0 (Untrained)");
    sleep(Duration::from_millis(1500));

    for epoch in 1..=epochs {
        let mut obs_vec = Vec::new();
        let mut act_vec = Vec::new();

        for b in 0..batch_size {
            let phase_shift = (b as f32) * 1.25;
            for t in 0..seq_len {
                let time = (t as f32) * 0.3;
                let angle = time + phase_shift;
                let noise: f32 = rng.random_range(-0.02..0.02);

                obs_vec.push(angle.cos() + noise);
                obs_vec.push(angle.sin() + noise);

                let act_noise: f32 = rng.random_range(-0.005..0.005);
                act_vec.push(-0.1 * angle.sin() + act_noise);
                act_vec.push(0.1 * angle.cos() + act_noise);
            }
        }

        let obs_tensor = Tensor::<MyAutodiffBackend, 3>::from_data(
            burn::tensor::TensorData::new(obs_vec, [batch_size, seq_len, obs_dim]),
            &device,
        );
        let act_tensor = Tensor::<MyAutodiffBackend, 3>::from_data(
            burn::tensor::TensorData::new(act_vec, [batch_size, seq_len, action_dim]),
            &device,
        );

        // ── JEPA forward pass ──
        // encoder: obs → z (latent space)
        // action_encoder: act → a (latent space)
        // fusion: [z, a] → u
        // SSM: u → predicted_z (predicts next latent)
        // decoder: z → reconstructed_x, predicted_z → predicted_x
        let (z, predicted_z, reconstructed_x, predicted_x) =
            explorer.forward(obs_tensor.clone(), act_tensor);

        // ── JEPA loss ──
        // L = MSE(z_pred, z_next)  ← latent prediction (core JEPA)
        //   + recon_w · (MSE(recon_x, x) + MSE(pred_x, x_next))  ← reconstruction
        //   + stability_w · VICReg(z)  ← prevents representation collapse
        //   + curvature_w · Σ|z_t - 2z_{t-1} + z_{t-2}|²  ← smooth trajectories
        let loss_args = LatentLossArgs {
            z,
            pred_z: predicted_z,
            reconstructed_x,
            predicted_x,
            original_x: obs_tensor,
            stability_weight,
            curvature_weight,
            recon_weight,
        };
        let loss = explorer.loss(loss_args);

        let current_loss: f32 = loss.clone().into_data().as_slice::<f32>().unwrap()[0];

        if epoch % 30 == 0 || epoch == 1 {
            println!("\n--- Training Progress Update ---");
            println!("Dream Epoch {:3}: (Loss: {:.6})", epoch, current_loss);
            let explorer_valid = explorer.valid();
            run_demo(
                &explorer_valid,
                &device,
                &format!("Training (Epoch {})", epoch),
            );
            sleep(Duration::from_millis(500));
        }

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &explorer);
        explorer = brain_optimizer.step(learning_rate, explorer, grads);
    }

    println!("\nThe Explorer has finished learning.");
    sleep(Duration::from_millis(1000));

    // --- Part 3: Imagination ---
    println!("\n[Part 3: Pure Imagination]");
    println!("No more observations. The Explorer imagines the future");
    println!("using only its latent state (JEPA latent loop).");
    sleep(Duration::from_millis(1500));

    run_demo(
        &explorer.valid(),
        &device,
        &format!("Final World Model (Epoch {})", epochs),
    );

    println!("--------------------------------------------------");
    println!("The Explorer traversed the unknown using only its mind.");
    println!("Mamba + JEPA. Latent loop with curvature + stability.");
}

fn run_demo<B: burn::tensor::backend::Backend>(
    model: &MultiScaleLatentPredictor<B>,
    device: &B::Device,
    title: &str,
) {
    println!("\n--- {} ---", title);

    // Encode initial observation → z_0 (enter latent space)
    let initial_obs = Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(vec![1.0f32, 0.0f32], [1, 2]),
        device,
    );
    let initial_z = model
        .encode(initial_obs.unsqueeze_dim::<3>(1))
        .reshape([1, model.d_model]); // [1, d_model]

    let mut z_current = initial_z;
    let mut state = model.ssm.zero_state(1, device);

    for t in 1..=20 {
        let angle = (t as f32) * 0.3;
        let real_x = angle.cos();
        let real_y = angle.sin();

        let action = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![-0.1 * angle.sin(), 0.1 * angle.cos()], [1, 2]),
            device,
        );

        // ── JEPA imagination step ──
        // z_current + action → SSM → z_next (predicted next latent)
        let (z_next, next_state) = model.step(z_current.clone(), action, state);
        state = next_state;

        // Decode z_next → observation (exit latent space for visualization)
        let pred_obs = model
            .decode(z_next.clone().unsqueeze_dim::<3>(1))
            .reshape([1, 2]);

        let pred_data = pred_obs.into_data();
        let pred_slice = pred_data.as_slice::<f32>().unwrap();

        draw_frame(
            title,
            t as usize,
            Some(([real_x, real_y], 'X', "Ground Truth")),
            Some(([pred_slice[0], pred_slice[1]], 'O', "Mental Map")),
        );
        sleep(Duration::from_millis(150));

        z_current = z_next;
    }
}
