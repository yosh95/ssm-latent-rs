#![recursion_limit = "256"]
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::Tensor;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use ssm_latent_model::predictor::MultiScaleMambaPredictor;
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
        {"Wgpu (GPU)"}
        #[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
        {"NdArray (CPU)"}
        #[cfg(all(not(feature = "wgpu"), not(feature = "ndarray")))]
        {"NdArray (CPU)"}
    };

    println!("==========================================================");
    println!("     📖 The Chronicles of the Digital Explorer");
    println!("     (Pure Mamba — Imagination loop in d_model space)");
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
    let obs_dim = 2;       // (x, y)
    let action_dim = 2;    // (ax, ay) — velocity tangent to circle
    let seq_len = 32;
    let batch_size = 4;
    let epochs = 160;
    let learning_rate = 1e-3;

    let ssm_config = MultiScaleSsmConfig::new(32, 8, 2, 2, 1)
        .with_n_layers(3)
        .with_use_conv(true)
        .with_conv_kernel(4);

    println!(
        "Config: d_model={}, d_state={}, expand={}, heads={}, layers={}, obs_dim={}, action_dim={}",
        ssm_config.d_model, ssm_config.d_state, ssm_config.expand,
        ssm_config.n_heads, ssm_config.n_layers, obs_dim, action_dim,
    );

    let mut explorer = MultiScaleMambaPredictor::<MyAutodiffBackend>::new(
        &ssm_config, obs_dim + action_dim, obs_dim, &device,
    );
    let mut brain_optimizer =
        AdamConfig::new().init::<MyAutodiffBackend, MultiScaleMambaPredictor<MyAutodiffBackend>>();

    // --- Part 2: Dreaming ---
    println!("\n[Part 2: Dreaming]");
    println!("The Explorer dreams. Training with obs+action → next-obs.");
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

        // Use forward_with_action: obs + action fused in d_model, then SSM → next obs
        let predictions = explorer.forward_with_action(obs_tensor.clone(), act_tensor);
        let loss = explorer.loss(predictions, obs_tensor);

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
    println!("using only its internal SSM state (d_model loop).");
    sleep(Duration::from_millis(1500));

    run_demo(
        &explorer.valid(),
        &device,
        &format!("Final World Model (Epoch {})", epochs),
    );

    println!("--------------------------------------------------");
    println!("The Explorer traversed the unknown using only its mind.");
    println!("Pure Mamba. d_model loop. No drift.");
}

fn run_demo<B: burn::tensor::backend::Backend>(
    model: &MultiScaleMambaPredictor<B>,
    device: &B::Device,
    title: &str,
) {
    println!("\n--- {} ---", title);

    // Encode initial observation to get the first SSM output y_0
    let initial_obs = Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(vec![1.0f32, 0.0f32], [1, 2]),
        device,
    );
    // We need y_0. Run a single forward pass through input_proj then SSM.
    // Since we don't have an action for t=0, use zero action.
    let zero_action = Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(vec![0.0f32, 0.0f32], [1, 2]),
        device,
    );

    let obs_enc = model.obs_proj.forward(initial_obs);        // [1, d_model]
    let act_enc = model.action_proj.forward(zero_action);       // [1, d_model]
    let u_concat = Tensor::cat(vec![obs_enc, act_enc], 1);      // [1, 2*d_model]
    let u = model.imagine_fusion.forward(u_concat);              // [1, d_model]

    // Get initial SSM output y_0 through the multi-scale stack
    let init_state = model.zero_state(1, device);
    let (mut y_current, next_ssms) = model.ssms.forward_step(u, &init_state.ssms);
    let mut state = ssm_latent_model::predictor::MultiScalePredictorState { ssms: next_ssms };

    for t in 1..=20 {
        let angle = (t as f32) * 0.3;
        let real_x = angle.cos();
        let real_y = angle.sin();

        let action = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(
                vec![-0.1 * angle.sin(), 0.1 * angle.cos()],
                [1, 2],
            ),
            device,
        );

        // Imagination step: y_current (d_model) + action → SSM → y_next (d_model)
        let (pred, y_next, next_state) = model.step_imagine(y_current, action, state);
        state = next_state;
        y_current = y_next;

        let pred_data = pred.into_data();
        let pred_slice = pred_data.as_slice::<f32>().unwrap();

        draw_frame(
            title,
            t as usize,
            Some(([real_x, real_y], 'X', "Ground Truth")),
            Some(([pred_slice[0], pred_slice[1]], 'O', "Mental Map")),
        );
        sleep(Duration::from_millis(150));
    }
}
