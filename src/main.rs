#![recursion_limit = "256"]
use burn::backend::{Autodiff, Wgpu, wgpu::WgpuDevice};
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::Tensor;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use ssm_latent_model::latent::{LatentPredictor, LatentState};
use ssm_latent_model::ssm::SsmConfig;
use std::thread::sleep;
use std::time::Duration;

type MyBackend = Wgpu;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn draw_frame(
    title: &str,
    step: usize,
    left_point: Option<([f32; 2], char, &str)>,
    right_point: Option<([f32; 2], char, &str)>,
) {
    let width = 40;
    let height = 15;

    let mut grids = vec![vec![vec![' '; width]; height]; 2];

    let points = [left_point, right_point];

    for (g_idx, pt_opt) in points.iter().enumerate() {
        if let Some(([x, y], marker, _)) = pt_opt {
            let grid = &mut grids[g_idx];
            // Draw axes
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

    println!("\x1B[H\x1B[2J"); // Clear screen
    println!("=== {} (Step: {}) ===", title, step);

    // Print labels
    if right_point.is_some() {
        let label_left = left_point.as_ref().map(|p| p.2).unwrap_or("");
        let label_right = right_point.as_ref().map(|p| p.2).unwrap_or("");
        println!(
            "  {:^width$}          {:^width$}",
            label_left,
            label_right,
            width = width
        );

        for (row_left, row_right) in grids[0].iter().zip(grids[1].iter()) {
            let s_left: String = row_left.iter().collect();
            let s_right: String = row_right.iter().collect();
            println!("  {}          {}", s_left, s_right);
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
    let device = WgpuDevice::default();
    let mut rng = StdRng::seed_from_u64(42);

    println!("==========================================================");
    println!("     📖 The Chronicles of the Digital Explorer");
    println!("==========================================================");
    sleep(Duration::from_millis(800));

    // --- Part 1: The Encounter ---
    println!("\n[Part 1: The Encounter]");
    println!("The Explorer is placed in a world where a mysterious signal pulses.");
    println!("It sees observations (x, y) that seem to dance in circles...");
    sleep(Duration::from_millis(1500));

    for t in 1..=20 {
        let angle = (t as f32) * 0.3;
        let x = angle.cos();
        let y = angle.sin();
        draw_frame(
            "Observation of the Phenomenon",
            t,
            Some(([x, y], 'X', "External World")),
            None,
        );
        sleep(Duration::from_millis(100));
    }

    let config = SsmConfig {
        d_model: 64,
        d_state: 32,
        expand: 2,
        n_heads: 4,
        mimo_rank: 2,
        use_conv: true,
        conv_kernel: 4,
    };
    let input_dim = 2;
    let action_dim = 2;
    let seq_len = 32;
    let batch_size = 4; // Increased for more stable gradient
    let epochs = 120;
    let learning_rate = 1e-3; // Balanced learning rate

    let mut explorer =
        LatentPredictor::<MyAutodiffBackend>::new(&config, input_dim, action_dim, &device);
    let mut brain_optimizer =
        AdamConfig::new().init::<MyAutodiffBackend, LatentPredictor<MyAutodiffBackend>>();

    // --- Part 2: Dreaming ---
    println!("\n[Part 2: Dreaming]");
    println!("The Explorer closes its eyes and begins to 'dream' about the data.");
    println!("It tries to condense messy observations into its mental map.");
    sleep(Duration::from_millis(1000));

    // SHOW INITIAL STATE (Epoch 0)
    println!("\n--- Initial Mental Map (Before Training) ---");
    run_demo(&explorer.valid(), &config, &device, "Epoch 0 (Untrained)");
    sleep(Duration::from_millis(1500));

    for epoch in 1..=epochs {
        let mut obs_vec = Vec::new();
        let mut act_vec = Vec::new();

        for b in 0..batch_size {
            // Constant phase shift per batch, no longer dependent on epoch to stabilize target
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

        let obs_data = Tensor::<MyAutodiffBackend, 3>::from_data(
            burn::tensor::TensorData::new(obs_vec, [batch_size, seq_len, input_dim]),
            &device,
        );
        let action_data = Tensor::<MyAutodiffBackend, 3>::from_data(
            burn::tensor::TensorData::new(act_vec, [batch_size, seq_len, action_dim]),
            &device,
        );

        let (z, predicted_z, reconstructed_x) = explorer.forward(obs_data.clone(), action_data);
        let loss = explorer.loss(z, predicted_z, reconstructed_x, obs_data, 1.2);

        let current_loss: f32 = loss.clone().into_data().as_slice::<f32>().unwrap()[0];

        // LOG EVERY 30 EPOCHS or EPOCH 1
        if epoch % 30 == 0 || epoch == 1 {
            println!("\n--- Training Progress Update ---");
            println!("Dream Epoch {:3}: (Loss: {:.6})", epoch, current_loss);

            let explorer_valid = explorer.valid();
            run_demo(
                &explorer_valid,
                &config,
                &device,
                &format!("Training (Epoch {})", epoch),
            );
            sleep(Duration::from_millis(500));
        }

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &explorer);
        explorer = brain_optimizer.step(learning_rate, explorer, grads);
    }

    println!("\nThe Explorer has finished learning. It now possesses a 'World Model'.");
    sleep(Duration::from_millis(1000));

    // --- Part 3: Imagination ---
    println!("\n[Part 3: Pure Imagination]");
    println!("Now, we take away the observations. The Explorer is blind.");
    println!("It will 'imagine' the future solely based on its internal world model.");
    sleep(Duration::from_millis(1500));

    run_demo(&explorer.valid(), &config, &device, "Final World Model");

    println!("--------------------------------------------------");
    println!("The Explorer successfully traversed the unknown using only its mind.");
    println!("It learned to mimic reality, bridging the gap between perception and thought.");
}

fn run_demo<B: burn::tensor::backend::Backend>(
    model: &LatentPredictor<B>,
    config: &SsmConfig,
    device: &B::Device,
    title: &str,
) {
    let batch_size = 1;

    // Initial memory
    let initial_obs = Tensor::<B, 3>::from_data(
        burn::tensor::TensorData::new(vec![1.0, 0.0], [1, 1, 2]),
        device,
    );

    let z_memory = model.encode(initial_obs);
    let [batch, _seq, d_model] = z_memory.dims();
    let mut current_latent = z_memory.reshape([batch, d_model]);

    let d_inner = config.d_model * config.expand;
    let d_head = d_inner / config.n_heads;

    let mut state = LatentState {
        h: Tensor::zeros(
            [
                batch_size,
                config.n_heads,
                config.d_state,
                d_head / config.mimo_rank,
            ],
            device,
        ),
        prev_bx: None,
        conv_state: if config.use_conv {
            Some(Tensor::zeros(
                [batch_size, d_inner, config.conv_kernel - 1],
                device,
            ))
        } else {
            None
        },
    };

    for t in 1..=20 {
        let angle = (t as f32) * 0.3;

        // Ground Truth for comparison
        let real_x = angle.cos();
        let real_y = angle.sin();

        // Action to take
        let action_val = vec![-0.1 * angle.sin(), 0.1 * angle.cos()];
        let action = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(action_val, [batch_size, 2]),
            device,
        );

        // One step of internal simulation
        let (next_latent, next_state) = model.step(current_latent, action, state);

        current_latent = next_latent;
        state = next_state;

        // Decode the 'thought'
        let decoded = model.decode(current_latent.clone().unsqueeze_dim::<3>(1));
        let pos = decoded.into_data();
        let pos_slice = pos.as_slice::<f32>().unwrap();
        let x = pos_slice[0];
        let y = pos_slice[1];

        draw_frame(
            title,
            t,
            Some(([real_x, real_y], 'X', "Ground Truth")),
            Some(([x, y], 'O', "Mental Map (Dream)")),
        );

        sleep(Duration::from_millis(100));
    }
}
