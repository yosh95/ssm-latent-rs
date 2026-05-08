use burn::backend::Autodiff;
use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData};
use rand::RngExt;
use ssm_latent_model::latent::{LatentLossArgs, LatentPredictor};
use ssm_latent_model::ssm::SsmConfig;

// --- Simple Game Environment ---
struct CatchGame {
    paddle_x: f32,
    target_x: f32,
    target_y: f32,
    width: usize,
    height: usize,
}

impl CatchGame {
    fn new(width: usize, height: usize) -> Self {
        let mut rng = rand::rng();
        Self {
            paddle_x: 0.5,
            target_x: rng.random_range(0.2..0.8),
            target_y: 0.0,
            width,
            height,
        }
    }

    fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool) {
        let step_size = 0.08;
        if action == 0 {
            self.paddle_x = (self.paddle_x - step_size).max(0.0);
        }
        if action == 2 {
            self.paddle_x = (self.paddle_x + step_size).min(1.0);
        }

        self.target_y += 0.1;

        let mut reward = 0.0;
        let mut done = false;

        // Hit range definition matching the visual representation "==="
        let paddle_hit_range = 0.1;

        if self.target_y >= 1.0 {
            done = true;
            // Check if the target is within the hit range of the paddle's center
            if (self.paddle_x - self.target_x).abs() < paddle_hit_range {
                reward = 1.0;
            } else {
                reward = -1.0;
            }
        }
        (
            vec![self.paddle_x, self.target_x, self.target_y],
            reward,
            done,
        )
    }

    fn render(&self) {
        let px = (self.paddle_x * (self.width - 1) as f32) as usize;
        let tx = (self.target_x * (self.width - 1) as f32) as usize;
        let ty = (self.target_y * (self.height - 1) as f32) as usize;

        let mut grid = vec![vec![' '; self.width]; self.height];
        if ty < self.height {
            grid[ty][tx.min(self.width - 1)] = '*';
        }

        // Display paddle as 3 characters (===) for better clarity
        for i in 0..3 {
            let offset_px = (px + i).saturating_sub(1);
            if offset_px < self.width {
                grid[self.height - 1][offset_px] = '=';
            }
        }

        print!("\x1B[H\x1B[2J");
        println!("+--------------------+");
        for row in grid {
            println!("|{}|", row.into_iter().collect::<String>());
        }
        println!("+--------------------+");
        println!(
            "Paddle: {:.2}, Target: ({:.2}, {:.2})",
            self.paddle_x, self.target_x, self.target_y
        );
    }
}

fn main() {
    type B = Wgpu;
    type AutodiffB = Autodiff<B>;
    let device = WgpuDevice::default();

    let config = SsmConfig {
        d_model: 64,
        d_state: 32,
        expand: 2,
        n_heads: 4,
        mimo_rank: 2,
        use_conv: true,
        conv_kernel: 4,
    };

    let mut predictor = LatentPredictor::<AutodiffB>::new(&config, 3, 3, &device);
    let mut optim = AdamConfig::new().init::<AutodiffB, LatentPredictor<AutodiffB>>();

    println!("Learning the world...");

    let base_lr = 2e-3;
    let epochs = 500;

    // 1. Pre-training Phase: Learn Physics from Random Actions
    for epoch in 0..epochs {
        // Learning Rate Scheduling: Linear decay to reduce oscillation at the end of training
        let lr = base_lr * (1.0 - (epoch as f32 / epochs as f32 * 0.9));

        let mut game = CatchGame::new(20, 10);
        let mut obs_vec = Vec::new();
        let mut act_vec = Vec::new();
        let mut current_obs = vec![game.paddle_x, game.target_x, game.target_y];

        for _ in 0..15 {
            let action = rand::rng().random_range(0..3);
            let action_vec = match action {
                0 => vec![1.0, 0.0, 0.0],
                1 => vec![0.0, 1.0, 0.0],
                _ => vec![0.0, 0.0, 1.0],
            };
            obs_vec.push(current_obs.clone());
            act_vec.push(action_vec);
            let (next_obs, _, done) = game.step(action);
            current_obs = next_obs;
            if done {
                break;
            }
        }

        let seq_len = obs_vec.len();
        let obs_tensor = Tensor::<AutodiffB, 3>::from_data(
            TensorData::new(
                obs_vec.into_iter().flatten().collect::<Vec<f32>>(),
                [1, seq_len, 3],
            ),
            &device,
        );
        let act_tensor = Tensor::<AutodiffB, 3>::from_data(
            TensorData::new(
                act_vec.into_iter().flatten().collect::<Vec<f32>>(),
                [1, seq_len, 3],
            ),
            &device,
        );

        let (z, pred_z, reconstructed_x, predicted_x) =
            predictor.forward(obs_tensor.clone(), act_tensor);
        let loss_args = LatentLossArgs {
            z,
            pred_z,
            reconstructed_x,
            predicted_x,
            original_x: obs_tensor,
            stability_weight: 1.0,
            curvature_weight: 0.5,
            recon_weight: 2.0,
        };

        let loss = predictor.loss(loss_args);
        let loss_val = loss.clone().into_data().as_slice::<f32>().unwrap()[0];

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &predictor);
        predictor = optim.step(lr as f64, predictor, grads);

        if epoch % 10 == 0 {
            println!("Epoch {:3} | Loss: {:.6} | LR: {:.6}", epoch, loss_val, lr);
        }
    }

    // 2. Playing Phase: Use World Model to Play
    println!("World Model learned. Starting play with Imagination...");
    std::thread::sleep(std::time::Duration::from_secs(2));

    for game_idx in 0..10 {
        let mut game = CatchGame::new(20, 10);
        let mut current_obs = vec![game.paddle_x, game.target_x, game.target_y];

        println!("Game {}", game_idx);

        loop {
            game.render();
            std::thread::sleep(std::time::Duration::from_millis(150));

            // Policy: Imagine results of actions
            let mut best_action = 1;
            let mut min_dist = f32::MAX;

            for action in [0, 1, 2] {
                let action_vec = match action {
                    0 => vec![1.0, 0.0, 0.0],
                    1 => vec![0.0, 1.0, 0.0],
                    _ => vec![0.0, 0.0, 1.0],
                };

                // Use the World Model to predict the next state given an action
                let act_t_3d = Tensor::<AutodiffB, 3>::from_data(
                    TensorData::new(action_vec, [1, 1, 3]),
                    &device,
                );

                let (_, _, _, predicted_x) = predictor.forward(
                    Tensor::from_data(TensorData::new(current_obs.clone(), [1, 1, 3]), &device),
                    act_t_3d,
                );

                let pred_val = predicted_x.into_data();
                let pred_slice = pred_val.as_slice::<f32>().unwrap();

                // Physical logic: pred_slice[0] is paddle_x, pred_slice[1] is target_x
                let predicted_paddle_x = pred_slice[0];
                let target_x = current_obs[1]; // The target's X position is constant in this game

                let dist = (predicted_paddle_x - target_x).abs();
                if dist < min_dist {
                    min_dist = dist;
                    best_action = action;
                }
            }

            let (next_obs, reward, done) = game.step(best_action);
            current_obs = next_obs;

            if done {
                game.render();
                if reward > 0.0 {
                    println!("SUCCESS! Caught the ball!");
                } else {
                    println!("FAIL... Missed.");
                }
                std::thread::sleep(std::time::Duration::from_secs(1));
                break;
            }
        }
    }
}
