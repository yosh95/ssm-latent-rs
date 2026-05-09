use burn::backend::Autodiff;
use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData};
use rand::Rng;
use ssm_latent_model::latent::{LatentLossArgs, LatentPredictor};
use ssm_latent_model::ssm::SsmConfig;

// --- Simple Ball Bouncing Game Environment ---
struct CatchGame {
    paddle_x: f32,
    ball_x: f32,
    ball_y: f32,
    ball_dx: f32,
    ball_dy: f32,
    width: usize,
    height: usize,
    friction_variation: bool,
}

impl CatchGame {
    fn new(width: usize, height: usize, friction_variation: bool) -> Self {
        let mut rng = rand::rng();
        Self {
            paddle_x: 0.5,
            ball_x: rng.random_range(0.2..0.8),
            ball_y: 0.1,
            ball_dx: rng.random_range(-0.05..0.05),
            ball_dy: 0.08,
            width,
            height,
            friction_variation,
        }
    }

    fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool) {
        let paddle_step = 0.08;
        if action == 0 {
            self.paddle_x = (self.paddle_x - paddle_step).max(0.0);
        } else if action == 2 {
            self.paddle_x = (self.paddle_x + paddle_step).min(1.0);
        }

        // Move ball
        self.ball_x += self.ball_dx;
        self.ball_y += self.ball_dy;

        // Wall bounce (Left/Right)
        if self.ball_x <= 0.0 || self.ball_x >= 1.0 {
            self.ball_dx *= -1.0;
            self.ball_x = self.ball_x.clamp(0.0, 1.0);
            
            // Simulating "friction variations" based on wall position if enabled
            if self.friction_variation {
                let mut rng = rand::rng();
                self.ball_dx *= rng.random_range(0.9..1.1);
            }
        }

        // Wall bounce (Top)
        if self.ball_y <= 0.0 {
            self.ball_dy *= -1.0;
            self.ball_y = 0.0;
        }

        let mut reward = 0.0;
        let mut done = false;
        let paddle_hit_range = 0.15;

        // Bottom boundary
        if self.ball_y >= 1.0 {
            if (self.paddle_x - self.ball_x).abs() < paddle_hit_range {
                reward = 1.0;
                // Bounce back up
                self.ball_dy *= -1.0;
                self.ball_y = 0.99;
                // Add some influence from paddle movement
                if action == 0 { self.ball_dx -= 0.02; }
                if action == 2 { self.ball_dx += 0.02; }
            } else {
                reward = -1.0;
                done = true;
            }
        }

        // Observation: [paddle_x, ball_x, ball_y, ball_dx, ball_dy]
        (
            vec![self.paddle_x, self.ball_x, self.ball_y, self.ball_dx, self.ball_dy],
            reward,
            done,
        )
    }

    fn render(&self) {
        let px = (self.paddle_x * (self.width - 1) as f32) as usize;
        let bx = (self.ball_x * (self.width - 1) as f32) as usize;
        let by = (self.ball_y * (self.height - 1) as f32) as usize;

        let mut grid = vec![vec![' '; self.width]; self.height];
        if by < self.height {
            grid[by][bx.min(self.width - 1)] = '*';
        }

        let paddle_width = 3;
        for i in 0..paddle_width {
            let offset_px = (px + i).saturating_sub(paddle_width / 2);
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
            "Paddle: {:.2}, Ball: ({:.2}, {:.2}), DX: {:.3}",
            self.paddle_x, self.ball_x, self.ball_y, self.ball_dx
        );
    }
}

fn main() {
    type B = Wgpu;
    type AutodiffB = Autodiff<B>;
    let device = WgpuDevice::default();

    let config = SsmConfig {
        d_model: 128, // Increased model capacity
        d_state: 64,
        expand: 2,
        n_heads: 4,
        mimo_rank: 1,
        use_conv: true,
        conv_kernel: 4,
    };

    let input_dim = 5; // [paddle_x, ball_x, ball_y, ball_dx, ball_dy]
    let action_dim = 3; 

    let mut predictor = LatentPredictor::<AutodiffB>::new(&config, input_dim, action_dim, &device);
    let mut optim = AdamConfig::new().init::<AutodiffB, LatentPredictor<AutodiffB>>();

    println!("Learning the Physics of the World (Mamba + JEPA)...");

    let base_lr = 1e-3;
    let epochs = 1000; // More epochs for harder task
    let batch_size = 1; // Simplified for this demo

    for epoch in 0..epochs {
        let lr = base_lr * (1.0 - (epoch as f32 / epochs as f32 * 0.9));

        let mut game = CatchGame::new(20, 10, epoch > epochs / 2);
        let mut obs_vec = Vec::new();
        let mut act_vec = Vec::new();
        let mut current_obs = vec![game.paddle_x, game.ball_x, game.ball_y, game.ball_dx, game.ball_dy];

        // Longer sequences to observe bounces
        for _ in 0..40 {
            let action = rand::rng().random_range(0..3);
            let action_vec = match action {
                0 => vec![1.0, 0.0, 0.0],
                1 => vec![0.0, 1.0, 0.0],
                _ => vec![0.0, 0.0, 1.0],
            };
            obs_vec.push(current_obs.clone());
            act_vec.push(action_vec);
            let (next_obs, _, _) = game.step(action);
            current_obs = next_obs;
        }

        let seq_len = obs_vec.len();
        let obs_tensor = Tensor::<AutodiffB, 3>::from_data(
            TensorData::new(obs_vec.into_iter().flatten().collect::<Vec<f32>>(), [1, seq_len, input_dim]),
            &device,
        );
        let act_tensor = Tensor::<AutodiffB, 3>::from_data(
            TensorData::new(act_vec.into_iter().flatten().collect::<Vec<f32>>(), [1, seq_len, action_dim]),
            &device,
        );

        let (z, pred_z, reconstructed_x, predicted_x) = predictor.forward(obs_tensor.clone(), act_tensor);
        let loss_args = LatentLossArgs {
            z,
            pred_z,
            reconstructed_x,
            predicted_x,
            original_x: obs_tensor,
            stability_weight: 0.1, // Reduced to prevent collapse without over-constraining
            curvature_weight: 0.1, // Lower curvature constraint
            recon_weight: 10.0,    // Strongly emphasize pixel-space physics reconstruction
        };

        let loss = predictor.loss(loss_args);
        let loss_val = loss.clone().into_data().as_slice::<f32>().unwrap()[0];

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &predictor);
        predictor = optim.step(lr as f64, predictor, grads);

        if epoch % 50 == 0 {
            println!("Epoch {:4} | Loss: {:.6} | LR: {:.6}", epoch, loss_val, lr);
        }
    }

    println!("World Model learned. Starting play with Predictive Control...");
    std::thread::sleep(std::time::Duration::from_secs(2));

    for game_idx in 0..5 {
        let mut game = CatchGame::new(20, 10, true);
        let mut current_obs = vec![game.paddle_x, game.ball_x, game.ball_y, game.ball_dx, game.ball_dy];
        let mut frames = 0;

        println!("Game {}", game_idx);

        loop {
            game.render();
            std::thread::sleep(std::time::Duration::from_millis(80));

            let mut best_action = 1;
            let mut min_error = f32::MAX;

            for action in [0, 1, 2] {
                let action_vec = match action {
                    0 => vec![1.0, 0.0, 0.0],
                    1 => vec![0.0, 1.0, 0.0],
                    _ => vec![0.0, 0.0, 1.0],
                };

                // One-step lookahead using the learned world model
                let (_, _, _, predicted_x) = predictor.forward(
                    Tensor::from_data(TensorData::new(current_obs.clone(), [1, 1, input_dim]), &device),
                    Tensor::from_data(TensorData::new(action_vec, [1, 1, action_dim]), &device),
                );

                let pred_val = predicted_x.into_data();
                let pred_slice = pred_val.as_slice::<f32>().unwrap();
                
                let pred_paddle_x = pred_slice[0];
                let pred_ball_x = pred_slice[1];
                let pred_ball_y = pred_slice[2];
                let pred_ball_dy = pred_slice[4];

                // Logic: 
                // 1. If ball is falling (dy > 0), try to match the predicted ball_x.
                // 2. We look at the *predicted* future state to decide.
                let error = if pred_ball_dy > 0.0 {
                    (pred_paddle_x - pred_ball_x).abs()
                } else {
                    // If ball is going up, stay near the center or follow loosely
                    (pred_paddle_x - 0.5).abs() * 0.2 + (pred_paddle_x - pred_ball_x).abs() * 0.8
                };
                
                if error < min_error {
                    min_error = error;
                    best_action = action;
                }
            }

            let (next_obs, _reward, done) = game.step(best_action);
            current_obs = next_obs;
            frames += 1;

            if done || frames > 500 {
                game.render();
                println!("GAME OVER | Frames: {}", frames);
                std::thread::sleep(std::time::Duration::from_secs(1));
                break;
            }
        }
    }
}
