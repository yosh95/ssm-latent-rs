use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData};
use rand::RngExt;
use ssm_latent_model::latent::{LatentLossArgs, LatentPredictor};
use ssm_latent_model::ssm::SsmConfig;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlButtonElement, HtmlCanvasElement};

macro_rules! log {
    ($($t:tt)*) => (web_sys::console::log_1(&format!($($t)*).into()))
}

#[cfg(feature = "wgpu")]
type RawBackend = burn::backend::Wgpu;
#[cfg(not(feature = "wgpu"))]
type RawBackend = burn::backend::NdArray<f32>;

type Backend = burn::backend::Autodiff<RawBackend>;

type ClosureLoop = Rc<RefCell<Option<Closure<dyn FnMut()>>>>;

// --- Physical Environment ---
struct CatchGame {
    paddle_x: f32,
    ball_x: f32,
    ball_y: f32,
    ball_dx: f32,
    ball_dy: f32,
}

impl CatchGame {
    fn new() -> Self {
        let mut rng = rand::rng();
        Self {
            paddle_x: 0.5,
            ball_x: rng.random_range(0.2..0.8),
            ball_y: 0.1,
            ball_dx: rng.random_range(-0.015..0.015),
            ball_dy: 0.02,
        }
    }

    fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool) {
        let paddle_step = 0.025; // Reduce step size to fix flickering
        if action == 0 {
            self.paddle_x = (self.paddle_x - paddle_step).max(0.0);
        } else if action == 2 {
            self.paddle_x = (self.paddle_x + paddle_step).min(1.0);
        }

        self.ball_x += self.ball_dx;
        self.ball_y += self.ball_dy;

        if self.ball_x <= 0.0 || self.ball_x >= 1.0 {
            self.ball_dx *= -1.0;
            self.ball_x = self.ball_x.clamp(0.0, 1.0);
        }

        if self.ball_y <= 0.0 {
            self.ball_dy *= -1.0;
            self.ball_y = 0.0;
        }

        let mut reward = 0.0;
        let mut done = false;
        let paddle_hit_range = 0.15;

        if self.ball_y >= 1.0 {
            let diff = self.ball_x - self.paddle_x;
            if diff.abs() < paddle_hit_range {
                reward = 1.0;

                // Typical breakout game physics:
                // Change reflection angle based on distance from the paddle center.
                // hit_pos ranges from -1.0 to 1.0.
                let hit_pos = diff / paddle_hit_range;
                let speed = (self.ball_dx.powi(2) + self.ball_dy.powi(2))
                    .sqrt()
                    .max(0.025);

                // Calculate new direction based on hit position while maintaining speed.
                self.ball_dx = hit_pos * (speed * 0.8); // Clamp X velocity to avoid extreme angles
                self.ball_dy = -(speed.powi(2) - self.ball_dx.powi(2)).sqrt().abs();

                self.ball_y = 0.99;
            } else {
                reward = -1.0;
                done = true;
            }
        }

        (
            vec![
                self.paddle_x,
                self.ball_x,
                self.ball_y,
                self.ball_dx,
                self.ball_dy,
            ],
            reward,
            done,
        )
    }
}

#[wasm_bindgen(start)]
pub async fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    log!("Game-Playing WASM Start");

    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let canvas = document
        .get_element_by_id("canvas")
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()?;
    let ctx = canvas
        .get_context("2d")?
        .unwrap()
        .dyn_into::<CanvasRenderingContext2d>()?;
    let status = document.get_element_by_id("status").unwrap();
    let start_button = document
        .get_element_by_id("startButton")
        .unwrap()
        .dyn_into::<HtmlButtonElement>()?;

    status.set_text_content(Some("Ready."));

    #[cfg(feature = "wgpu")]
    let device = burn::backend::wgpu::WgpuDevice::default();
    #[cfg(not(feature = "wgpu"))]
    let device = burn::backend::ndarray::NdArrayDevice::default();

    let config = SsmConfig::new(64, 16, 2, 4, 1);
    let model = LatentPredictor::new(&config, 5, 3, &device);
    let optim = AdamConfig::new().init();

    let app = Rc::new(RefCell::new(AppState {
        model,
        optim,
        device,
        game: CatchGame::new(),
        pred_pos: [0.5, 0.5, 0.5], // paddle_x, ball_x, ball_y
        config,
        history_obs: Vec::new(),
        history_act: Vec::new(),
        epoch_count: 0,
        loss: 0.0,
        frame_count: 0,
        current_obs: vec![0.5, 0.5, 0.1, 0.0, 0.05],
    }));

    let f: ClosureLoop = Rc::new(RefCell::new(None));
    let g = f.clone();

    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        let mut app = app.borrow_mut();
        app.update();
        app.draw(&ctx);

        status.set_text_content(Some(&format!(
            "Epoch: {} | Loss: {:.6} | Mode: {}",
            app.epoch_count,
            app.loss,
            if app.epoch_count < 100 {
                "Initial Learning..."
            } else {
                "Predictive Control"
            }
        )));

        web_sys::window()
            .unwrap()
            .request_animation_frame(f.borrow().as_ref().unwrap().as_ref().unchecked_ref())
            .unwrap();
    }) as Box<dyn FnMut()>));

    let on_click = {
        let start_button = start_button.clone();
        Closure::wrap(Box::new(move || {
            start_button.set_disabled(true);
            start_button.set_inner_text("Running...");
            web_sys::window()
                .unwrap()
                .request_animation_frame(g.borrow().as_ref().unwrap().as_ref().unchecked_ref())
                .unwrap();
        }) as Box<dyn FnMut()>)
    };

    start_button.set_onclick(Some(on_click.as_ref().unchecked_ref()));
    on_click.forget();

    Ok(())
}

struct AppState {
    model: LatentPredictor<Backend>,
    optim: burn::optim::adaptor::OptimizerAdaptor<
        burn::optim::Adam,
        LatentPredictor<Backend>,
        Backend,
    >,
    #[cfg(feature = "wgpu")]
    device: burn::backend::wgpu::WgpuDevice,
    #[cfg(not(feature = "wgpu"))]
    device: burn::backend::ndarray::NdArrayDevice,
    game: CatchGame,
    pred_pos: [f32; 3],
    #[allow(dead_code)]
    config: SsmConfig,
    history_obs: Vec<f32>,
    history_act: Vec<f32>,
    epoch_count: usize,
    loss: f32,
    frame_count: usize,
    current_obs: Vec<f32>,
}

impl AppState {
    fn update(&mut self) {
        self.frame_count += 1;

        // Use past context (history) to leverage SSM's state capabilities
        let context_size = 10; // Use last 10 steps as temporal context
        // Choose action
        let mut best_action = 1;
        if self.epoch_count < 40 {
            // Exploration: Random moves during early training
            best_action = rand::rng().random_range(0..3);
        } else {
            // Predictive Control using the model with temporal context
            // Use the model in validation mode to avoid building a computational graph
            let model_valid = self.model.valid();

            let mut obs_seq = self.history_obs.clone();
            obs_seq.extend_from_slice(&self.current_obs);
            let total_len = obs_seq.len() / 5;

            let (input_obs, seq_len) = if total_len >= context_size {
                let start = (total_len - context_size) * 5;
                (obs_seq[start..].to_vec(), context_size)
            } else {
                (obs_seq, total_len)
            };

            let mut min_error = f32::MAX;
            for action in [0, 1, 2] {
                let action_vec = match action {
                    0 => vec![1.0, 0.0, 0.0],
                    1 => vec![0.0, 1.0, 0.0],
                    _ => vec![0.0, 0.0, 1.0],
                };

                // Append the candidate action to the recent history
                let mut trial_act_seq_full = self.history_act.clone();
                trial_act_seq_full.extend_from_slice(&action_vec);
                let trial_act_seq = if total_len >= context_size {
                    let start = (total_len - context_size) * 3;
                    trial_act_seq_full[start..].to_vec()
                } else {
                    trial_act_seq_full
                };

                let (_, _, _, predicted_x) = model_valid.forward(
                    Tensor::from_data(
                        TensorData::new(input_obs.clone(), [1, seq_len, 5]),
                        &self.device,
                    ),
                    Tensor::from_data(
                        TensorData::new(trial_act_seq, [1, seq_len, 3]),
                        &self.device,
                    ),
                );

                let pred_slice = predicted_x.into_data();
                let pred_slice = pred_slice.as_slice::<f32>().unwrap();

                // Get prediction for the last step (the predicted future)
                let last_idx = (seq_len - 1) * 5;
                let p_x = pred_slice[last_idx];
                let b_x = pred_slice[last_idx + 1];
                let b_y = pred_slice[last_idx + 2];
                let b_dy = pred_slice[last_idx + 4];

                let error = if b_dy > 0.0 {
                    (p_x - b_x).abs()
                } else {
                    (p_x - 0.5).abs() * 0.2 + (p_x - b_x).abs() * 0.8
                };

                if error < min_error {
                    min_error = error;
                    best_action = action;
                    self.pred_pos = [p_x, b_x, b_y];
                }
            }
        }

        // Environment Step
        let (next_obs, _reward, done) = self.game.step(best_action);

        // Data Collection
        let action_vec = match best_action {
            0 => vec![1.0, 0.0, 0.0],
            1 => vec![0.0, 1.0, 0.0],
            _ => vec![0.0, 0.0, 1.0],
        };
        self.history_obs.extend_from_slice(&self.current_obs);
        self.history_act.extend_from_slice(&action_vec);
        self.current_obs = next_obs;

        if done {
            // Reset game and clear history to start a fresh SSM context
            self.game = CatchGame::new();
            self.current_obs = vec![
                self.game.paddle_x,
                self.game.ball_x,
                self.game.ball_y,
                self.game.ball_dx,
                self.game.ball_dy,
            ];
            self.history_obs.clear();
            self.history_act.clear();
        }

        let max_history = 64;
        if self.history_act.len() / 3 > max_history {
            self.history_obs.drain(0..5);
            self.history_act.drain(0..3);
        }

        // Online Training Step
        if self.frame_count.is_multiple_of(5) && self.history_act.len() / 3 >= 20 {
            self.train_step();
            self.epoch_count += 1;
        }
    }

    fn train_step(&mut self) {
        let seq_len = self.history_act.len() / 3;
        let obs_tensor = Tensor::<Backend, 3>::from_data(
            TensorData::new(self.history_obs.clone(), [1, seq_len, 5]),
            &self.device,
        );
        let act_tensor = Tensor::<Backend, 3>::from_data(
            TensorData::new(self.history_act.clone(), [1, seq_len, 3]),
            &self.device,
        );

        let (z, pred_z, recons_x, pred_x) = self.model.forward(obs_tensor.clone(), act_tensor);
        let loss = self.model.loss(LatentLossArgs {
            z,
            pred_z,
            reconstructed_x: recons_x,
            predicted_x: pred_x,
            original_x: obs_tensor,
            stability_weight: 0.1,
            curvature_weight: 0.1,
            recon_weight: 10.0,
        });

        self.loss = loss.clone().into_data().as_slice::<f32>().unwrap()[0];

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        self.model = self.optim.step(1e-3, self.model.clone(), grads);
    }

    fn draw(&self, ctx: &CanvasRenderingContext2d) {
        let width = 600.0f32;
        let height = 400.0f32;
        ctx.clear_rect(0.0, 0.0, width as f64, height as f64);

        // Paddle
        ctx.set_fill_style_str("#fff");
        let pw = 60.0f32;
        let ph = 10.0f32;
        ctx.fill_rect(
            (self.game.paddle_x * width) as f64 - (pw / 2.0f32) as f64,
            (height - ph) as f64,
            pw as f64,
            ph as f64,
        );

        // Real Ball (Blue)
        ctx.begin_path();
        ctx.set_fill_style_str("#4a90e2");
        let _ = ctx.arc(
            (self.game.ball_x * width) as f64,
            (self.game.ball_y * height) as f64,
            8.0,
            0.0,
            std::f64::consts::PI * 2.0,
        );
        ctx.fill();

        // Predicted Ball (Orange)
        if self.epoch_count > 10 {
            ctx.begin_path();
            ctx.set_fill_style_str("rgba(245, 166, 35, 0.6)");
            let _ = ctx.arc(
                (self.pred_pos[1] * width) as f64,
                (self.pred_pos[2] * height) as f64,
                6.0,
                0.0,
                std::f64::consts::PI * 2.0,
            );
            ctx.fill();
        }
    }
}
