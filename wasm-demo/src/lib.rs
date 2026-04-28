use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, HtmlButtonElement};
use burn::tensor::{Tensor, TensorData};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use ssm_latent_model::latent::{LatentPredictor, LatentState};
use ssm_latent_model::ssm::SsmConfig;
use std::rc::Rc;
use std::cell::RefCell;

macro_rules! log {
    ($($t:tt)*) => (web_sys::console::log_1(&format!($($t)*).into()))
}

// Backend with Autodiff for online training
#[cfg(feature = "wgpu")]
type RawBackend = burn::backend::Wgpu;
#[cfg(not(feature = "wgpu"))]
type RawBackend = burn::backend::NdArray<f32>;

type Backend = burn::backend::Autodiff<RawBackend>;

#[wasm_bindgen(start)]
pub async fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    log!("WASM World Model: Real-time Online Training Start");

    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let canvas = document.get_element_by_id("canvas").unwrap().dyn_into::<HtmlCanvasElement>()?;
    let ctx = canvas.get_context("2d")?.unwrap().dyn_into::<CanvasRenderingContext2d>()?;
    let status = document.get_element_by_id("status").unwrap();
    let start_button = document.get_element_by_id("startButton").unwrap().dyn_into::<HtmlButtonElement>()?;

    status.set_text_content(Some("Model Ready. Click Start to Begin."));

    #[cfg(feature = "wgpu")]
    let device = burn::backend::wgpu::WgpuDevice::default();
    #[cfg(not(feature = "wgpu"))]
    let device = burn::backend::ndarray::NdArrayDevice::default();

    let config = SsmConfig::new(32, 8, 4, 4, 2); 
    let model = LatentPredictor::new(&config, 4, 1, &device);
    let optim = AdamConfig::new().init();
    
    let d_inner = config.d_model * config.expand;
    let d_head = d_inner / config.n_heads;
    let latent_state = LatentState {
        h: Tensor::zeros([1, config.n_heads, config.d_state, d_head / config.mimo_rank], &device),
        prev_bx: None,
        conv_state: Some(Tensor::zeros([1, d_inner, config.conv_kernel - 1], &device)),
    };

    let app = Rc::new(RefCell::new(AppState {
        model,
        optim,
        device,
        theta: 0.0,
        time: 0.0,
        pred_theta: 0.0,
        latent_state: Some(latent_state),
        z_prev: None,
        config,
        history_obs: Vec::new(),
        history_act: Vec::new(),
        epoch_count: 0,
        loss: 0.0,
        frame_count: 0,
    }));

    app.borrow().draw(&ctx);

    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();

    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        let mut app = app.borrow_mut();
        app.update();
        app.draw(&ctx);
        
        status.set_text_content(Some(&format!(
            "Epoch: {} | Loss: {:.6} | Mode: {}", 
            app.epoch_count, 
            app.loss,
            if app.epoch_count < 50 { "Initial Training..." } else { "Reproducing Metronome" }
        )));

        web_sys::window().unwrap().request_animation_frame(f.borrow().as_ref().unwrap().as_ref().unchecked_ref()).unwrap();
    }) as Box<dyn FnMut()>));

    let start_button_clone = start_button.clone();
    let on_click = Closure::wrap(Box::new(move || {
        start_button_clone.set_disabled(true);
        start_button_clone.set_inner_text("Running...");
        let window = web_sys::window().unwrap();
        window.request_animation_frame(g.borrow().as_ref().unwrap().as_ref().unchecked_ref()).unwrap();
    }) as Box<dyn FnMut()>);

    start_button.set_onclick(Some(on_click.as_ref().unchecked_ref()));
    on_click.forget();

    Ok(())
}

struct AppState {
    model: LatentPredictor<Backend>,
    optim: burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, LatentPredictor<Backend>, Backend>,
    #[cfg(feature = "wgpu")]
    device: burn::backend::wgpu::WgpuDevice,
    #[cfg(not(feature = "wgpu"))]
    device: burn::backend::ndarray::NdArrayDevice,
    theta: f64,
    time: f64,
    pred_theta: f64,
    latent_state: Option<LatentState<Backend>>,
    z_prev: Option<Tensor<Backend, 2>>,
    config: SsmConfig,
    history_obs: Vec<f32>,
    history_act: Vec<f32>,
    epoch_count: usize,
    loss: f32,
    frame_count: usize,
}

impl AppState {
    fn update(&mut self) {
        self.frame_count += 1;
        // 1. Metronome Physics
        self.time += 0.05;
        self.theta = (self.time).sin() * 1.2;
        let action = 0.0; 
        
        let current_obs = [0.0, 0.0, self.theta as f32, (self.time).cos() as f32];
        self.history_obs.extend_from_slice(&current_obs);
        self.history_act.push(action);

        let max_history = 128; // Increased for better periodic learning
        if self.history_act.len() > max_history {
            self.history_obs.drain(0..4);
            self.history_act.remove(0);
        }

        // 2. Continuous Training (Every 10 frames)
        if self.history_act.len() == max_history && self.frame_count % 10 == 0 {
            self.train_step();
            self.epoch_count += 1;
        }

        // Softly sync imagination with reality
        let obs = Tensor::<Backend, 3>::from_data(TensorData::new(current_obs.to_vec(), [1, 1, 4]), &self.device);
        let encoded_obs = self.model.encode(obs).reshape([1, self.config.d_model]);

        if let Some(z) = self.z_prev.take() {
            // 98% imagination, 2% reality (softer sync for smoothness)
            let mix_ratio = if self.epoch_count < 100 { 0.02 } else { 0.005 };
            self.z_prev = Some(z * (1.0 - mix_ratio) + encoded_obs * mix_ratio);
        } else {
            self.z_prev = Some(encoded_obs);
            
            let d_inner = self.config.d_model * self.config.expand;
            let d_head = d_inner / self.config.n_heads;
            self.latent_state = Some(LatentState {
                h: Tensor::zeros([1, self.config.n_heads, self.config.d_state, d_head / self.config.mimo_rank], &self.device),
                prev_bx: None,
                conv_state: Some(Tensor::zeros([1, d_inner, self.config.conv_kernel - 1], &self.device)),
            });
        }

        if let (Some(z_curr), Some(state)) = (self.z_prev.take(), self.latent_state.take()) {
            // Decode CURRENT estimate for visualization to avoid 1-frame lag
            let decoded = self.model.decode(z_curr.clone().unsqueeze_dim::<3>(1));
            self.pred_theta = decoded.into_data().as_slice::<f32>().unwrap()[2] as f64;

            let action_tensor = Tensor::<Backend, 2>::from_data(TensorData::new(vec![action], [1, 1]), &self.device);
            
            // Step to predict NEXT state
            let (y, next_state) = self.model.step(z_curr, action_tensor, state);
            
            // Detach tensors from the computation graph to save memory during pure inference
            self.z_prev = Some(y.detach());
            self.latent_state = Some(LatentState {
                h: next_state.h.detach(),
                prev_bx: next_state.prev_bx.map(|t| t.detach()),
                conv_state: next_state.conv_state.map(|t| t.detach()),
            });
        }
    }

    fn train_step(&mut self) {
        let seq_len = self.history_act.len();
        let obs_tensor = Tensor::<Backend, 3>::from_data(
            TensorData::new(self.history_obs.clone(), [1, seq_len, 4]),
            &self.device
        );
        let act_tensor = Tensor::<Backend, 3>::from_data(
            TensorData::new(self.history_act.clone(), [1, seq_len, 1]),
            &self.device
        );

        // Forward pass
        let (z, pred_z, recons_x) = self.model.forward(obs_tensor.clone(), act_tensor);
        let loss = self.model.loss(z, pred_z, recons_x, obs_tensor, 0.05);
        
        self.loss = loss.clone().into_data().as_slice::<f32>().unwrap()[0];

        // Backward and Optimize with a smaller learning rate for stability
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        self.model = self.optim.step(5e-4, self.model.clone(), grads);
    }

    fn draw(&self, ctx: &CanvasRenderingContext2d) {
        ctx.clear_rect(0.0, 0.0, 600.0, 400.0);
        let cx = 300.0;
        let cy = 200.0;
        let len = 150.0;

        // Base
        ctx.set_fill_style_str("#333");
        ctx.fill_rect(cx - 25.0, cy - 5.0, 50.0, 10.0);

        // Reality: Blue
        ctx.set_line_width(6.0);
        ctx.set_stroke_style_str("#4a90e2");
        ctx.begin_path();
        ctx.move_to(cx, cy);
        ctx.line_to(cx + self.theta.sin() * len, cy - self.theta.cos() * len);
        ctx.stroke();

        // Imagination: Orange
        ctx.set_line_width(3.0);
        ctx.set_stroke_style_str("#f5a623");
        ctx.begin_path();
        ctx.move_to(cx, cy);
        ctx.line_to(cx + self.pred_theta.sin() * len, cy - self.pred_theta.cos() * len);
        ctx.stroke();
        
        ctx.set_font("16px sans-serif");
        ctx.set_fill_style_str("#4a90e2");
        let _ = ctx.fill_text("Reality (Blue line)", 20.0, 40.0);
        ctx.set_fill_style_str("#f5a623");
        let _ = ctx.fill_text("Imagination (Orange line)", 20.0, 70.0);
    }
}
