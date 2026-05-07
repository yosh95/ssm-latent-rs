#![recursion_limit = "256"]
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use burn::optim::AdamConfig;
use burn::optim::Optimizer;
use burn::tensor::{Tensor, TensorData};
use hf_hub::api::sync::Api;
use ndarray::{Array2, Axis};
use ort::session::Session;
use ort::value::Tensor as OrtTensor;
use ort::execution_providers::CUDAExecutionProvider;
use rand::seq::IndexedRandom;
use rand::RngExt;
use ssm_latent_model::latent::LatentPredictor;
use ssm_latent_model::ssm::SsmConfig;
use std::fs;
use std::io::{self, Write};
use std::thread::sleep;
use std::time::Duration;
use tokenizers::Tokenizer;

type MyBackend = Autodiff<Wgpu>;

const RESET: &str = "\x1b[0m";
const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
// const BLUE: &str = "\x1b[34m";
const BOLD: &str = "\x1b[1m";

pub struct LogEmbedder {
    session: Session,
    tokenizer: Tokenizer,
}

impl LogEmbedder {
    pub fn new(model_id: &str) -> Self {
        let api = Api::new().unwrap();
        let repo = api.model(model_id.to_string());
        let model_path = repo.get("onnx/model.onnx").or_else(|_| repo.get("model.onnx")).unwrap();
        let tokenizer_path = repo.get("tokenizer.json").unwrap();
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
        let model_bytes = fs::read(model_path).unwrap();
        
        // CUDAの初期化を試み、失敗した場合は標準（CPU）セッションを使用
        let session = match Session::builder().unwrap().with_execution_providers([CUDAExecutionProvider::default().build()]) {
            Ok(builder) => builder.with_intra_threads(1).unwrap().commit_from_memory(&model_bytes).unwrap(),
            Err(_) => {
                println!("{}CUDA execution provider is not available. Falling back to CPU for embeddings.{}", YELLOW, RESET);
                Session::builder().unwrap().with_intra_threads(1).unwrap().commit_from_memory(&model_bytes).unwrap()
            }
        };
        
        Self { session, tokenizer }
    }

    pub fn embed(&mut self, text: &str) -> Vec<f32> {
        let encoding = self.tokenizer.encode(text, true).unwrap();
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
        let tids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();
        let seq = ids.len();
        
        let input_ids = Array2::from_shape_vec((1, seq), ids).unwrap();
        let attention_mask = Array2::from_shape_vec((1, seq), mask).unwrap();
        let token_type_ids = Array2::from_shape_vec((1, seq), tids).unwrap();

        let inputs = ort::inputs![
            "input_ids" => OrtTensor::from_array(input_ids).unwrap(),
            "attention_mask" => OrtTensor::from_array(attention_mask).unwrap(),
            "token_type_ids" => OrtTensor::from_array(token_type_ids).unwrap(),
        ];

        let outputs = self.session.run(inputs).expect("Session run failed");
        let embeddings = outputs[0].try_extract_array::<f32>().unwrap();
        embeddings.mean_axis(Axis(1)).unwrap().to_owned().into_raw_vec_and_offset().0
    }
}

fn pause_if_enabled(enabled: bool, message: &str) {
    if !enabled { return; }
    print!("\n{}[PAUSE]{} {} (Press Enter to continue...){}", YELLOW, BOLD, message, RESET);
    io::stdout().flush().unwrap();
    let mut _s = String::new();
    io::stdin().read_line(&mut _s).unwrap();
}

fn generate_logs_in_memory() -> (Vec<String>, Vec<String>) {
    let mut rng = rand::rng();
    let users = ["admin", "user1", "user2", "api_worker", "guest", "operator_01"];
    let services = ["auth", "database", "gateway", "storage", "cache", "web-ui"];
    let status = ["OK", "Success", "Processed", "Redirected"];
    let actions = ["GET", "POST", "PUT", "DELETE", "UPDATE"];
    
    let gen_ip = |r: &mut rand::rngs::ThreadRng| -> String {
        format!("{}.{}.{}.{}", r.random_range(1..255), r.random_range(0..255), r.random_range(0..255), r.random_range(1..255))
    };

    let mut normal_logs = Vec::new();
    for _ in 0..500 {
        let user = users.choose(&mut rng).unwrap();
        let svc = services.choose(&mut rng).unwrap();
        let st = status.choose(&mut rng).unwrap();
        let act = actions.choose(&mut rng).unwrap();
        let ms = rng.random_range(2..150);
        let ip = gen_ip(&mut rng);
        normal_logs.push(format!("INFO: [{}] {} request from {} by {} - {}. ({}ms)", svc, act, ip, user, st, ms));
    }
    
    let mut anomalous_logs = Vec::new();
    for _ in 0..200 {
        if rng.random_bool(0.12) {
            let user = users.choose(&mut rng).unwrap();
            let svc = services.choose(&mut rng).unwrap();
            let ip = gen_ip(&mut rng);
            let anomalies = [
                format!("INFO: [kernel] CRITICAL memory corruption at 0x{:X}", rng.random_range(0..0xFFFFFF)),
                format!("INFO: potential session hijacking from {} origin mismatch", ip),
                format!("INFO: unusual latency spike ({}ms) in auth module", rng.random_range(5000..9999)),
                format!("INFO: failed to verify signature for package update from {}", ip),
                format!("INFO: unexpected binary execution in /tmp by {}", user),
                format!("INFO: internal consistency check failed in {} filesystem", svc),
            ];
            anomalous_logs.push(anomalies.choose(&mut rng).unwrap().to_string());
        } else {
            let user = users.choose(&mut rng).unwrap();
            let svc = services.choose(&mut rng).unwrap();
            let st = status.choose(&mut rng).unwrap();
            let act = actions.choose(&mut rng).unwrap();
            let ip = gen_ip(&mut rng);
            anomalous_logs.push(format!("INFO: [{}] {} request from {} by {}: {}.", svc, act, ip, user, st));
        }
    }
    (normal_logs, anomalous_logs)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let use_pause = args.contains(&"--pause".to_string());
    
    // WGPUは自動的に最適なバックエンド（Vulkan/Metal/DX12/etc）を選択する
    let device = WgpuDevice::default();
    println!("\n{}=== SSM LOG WORLD MODEL: END-TO-END DEMO ==={}", BOLD, RESET);
    println!("Device: {:?}", device);
    
    // --- STEP 0: GENERATE DATA ---
    println!("\n{}[Step 0/4]{} Generating Randomized Log Data...", BOLD, RESET);
    let (normal_data, anomalous_data) = generate_logs_in_memory();
    println!("Data generated successfully (in-memory).");

    println!("\n{}--- Data Samples (First 20) ---{}", BOLD, RESET);
    println!("{}[Training Samples (Normal)]{}", GREEN, RESET);
    for i in 0..20 {
        println!("  {:02}: {}", i, normal_data[i]);
    }
    
    println!("\n{}[Inference Samples (Mixed)]{}", YELLOW, RESET);
    for i in 0..20 {
        println!("  {:02}: {}", i, anomalous_data[i]);
    }
    println!("----------------------");

    pause_if_enabled(use_pause, "Datasets created.");

    // --- STEP 1: LOAD DATA ---
    println!("\n{}[Step 1/4]{} Model Initialization...", BOLD, RESET);
    let mut embedder = LogEmbedder::new("sentence-transformers/all-MiniLM-L6-v2");
    let config = SsmConfig { d_model: 64, d_state: 16, expand: 2, n_heads: 4, mimo_rank: 1, use_conv: true, conv_kernel: 4 };
    let mut model = LatentPredictor::<MyBackend>::new(&config, 384, 2, &device);
    println!("SentenceTransformer & SSM model are ready.");
    pause_if_enabled(use_pause, "Start training?");

    // --- STEP 2: ACTUAL TRAINING ---
    println!("\n{}[Step 2/4]{} Learning Phase: Backpropagation (200 Epochs)...", BOLD, RESET);
    let mut train_vecs = Vec::new();
    for i in 0..200 {
        train_vecs.push(embedder.embed(&normal_data[i]));
    }
    
    let input_tensor = Tensor::<MyBackend, 3>::from_data(TensorData::new(train_vecs.concat(), [1, 200, 384]), &device);
    let actions = Tensor::zeros([1, 200, 2], &device);

    let mut optim = AdamConfig::new().init();
    let lr = 1e-3;

    for epoch in 1..=200 {
        let (z, pred_z, reconstructed_x) = model.forward(input_tensor.clone(), actions.clone());
        let loss = model.loss(z, pred_z, reconstructed_x, input_tensor.clone(), 1.0, 1.0);
        
        let grads = loss.backward();
        let grads_params = burn::optim::GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads_params);

        if epoch % 10 == 0 || epoch == 1 {
            println!("Epoch {:>2}/200 | Loss: {}{:.8}{}", epoch, GREEN, loss.into_data().as_slice::<f32>().unwrap()[0], RESET);
        }
    }
    
    let (_, _, reconstructed) = model.forward(input_tensor.clone(), actions);
    let mse: f32 = (input_tensor - reconstructed).powf_scalar(2.0).mean().into_data().as_slice::<f32>().unwrap()[0];
    let threshold = mse * 4.0; 
    println!("\nCalibration DONE. Dynamic Threshold: {}{:.6}{}", YELLOW, threshold, RESET);
    pause_if_enabled(use_pause, "Start inference?");

    // --- STEP 3: INFERENCE ---
    println!("\n{}[Step 3/4]{} Real-time Inference...", BOLD, RESET);
    println!("{:<4} | {:<8} | {:<5} | {:<50}", "IDX", "MSE", "RES", "LOG SAMPLE");
    println!("{}", "-".repeat(85));

    let mut tp = 0; let mut fp = 0; let mut fn_count = 0;
    for (i, log) in anomalous_data.iter().enumerate().take(60) {
        let vec = embedder.embed(log);
        let log_tensor = Tensor::<MyBackend, 3>::from_data(TensorData::new(vec, [1, 1, 384]), &device);
        let (_, _, reconstructed) = model.forward(log_tensor.clone(), Tensor::zeros([1, 1, 2], &device));
        let score: f32 = (log_tensor - reconstructed).powf_scalar(2.0).mean().into_data().as_slice::<f32>().unwrap()[0];
        
        // Revised anomaly criteria: An anomaly is a log that doesn't follow the typical "INFO: [svc] ACT..." pattern
        let actual_anomaly = !log.contains("] ") || log.contains("CRITICAL") || log.contains("spike");
        let predicted_anomaly = score > threshold;

        let (tag, color) = match (actual_anomaly, predicted_anomaly) {
            (true, true) => { tp += 1; ("TP", RED) },
            (false, true) => { fp += 1; ("FP", YELLOW) },
            (true, false) => { fn_count += 1; ("FN", YELLOW) },
            (false, false) => { ("TN", GREEN) },
        };

        println!("{:<4} | {:>8.6} | {}{:<3}{} | {}{}{}", i, score, color, tag, RESET, color, if log.len() > 60 { &log[..60] } else { log }, RESET);
        
        if actual_anomaly { sleep(Duration::from_millis(400)); }
        else { sleep(Duration::from_millis(20)); }
    }

    // --- STEP 4: FINAL REPORT ---
    let total_anomalies = tp + fn_count;
    let total_normals = 60 - total_anomalies;
    
    let precision = if tp + fp > 0 { tp as f32 / (tp + fp) as f32 } else { 0.0 };
    let recall = if tp + fn_count > 0 { tp as f32 / (tp + fn_count) as f32 } else { 0.0 };
    let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
    let fpr = if total_normals > 0 { fp as f32 / total_normals as f32 } else { 0.0 };
    let fnr = if total_anomalies > 0 { fn_count as f32 / total_anomalies as f32 } else { 0.0 };

    println!("\n{}[Step 4/4]{} Final Report", BOLD, RESET);
    println!("-------------------------------------------");
    println!("{:<20} : {}{:.4}{}", "Precision", GREEN, precision, RESET);
    println!("{:<20} : {}{:.4}{}", "Recall", GREEN, recall, RESET);
    println!("{:<20} : {}{:.4}{}", "F1-Score", GREEN, f1, RESET);
    println!("-------------------------------------------");
    println!("{:<20} : {}{:.4}{}", "False Positive Rate", YELLOW, fpr, RESET);
    println!("{:<20} : {}{:.4}{}", "False Negative Rate", YELLOW, fnr, RESET);
    println!("-------------------------------------------");
    println!("TP: {}, FP: {}, TN: {}, FN: {}", tp, fp, 60 - (tp + fp + fn_count), fn_count);
    println!("\nSemantic Anomaly Detection complete.");
}
