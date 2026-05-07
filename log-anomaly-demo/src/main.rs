#![recursion_limit = "256"]
use burn::backend::Autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::optim::AdamConfig;
use burn::optim::Optimizer;
use burn::tensor::{Tensor, TensorData};
use hf_hub::api::sync::Api;
use ndarray::{Array2, Axis};
use ort::execution_providers::CUDAExecutionProvider;
use ort::session::Session;
use ort::value::Tensor as OrtTensor;
use rand::RngExt;
use rand::seq::IndexedRandom;
use ssm_latent_model::latent::{LatentLossArgs, LatentPredictor};
use ssm_latent_model::ssm::SsmConfig;
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
        let model_path = repo
            .get("onnx/model.onnx")
            .or_else(|_| repo.get("model.onnx"))
            .unwrap();

        // Ensure the external data files are also downloaded
        // Large ONNX models split their weights into .onnx_data files.
        let _ = repo.get("onnx/model.onnx_data_1");
        let _ = repo.get("onnx/model.onnx_data_2"); // Just in case there's more than one

        let tokenizer_path = repo.get("tokenizer.json").unwrap();
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

        // Attempt to initialize CUDA; fall back to a standard (CPU) session if it fails
        let session = match Session::builder()
            .unwrap()
            .with_execution_providers([CUDAExecutionProvider::default().build()])
        {
            Ok(builder) => builder
                .with_intra_threads(1)
                .unwrap()
                .commit_from_file(model_path)
                .unwrap(),
            Err(_) => {
                println!(
                    "{}CUDA execution provider is not available. Falling back to CPU for embeddings.{}",
                    YELLOW, RESET
                );
                Session::builder()
                    .unwrap()
                    .with_intra_threads(1)
                    .unwrap()
                    .commit_from_file(model_path)
                    .unwrap()
            }
        };

        Self { session, tokenizer }
    }

    pub fn embed(&mut self, text: &str) -> Vec<f32> {
        let encoding = self.tokenizer.encode(text, true).unwrap();
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();
        let seq = ids.len();

        let input_ids = Array2::from_shape_vec((1, seq), ids).unwrap();
        let attention_mask = Array2::from_shape_vec((1, seq), mask).unwrap();

        let mut inputs = ort::inputs![
            "input_ids" => OrtTensor::from_array(input_ids).unwrap(),
            "attention_mask" => OrtTensor::from_array(attention_mask).unwrap(),
        ];

        // Check input names expected by the session
        let input_names: Vec<String> = self
            .session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();

        if input_names.iter().any(|n| n == "token_type_ids") {
            let token_type_ids = encoding.get_type_ids();
            let type_ids: Vec<i64> = token_type_ids.iter().map(|&x| x as i64).collect();
            let type_ids_array = Array2::from_shape_vec((1, seq), type_ids).unwrap();
            inputs.push((
                "token_type_ids".into(),
                OrtTensor::from_array(type_ids_array).unwrap().into(),
            ));
        }

        let outputs = self.session.run(inputs).expect("Session run failed");
        let embeddings = outputs[0].try_extract_array::<f32>().unwrap();
        embeddings
            .mean_axis(Axis(1))
            .unwrap()
            .to_owned()
            .into_raw_vec_and_offset()
            .0
    }
}

fn pause_if_enabled(enabled: bool, message: &str) {
    if !enabled {
        return;
    }
    print!(
        "\n{}[PAUSE]{} {} (Press Enter to continue...){}",
        YELLOW, BOLD, message, RESET
    );
    io::stdout().flush().unwrap();
    let mut _s = String::new();
    io::stdin().read_line(&mut _s).unwrap();
}

// ---------------------------------------------------------------------------
// Adaptive Anomaly Threshold: MAD (calibration) + EWMA (online tracking)
// ---------------------------------------------------------------------------

/// Compute the Median Absolute Deviation (MAD) of a slice of f32 scores.
fn mad(scores: &[f32]) -> f32 {
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let mut deviations: Vec<f32> = sorted.iter().map(|&x| (x - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    deviations[deviations.len() / 2]
}

/// Compute a robust threshold using MAD: median + k * 1.4826 * MAD.
/// The constant 1.4826 makes MAD consistent with standard deviation for normal distributions.
fn compute_mad_threshold(scores: &[f32], k: f32) -> f32 {
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let mut deviations: Vec<f32> = sorted.iter().map(|&x| (x - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad_val = deviations[deviations.len() / 2];
    median + k * 1.4826 * mad_val
}

/// Exponential Weighted Moving Average threshold tracker.
/// Updates mean and variance online with O(1) memory, suitable for edge/streaming use.
struct EwmaThreshold {
    mean: f64,
    var: f64,
    alpha: f64,
    k: f64,
    warmup: usize,
    count: usize,
}

impl EwmaThreshold {
    fn new(alpha: f64, k: f64, warmup: usize, initial_mean: f64, initial_var: f64) -> Self {
        Self {
            mean: initial_mean,
            var: initial_var,
            alpha,
            k,
            warmup,
            count: 0,
        }
    }

    /// Return the current EWMA-based threshold without updating any statistics.
    /// Returns `f32::NAN` during the warmup period.
    fn current_threshold(&self) -> f32 {
        if self.count < self.warmup {
            return f32::NAN;
        }
        let std = self.var.max(0.0).sqrt() as f32;
        (self.mean + self.k * std as f64) as f32
    }

    /// Update EWMA statistics with a new observation (no anomaly decision).
    fn observe(&mut self, score: f32) {
        let s = score as f64;
        self.count += 1;
        let old_mean = self.mean;
        self.mean = self.alpha * s + (1.0 - self.alpha) * self.mean;
        let delta = s - old_mean;
        self.var = self.alpha * delta * (s - self.mean) + (1.0 - self.alpha) * self.var;
    }
}

/// Hybrid adaptive threshold combining MAD-based calibration with EWMA online tracking.
/// - Initial threshold is set via MAD on training reconstruction scores (robust to outliers).
/// - Online phase uses EWMA to track distribution drift.
/// - Only **normal** observations update the EWMA, preventing anomaly contamination.
/// - The adaptive threshold never drops below the MAD baseline (hard floor).
struct HybridAdaptiveThreshold {
    baseline_threshold: f32,
    ewma: EwmaThreshold,
}

impl HybridAdaptiveThreshold {
    /// Build from calibration scores (training reconstruction errors).
    /// - `k_mad`: sensitivity for MAD-based initial threshold (default 3.0)
    /// - `alpha`: EWMA smoothing factor (0.05–0.2; smaller = more stable)
    /// - `k_ewma`: sensitivity for EWMA online threshold (default 3.0)
    ///
    /// The baseline is the **maximum** of the MAD threshold and mean*4.0,
    /// so it is never weaker than the old hardcoded threshold.
    fn from_calibration(scores: &[f32], k_mad: f32, alpha: f64, k_ewma: f64) -> Self {
        let mad_threshold = compute_mad_threshold(scores, k_mad);
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let old_threshold = mean * 4.0;
        // Take the higher of the two so we never regress below the old threshold
        let baseline = mad_threshold.max(old_threshold);
        let var = scores.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / scores.len() as f32;
        Self {
            baseline_threshold: baseline,
            ewma: EwmaThreshold::new(alpha, k_ewma, 10, mean as f64, var as f64),
        }
    }

    /// Classify `score` as normal or anomalous.
    /// - Anomalous scores are **not** fed into the EWMA (prevents contamination).
    /// - The effective threshold is the maximum of the EWMA threshold and
    ///   the MAD baseline — the threshold can only rise, never fall below baseline.
    fn update_and_check(&mut self, score: f32) -> (f32, bool) {
        // Peek at the EWMA threshold before deciding
        let ewma_thresh = self.ewma.current_threshold();
        let threshold = if ewma_thresh.is_nan() {
            self.baseline_threshold
        } else {
            // The threshold can only rise; it never drops below baseline
            ewma_thresh.max(self.baseline_threshold)
        };

        let is_anomaly = score > threshold;

        // Only feed normal observations into the EWMA to keep it clean
        if !is_anomaly {
            self.ewma.observe(score);
        }

        (threshold, is_anomaly)
    }
}

fn generate_logs_in_memory() -> (Vec<String>, Vec<String>) {
    let mut rng = rand::rng();
    let users = [
        "admin",
        "user1",
        "user2",
        "api_worker",
        "guest",
        "operator_01",
    ];
    let services = ["auth", "database", "gateway", "storage", "cache", "web-ui"];
    let status = ["OK", "Success", "Processed", "Redirected"];
    let actions = ["GET", "POST", "PUT", "DELETE", "UPDATE"];

    let gen_ip = |r: &mut rand::rngs::ThreadRng| -> String {
        format!(
            "{}.{}.{}.{}",
            r.random_range(1..255),
            r.random_range(0..255),
            r.random_range(0..255),
            r.random_range(1..255)
        )
    };

    let mut normal_logs = Vec::new();
    for _ in 0..500 {
        let user = users.choose(&mut rng).unwrap();
        let svc = services.choose(&mut rng).unwrap();
        let st = status.choose(&mut rng).unwrap();
        let act = actions.choose(&mut rng).unwrap();
        let ms = rng.random_range(2..150);
        let ip = gen_ip(&mut rng);
        normal_logs.push(format!(
            "INFO: [{}] {} request from {} by {} - {}. ({}ms)",
            svc, act, ip, user, st, ms
        ));
    }

    let mut anomalous_logs = Vec::new();
    for _ in 0..200 {
        if rng.random_bool(0.12) {
            let user = users.choose(&mut rng).unwrap();
            let svc = services.choose(&mut rng).unwrap();
            let ip = gen_ip(&mut rng);
            let anomalies = [
                format!(
                    "INFO: [kernel] CRITICAL memory corruption at 0x{:X}",
                    rng.random_range(0..0xFFFFFF)
                ),
                format!(
                    "INFO: potential session hijacking from {} origin mismatch",
                    ip
                ),
                format!(
                    "INFO: unusual latency spike ({}ms) in auth module",
                    rng.random_range(5000..9999)
                ),
                format!(
                    "INFO: failed to verify signature for package update from {}",
                    ip
                ),
                format!("INFO: unexpected binary execution in /tmp by {}", user),
                format!(
                    "INFO: internal consistency check failed in {} filesystem",
                    svc
                ),
            ];
            anomalous_logs.push(anomalies.choose(&mut rng).unwrap().to_string());
        } else {
            let user = users.choose(&mut rng).unwrap();
            let svc = services.choose(&mut rng).unwrap();
            let st = status.choose(&mut rng).unwrap();
            let act = actions.choose(&mut rng).unwrap();
            let ip = gen_ip(&mut rng);
            anomalous_logs.push(format!(
                "INFO: [{}] {} request from {} by {}: {}.",
                svc, act, ip, user, st
            ));
        }
    }
    (normal_logs, anomalous_logs)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let use_pause = args.contains(&"--pause".to_string());

    // WGPU automatically selects the best available backend (Vulkan/Metal/DX12/etc)
    let device = WgpuDevice::default();
    println!(
        "\n{}=== SSM LOG WORLD MODEL: END-TO-END DEMO ==={}",
        BOLD, RESET
    );
    println!("Device: {:?}", device);

    // --- STEP 0: GENERATE DATA ---
    println!(
        "\n{}[Step 0/4]{} Generating Randomized Log Data...",
        BOLD, RESET
    );
    let (normal_data, anomalous_data) = generate_logs_in_memory();
    println!("Data generated successfully (in-memory).");

    println!("\n{}--- Data Samples (First 20) ---{}", BOLD, RESET);
    println!("{}[Training Samples (Normal)]{}", GREEN, RESET);
    for (i, log) in normal_data.iter().enumerate().take(20) {
        println!("  {:02}: {}", i, log);
    }

    println!("\n{}[Inference Samples (Mixed)]{}", YELLOW, RESET);
    for (i, log) in anomalous_data.iter().enumerate().take(20) {
        println!("  {:02}: {}", i, log);
    }
    println!("----------------------");

    pause_if_enabled(use_pause, "Datasets created.");

    // --- STEP 1: LOAD DATA ---
    println!("\n{}[Step 1/4]{} Model Initialization...", BOLD, RESET);
    // let mut embedder = LogEmbedder::new("sentence-transformers/all-MiniLM-L6-v2");
    let mut embedder = LogEmbedder::new("intfloat/multilingual-e5-small");

    // デバッグ: ベクトル次元数を確認
    let test_emb = embedder.embed("test");
    let emb_dim = test_emb.len();
    println!("Embedding dimension: {}", emb_dim);

    let config = SsmConfig {
        d_model: 64,
        d_state: 16,
        expand: 2,
        n_heads: 4,
        mimo_rank: 1,
        use_conv: true,
        conv_kernel: 4,
    };
    let mut model = LatentPredictor::<MyBackend>::new(&config, emb_dim, 2, &device);
    println!("SentenceTransformer & SSM model are ready.");
    pause_if_enabled(use_pause, "Start training?");

    // --- STEP 2: ACTUAL TRAINING ---
    println!(
        "\n{}[Step 2/4]{} Learning Phase: Backpropagation (200 Epochs)...",
        BOLD, RESET
    );
    let mut train_vecs = Vec::new();
    for log in normal_data.iter().take(200) {
        train_vecs.push(embedder.embed(log));
    }

    let input_tensor = Tensor::<MyBackend, 3>::from_data(
        TensorData::new(train_vecs.concat(), [1, 200, emb_dim]),
        &device,
    );
    let actions = Tensor::zeros([1, 200, 2], &device);

    let mut optim = AdamConfig::new().init();
    let lr = 1e-3;

    for epoch in 1..=200 {
        let (z, pred_z, reconstructed_x, predicted_x) =
            model.forward(input_tensor.clone(), actions.clone());
        let loss = model.loss(LatentLossArgs {
            z,
            pred_z,
            reconstructed_x,
            predicted_x,
            original_x: input_tensor.clone(),
            stability_weight: 1.0,
            curvature_weight: 1.0,
            recon_weight: 1.0,
        });

        let grads = loss.backward();
        let grads_params = burn::optim::GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads_params);

        if epoch % 10 == 0 || epoch == 1 {
            println!(
                "Epoch {:>2}/200 | Loss: {}{:.8}{}",
                epoch,
                GREEN,
                loss.into_data().as_slice::<f32>().unwrap()[0],
                RESET
            );
        }
    }

    // --- STEP 2b: CALIBRATION ---
    // Compute per-sample reconstruction scores on training data for MAD-based threshold
    println!(
        "\n{}[Step 2b/4]{} Calibrating adaptive threshold (MAD + EWMA)...",
        BOLD, RESET
    );
    let mut calibration_scores: Vec<f32> = Vec::new();
    for (i, vec) in train_vecs.iter().enumerate() {
        let log_tensor = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(vec.clone(), [1, 1, emb_dim]),
            &device,
        );
        let (_, _, reconstructed, _) =
            model.forward(log_tensor.clone(), Tensor::zeros([1, 1, 2], &device));
        let score: f32 = (log_tensor - reconstructed)
            .powf_scalar(2.0)
            .mean()
            .into_data()
            .as_slice::<f32>()
            .unwrap()[0];
        calibration_scores.push(score);
        if i < 10 {
            println!("  calibration sample {:>3}: score = {:.6}", i, score);
        }
    }

    let mut threshold_engine = HybridAdaptiveThreshold::from_calibration(
        &calibration_scores,
        3.0, // k_mad: MAD sensitivity (3.0 ≈ 99.7% for normal dist)
        0.1, // alpha: EWMA smoothing (0.1 = moderate adaptivity)
        3.0, // k_ewma: EWMA sensitivity
    );

    let baseline = threshold_engine.baseline_threshold;
    let cal_mean = calibration_scores.iter().sum::<f32>() / calibration_scores.len() as f32;
    let cal_std = {
        let variance = calibration_scores
            .iter()
            .map(|&x| (x - cal_mean).powi(2))
            .sum::<f32>()
            / calibration_scores.len() as f32;
        variance.sqrt()
    };
    let old_hardcoded_threshold = cal_mean * 4.0;
    println!(
        "  Calibration statistics: mean={:.6}, std={:.6}, median={:.6}, MAD={:.6}",
        cal_mean,
        cal_std,
        {
            let mut s = calibration_scores.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s[s.len() / 2]
        },
        mad(&calibration_scores),
    );
    println!(
        "  {}Baseline threshold (MAD): {:.6}{}",
        YELLOW, baseline, RESET
    );
    println!(
        "  (old hardcoded: mean*4.0 = {:.6})",
        old_hardcoded_threshold
    );
    pause_if_enabled(use_pause, "Start inference?");

    // --- STEP 3: INFERENCE ---
    println!(
        "\n{}[Step 3/4]{} Real-time Inference (Adaptive Threshold)...",
        BOLD, RESET
    );
    println!(
        "{:<4} | {:<8} | {:<8} | {:<5} | {:<50}",
        "IDX", "MSE", "THRESH", "RES", "LOG SAMPLE"
    );
    println!("{}", "-".repeat(95));

    let mut tp = 0;
    let mut fp = 0;
    let mut fn_count = 0;
    for (i, log) in anomalous_data.iter().enumerate().take(60) {
        let vec = embedder.embed(log);
        let log_tensor =
            Tensor::<MyBackend, 3>::from_data(TensorData::new(vec, [1, 1, emb_dim]), &device);
        let (_, _, reconstructed, _) =
            model.forward(log_tensor.clone(), Tensor::zeros([1, 1, 2], &device));
        let score: f32 = (log_tensor - reconstructed)
            .powf_scalar(2.0)
            .mean()
            .into_data()
            .as_slice::<f32>()
            .unwrap()[0];

        // Revised anomaly criteria: An anomaly is a log that doesn't follow the typical "INFO: [svc] ACT..." pattern
        let actual_anomaly =
            !log.contains("] ") || log.contains("CRITICAL") || log.contains("spike");

        // Adaptive threshold: MAD baseline + EWMA online tracking
        let (current_threshold, predicted_anomaly) = threshold_engine.update_and_check(score);

        let (tag, color) = match (actual_anomaly, predicted_anomaly) {
            (true, true) => {
                tp += 1;
                ("TP", RED)
            }
            (false, true) => {
                fp += 1;
                ("FP", YELLOW)
            }
            (true, false) => {
                fn_count += 1;
                ("FN", YELLOW)
            }
            (false, false) => ("TN", GREEN),
        };

        println!(
            "{:<4} | {:>8.6} | {:>8.6} | {}{:<3}{} | {}{}{}",
            i,
            score,
            current_threshold,
            color,
            tag,
            RESET,
            color,
            if log.len() > 50 { &log[..50] } else { log },
            RESET
        );

        if actual_anomaly {
            sleep(Duration::from_millis(400));
        } else {
            sleep(Duration::from_millis(20));
        }
    }

    // --- STEP 4: FINAL REPORT ---
    let total_anomalies = tp + fn_count;
    let total_normals = 60 - total_anomalies;

    let precision = if tp + fp > 0 {
        tp as f32 / (tp + fp) as f32
    } else {
        0.0
    };
    let recall = if tp + fn_count > 0 {
        tp as f32 / (tp + fn_count) as f32
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    let fpr = if total_normals > 0 {
        fp as f32 / total_normals as f32
    } else {
        0.0
    };
    let fnr = if total_anomalies > 0 {
        fn_count as f32 / total_anomalies as f32
    } else {
        0.0
    };

    println!(
        "\n{}[Step 4/4]{} Final Report (Adaptive Threshold: MAD+EWMA)",
        BOLD, RESET
    );
    println!("-------------------------------------------");
    println!("{:<20} : {}{:.4}{}", "Precision", GREEN, precision, RESET);
    println!("{:<20} : {}{:.4}{}", "Recall", GREEN, recall, RESET);
    println!("{:<20} : {}{:.4}{}", "F1-Score", GREEN, f1, RESET);
    println!("-------------------------------------------");
    println!(
        "{:<20} : {}{:.4}{}",
        "False Positive Rate", YELLOW, fpr, RESET
    );
    println!(
        "{:<20} : {}{:.4}{}",
        "False Negative Rate", YELLOW, fnr, RESET
    );
    println!("-------------------------------------------");
    println!(
        "TP: {}, FP: {}, TN: {}, FN: {}",
        tp,
        fp,
        60 - (tp + fp + fn_count),
        fn_count
    );
    println!("-------------------------------------------");
    println!("{:<20} : {:.6}", "Baseline (MAD k=3)", baseline);
    println!("{:<20} : {:.6}", "Old (mean*4.0)", old_hardcoded_threshold);
    println!("{:<20} : {:.6}", "Calibration Mean±Std", cal_mean);
    println!("\nSemantic Anomaly Detection complete.");
}
