use burn::backend::Autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::optim::AdamConfig;
use burn::optim::Optimizer;
use burn::tensor::{Tensor, TensorData};
use serde::Deserialize;
use ssm_latent_model::latent::{LatentLossArgs, LatentPredictor};
use ssm_latent_model::ssm::SsmConfig;
use std::error::Error;
use std::fs::File;
use std::path::Path;

type MyBackend = Autodiff<Wgpu>;

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";

#[derive(Debug, Deserialize, serde::Serialize)]
struct NabRecord {
    timestamp: String,
    value: f32,
}

fn load_nab_records(path: &Path) -> Result<Vec<NabRecord>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut records = Vec::new();
    for result in rdr.deserialize() {
        let record: NabRecord = result?;
        records.push(record);
    }
    Ok(records)
}

fn normalize(data: &[f32]) -> (Vec<f32>, f32, f32) {
    let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let range = (max - min).max(1e-6);
    let normalized = data.iter().map(|&x| (x - min) / range).collect();
    (normalized, min, max)
}

fn find_csv_files(dir: &Path, files: &mut Vec<std::path::PathBuf>) -> Result<(), Box<dyn Error>> {
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                find_csv_files(&path, files)?;
            } else if path.extension().and_then(|s| s.to_str()) == Some("csv") {
                files.push(path);
            }
        }
    }
    Ok(())
}

/// Compute anomaly scores using MAD calibration + EWMA with contamination prevention.
///
/// This replaces the broken sigmoid(z-4) scoring with a proper statistical approach:
/// 1. Use the probationary period to calibrate baseline statistics (median, MAD)
/// 2. Compute z-scores relative to the robust baseline
/// 3. Update EWMA only on non-anomalous observations (contamination prevention)
/// 4. Use a gentler nonlinear transform (sqrt) that preserves score discriminability
/// 5. Min-max normalize to [0, 1] for NAB compatibility
fn compute_anomaly_scores(
    recon_errors: &[f32],
    latent_errors: &[f32],
    probation_len: usize,
    recon_weight: f32,
    latent_weight: f32,
) -> Vec<f32> {
    let seq_len = recon_errors.len();
    assert_eq!(seq_len, latent_errors.len());

    // Combine errors with configurable weights
    let combined: Vec<f32> = recon_errors
        .iter()
        .zip(latent_errors.iter())
        .map(|(&r, &l)| r * recon_weight + l * latent_weight)
        .collect();

    // === Phase 1: MAD calibration on probationary period ===
    let calibration = &combined[..probation_len.min(combined.len())];
    let mut sorted_cal = calibration.to_vec();
    sorted_cal.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted_cal[sorted_cal.len() / 2];

    // MAD (Median Absolute Deviation) — robust estimator of spread
    let mut abs_devs: Vec<f32> = sorted_cal.iter().map(|x| (x - median).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = abs_devs[abs_devs.len() / 2] * 1.4826; // Scale factor for consistency with std dev
    let robust_std = mad.max(1e-6); // Floor to prevent division by zero

    // === Phase 2: EWMA scoring with contamination prevention ===
    let mut ewma_mean = median; // Initialize with robust calibration value
    let mut ewma_var = robust_std * robust_std; // Initialize with MAD-based variance
    let beta = 0.05; // EWMA smoothing factor (same as original, but better initialized)

    let mut z_scores: Vec<f32> = Vec::with_capacity(seq_len);

    for i in 0..seq_len {
        let err = combined[i];
        let diff = err - ewma_mean;
        let std = ewma_var.sqrt().max(1e-6);
        let z = diff.abs() / std; // Use |z| so anomalies in either direction are detected

        z_scores.push(z);

        // Contamination prevention: only update EWMA on non-anomalous observations
        // This prevents anomalies from polluting the baseline
        if z < 3.0 {
            ewma_mean = (1.0 - beta) * ewma_mean + beta * err;
            let dev = err - ewma_mean;
            ewma_var = (1.0 - beta) * ewma_var + beta * dev * dev;
        }
    }

    // === Phase 3: Min-Max normalization to [0, 1] for NAB ===
    // NAB's sweeper.py expects scores in [0, 1] where higher = more anomalous
    let max_z = z_scores.iter().fold(0.0f32, |a, &b| a.max(b));
    let min_z = z_scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let range = (max_z - min_z).max(1e-8);

    // Use sqrt transform for gentle nonlinear stretching (preserves discriminability)
    let scores: Vec<f32> = z_scores
        .iter()
        .map(|&z| {
            let normalized = (z - min_z) / range;
            // Square root stretches low values and compresses high values
            // This is a much gentler transform than sigmoid(z-4)
            normalized.sqrt().max(0.0).min(1.0)
        })
        .collect();

    // Set probationary period scores to 0 (not evaluated by NAB anyway)
    let mut result = scores;
    for i in 0..probation_len.min(result.len()) {
        result[i] = 0.0;
    }
    result
}

fn main() -> Result<(), Box<dyn Error>> {
    let device = WgpuDevice::default();
    println!(
        "\n{}=== NAB SSM+Latent Anomaly Detection (Phase 1+2 Fix) ==={}",
        BOLD, RESET
    );

    let detector_name = "ssm_latent";
    let base_data_path = Path::new("nab-demo/data");
    let base_result_path = Path::new("nab-demo/results").join(detector_name);

    let mut csv_files = Vec::new();
    find_csv_files(base_data_path, &mut csv_files)?;
    csv_files.sort();

    // ===== Training & Inference Configuration =====
    let train_ratio = 0.50; // Use 50% for training (up from 15%)
    let min_train = 100; // Minimum training samples
    let epochs = 100; // More epochs (up from 50)
    let learning_rate = 5e-4; // Lower LR for more stable training

    // Model configuration — slightly larger for better capacity
    let config = SsmConfig {
        d_model: 64,
        d_state: 16,
        expand: 2,
        n_heads: 4,
        mimo_rank: 1,
        use_conv: true,
        conv_kernel: 4,
    };

    println!(
        "{}Config: d_model={}, d_state={}, expand={}, heads={}, epochs={}, lr={}",
        CYAN, config.d_model, config.d_state, config.expand, config.n_heads, epochs, learning_rate
    );

    for data_path in csv_files {
        let relative_path = data_path.strip_prefix(base_data_path)?;
        if relative_path.to_str().unwrap().contains("README") {
            continue;
        }

        let parts: Vec<_> = relative_path.components().collect();
        if parts.len() < 2 {
            continue;
        }
        let category = parts[0].as_os_str().to_str().unwrap();
        let filename = parts[parts.len() - 1].as_os_str().to_str().unwrap();

        let result_dir = base_result_path.join(category);
        let result_filename = format!("{}_{}", detector_name, filename);
        let final_result_path = result_dir.join(&result_filename);

        if final_result_path.exists() {
            println!("{}[Skipping]{} {}/{}", GREEN, RESET, category, filename);
            continue;
        }

        println!("\n{}[Processing]{} {}/{}", BOLD, RESET, category, filename);

        let raw_records = load_nab_records(&data_path)?;
        let raw_values: Vec<f32> = raw_records.iter().map(|r| r.value).collect();
        let (normalized_values, _, _) = normalize(&raw_values);
        let seq_len = normalized_values.len();

        let train_count = ((seq_len as f32 * train_ratio) as usize).max(min_train);
        let train_data = &normalized_values[..train_count];

        // ===== Phase 1: Training =====
        let mut model = LatentPredictor::<MyBackend>::new(&config, 1, 1, &device);
        let mut optim = AdamConfig::new().init();

        let train_tensor = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(train_data.to_vec(), [1, train_count, 1]),
            &device,
        );

        // WGPU training requires autodiff backend
        let zero_actions = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(vec![0.0f32; train_count], [1, train_count, 1]),
            &device,
        );

        println!(
            "  {}Training:{} {} points, {} epochs",
            YELLOW, RESET, train_count, epochs
        );

        let mut last_loss = f32::INFINITY;
        for epoch in 1..=epochs {
            let (z, pred_z, reconstructed_x, predicted_x) =
                model.forward(train_tensor.clone(), zero_actions.clone());

            let loss = model.loss(LatentLossArgs {
                z,
                pred_z,
                reconstructed_x,
                predicted_x,
                original_x: train_tensor.clone(),
                stability_weight: 0.1,  // Reduced from 1.0 — stability loss was too dominant
                curvature_weight: 0.05, // Reduced from 0.5 — curvature was over-smoothing
                recon_weight: 5.0,      // Increased from 2.0 — reconstruction quality matters
            });

            let loss_val = loss.to_data().as_slice::<f32>().unwrap()[0];
            let grads = loss.backward();
            let grads_params = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optim.step(learning_rate, model, grads_params);

            if epoch % 20 == 0 || epoch == 1 {
                println!("    Epoch {}: loss = {:.6}", epoch, loss_val);
            }
            last_loss = loss_val;
        }
        println!("  Final training loss: {:.6}", last_loss);

        // ===== Phase 2: Inference =====
        let mut full_recon_err = Vec::with_capacity(seq_len);
        let mut full_latent_err = Vec::with_capacity(seq_len);

        // Process in larger chunks (10000 vs old 5000) to reduce boundary artifacts.
        // For sequences shorter than 10000, this processes the entire sequence at once.
        println!("  {}Inference:{} processing {} points", YELLOW, RESET, seq_len);

        let chunk_size = 10000;
        for start in (0..seq_len).step_by(chunk_size) {
            let end = (start + chunk_size).min(seq_len);
            let current_chunk = &normalized_values[start..end];
            let chunk_tensor = Tensor::<MyBackend, 3>::from_data(
                TensorData::new(current_chunk.to_vec(), [1, current_chunk.len(), 1]),
                &device,
            );
            let chunk_actions = Tensor::<MyBackend, 3>::from_data(
                TensorData::new(vec![0.0f32; current_chunk.len()], [1, current_chunk.len(), 1]),
                &device,
            );

            let (z, pred_z, reconstructed, _) =
                model.forward(chunk_tensor.clone(), chunk_actions);

            // Compute per-timestep scalar errors by taking mean over the feature dimension.
            // reconstructed shape: [1, seq_len, 1], z/pred_z shape: [1, seq_len, d_model]
            // We need one error value per timestep, so we mean-reduce over the last dim.
            let r_err = (chunk_tensor.clone() - reconstructed)
                .powf_scalar(2.0)
                .mean_dim(2) // [1, seq_len, 1] → [1, seq_len] after squeeze
                .into_data()
                .as_slice::<f32>()
                .unwrap()
                .to_vec();
            let l_err = (z - pred_z)
                .powf_scalar(2.0)
                .mean_dim(2) // [1, seq_len, d_model] → [1, seq_len] after mean over features
                .into_data()
                .as_slice::<f32>()
                .unwrap()
                .to_vec();

            full_recon_err.extend(r_err);
            full_latent_err.extend(l_err);
        }

        // ===== Phase 3: Compute Anomaly Scores (MAD + EWMA + Contamination Prevention) =====
        let probation_len = ((seq_len as f32 * 0.15) as usize).max(50);
        let scores = compute_anomaly_scores(
            &full_recon_err,
            &full_latent_err,
            probation_len,
            0.5, // recon_weight — balanced with latent
            0.5, // latent_weight — balanced with recon
        );

        println!(
            "  {}Scores:{} min={:.4}, max={:.4}, mean={:.4}",
            YELLOW,
            RESET,
            scores.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            scores.iter().fold(0.0f32, |a, &b| a.max(b)),
            scores.iter().sum::<f32>() / scores.len() as f32
        );

        // ===== Save Results =====
        std::fs::create_dir_all(&result_dir)?;
        let mut wtr = csv::Writer::from_path(&final_result_path)?;
        wtr.write_record(&["timestamp", "value", "anomaly_score"])?;
        for i in 0..raw_records.len() {
            wtr.write_record(&[
                &raw_records[i].timestamp,
                &raw_values[i].to_string(),
                &scores[i].to_string(),
            ])?;
        }
        println!(
            "  {}Saved:{} {}",
            GREEN,
            RESET,
            final_result_path.display()
        );
    }

    println!("\n{}All datasets processed.{}  Run `python evaluate_nab.py` for scoring.", BOLD, RESET);
    Ok(())
}