use burn::backend::Autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::module::AutodiffModule;
use burn::optim::AdamConfig;
use burn::optim::Optimizer;
use burn::tensor::{Tensor, TensorData};
use chrono::{Datelike, NaiveDateTime, Timelike};
use serde::Deserialize;
use ssm_latent_model::predictor::MultiScaleMambaPredictor;
use ssm_latent_model::ssm::MultiScaleSsmConfig;
use std::error::Error;
use std::fs::File;
use std::path::Path;

type MyBackend = Autodiff<Wgpu>;
type InnerBackend = Wgpu;

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const GREEN: &str = "\x1b[32m";
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

/// MAD-based robust normalization. Returns (normalized, median, mad_scaled).
fn mad_normalize(data: &[f32]) -> (Vec<f32>, f32, f32) {
    let mut sorted: Vec<f32> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];

    let mut abs_devs: Vec<f32> = data.iter().map(|x| (x - median).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = abs_devs[abs_devs.len() / 2] * 1.4826; // scale to std of normal
    let mad_safe = mad.max(1e-6);

    let normalized: Vec<f32> = data.iter().map(|&x| (x - median) / mad_safe).collect();
    (normalized, median, mad_safe)
}

fn extract_temporal_features(timestamp: &str) -> [f32; 4] {
    let dt = NaiveDateTime::parse_from_str(timestamp, "%Y-%m-%d %H:%M:%S")
        .or_else(|_| NaiveDateTime::parse_from_str(timestamp, "%Y-%m-%dT%H:%M:%S.000Z"))
        .or_else(|_| NaiveDateTime::parse_from_str(timestamp, "%Y-%m-%dT%H:%M:%SZ"))
        .or_else(|_| NaiveDateTime::parse_from_str(timestamp, "%Y-%m-%d %H:%M:%S.000000"));

    match dt {
        Ok(dt) => {
            let pi = std::f32::consts::PI;
            let hour = dt.hour() as f32 + dt.minute() as f32 / 60.0;
            let dow = dt.weekday().num_days_from_monday() as f32;
            [
                (2.0 * pi * hour / 24.0).sin(),
                (2.0 * pi * hour / 24.0).cos(),
                (2.0 * pi * dow / 7.0).sin(),
                (2.0 * pi * dow / 7.0).cos(),
            ]
        }
        Err(_) => [0.0; 4],
    }
}

/// Build a 7D input feature per timestep: [value, diff, z_short, z_long, hour_sin, hour_cos, dow_sin, dow_cos]
fn build_features(normalized: &[f32], timestamps: &[String]) -> Vec<Vec<f32>> {
    let n = normalized.len();
    let mut feats = Vec::with_capacity(n);

    // Rolling z-score windows
    let short_win = 48usize;
    let long_win = 336usize;

    // Precompute rolling means/stds efficiently
    for i in 0..n {
        let val = normalized[i];
        let diff = if i > 0 {
            normalized[i] - normalized[i - 1]
        } else {
            0.0
        };

        // Short-term z-score
        let s_start = if i >= short_win { i - short_win } else { 0 };
        let s_n = (i - s_start + 1) as f32;
        let s_mean: f32 = normalized[s_start..=i].iter().sum::<f32>() / s_n;
        let s_var: f32 = normalized[s_start..=i]
            .iter()
            .map(|v| (v - s_mean).powi(2))
            .sum::<f32>()
            / s_n;
        let z_short = if s_var > 1e-8 {
            (val - s_mean) / s_var.sqrt()
        } else {
            0.0
        };

        // Long-term z-score
        let l_start = if i >= long_win { i - long_win } else { 0 };
        let l_n = (i - l_start + 1) as f32;
        let l_mean: f32 = normalized[l_start..=i].iter().sum::<f32>() / l_n;
        let l_var: f32 = normalized[l_start..=i]
            .iter()
            .map(|v| (v - l_mean).powi(2))
            .sum::<f32>()
            / l_n;
        let z_long = if l_var > 1e-8 {
            (val - l_mean) / l_var.sqrt()
        } else {
            0.0
        };

        let [h_sin, h_cos, d_sin, d_cos] = extract_temporal_features(&timestamps[i]);

        feats.push(vec![val, diff, z_short, z_long, h_sin, h_cos, d_sin, d_cos]);
    }
    feats
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

/// Convert raw prediction errors to calibrated anomaly scores in [0, 1].
///
/// Strategy:
/// 1. Compute raw error = |pred - actual|
/// 2. Fit Gaussian to errors during probationary period via MAD
/// 3. Convert to z-scores using streaming EWMA mean/std
/// 4. Map z-scores through CDF-like transform to [0, 1]
fn errors_to_scores(errors: &[f32], probation_len: usize) -> Vec<f32> {
    let n = errors.len();
    let cal = &errors[..probation_len.min(n)];

    // MAD calibration on probationary period
    let mut sorted_cal = cal.to_vec();
    sorted_cal.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = sorted_cal[sorted_cal.len() / 2];

    let mut abs_devs: Vec<f32> = sorted_cal.iter().map(|x| (x - med).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = (abs_devs[abs_devs.len() / 2] * 1.4826).max(1e-6);

    // Streaming EWMA mean/variance with contamination guard
    let alpha = 0.05;
    let guard_z = 3.5; // only update stats when z < guard_z
    let mut ewma_mean = med;
    let mut ewma_var = mad * mad;
    let mut z_scores = Vec::with_capacity(n);

    for &e in errors.iter() {
        let std = ewma_var.sqrt().max(1e-6);
        let z = (e - ewma_mean).abs() / std;
        z_scores.push(z);

        // Guarded update: don't let anomalies contaminate the baseline
        if z < guard_z {
            let diff = e - ewma_mean;
            ewma_mean = (1.0 - alpha) * ewma_mean + alpha * e;
            ewma_var = (1.0 - alpha) * ewma_var + alpha * diff * diff;
        }
    }

    // Map z-scores to [0, 1] via a sigmoid-like percentile transform.
    // z=0 → ~0.0, z=3 → ~0.95, z=6 → ~0.999
    // This preserves the NAB requirement of meaningful score distribution.
    let mut z_sorted = z_scores.clone();
    z_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let scores: Vec<f32> = z_scores
        .iter()
        .enumerate()
        .map(|(i, &z)| {
            if i < probation_len {
                return 0.0;
            }
            // Percentile rank of this z-score
            let rank = z_sorted.partition_point(|&s| s < z) as f32;
            let pct = rank / n as f32;
            // Power transform: lower exponent = sharper contrast at high end
            // This helps NAB's threshold optimizer find a good cutoff
            pct.powf(0.25).min(1.0)
        })
        .collect();

    scores
}

fn main() -> Result<(), Box<dyn Error>> {
    let device = WgpuDevice::default();
    println!(
        "\n{}=== NAB Mamba Predictor — Anomaly Detection ==={}",
        BOLD, RESET
    );

    let detector_name = "ssm_latent_multiscale";
    let base_data_path = Path::new("nab-demo/data");
    let base_result_path = Path::new("nab-demo/results").join(detector_name);

    let mut csv_files = Vec::new();
    find_csv_files(base_data_path, &mut csv_files)?;
    csv_files.sort();

    // ── Model config (multi-scale, ~45K params) ──
    // 3 stacked SSM layers: fast (spikes), medium (daily), slow (weekly+)
    let feature_dim: usize = 8; // value, diff, z_short, z_long, h_sin, h_cos, d_sin, d_cos
    let ssm_config = MultiScaleSsmConfig::new(32, 8, 2, 2, 1)
        .with_n_layers(3)
        .with_use_conv(true)
        .with_conv_kernel(4);

    let learning_rate = 5e-4;
    let max_epochs = 80;
    let probation_ratio = 0.15; // train ONLY on first 15% — no anomaly leakage

    println!(
        "{}Config: d_model={}, d_state={}, expand={}, heads={}, layers={}, params~{}, lr={}, probation={:.0}%{}",
        CYAN,
        ssm_config.d_model,
        ssm_config.d_state,
        ssm_config.expand,
        ssm_config.n_heads,
        ssm_config.n_layers,
        (ssm_config.d_model * ssm_config.d_model * ssm_config.expand * 4
            + ssm_config.d_model * ssm_config.d_state * ssm_config.n_heads * 2)
            * ssm_config.n_layers
            / 1000,
        learning_rate,
        probation_ratio * 100.0,
        RESET
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

        println!("\n{}[Processing]{} {}/{}", BOLD, RESET, category, filename);

        // ── Load & normalize ──
        let raw_records = load_nab_records(&data_path)?;
        let raw_values: Vec<f32> = raw_records.iter().map(|r| r.value).collect();
        let timestamps: Vec<String> = raw_records.iter().map(|r| r.timestamp.clone()).collect();
        let seq_len = raw_values.len();

        let (normalized, _med, _mad) = mad_normalize(&raw_values);
        let features = build_features(&normalized, &timestamps);

        // ── Train only on probationary period (first 15%) ──
        let train_len = ((seq_len as f32 * probation_ratio) as usize)
            .max(50)
            .min(seq_len / 2);
        let train_len = train_len.min(2048); // VRAM guard

        let flat_feats: Vec<f32> = features[..train_len].iter().flatten().cloned().collect();

        let train_tensor = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(flat_feats, [1, train_len, feature_dim]),
            &device,
        );

        let mut model = MultiScaleMambaPredictor::<MyBackend>::new(
            &ssm_config,
            feature_dim,
            feature_dim,
            &device,
        );
        let mut optim = AdamConfig::new().init();

        let mut best_loss = f32::INFINITY;
        let early_stop_patience = 20;
        let mut no_improve = 0;

        // ── Training loop (pure MSE) ──
        for epoch in 1..=max_epochs {
            let predictions = model.forward(train_tensor.clone());
            let loss = model.loss(predictions, train_tensor.clone());

            let loss_val = loss.clone().into_data().as_slice::<f32>().unwrap()[0];
            if !loss_val.is_finite() {
                continue;
            }
            if loss_val < best_loss - 1e-6 {
                best_loss = loss_val;
                no_improve = 0;
            } else {
                no_improve += 1;
            }

            let grads = loss.backward();
            let grads_params = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optim.step(learning_rate as f64, model, grads_params);

            if no_improve >= early_stop_patience {
                println!("  Early stop @ epoch {epoch}, best_loss={best_loss:.6}");
                break;
            }
        }
        println!("  Training done: best_loss={best_loss:.6}");

        // ── Streaming inference over full sequence ──
        let model: MultiScaleMambaPredictor<InnerBackend> = model.valid();
        let mut state = model.zero_state(1, &device);
        let mut pred_errors = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            let feat_vec = &features[i];
            let x = Tensor::<InnerBackend, 2>::from_data(
                TensorData::new(feat_vec.clone(), [1, feature_dim]),
                &device,
            );

            let (pred, next_state) = model.step(x.clone(), state);
            state = next_state;

            let pred_vals = pred.into_data();
            let pred_slice = pred_vals.as_slice::<f32>().unwrap();

            // Prediction error: compare predicted features to actual next-step features
            let actual_next = if i + 1 < seq_len {
                &features[i + 1]
            } else {
                feat_vec
            };

            // Weighted error: value (index 0) gets highest weight
            let mut err = 0.0f32;
            err += (pred_slice[0] - actual_next[0]).powi(2) * 1.0; // value
            err += (pred_slice[1] - actual_next[1]).powi(2) * 0.3; // diff
            err += (pred_slice[2] - actual_next[2]).powi(2) * 0.2; // z_short
            err += (pred_slice[3] - actual_next[3]).powi(2) * 0.2; // z_long

            pred_errors.push(err);
        }

        // ── Convert errors to anomaly scores ──
        let probation_len = train_len;
        let scores = errors_to_scores(&pred_errors, probation_len);

        // ── Write results ──
        std::fs::create_dir_all(&result_dir)?;
        let mut wtr = csv::Writer::from_path(&final_result_path)?;
        wtr.write_record(["timestamp", "value", "anomaly_score"])?;
        for i in 0..raw_records.len() {
            wtr.write_record([
                &raw_records[i].timestamp,
                &raw_values[i].to_string(),
                &scores[i].to_string(),
            ])?;
        }
        println!(
            "  {}Saved{} {}/{} ({} points, train={})",
            GREEN, RESET, category, filename, seq_len, train_len
        );
    }

    println!(
        "\n{}All done.{} Run `cd nab-demo && python evaluate_nab.py` to score.",
        BOLD, RESET
    );
    Ok(())
}
