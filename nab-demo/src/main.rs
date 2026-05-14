use burn::backend::Autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::module::AutodiffModule;
use burn::optim::AdamConfig;
use burn::optim::Optimizer;
use burn::tensor::{Tensor, TensorData};
use serde::Deserialize;
use chrono::{NaiveDateTime, Timelike, Datelike};
use ssm_latent_model::latent::{LatentLossArgs, MultiScaleLatentPredictor};
use ssm_latent_model::ssm::{MultiScaleSsmConfig, MultiScaleState};
use std::error::Error;
use std::fs::File;
use std::path::Path;

type MyBackend = Autodiff<Wgpu>;
type InnerBackend = Wgpu;

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const GREEN: &str = "\x1b[32m";
const CYAN: &str = "\x1b[36m";
const YELLOW: &str = "\x1b[33m";

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

fn minmax_normalize(data: &[f32]) -> (Vec<f32>, f32, f32) {
    let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let range = (max - min).max(1e-6);
    let normalized = data.iter().map(|&x| (x - min) / range).collect();
    (normalized, min, max)
}

fn extract_temporal_features(timestamp: &str) -> (f32, f32, f32, f32) {
    let dt = NaiveDateTime::parse_from_str(timestamp, "%Y-%m-%d %H:%M:%S")
        .or_else(|_| NaiveDateTime::parse_from_str(timestamp, "%Y-%m-%dT%H:%M:%S.000Z"))
        .or_else(|_| NaiveDateTime::parse_from_str(timestamp, "%Y-%m-%dT%H:%M:%SZ"))
        .or_else(|_| NaiveDateTime::parse_from_str(timestamp, "%Y-%m-%d %H:%M:%S.000000"));

    match dt {
        Ok(dt) => {
            let hour = dt.hour() as f32 + dt.minute() as f32 / 60.0;
            let dow = dt.weekday().num_days_from_monday() as f32;
            let pi = std::f32::consts::PI;
            let hour_sin = (2.0 * pi * hour / 24.0).sin();
            let hour_cos = (2.0 * pi * hour / 24.0).cos();
            let dow_sin = (2.0 * pi * dow / 7.0).sin();
            let dow_cos = (2.0 * pi * dow / 7.0).cos();
            (hour_sin, hour_cos, dow_sin, dow_cos)
        }
        Err(_) => (0.0, 0.0, 0.0, 0.0),
    }
}

struct RollingStatsTracker {
    window: std::collections::VecDeque<f32>,
    window_size: usize,
    sum: f32,
    sum_sq: f32,
}

impl RollingStatsTracker {
    fn new(window_size: usize) -> Self {
        Self {
            window: std::collections::VecDeque::with_capacity(window_size),
            window_size,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    fn update(&mut self, value: f32, prev_value: f32) -> (f32, f32, f32) {
        if self.window.len() >= self.window_size {
            let old = self.window.pop_front().unwrap();
            self.sum -= old;
            self.sum_sq -= old * old;
        }
        self.window.push_back(value);
        self.sum += value;
        self.sum_sq += value * value;

        let n = self.window.len() as f32;
        let mean = self.sum / n;
        let variance = (self.sum_sq / n - mean * mean).max(0.0);
        let std = variance.sqrt().max(1e-6);
        let z_score = (value - mean) / std;
        let diff = value - prev_value;
        (z_score, diff, mean)
    }
}

/// 7D feature vector (stable, well-tested from Phase 3)
fn build_features(
    normalized_values: &[f32],
    timestamps: &[String],
    rolling_window: usize,
) -> Vec<Vec<f32>> {
    let seq_len = normalized_values.len();
    let mut tracker = RollingStatsTracker::new(rolling_window);
    let mut features = Vec::with_capacity(seq_len);

    for i in 0..seq_len {
        let prev = if i > 0 {
            normalized_values[i - 1]
        } else {
            normalized_values[i]
        };
        let (z_score, diff, _) = tracker.update(normalized_values[i], prev);
        let (h_sin, h_cos, d_sin, d_cos) = extract_temporal_features(&timestamps[i]);

        features.push(vec![
            normalized_values[i], // 0: value
            z_score,              // 1: z-score
            diff,                 // 2: first difference
            h_sin,                // 3: hour sin
            h_cos,                // 4: hour cos
            d_sin,                // 5: day-of-week sin
            d_cos,                // 6: day-of-week cos
        ]);
    }
    features
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

fn compute_anomaly_scores(
    pred_errors: &[f32],
    recon_errors: &[f32],
    latent_errors: &[f32],
    probation_len: usize,
) -> Vec<f32> {
    let seq_len = pred_errors.len();

    // Weighted combination
    let combined: Vec<f32> = (0..seq_len)
        .map(|i| 0.5 * pred_errors[i] + 0.3 * recon_errors[i] + 0.2 * latent_errors[i])
        .collect();

    // MAD-based calibration
    let cal_len = probation_len.min(seq_len);
    let calibration = &combined[..cal_len];
    let mut sorted_cal = calibration.to_vec();
    sorted_cal.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted_cal[sorted_cal.len() / 2];

    let mut abs_devs: Vec<f32> = sorted_cal.iter().map(|x| (x - median).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = abs_devs[abs_devs.len() / 2] * 1.4826;
    let robust_std = mad.max(1e-6);

    // Contamination-guarded EWMA
    let mut ewma_mean = median;
    let mut ewma_var = robust_std * robust_std;
    let beta = 0.03;
    let guard_z = 3.0;

    let mut z_scores: Vec<f32> = Vec::with_capacity(seq_len);
    for &err in combined.iter() {
        let std = ewma_var.sqrt().max(1e-6);
        let z = (err - ewma_mean).abs() / std;
        z_scores.push(z);

        if z < guard_z {
            let diff = err - ewma_mean;
            ewma_mean = (1.0 - beta) * ewma_mean + beta * err;
            ewma_var = (1.0 - beta) * ewma_var + beta * diff * diff;
        }
    }

    // Percentile rank + power transform (Phase 3 proven formula)
    let mut z_sorted = z_scores.clone();
    z_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let scores: Vec<f32> = z_scores
        .iter()
        .map(|&z| {
            let rank = z_sorted.partition_point(|&s| s < z) as f32;
            let percentile = rank / seq_len as f32;
            percentile.powf(0.3).min(1.0)
        })
        .collect();

    // Zero out probationary period
    let mut result = scores;
    for i in 0..probation_len.min(result.len()) {
        result[i] = 0.0;
    }
    result
}

fn main() -> Result<(), Box<dyn Error>> {
    let device = WgpuDevice::default();
    println!(
        "\n{}=== NAB SSM+Latent Multi-Scale Anomaly Detection (Phase 4) ==={}",
        BOLD, RESET
    );

    let detector_name = "ssm_latent_multiscale";
    let base_data_path = Path::new("nab-demo/data");
    let base_result_path = Path::new("nab-demo/results").join(detector_name);

    let mut csv_files = Vec::new();
    find_csv_files(base_data_path, &mut csv_files)?;
    csv_files.sort();

    let train_ratio = 0.75;
    let min_train = 100;
    let learning_rate = 3e-4;
    let max_epochs = 120;
    let feature_dim: usize = 7; // [value, z_score, diff, h_sin, h_cos, d_sin, d_cos]
    let rolling_window: usize = 48;

    // Multi-scale SSM: 2 layers, conservative sizing
    let config = MultiScaleSsmConfig::new(64, 16, 2, 4, 1)
        .with_n_layers(2)
        .with_use_conv(true)
        .with_conv_kernel(4);

    println!(
        "{}Config: d_model={}, d_state={}, expand={}, heads={}, layers={}, max_epochs={}, lr={}{}",
        CYAN,
        config.d_model,
        config.d_state,
        config.expand,
        config.n_heads,
        config.n_layers,
        max_epochs,
        learning_rate,
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

        println!(
            "\n{}[Processing]{} {}/{}",
            BOLD, RESET, category, filename
        );

        let raw_records = load_nab_records(&data_path)?;
        let raw_values: Vec<f32> = raw_records.iter().map(|r| r.value).collect();
        let timestamps: Vec<String> =
            raw_records.iter().map(|r| r.timestamp.clone()).collect();

        let (normalized_values, _, _) = minmax_normalize(&raw_values);
        let seq_len = normalized_values.len();
        let features = build_features(&normalized_values, &timestamps, rolling_window);
        let total_train_count = ((seq_len as f32 * train_ratio) as usize).max(min_train);

        // GPU VRAM protection: chunk if > 4096
        let max_train_seq: usize = 4096;
        let use_chunked_training = total_train_count > max_train_seq;
        let adaptive_epochs = if use_chunked_training {
            (max_epochs / 2).max(30)
        } else {
            let ratio = seq_len as f32 / 4032.0;
            ((max_epochs as f32 * ratio.sqrt()) as usize).clamp(30, max_epochs)
        };
        let early_stop_patience = (adaptive_epochs / 5).max(10);

        let mut best_loss = f32::INFINITY;
        let mut no_improve_count = 0;

        let mut model = MultiScaleLatentPredictor::<MyBackend>::new(
            &config,
            feature_dim,
            feature_dim,
            &device,
        );
        let mut optim = AdamConfig::new().init();

        // ─── Training ────────────────────────────────────────────────
        if use_chunked_training {
            let n_chunks = (total_train_count + max_train_seq - 1) / max_train_seq;
            for epoch in 1..=adaptive_epochs {
                let progress = epoch as f32 / adaptive_epochs as f32;
                let current_lr = learning_rate
                    * (0.1 + 0.9 * 0.5 * (1.0 + (progress * std::f32::consts::PI).cos()));
                let mut epoch_loss = 0.0f32;
                let mut n_updates = 0;

                for chunk_idx in 0..n_chunks {
                    let start = chunk_idx * max_train_seq;
                    let end = (start + max_train_seq).min(total_train_count);
                    let chunk_len = end - start;
                    if chunk_len < 10 {
                        continue;
                    }

                    let chunk_features: Vec<f32> =
                        features[start..end].iter().flatten().cloned().collect();
                    let train_tensor = Tensor::<MyBackend, 3>::from_data(
                        TensorData::new(chunk_features, [1, chunk_len, feature_dim]),
                        &device,
                    );
                    let zero_actions = Tensor::<MyBackend, 3>::from_data(
                        TensorData::new(
                            vec![0.0f32; chunk_len * feature_dim],
                            [1, chunk_len, feature_dim],
                        ),
                        &device,
                    );

                    let (z, pred_z, reconstructed_x, predicted_x) =
                        model.forward(train_tensor.clone(), zero_actions);
                    let loss = model.loss(LatentLossArgs {
                        z,
                        pred_z,
                        reconstructed_x,
                        predicted_x,
                        original_x: train_tensor,
                        stability_weight: 0.1,
                        curvature_weight: 0.05,
                        recon_weight: 5.0,
                    });

                    let loss_val =
                        loss.clone().into_data().as_slice::<f32>().unwrap()[0];
                    if loss_val.is_finite() {
                        epoch_loss += loss_val;
                        n_updates += 1;
                        let grads = loss.backward();
                        let grads_params =
                            burn::optim::GradientsParams::from_grads(grads, &model);
                        model = optim.step(current_lr as f64, model, grads_params);
                    }
                }

                if n_updates == 0 {
                    continue;
                }
                let avg_loss = epoch_loss / n_updates as f32;
                if avg_loss < best_loss - 1e-5 {
                    best_loss = avg_loss;
                    no_improve_count = 0;
                } else {
                    no_improve_count += 1;
                }
                if no_improve_count >= early_stop_patience {
                    println!("  Early stop epoch {}/{}, best={:.6}", epoch, adaptive_epochs, best_loss);
                    break;
                }
            }
        } else {
            let flat_features: Vec<f32> =
                features[..total_train_count].iter().flatten().cloned().collect();
            let train_tensor = Tensor::<MyBackend, 3>::from_data(
                TensorData::new(flat_features, [1, total_train_count, feature_dim]),
                &device,
            );
            let zero_actions = Tensor::<MyBackend, 3>::from_data(
                TensorData::new(
                    vec![0.0f32; total_train_count * feature_dim],
                    [1, total_train_count, feature_dim],
                ),
                &device,
            );

            for epoch in 1..=adaptive_epochs {
                let progress = epoch as f32 / adaptive_epochs as f32;
                let current_lr = learning_rate
                    * (0.1 + 0.9 * 0.5 * (1.0 + (progress * std::f32::consts::PI).cos()));

                let (z, pred_z, reconstructed_x, predicted_x) =
                    model.forward(train_tensor.clone(), zero_actions.clone());
                let loss = model.loss(LatentLossArgs {
                    z,
                    pred_z,
                    reconstructed_x,
                    predicted_x,
                    original_x: train_tensor.clone(),
                    stability_weight: 0.1,
                    curvature_weight: 0.05,
                    recon_weight: 5.0,
                });

                let loss_val =
                    loss.clone().into_data().as_slice::<f32>().unwrap()[0];
                if !loss_val.is_finite() {
                    continue;
                }
                if loss_val < best_loss - 1e-5 {
                    best_loss = loss_val;
                    no_improve_count = 0;
                } else {
                    no_improve_count += 1;
                }

                let grads = loss.backward();
                let grads_params =
                    burn::optim::GradientsParams::from_grads(grads, &model);
                model = optim.step(current_lr as f64, model, grads_params);

                if no_improve_count >= early_stop_patience {
                    println!("  Early stop epoch {}/{}, best={:.6}", epoch, adaptive_epochs, best_loss);
                    break;
                }
            }
        }

        // ─── Inference Phase ─────────────────────────────────────────
        let model: MultiScaleLatentPredictor<InnerBackend> = model.valid();
        let mut pred_errors: Vec<f32> = Vec::with_capacity(seq_len);
        let mut recon_errors: Vec<f32> = Vec::with_capacity(seq_len);
        let mut latent_errors: Vec<f32> = Vec::with_capacity(seq_len);

        let d_inner = config.d_model * config.expand;
        let n_heads = config.n_heads;
        let d_head = d_inner / n_heads;
        let d_head_mimo = d_head / config.mimo_rank;

        let mut state = MultiScaleState::<InnerBackend>::zeros(
            1,
            config.n_layers,
            n_heads,
            config.d_state,
            d_head_mimo,
            config.use_conv,
            d_inner,
            config.conv_kernel,
            &device,
        );

        for i in 0..seq_len {
            let feature_vec = &features[i];
            let x = Tensor::<InnerBackend, 2>::from_data(
                TensorData::new(feature_vec.clone(), [1, feature_dim]),
                &device,
            );
            let action = Tensor::<InnerBackend, 2>::from_data(
                TensorData::new(vec![0.0f32; feature_dim], [1, feature_dim]),
                &device,
            );

            let z = model.encode(x.clone().unsqueeze_dim::<3>(1).swap_dims(0, 1));
            let z_2d = z.clone().squeeze_dims::<2>(&[1]);
            let (y, next_state) = model.step(z_2d.clone(), action, state);
            state = next_state;

            let reconstructed_x = model.decode(z);
            let predicted_x = model.decode(y.clone().unsqueeze_dim::<3>(1));

            let recon_data = reconstructed_x.into_data();
            let pred_data = predicted_x.into_data();
            let recon_vals = recon_data.as_slice::<f32>().unwrap();
            let pred_vals = pred_data.as_slice::<f32>().unwrap();

            let actual_next = if i + 1 < seq_len {
                normalized_values[i + 1]
            } else {
                normalized_values[i]
            };

            pred_errors.push((pred_vals[0] - actual_next).powi(2));
            recon_errors.push((recon_vals[0] - normalized_values[i]).powi(2));

            // Latent prediction error
            let next_vec = if i + 1 < seq_len { &features[i + 1] } else { &features[i] };
            let z_next = Tensor::<InnerBackend, 2>::from_data(
                TensorData::new(next_vec.clone(), [1, feature_dim]),
                &device,
            );
            let z_next_enc = model.encode(z_next.unsqueeze_dim::<3>(1).swap_dims(0, 1));
            let z_next_2d = z_next_enc.squeeze_dims::<2>(&[1]);
            let latent_diff = y.clone() - z_next_2d;
            let latent_err = latent_diff
                .powf_scalar(2.0)
                .sum_dim(1)
                .squeeze_dims::<1>(&[1])
                .into_data()
                .as_slice::<f32>()
                .unwrap()[0];
            latent_errors.push(latent_err);
        }

        let probation_len = ((seq_len as f32 * 0.15) as usize).max(50);
        for i in 0..probation_len.min(seq_len) {
            pred_errors[i] = 0.0;
            recon_errors[i] = 0.0;
            latent_errors[i] = 0.0;
        }

        let scores = compute_anomaly_scores(
            &pred_errors,
            &recon_errors,
            &latent_errors,
            probation_len,
        );

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
            "  {}Saved {}:{} {}/{}",
            GREEN, detector_name, RESET, category, filename
        );
    }

    println!(
        "\n{}All datasets processed.{}  Run `python evaluate_nab.py`",
        BOLD, RESET
    );
    Ok(())
}
