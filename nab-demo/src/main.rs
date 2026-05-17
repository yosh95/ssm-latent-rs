use burn::backend::Autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::module::AutodiffModule;
use burn::optim::AdamConfig;
use burn::optim::Optimizer;
use burn::tensor::{Tensor, TensorData};
use chrono::{Datelike, NaiveDateTime, Timelike};
use serde::Deserialize;
use ssm_latent_model::latent::{LatentLossArgs, MultiScaleLatentPredictor};
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
const YELLOW: &str = "\x1b[33m";

// ─── Config structs (deserialized from nab_config.toml) ─────────────────

#[derive(Debug, Deserialize)]
struct NabConfig {
    mode: ModeConfig,
    model: ModelConfig,
    train: TrainConfig,
    loss: LossConfig,
    scoring: ScoringConfig,
    calibration: CalibrationConfig,
    datasets: DatasetsConfig,
}

#[derive(Debug, Deserialize)]
struct ModeConfig {
    current: String,
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    d_model: usize,
    d_state: usize,
    expand: usize,
    n_heads: usize,
    mimo_rank: usize,
    n_layers: usize,
    use_conv: bool,
    conv_kernel: usize,
}

#[derive(Debug, Deserialize)]
struct TrainConfig {
    learning_rate: f64,
    max_epochs: usize,
    probation_ratio: f64,
    early_stop_patience: usize,
}

#[derive(Debug, Deserialize)]
struct LossConfig {
    stability_weight: f64,
    curvature_weight: f64,
    recon_weight: f64,
}

#[derive(Debug, Deserialize)]
struct ScoringConfig {
    alpha_recon: f64,
    beta_latent: f64,
    gamma_obs: f64,
}

#[derive(Debug, Deserialize)]
struct CalibrationConfig {
    k_mad: f32,
    alpha_ewma: f64,
    k_ewma: f64,
    guard_z: f32,
    power: f32,
}

#[derive(Debug, Deserialize)]
struct DatasetsConfig {
    quick: Vec<String>,
}

// ─── Data loading ───────────────────────────────────────────────────────

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

// ─── Feature engineering ────────────────────────────────────────────────

/// MAD-based robust normalization. Returns (normalized, median, mad_scaled).
fn mad_normalize(data: &[f32]) -> (Vec<f32>, f32, f32) {
    let mut sorted: Vec<f32> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];

    let mut abs_devs: Vec<f32> = data.iter().map(|x| (x - median).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = abs_devs[abs_devs.len() / 2] * 1.4826;
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

/// Build 8D input: [value, diff, z_short, z_long, hour_sin, hour_cos, dow_sin, dow_cos]
fn build_features(normalized: &[f32], timestamps: &[String]) -> Vec<Vec<f32>> {
    let n = normalized.len();
    let mut feats = Vec::with_capacity(n);
    let short_win = 48usize;
    let long_win = 336usize;

    for i in 0..n {
        let val = normalized[i];
        let diff = if i > 0 {
            normalized[i] - normalized[i - 1]
        } else {
            0.0
        };

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

// ─── File discovery ─────────────────────────────────────────────────────

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

// ─── Anomaly score calibration: triple-signal with independent normalization ──

/// Convert three raw error streams to anomaly scores.
///
/// Each signal stream is **independently** calibrated using MAD from the probation
/// period, then combined with configurable weights. The output is a raw anomaly
/// score (unbounded z-score), NOT a [0,1] probability. NAB's optimizer handles
/// threshold selection itself.
///
/// Pipeline:
///   1. Calibrate each signal on probation: median, MAD
///   2. Per-point: z_i = |error_i - median| / MAD  (robust z-score)
///   3. Combine: score[t] = α·z_recon[t] + β·z_latent[t] + γ·z_obs[t]
///   4. Output raw score (no percentile, no power — NAB thresholder does its job)
fn triple_signal_to_scores(
    recon: &[f32],
    latent: &[f32],
    obs_pred: &[f32],
    probation_len: usize,
    _cal: &CalibrationConfig,
    weights: (f64, f64, f64),
) -> Vec<f32> {
    let n = recon.len();
    let cal_len = probation_len.min(n);
    let (alpha_recon, beta_latent, gamma_obs) = weights;

    // ── Calibrate each signal independently on probation period ──
    let (med_r, mad_r) = calc_mad_robust(&recon[..cal_len]);
    let (med_l, mad_l) = calc_mad_robust(&latent[..cal_len]);
    let (med_o, mad_o) = calc_mad_robust(&obs_pred[..cal_len]);

    // ── Compute robust z-scores for the full sequence ──
    (0..n)
        .map(|i| {
            let zr = ((recon[i] - med_r).abs() / mad_r) as f64;
            let zl = ((latent[i] - med_l).abs() / mad_l) as f64;
            let zo = ((obs_pred[i] - med_o).abs() / mad_o) as f64;
            (alpha_recon * zr + beta_latent * zl + gamma_obs * zo) as f32
        })
        .collect()
}

/// Compute median and MAD (with consistency factor 1.4826) for a slice.
fn calc_mad_robust(data: &[f32]) -> (f32, f32) {
    let n = data.len();
    if n == 0 {
        return (0.0, 1e-6);
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = sorted[n / 2];
    let mut abs_devs: Vec<f32> = data.iter().map(|x| (x - med).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = (abs_devs[n / 2] * 1.4826).max(1e-6);
    (med, mad)
}

/// Stream a single error signal through contamination-guarded EWMA, returning z-scores.
#[allow(dead_code)]
fn stream_zscore(errors: &[f32], cal_len: usize, cal: &CalibrationConfig) -> Vec<f32> {
    let n = errors.len();
    let cal_data = &errors[..cal_len.min(n)];

    let (med, mad) = calc_mad_robust(cal_data);

    let alpha = cal.alpha_ewma;
    let guard_z = cal.guard_z;
    let mut ewma_mean = med as f64;
    let mut ewma_var = (mad * mad) as f64;
    let mut z_scores = Vec::with_capacity(n);

    for &e in errors.iter() {
        let std = (ewma_var.sqrt() as f32).max(1e-6);
        let z = (e - ewma_mean as f32).abs() / std;
        z_scores.push(z);

        if z < guard_z {
            let e_f64 = e as f64;
            let diff = e_f64 - ewma_mean;
            ewma_mean = (1.0 - alpha) * ewma_mean + alpha * e_f64;
            ewma_var = (1.0 - alpha) * ewma_var + alpha * diff * diff;
        }
    }
    z_scores
}

fn compute_mad_threshold(scores: &[f32], k: f32) -> f32 {
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let mut deviations: Vec<f32> = sorted.iter().map(|&x| (x - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad_val = deviations[deviations.len() / 2];
    median + k * 1.4826 * mad_val
}

// ─── Main ───────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn Error>> {
    let device = WgpuDevice::default();

    // ── Load config ──
    let config_path = Path::new("nab-demo/config/nab_config.toml");
    let config_str = std::fs::read_to_string(config_path)?;
    let cfg: NabConfig = toml::from_str(&config_str)?;

    let mode = &cfg.mode.current;
    let feature_dim: usize = 8;
    let action_dim: usize = 1; // dummy action (zero) — JEPA fusion requires action_dim > 0

    println!(
        "\n{}=== NAB JEPA Anomaly Detection (mode: {}) ==={}",
        BOLD, mode, RESET
    );

    let detector_name = "ssm_latent_jepa";
    let base_data_path = Path::new("nab-demo/data");
    let base_result_path = Path::new("nab-demo/results").join(detector_name);

    // ── Collect datasets ──
    let mut all_csv_files = Vec::new();
    find_csv_files(base_data_path, &mut all_csv_files)?;
    all_csv_files.sort();

    let csv_files: Vec<std::path::PathBuf> = if mode == "quick" {
        all_csv_files
            .into_iter()
            .filter(|p| {
                let relative = p.strip_prefix(base_data_path).unwrap_or(p);
                let s = relative.to_str().unwrap_or("");
                cfg.datasets
                    .quick
                    .iter()
                    .any(|q| s.ends_with(q) || s == q.as_str())
            })
            .collect()
    } else {
        all_csv_files
    };

    println!("{}Datasets to process: {}{}", CYAN, csv_files.len(), RESET);
    for f in &csv_files {
        let relative = f.strip_prefix(base_data_path).unwrap_or(f);
        println!("  - {}", relative.display());
    }

    // ── Model config ──
    let ssm_config = MultiScaleSsmConfig::new(
        cfg.model.d_model,
        cfg.model.d_state,
        cfg.model.expand,
        cfg.model.n_heads,
        cfg.model.mimo_rank,
    )
    .with_n_layers(cfg.model.n_layers)
    .with_use_conv(cfg.model.use_conv)
    .with_conv_kernel(cfg.model.conv_kernel);

    let learning_rate = cfg.train.learning_rate;
    let max_epochs = cfg.train.max_epochs;
    let probation_ratio = cfg.train.probation_ratio;
    let early_stop_patience = cfg.train.early_stop_patience;

    // JEPA loss weights
    let stability_weight = cfg.loss.stability_weight;
    let curvature_weight = cfg.loss.curvature_weight;
    let recon_weight = cfg.loss.recon_weight;

    // Anomaly score composition
    let alpha_recon = cfg.scoring.alpha_recon;
    let beta_latent = cfg.scoring.beta_latent;
    let gamma_obs = cfg.scoring.gamma_obs;

    println!(
        "{}JEPA config: d_model={}, d_state={}, expand={}, heads={}, layers={}{}",
        CYAN,
        ssm_config.d_model,
        ssm_config.d_state,
        ssm_config.expand,
        ssm_config.n_heads,
        ssm_config.n_layers,
        RESET
    );
    println!(
        "{}Loss weights: stability={}, curvature={}, recon={}{}",
        CYAN, stability_weight, curvature_weight, recon_weight, RESET
    );
    println!(
        "{}Score: α_recon={}, β_latent={}, γ_obs={}{}",
        CYAN, alpha_recon, beta_latent, gamma_obs, RESET
    );
    println!(
        "{}Train: lr={}, epochs={}, probation={:.0}%, patience={}{}",
        CYAN,
        learning_rate,
        max_epochs,
        probation_ratio * 100.0,
        early_stop_patience,
        RESET
    );

    // ── Process each dataset ──
    for data_path in &csv_files {
        let relative_path = data_path.strip_prefix(base_data_path)?;
        let parts: Vec<_> = relative_path.components().collect();
        if parts.len() < 2 {
            continue;
        }
        let category = parts[0].as_os_str().to_str().unwrap();
        let filename = parts[parts.len() - 1].as_os_str().to_str().unwrap();

        let result_dir = base_result_path.join(category);
        let result_filename = format!("{}_{}", detector_name, filename);
        let final_result_path = result_dir.join(&result_filename);

        // Skip if already generated
        if final_result_path.exists() {
            println!(
                "  {}Skip{} {}/{} (already exists)",
                YELLOW, RESET, category, filename
            );
            continue;
        }

        println!("\n{}[Processing]{} {}/{}", BOLD, RESET, category, filename);

        // ── Load & normalize ──
        let raw_records = load_nab_records(data_path)?;
        let raw_values: Vec<f32> = raw_records.iter().map(|r| r.value).collect();
        let timestamps: Vec<String> = raw_records.iter().map(|r| r.timestamp.clone()).collect();
        let seq_len = raw_values.len();

        let (normalized, _med, _mad) = mad_normalize(&raw_values);
        let features = build_features(&normalized, &timestamps);

        // ── Train only on probationary period (first 15%) ──
        let train_len = ((seq_len as f64 * probation_ratio) as usize)
            .max(50)
            .min(seq_len / 2)
            .min(2048);

        let flat_feats: Vec<f32> = features[..train_len].iter().flatten().cloned().collect();
        // Pad to multiple of feature_dim if needed
        let actual_len = flat_feats.len() / feature_dim;
        let train_len_aligned = actual_len;

        let train_tensor = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(flat_feats, [1, train_len_aligned, feature_dim]),
            &device,
        );
        // Dummy zero actions for JEPA
        let zero_actions =
            Tensor::<MyBackend, 3>::zeros([1, train_len_aligned, action_dim], &device);

        let mut model = MultiScaleLatentPredictor::<MyBackend>::new(
            &ssm_config,
            feature_dim,
            action_dim,
            &device,
        );
        let mut optim = AdamConfig::new().init();

        let mut best_loss = f32::INFINITY;
        let mut no_improve = 0;

        // ── JEPA training loop ──
        for epoch in 1..=max_epochs {
            let (z, pred_z, reconstructed_x, predicted_x) =
                model.forward(train_tensor.clone(), zero_actions.clone());

            let loss = model.loss(LatentLossArgs {
                z,
                pred_z,
                reconstructed_x,
                predicted_x,
                original_x: train_tensor.clone(),
                stability_weight,
                curvature_weight,
                recon_weight,
            });

            let loss_val = loss.clone().into_data().as_slice::<f32>().unwrap()[0];
            if !loss_val.is_finite() {
                continue;
            }
            if loss_val < best_loss - 1e-7 {
                best_loss = loss_val;
                no_improve = 0;
            } else {
                no_improve += 1;
            }

            let grads = loss.backward();
            let grads_params = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optim.step(learning_rate, model, grads_params);

            if no_improve >= early_stop_patience {
                println!("  Early stop @ epoch {epoch}, best_loss={best_loss:.6}");
                break;
            }
        }
        println!("  Training done: best_loss={best_loss:.6}");

        // ── JEPA streaming inference with triple-signal anomaly scoring ──
        let model: MultiScaleLatentPredictor<InnerBackend> = model.valid();
        let d_model = cfg.model.d_model;

        let mut state = model.ssm.zero_state(1, &device);
        // Previous predictions (used in next step's error computation)
        let mut prev_pred_z: Option<Tensor<InnerBackend, 2>> = None;
        let mut prev_pred_x: Option<Tensor<InnerBackend, 2>> = None;
        let mut anomaly_scores = Vec::with_capacity(seq_len);
        let mut recon_errs = Vec::with_capacity(seq_len);
        let mut latent_pred_errs = Vec::with_capacity(seq_len);
        let mut obs_pred_errs = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            let feat_vec = &features[i];
            let x = Tensor::<InnerBackend, 2>::from_data(
                TensorData::new(feat_vec.clone(), [1, feature_dim]),
                &device,
            );

            // 1. Encode + decode for reconstruction error
            let z = model
                .encode(x.clone().unsqueeze_dim::<3>(1))
                .reshape([1, d_model]);
            let recon_x = model
                .decode(z.clone().unsqueeze_dim::<3>(1))
                .reshape([1, feature_dim]);

            let recon_err: f32 = (x.clone() - recon_x)
                .powf_scalar(2.0)
                .mean()
                .into_data()
                .as_slice::<f32>()
                .unwrap()[0];
            recon_errs.push(recon_err);

            // 2. Prediction errors from previous step
            let (latent_pred_err, obs_pred_err) =
                if let (Some(pz), Some(px)) = (&prev_pred_z, &prev_pred_x) {
                    let lpe: f32 = (z.clone() - pz.clone())
                        .powf_scalar(2.0)
                        .mean()
                        .into_data()
                        .as_slice::<f32>()
                        .unwrap()[0];
                    let ope: f32 = (x.clone() - px.clone())
                        .powf_scalar(2.0)
                        .mean()
                        .into_data()
                        .as_slice::<f32>()
                        .unwrap()[0];
                    (lpe, ope)
                } else {
                    (0.0, 0.0)
                };
            latent_pred_errs.push(latent_pred_err);
            obs_pred_errs.push(obs_pred_err);

            // 3. Combined anomaly score
            let score = (alpha_recon as f32) * recon_err
                + (beta_latent as f32) * latent_pred_err
                + (gamma_obs as f32) * obs_pred_err;
            anomaly_scores.push(score);

            // 4. Step the model forward for next timestep's prediction
            let action = Tensor::<InnerBackend, 2>::zeros([1, action_dim], &device);
            let (z_next, next_state) = model.step(z.clone(), action, state);

            let pred_x_next = model
                .decode(z_next.clone().unsqueeze_dim::<3>(1))
                .reshape([1, feature_dim]);

            state = next_state;
            prev_pred_z = Some(z_next);
            prev_pred_x = Some(pred_x_next);
        }

        // ── Calibrate anomaly scores (triple-signal, independently normalized) ──
        let probation_len = train_len_aligned;
        let scores = triple_signal_to_scores(
            &recon_errs,
            &latent_pred_errs,
            &obs_pred_errs,
            probation_len,
            &cfg.calibration,
            (alpha_recon, beta_latent, gamma_obs),
        );

        // ── Write results ──
        std::fs::create_dir_all(&result_dir)?;
        let mut wtr = csv::Writer::from_path(&final_result_path)?;
        wtr.write_record([
            "timestamp",
            "value",
            "anomaly_score",
            "recon_err_raw",
            "latent_pred_err_raw",
            "obs_pred_err_raw",
        ])?;
        for i in 0..raw_records.len() {
            wtr.write_record([
                &raw_records[i].timestamp,
                &raw_values[i].to_string(),
                &scores[i].to_string(),
                &recon_errs[i].to_string(),
                &latent_pred_errs[i].to_string(),
                &obs_pred_errs[i].to_string(),
            ])?;
        }
        println!(
            "  {}Saved{} {}/{} ({} points, train={})",
            GREEN, RESET, category, filename, seq_len, train_len_aligned
        );
    }

    println!(
        "\n{}All done.{} Run `cd nab-demo && python evaluate_nab.py` to score.",
        BOLD, RESET
    );
    Ok(())
}
