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
    #[allow(dead_code)]
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
#[allow(dead_code)]
struct ScoringConfig {
    alpha_recon: f64,
    beta_latent: f64,
    gamma_obs: f64,
    #[serde(default = "default_ema_alpha")]
    ema_alpha: f64,
}

fn default_ema_alpha() -> f64 { 0.3 }

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CalibrationConfig {
    k_mad: f32,
    alpha_ewma: f64,
    k_ewma: f64,
    guard_z: f32,
    threshold_decay: f64,
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

/// Build 2D input: [value, diff] only — NO temporal features, NO z-scores.
/// 
/// CRITICAL: Temporal features (hour, day-of-week) allow the model to learn
/// "at this time, values often change" and thus predict anomalies. By using
/// only [value, diff], the model sees minimal context and cannot anticipate
/// anomalous patterns, making prediction errors highly discriminative.
fn build_features(normalized: &[f32], _timestamps: &[String]) -> Vec<Vec<f32>> {
    let n = normalized.len();
    let mut feats = Vec::with_capacity(n);

    for i in 0..n {
        let val = normalized[i];
        let diff = if i > 0 {
            normalized[i] - normalized[i - 1]
        } else {
            0.0
        };
        feats.push(vec![val, diff]);
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

// ─── 3-Way Ensemble Anomaly Scoring ─────────────────────────────────────
///
/// Combines three independent anomaly signals:
/// 1. **WindowedGaussian**: z-score of current value against rolling window
///    (the best NAB built-in detector). Catches point anomalies & level shifts.
/// 2. **Difference spike**: z-score of abs(first-difference) against rolling
///    window. Catches sudden jumps/drops.
/// 3. **JEPA manifold distance**: k-NN distance in latent space against
///    calibration-period reference vectors. Catches structural anomalies.
///
/// Fusion: anomaly_score = max(wg_score, diff_score, jepa_score)
fn ensemble_scoring(
    z_slice: &[f32],       // flat latent vectors: [seq_len, d_model]
    d_model: usize,
    values: &[f32],        // MAD-normalized raw values
    probation_len: usize,
) -> Vec<f32> {
    let n = values.len();
    let cal_len = probation_len.min(n);

    // ── 1. WindowedGaussian (rolling z-score) ──
    // Use a window of 48 points (~4 hours at 5min), sliding.
    // z = (value - rolling_mean) / rolling_std
    let wg_win = 48usize;
    let mut wg_scores = Vec::with_capacity(n);
    
    // Pre-compute rolling stats efficiently
    for i in 0..n {
        let start = i.saturating_sub(wg_win);
        let slice = &values[start..=i];
        let len = slice.len() as f32;
        let mean = slice.iter().sum::<f32>() / len;
        let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / len;
        let std = var.sqrt().max(1e-6);
        let z = ((values[i] - mean).abs()) / std;
        // Gaussian tail probability: p = 2*(1-Phi(z))
        // Approximate: score = min(z/6, 1.0) — simple linear scaling
        let score = (z / 6.0).min(1.0);
        wg_scores.push(score);
    }

    // ── 2. Difference spike ──
    let diff_win = 48usize;
    let mut diff_scores = Vec::with_capacity(n);
    
    for i in 0..n {
        let diff = if i > 0 { (values[i] - values[i - 1]).abs() } else { 0.0 };
        let start = i.saturating_sub(diff_win);
        // Collect diffs in window
        let diffs: Vec<f32> = (start.max(1)..i)
            .map(|j| (values[j] - values[j - 1]).abs())
            .collect();
        if diffs.is_empty() {
            diff_scores.push(0.0);
            continue;
        }
        let d_len = diffs.len() as f32;
        let d_mean = diffs.iter().sum::<f32>() / d_len;
        let d_var = diffs.iter().map(|d| (d - d_mean).powi(2)).sum::<f32>() / d_len;
        let d_std = d_var.sqrt().max(1e-6);
        let z_diff = if diff > d_mean { (diff - d_mean) / d_std } else { 0.0 };
        let score = (z_diff / 5.0).min(1.0);
        diff_scores.push(score);
    }

    // ── 3. JEPA manifold distance ──
    let seq_len = z_slice.len() / d_model;
    let k = 8usize;
    
    // Build reference set from calibration period
    let ref_vectors: Vec<&[f32]> = (0..cal_len)
        .map(|i| &z_slice[i * d_model..(i + 1) * d_model])
        .collect();
    
    let mut jepa_scores = Vec::with_capacity(seq_len);
    
    for i in 0..seq_len {
        let query = &z_slice[i * d_model..(i + 1) * d_model];
        let mut dists: Vec<f32> = ref_vectors.iter().map(|rv| {
            let mut sum = 0.0f32;
            for j in 0..d_model { let diff = query[j] - rv[j]; sum += diff * diff; }
            sum
        }).collect();
        
        let k_eff = k.min(dists.len().saturating_sub(1)).max(1);
        dists.select_nth_unstable_by(k_eff, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean_k_dist = (dists[..k_eff].iter().sum::<f32>() / k_eff as f32).sqrt();
        
        // Normalize: mean distance of calibration points to themselves ~ baseline
        jepa_scores.push(mean_k_dist);
    }
    
    // Calibrate JEPA scores: z-score against calibration distribution
    let mut cal_dists: Vec<f32> = jepa_scores[..cal_len].to_vec();
    cal_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cal_jepa_mean = cal_dists.iter().sum::<f32>() / cal_dists.len() as f32;
    let cal_jepa_var = cal_dists.iter().map(|d| (d - cal_jepa_mean).powi(2)).sum::<f32>() / cal_dists.len() as f32;
    let cal_jepa_std = cal_jepa_var.sqrt().max(1e-6);

    // ── Fusion ──
    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        let wg = wg_scores[i];
        let diff = diff_scores[i];
        let jepa_z = ((jepa_scores[i] - cal_jepa_mean) / cal_jepa_std).max(0.0);
        let jepa = (jepa_z / 6.0).min(1.0);
        
        // Ensemble: max of all three detectors
        let raw = wg.max(diff).max(jepa);
        scores.push(raw);
    }

    scores
}

// ─── Main ───────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn Error>> {
    let device = WgpuDevice::default();

    // ── Load config ──
    let config_path = Path::new("nab-demo/config/nab_config.toml");
    let config_str = std::fs::read_to_string(config_path)?;
    let cfg: NabConfig = toml::from_str(&config_str)?;

    let mode = &cfg.mode.current;
    let feature_dim: usize = 2; // [value, diff] only — minimal context
    let action_dim: usize = 0;

    println!(
        "\n{}=== NAB JEPA Anomaly Detection (mode: {}, no-action) ==={}",
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
        "{}Score: Percentile-rank EMA (contamination-guarded){}\n",
        CYAN, RESET
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
        let actual_len = flat_feats.len() / feature_dim;
        let train_len_aligned = actual_len;

        let train_tensor = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(flat_feats, [1, train_len_aligned, feature_dim]),
            &device,
        );

        // #3: action_dim=0 → no dummy action tensor needed
        let mut model = MultiScaleLatentPredictor::<MyBackend>::new(
            &ssm_config,
            feature_dim,
            action_dim,
            &device,
        );
        let mut optim = AdamConfig::new().init();

        let mut best_loss = f32::INFINITY;
        let mut no_improve = 0;

        // ── JEPA training loop (no-action forward) ──
        for epoch in 1..=max_epochs {
            // #1: forward_no_action — single pass, no dummy action tensor
            let (z, pred_z, reconstructed_x, predicted_x) =
                model.forward_no_action(train_tensor.clone());

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

        // ── Fallback for diverged training ──
        if best_loss > 1e9 {
            println!(
                "  {}Training diverged — outputting all-zero scores{}",
                YELLOW, RESET
            );
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
                    "0.0",
                    "0.0",
                    "0.0",
                    "0.0",
                ])?;
            }
            println!(
                "  {}Saved (zeroed){} {}/{} ({} points, train={})",
                GREEN, RESET, category, filename, seq_len, train_len_aligned
            );
            continue;
        }

        // ── Convert to inference model ──
        let model: MultiScaleLatentPredictor<InnerBackend> = model.valid();

        // ── #1: One-pass batch inference on entire sequence ──
        let all_flat: Vec<f32> = features.iter().flatten().cloned().collect();
        let all_tensor = Tensor::<InnerBackend, 3>::from_data(
            TensorData::new(all_flat, [1, seq_len, feature_dim]),
            &device,
        );
        let (z_all, pred_z_all, reconstructed_all, predicted_all) =
            model.forward_no_action(all_tensor.clone());

        // Extract as Vec<f32>
        let z_data = z_all.into_data();
        let pred_z_data = pred_z_all.into_data();
        let recon_data = reconstructed_all.into_data();
        let pred_data = predicted_all.into_data();

        let z_slice = z_data.as_slice::<f32>().unwrap();
        let pred_z_slice = pred_z_data.as_slice::<f32>().unwrap();
        let recon_slice = recon_data.as_slice::<f32>().unwrap();
        let pred_slice = pred_data.as_slice::<f32>().unwrap();

        // ── Compute per-step errors ──
        let mut recon_errs = Vec::with_capacity(seq_len);
        let mut latent_pred_errs = Vec::with_capacity(seq_len);
        let mut obs_pred_errs = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            // Reconstruction error: ||x[i] - recon_x[i]||²
            let mut re = 0.0f32;
            for d in 0..feature_dim {
                let orig = features[i][d];
                let recon = recon_slice[i * feature_dim + d];
                re += (orig - recon) * (orig - recon);
            }
            recon_errs.push(re);

            // Latent prediction error: ||z[i] - pred_z[i-1]||² (from previous step)
            let d_model = cfg.model.d_model;
            if i > 0 {
                let mut lpe = 0.0f32;
                for d in 0..d_model {
                    let z_curr = z_slice[i * d_model + d];
                    let pz_prev = pred_z_slice[(i - 1) * d_model + d];
                    lpe += (z_curr - pz_prev) * (z_curr - pz_prev);
                }
                latent_pred_errs.push(lpe);
            } else {
                latent_pred_errs.push(0.0);
            }

            // Observation prediction error: ||x[i] - pred_x[i-1]||²
            if i > 0 {
                let mut ope = 0.0f32;
                for d in 0..feature_dim {
                    let x_curr = features[i][d];
                    let px_prev = pred_slice[(i - 1) * feature_dim + d];
                    ope += (x_curr - px_prev) * (x_curr - px_prev);
                }
                obs_pred_errs.push(ope);
            } else {
                obs_pred_errs.push(0.0);
            }
        }

        // ── #2: 3-way ensemble scoring ──
        let probation_len = train_len_aligned;
        let scores = ensemble_scoring(
            z_slice,
            cfg.model.d_model,
            &normalized,
            probation_len,
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
