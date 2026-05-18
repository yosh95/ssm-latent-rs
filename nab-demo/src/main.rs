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

// ─── Mahalanobis-based triple-signal fusion ─────────────────────────────

/// Compute median and robust MAD. MAD has a floor of 1e-4.
fn calc_mad_robust(data: &[f32]) -> (f32, f32) {
    let n = data.len();
    if n == 0 {
        return (0.0, 1.0);
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = sorted[n / 2];
    let mut abs_devs: Vec<f32> = data.iter().map(|x| (x - med).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let raw_mad = abs_devs[n / 2] * 1.4826;
    let data_range = sorted[n - 1] - sorted[0];
    let floor = (data_range * 0.01).max(1e-4);
    let mad = raw_mad.max(floor);
    (med, mad)
}

/// Robust covariance matrix estimation on calibration data (MCD-inspired).
///
/// Uses iterative MAD-based outlier rejection to compute a clean covariance
/// estimate, then returns (mean_vector, covariance_matrix) for the 3-signal
/// vector [recon_err, latent_pred_err, obs_pred_err].
fn robust_covariance_3d(data: &[[f32; 3]], max_iter: usize) -> ([f32; 3], [[f32; 3]; 3]) {
    let n = data.len();
    if n < 10 {
        // Fallback: identity covariance
        return (
            [0.0; 3],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        );
    }

    // Initial estimate using all data
    let mut keep: Vec<bool> = vec![true; n];

    for _iter in 0..max_iter {
        let kept_data: Vec<&[f32; 3]> = data
            .iter()
            .enumerate()
            .filter(|(i, _)| keep[*i])
            .map(|(_, d)| d)
            .collect();
        let k = kept_data.len();
        if k < 5 {
            break;
        }

        // Mean
        let mut mean = [0.0f32; 3];
        for d in &kept_data {
            for j in 0..3 {
                mean[j] += d[j];
            }
        }
        for j in 0..3 {
            mean[j] /= k as f32;
        }

        // Covariance
        let mut cov = [[0.0f32; 3]; 3];
        for d in &kept_data {
            let diff = [d[0] - mean[0], d[1] - mean[1], d[2] - mean[2]];
            for a in 0..3 {
                for b in 0..3 {
                    cov[a][b] += diff[a] * diff[b];
                }
            }
        }
        for a in 0..3 {
            for b in 0..3 {
                cov[a][b] /= (k - 1) as f32;
            }
        }

        // Compute Mahalanobis distances
        let inv_cov = invert_3x3(&cov);
        let mut distances: Vec<(usize, f32)> = Vec::with_capacity(n);
        for (i, d) in data.iter().enumerate() {
            let diff = [d[0] - mean[0], d[1] - mean[1], d[2] - mean[2]];
            let md = mahalanobis_sq_3d(&diff, &inv_cov);
            distances.push((i, md));
        }

        // Keep points within chi2(3)_0.975 ≈ 3.0 (robust threshold)
        // chi2(3)_0.975 = 9.35; we use a tighter 3.0 for outlier removal
        let threshold = 3.0f32;
        let mut new_keep = vec![false; n];
        for &(i, md) in &distances {
            if md < threshold {
                new_keep[i] = true;
            }
        }

        if new_keep == keep {
            // Converged
            keep = new_keep;
            break;
        }
        keep = new_keep;
    }

    // Final estimate on kept data
    let kept_data: Vec<&[f32; 3]> = data
        .iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, d)| d)
        .collect();
    let k = kept_data.len().max(5);

    let mut mean = [0.0f32; 3];
    for d in &kept_data {
        for j in 0..3 {
            mean[j] += d[j];
        }
    }
    for j in 0..3 {
        mean[j] /= k as f32;
    }

    let mut cov = [[0.0f32; 3]; 3];
    for d in &kept_data {
        let diff = [d[0] - mean[0], d[1] - mean[1], d[2] - mean[2]];
        for a in 0..3 {
            for b in 0..3 {
                cov[a][b] += diff[a] * diff[b];
            }
        }
    }
    for a in 0..3 {
        for b in 0..3 {
            cov[a][b] /= (k - 1) as f32;
        }
    }

    // Regularize: add small identity to ensure positive definiteness
    for i in 0..3 {
        cov[i][i] += 1e-6;
    }

    (mean, cov)
}

fn invert_3x3(m: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    let inv_det = if det.abs() < 1e-10 { 0.0 } else { 1.0 / det };

    [
        [
            (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * inv_det,
            (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * inv_det,
        ],
    ]
}

fn mahalanobis_sq_3d(diff: &[f32; 3], inv_cov: &[[f32; 3]; 3]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..3 {
        let mut row_sum = 0.0f32;
        for j in 0..3 {
            row_sum += inv_cov[i][j] * diff[j];
        }
        sum += diff[i] * row_sum;
    }
    sum.max(0.0)
}

/// Convert triple-signal errors to anomaly scores via Mahalanobis distance.
///
///   1. Calibrate: estimate robust mean and covariance of 3-signal vector
///      on probation period.
///   2. For each point, compute Mahalanobis distance from the calibration
///      distribution.
///   3. Convert to [0, 1] via percentile rank + power transform.
fn triple_signal_mahalanobis(
    recon: &[f32],
    latent: &[f32],
    obs_pred: &[f32],
    probation_len: usize,
    cal: &CalibrationConfig,
) -> Vec<f32> {
    let n = recon.len();
    let cal_len = probation_len.min(n);

    // ── Build 3D data for calibration ──
    let cal_data: Vec<[f32; 3]> = (0..cal_len)
        .map(|i| [recon[i], latent[i], obs_pred[i]])
        .collect();
    let (mean, cov) = robust_covariance_3d(&cal_data, 5);
    let inv_cov = invert_3x3(&cov);

    // ── Compute Mahalanobis distances for all points ──
    let mut distances: Vec<f32> = (0..n)
        .map(|i| {
            let diff = [
                recon[i] - mean[0],
                latent[i] - mean[1],
                obs_pred[i] - mean[2],
            ];
            mahalanobis_sq_3d(&diff, &inv_cov).sqrt()
        })
        .collect();

    // ── Percentile rank on calibration distances, then power transform ──
    let mut cal_dists: Vec<f32> = distances[..cal_len].to_vec();
    cal_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cal_max = cal_dists[cal_dists.len() - 1];

    let power = cal.power.max(0.1).min(2.0);
    let scores: Vec<f32> = distances
        .iter()
        .map(|&d| {
            // Normalize by calibration max (with 50% headroom)
            let normalized = (d / (cal_max * 1.5)).min(1.0).max(0.0);
            normalized.powf(power)
        })
        .collect();

    scores
}

// ─── Contamination-guarded EWMA with threshold decay ────────────────────

/// Streaming z-score normalizer with contamination guard and threshold decay.
///
/// - **Contamination guard** (`guard_z`): scores with z ≥ guard_z are NOT fed
///   into the EWMA, preventing anomaly contamination.
/// - **Threshold decay** (`threshold_decay`): the EWMA-based threshold slowly
///   decays toward the MAD baseline, preventing threshold inflation after
///   false positives or borderline events.
struct GuardedEwma {
    mean: f64,
    var: f64,
    alpha: f64,
    guard_z: f32,
    decay: f64,
    baseline_mean: f64,
    baseline_var: f64,
}

impl GuardedEwma {
    fn new(alpha: f64, guard_z: f32, decay: f64, initial_mean: f64, initial_var: f64) -> Self {
        Self {
            mean: initial_mean,
            var: initial_var,
            alpha,
            guard_z,
            decay,
            baseline_mean: initial_mean,
            baseline_var: initial_var,
        }
    }

    fn current_std(&self) -> f64 {
        self.var.max(0.0).sqrt()
    }

    fn observe(&mut self, value: f32) -> f32 {
        let v = value as f64;
        let std = self.current_std().max(1e-6);
        let z = (v - self.mean).abs() / std;

        // Only update EWMA with non-anomalous values
        if (z as f32) < self.guard_z {
            let diff = v - self.mean;
            self.mean = (1.0 - self.alpha) * self.mean + self.alpha * v;
            self.var = (1.0 - self.alpha) * self.var + self.alpha * diff * diff;
        }

        // Apply threshold decay: slowly pull mean/var back toward baseline
        // This prevents permanent threshold inflation
        if self.decay > 0.0 {
            self.mean = (1.0 - self.decay) * self.mean + self.decay * self.baseline_mean;
            self.var = (1.0 - self.decay) * self.var + self.decay * self.baseline_var;
        }

        z as f32
    }
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
    // #3: No dummy action — use action_dim=0 (unsupervised mode)
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
        "{}Score: Mahalanobis triple-signal (robust covariance){}\n",
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

        // ── #2: Mahalanobis triple-signal fusion ──
        let probation_len = train_len_aligned;
        let scores = triple_signal_mahalanobis(
            &recon_errs,
            &latent_pred_errs,
            &obs_pred_errs,
            probation_len,
            &cfg.calibration,
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
