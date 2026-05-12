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

fn main() -> Result<(), Box<dyn Error>> {
    let device = WgpuDevice::default();
    println!("\n{}=== NAB SOTA EVALUATION PIPELINE (Advanced Scoring) ==={}", BOLD, RESET);

    let detector_name = "ssm_latent";
    let base_data_path = Path::new("nab-demo/data");
    let base_result_path = Path::new("nab-demo/results").join(detector_name);

    let mut csv_files = Vec::new();
    find_csv_files(base_data_path, &mut csv_files)?;
    csv_files.sort();

    for data_path in csv_files {
        let relative_path = data_path.strip_prefix(base_data_path)?;
        if relative_path.to_str().unwrap().contains("README") { continue; }
        
        let parts: Vec<_> = relative_path.components().collect();
        if parts.len() < 2 { continue; }
        let category = parts[0].as_os_str().to_str().unwrap();
        let filename = parts[parts.len()-1].as_os_str().to_str().unwrap();

        let result_dir = base_result_path.join(category);
        let result_filename = format!("{}_{}", detector_name, filename);
        let final_result_path = result_dir.join(result_filename);

        if final_result_path.exists() {
            println!("{}[Skipping]{} {}/{}", GREEN, RESET, category, filename);
            continue;
        }

        println!("\n{}[Processing]{} {}/{}", BOLD, RESET, category, filename);

        let raw_records = load_nab_records(&data_path)?;
        let raw_values: Vec<f32> = raw_records.iter().map(|r| r.value).collect();
        let (normalized_values, _, _) = normalize(&raw_values);

        let train_count = (raw_values.len() as f32 * 0.15).max(50.0) as usize;
        let train_data = &normalized_values[..train_count];

        let config = SsmConfig {
            d_model: 64, d_state: 16, expand: 2, n_heads: 4, mimo_rank: 1, use_conv: true, conv_kernel: 4,
        };
        let mut model = LatentPredictor::<MyBackend>::new(&config, 1, 1, &device);
        let mut optim = AdamConfig::new().init();
        
        let train_tensor = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(train_data.to_vec(), [1, train_count, 1]), &device,
        );

        // Learning
        for _ in 1..=50 {
            let (z, pred_z, reconstructed_x, predicted_x) = model.forward(train_tensor.clone(), Tensor::zeros([1, train_count, 1], &device));
            let loss = model.loss(LatentLossArgs {
                z, pred_z, reconstructed_x, predicted_x,
                original_x: train_tensor.clone(),
                stability_weight: 1.0, curvature_weight: 0.5, recon_weight: 2.0,
            });
            let grads = loss.backward();
            let grads_params = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optim.step(1e-3, model, grads_params);
        }

        // --- Optimized Inference with Chunking to avoid OOM ---
        let seq_len = normalized_values.len();
        let chunk_size = 5000;
        let mut full_recon_err = Vec::with_capacity(seq_len);
        let mut full_latent_err = Vec::with_capacity(seq_len);

        for start in (0..seq_len).step_by(chunk_size) {
            let end = (start + chunk_size).min(seq_len);
            let current_chunk = &normalized_values[start..end];
            let chunk_tensor = Tensor::<MyBackend, 3>::from_data(
                TensorData::new(current_chunk.to_vec(), [1, current_chunk.len(), 1]), &device,
            );
            
            let (z, pred_z, reconstructed, _) = model.forward(chunk_tensor.clone(), Tensor::zeros([1, current_chunk.len(), 1], &device));
            
            let r_err = (chunk_tensor - reconstructed).powf_scalar(2.0).into_data().as_slice::<f32>().unwrap().to_vec();
            let l_err = (z - pred_z).powf_scalar(2.0).into_data().as_slice::<f32>().unwrap().to_vec();
            
            full_recon_err.extend(r_err);
            full_latent_err.extend(l_err);
        }
        
        // --- SOTA Scoring Logic: Dynamic Z-Score + Nonlinear Scaling ---
        let mut scores = Vec::with_capacity(seq_len);
        let mut moving_mean = 0.0f32;
        let mut moving_var = 0.05f32; // 初期分散
        let alpha = 0.05; // 統計量の更新感度

        for i in 0..seq_len {
            let err = full_recon_err[i] * 0.7 + full_latent_err[i] * 0.3;
            
            // 偏差の計算
            let diff = err - moving_mean;
            let std = moving_var.sqrt().max(1e-6);
            let z_score = diff / std;

            // 統計情報の更新（EMA）
            moving_mean = (1.0 - alpha) * moving_mean + alpha * err;
            moving_var = (1.0 - alpha) * moving_var + alpha * diff.powi(2);

            // スコアの非線形変換: シグモイド曲線で強調
            // Z-scoreが3を超えると急激に1.0に近づく設定
            let s = 1.0 / (1.0 + (-(z_score - 4.0)).exp());
            scores.push(s);
        }

        // 保存
        std::fs::create_dir_all(&result_dir)?;
        let mut wtr = csv::Writer::from_path(&final_result_path)?;
        wtr.write_record(&["timestamp", "value", "anomaly_score"])?;
        for i in 0..raw_records.len() {
            wtr.write_record(&[&raw_records[i].timestamp, &raw_values[i].to_string(), &scores[i].to_string()])?;
        }
    }

    println!("\nAll datasets processed with advanced scoring.");
    Ok(())
}
