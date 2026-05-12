use burn::backend::Autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::module::AutodiffModule;
use burn::optim::AdamConfig;
use burn::optim::Optimizer;
use burn::tensor::{Tensor, TensorData};
use serde::Deserialize;
use ssm_latent_model::latent::{LatentLossArgs, LatentPredictor, LatentState};
use ssm_latent_model::ssm::SsmConfig;
use std::error::Error;
use std::fs::File;
use std::path::Path;

type MyBackend = Autodiff<Wgpu>;
type InnerBackend = Wgpu;

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

fn compute_anomaly_scores(
    recon_errors: &[f32],
    probation_len: usize,
) -> Vec<f32> {
    let seq_len = recon_errors.len();
    let calibration = &recon_errors[..probation_len.min(recon_errors.len())];
    let mut sorted_cal = calibration.to_vec();
    sorted_cal.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted_cal[sorted_cal.len() / 2];

    let mut abs_devs: Vec<f32> = sorted_cal.iter().map(|x| (x - median).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = abs_devs[abs_devs.len() / 2] * 1.4826;
    let robust_std = mad.max(1e-6);

    let mut ewma_mean = median;
    let mut ewma_var = robust_std * robust_std;
    let beta = 0.03;

    let mut z_scores: Vec<f32> = Vec::with_capacity(seq_len);

    for i in 0..seq_len {
        let err = recon_errors[i];
        let diff = err - ewma_mean;
        let std = ewma_var.sqrt().max(1e-6);
        let z = diff.abs() / std;
        z_scores.push(z);

        if z < 3.0 {
            ewma_mean = (1.0 - beta) * ewma_mean + beta * err;
            let dev = err - ewma_mean;
            ewma_var = (1.0 - beta) * ewma_var + beta * dev * dev;
        }
    }

    let mut z_sorted = z_scores.clone();
    z_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let scores: Vec<f32> = z_scores
        .iter()
        .map(|&z| {
            let rank = z_sorted.partition_point(|&s| s < z) as f32;
            let percentile = rank / seq_len as f32;
            (percentile).powf(0.3).min(1.0)
        })
        .collect();

    let mut result = scores;
    for i in 0..probation_len.min(result.len()) {
        result[i] = 0.0;
    }
    result
}

fn main() -> Result<(), Box<dyn Error>> {
    let device = WgpuDevice::default();
    println!(
        "\n{}=== NAB SSM+Latent Anomaly Detection ==={}",
        BOLD, RESET
    );

    let detector_name = "ssm_latent";
    let base_data_path = Path::new("nab-demo/data");
    let base_result_path = Path::new("nab-demo/results").join(detector_name);

    let mut csv_files = Vec::new();
    find_csv_files(base_data_path, &mut csv_files)?;
    csv_files.sort();

    let train_ratio = 0.50;
    let min_train = 100;
    let epochs = 100;
    let learning_rate = 5e-4;

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
        if parts.len() < 2 { continue; }
        let category = parts[0].as_os_str().to_str().unwrap();
        let filename = parts[parts.len() - 1].as_os_str().to_str().unwrap();

        let result_dir = base_result_path.join(category);
        let result_filename = format!("{}_{}", detector_name, filename);
        let final_result_path = result_dir.join(&result_filename);

        println!("\n{}[Processing]{} {}/{}", BOLD, RESET, category, filename);

        let raw_records = load_nab_records(&data_path)?;
        let raw_values: Vec<f32> = raw_records.iter().map(|r| r.value).collect();
        let (normalized_values, _, _) = normalize(&raw_values);
        let seq_len = normalized_values.len();

        let train_count = ((seq_len as f32 * train_ratio) as usize).max(min_train);
        let train_data = &normalized_values[..train_count];

        let mut model = LatentPredictor::<MyBackend>::new(&config, 1, 1, &device);
        let mut optim = AdamConfig::new().init();

        let train_tensor = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(train_data.to_vec(), [1, train_count, 1]),
            &device,
        );

        let zero_actions = Tensor::<MyBackend, 3>::from_data(
            TensorData::new(vec![0.0f32; train_count], [1, train_count, 1]),
            &device,
        );

        for epoch in 1..=epochs {
            let (z, pred_z, reconstructed_x, predicted_x) =
                model.forward(train_tensor.clone(), zero_actions.clone());

            let loss = model.loss(LatentLossArgs {
                z, pred_z, reconstructed_x, predicted_x,
                original_x: train_tensor.clone(),
                stability_weight: 0.1, curvature_weight: 0.05, recon_weight: 5.0,
            });

            let grads = loss.backward();
            let grads_params = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optim.step(learning_rate, model, grads_params);
        }

        let model: LatentPredictor<InnerBackend> = model.valid();
        let mut full_recon_err = Vec::with_capacity(seq_len);

        let d_inner = config.d_model * config.expand;
        let n_heads = config.n_heads;
        let d_head = d_inner / n_heads;
        let d_head_mimo = d_head / config.mimo_rank;

        let mut state = LatentState::<InnerBackend> {
            h: Tensor::zeros([1, n_heads, config.d_state, d_head_mimo], &device),
            prev_bx: None,
            conv_state: None,
        };

        for i in 0..seq_len {
            let x = Tensor::<InnerBackend, 2>::from_data(TensorData::new(vec![normalized_values[i]], [1, 1]), &device);
            let action = Tensor::<InnerBackend, 2>::from_data(TensorData::new(vec![0.0f32], [1, 1]), &device);

            let z = model.encode(x.clone().unsqueeze_dim::<3>(1).swap_dims(0, 1));
            let z_2d = z.clone().squeeze_dims::<2>(&[1]);

            let (y, next_state) = model.step(z_2d.clone(), action, state);
            state = next_state;

            let predicted_x = model.decode(y.unsqueeze_dim::<3>(1));
            let reconstructed_x = model.decode(z);

            let predicted_val = predicted_x.into_data().as_slice::<f32>().unwrap()[0];
            let reconstructed_val = reconstructed_x.into_data().as_slice::<f32>().unwrap()[0];

            let actual_next = if i + 1 < seq_len { normalized_values[i + 1] } else { normalized_values[i] };
            let pred_err = (predicted_val - actual_next).powi(2);
            let recon_err = (reconstructed_val - normalized_values[i]).powi(2);

            full_recon_err.push(0.7 * pred_err + 0.3 * recon_err);
        }

        let probation_len = ((seq_len as f32 * 0.15) as usize).max(50);
        for i in 0..50.min(full_recon_err.len()) { full_recon_err[i] = 0.0; }

        let scores = compute_anomaly_scores(&full_recon_err, probation_len);

        std::fs::create_dir_all(&result_dir)?;
        let mut wtr = csv::Writer::from_path(&final_result_path)?;
        wtr.write_record(&["timestamp", "value", "anomaly_score"])?;
        for i in 0..raw_records.len() {
            wtr.write_record(&[&raw_records[i].timestamp, &raw_values[i].to_string(), &scores[i].to_string()])?;
        }
        println!("  {}Saved {}:{} {}/{}", GREEN, detector_name, RESET, category, filename);
    }
    println!("\n{}All datasets processed.{}  Run `python evaluate_nab.py`.", BOLD, RESET);
    Ok(())
}
