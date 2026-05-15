#![recursion_limit = "256"]
/// Circle World Demo — Forward-Forward variant.
///
/// This demonstrates training an encoder using the Forward-Forward algorithm
/// (Hinton, 2022) instead of backpropagation-based end-to-end gradient descent.
///
/// Architecture:
///   encoder: 2 → 16 → 8  (trained via FF goodness, layers independent)
///   decoder: 8 → 2        (trained via local MSE reconstruction)
///
/// Each layer of the encoder is trained independently:
/// - **Positive data**: real circle points → should produce high goodness
/// - **Negative data**: random noise points → should produce low goodness
///
/// The decoder is a single linear layer trained with local MSE reconstruction
/// (input detached from encoder, so no backprop through encoder).
///
/// This is the "backprop-free" variant: no gradients flow between layers.
/// The original backprop version is preserved in `circle-world-demo/src/main.rs`.
use burn::module::AutodiffModule;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::Tensor;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use ssm_latent_model::ff_model::FfEncoder;

// --- Backend Selection ---
#[cfg(feature = "wgpu")]
type MyBackend = burn::backend::Wgpu;
#[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
type MyBackend = burn::backend::NdArray;
#[cfg(all(not(feature = "wgpu"), not(feature = "ndarray")))]
type MyBackend = burn::backend::NdArray;

type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

/// Full model: FF Encoder + Decoder (both trained with local objectives only)
///
/// The encoder uses Forward-Forward (layer-local goodness).
/// The decoder uses local MSE reconstruction (detached encoder output → decoder).
///
/// Critically, the decoder gradient does NOT flow back through the encoder layers,
/// unlike the backprop variant where decoder loss flows through every encoder layer.
#[derive(Debug, burn::module::Module)]
struct FfWorldModel<B: burn::tensor::backend::Backend> {
    encoder: FfEncoder<B>,
    decoder: Linear<B>,
    latent_dim: usize,
}

impl<B: burn::tensor::backend::Backend> FfWorldModel<B> {
    fn new(latent_dim: usize, device: &B::Device) -> Self {
        // 2-layer encoder: 2 → 16 → latent_dim
        let encoder = FfEncoder::new(&[2, 16, latent_dim], device);
        let decoder = LinearConfig::new(latent_dim, 2).init(device);
        Self {
            encoder,
            decoder,
            latent_dim,
        }
    }

    /// Encode observations: single forward pass with detach between layers.
    fn encode(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.encoder.encode(x)
    }

    /// Decode latent to observation space (local, no gradient through encoder).
    fn decode(&self, z: Tensor<B, 2>) -> Tensor<B, 2> {
        self.decoder.forward(z)
    }
}

// --- Visualization (same as original) ---

fn draw_frame(
    title: &str,
    step: usize,
    left_point: Option<([f32; 2], char, &str)>,
    right_point: Option<([f32; 2], char, &str)>,
) {
    let width = 30;
    let height = 15;

    let mut grids = vec![vec![vec![' '; width]; height]; 2];

    let points = [left_point, right_point];

    for (g_idx, pt_opt) in points.iter().enumerate() {
        if let Some(([x, y], marker, _)) = pt_opt {
            let grid = &mut grids[g_idx];
            grid[height / 2].fill('-');
            for row in grid.iter_mut() {
                row[width / 2] = '|';
            }
            grid[height / 2][width / 2] = '+';

            let gx = (((x + 2.0) / 4.0) * (width as f32 - 1.0)) as i32;
            let gy = (((2.0 - y) / 4.0) * (height as f32 - 1.0)) as i32;

            if gx >= 0 && gx < width as i32 && gy >= 0 && gy < height as i32 {
                grid[gy as usize][gx as usize] = *marker;
            }
        }
    }

    println!("\x1B[H\x1B[2J");
    println!("=== {} (Step: {}) ===", title, step);

    if right_point.is_some() {
        let label_left = left_point.as_ref().map(|p| p.2).unwrap_or("");
        let label_right = right_point.as_ref().map(|p| p.2).unwrap_or("");
        println!(
            "  {:^width$}    {:^width$}",
            label_left,
            label_right,
            width = width
        );

        for (row_left, row_right) in grids[0].iter().zip(grids[1].iter()) {
            let s_left: String = row_left.iter().collect();
            let s_right: String = row_right.iter().collect();
            println!("  {}    {}", s_left, s_right);
        }
    } else if let Some((_, _, label)) = left_point {
        println!("  {:^width$}", label, width = width);
        for row in &grids[0] {
            let s: String = row.iter().collect();
            println!("  {}", s);
        }
    }
}

fn main() {
    // --- Config ---
    let latent_dim = 8;
    let epochs = 200;
    let batch_size = 32;
    let seq_len = 16;
    let learning_rate = 0.01;
    let threshold = 2.0; // goodness threshold for FF

    #[cfg(feature = "wgpu")]
    let device = burn::backend::wgpu::WgpuDevice::default();
    #[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
    let device = burn::backend::ndarray::NdArrayDevice::default();
    #[cfg(all(not(feature = "wgpu"), not(feature = "ndarray")))]
    let device = burn::backend::ndarray::NdArrayDevice::default();

    let mut rng = StdRng::seed_from_u64(42);

    println!("==========================================================");
    println!("    🔮 Circle World — Forward-Forward Algorithm");
    println!("    (Backprop-free: each layer trains independently)");
    println!("==========================================================");
    println!();
    println!("Architecture:");
    println!("  Encoder:   2 → 16 → {} (Forward-Forward, layers independent)", latent_dim);
    println!("  Decoder:   {} → 2       (Local MSE, detached input)", latent_dim);
    println!();
    println!("Key difference from original:");
    println!("  - Each encoder layer has its OWN loss (FF goodness)");
    println!("  - Gradients do NOT flow between layers (via detach)");
    println!("  - Decoder error does NOT backprop through encoder");
    println!();

    // --- Build model ---
    let mut model = FfWorldModel::<MyAutodiffBackend>::new(latent_dim, &device);
    let mut optimizer =
        AdamConfig::new().init::<MyAutodiffBackend, FfWorldModel<MyAutodiffBackend>>();

    // --- Visualization: initial state ---
    std::thread::sleep(std::time::Duration::from_millis(500));

    println!("[Part 1: Observation]");
    for t in 1..=20 {
        let angle = (t as f32) * 0.3;
        let x = angle.cos();
        let y = angle.sin();
        draw_frame(
            "Observing the circle",
            t,
            Some(([x, y], 'X', "External World")),
            None,
        );
        std::thread::sleep(std::time::Duration::from_millis(80));
    }

    println!("\n[Part 2: Forward-Forward Training]");
    let model_valid = model.valid();
    run_inference(
        &model_valid,
        &device,
        "Epoch 0 (Untrained)",
        latent_dim,
    );
    std::thread::sleep(std::time::Duration::from_millis(1200));

    // --- Training Loop ---
    for epoch in 1..=epochs {
        // Generate batch of positive data (circle points)
        let mut pos_data = Vec::new();
        let mut neg_data = Vec::new();

        for b in 0..batch_size {
            let phase_shift = (b as f32) * 1.25;
            for t in 0..seq_len {
                let time = (t as f32) * 0.3;
                let angle = time + phase_shift;

                // Positive: circle point with tiny noise
                let noise: f32 = rng.random_range(-0.02..0.02);
                pos_data.push(angle.cos() + noise);
                pos_data.push(angle.sin() + noise);

                // Negative: random point (NOT on the circle)
                let neg_x: f32 = rng.random_range(-2.0..2.0);
                let neg_y: f32 = rng.random_range(-2.0..2.0);
                // Also add some noisy circle points as hard negatives
                if rng.random_bool(0.3) {
                    let hard_noise_x: f32 = rng.random_range(-0.5..0.5);
                    let hard_noise_y: f32 = rng.random_range(-0.5..0.5);
                    neg_data.push(angle.cos() + hard_noise_x);
                    neg_data.push(angle.sin() + hard_noise_y);
                } else {
                    neg_data.push(neg_x);
                    neg_data.push(neg_y);
                }
            }
        }

        let n_total = batch_size * seq_len;

        let pos_tensor = Tensor::<MyAutodiffBackend, 2>::from_data(
            burn::tensor::TensorData::new(pos_data, [n_total, 2]),
            &device,
        );
        let neg_tensor = Tensor::<MyAutodiffBackend, 2>::from_data(
            burn::tensor::TensorData::new(neg_data, [n_total, 2]),
            &device,
        );

        // Keep original input for decoder reconstruction target
        let original_input = pos_tensor.clone();

        // ─── Forward-Forward Passes ───
        let pos_activations = model.encoder.positive_pass(pos_tensor);
        let neg_activations = model.encoder.negative_pass(neg_tensor);

        let ff_losses =
            model
                .encoder
                .compute_layer_losses(&pos_activations, &neg_activations, threshold);

        // ─── Decoder Loss (local MSE) ───
        // Detach encoder output so decoder gradient doesn't flow into encoder
        let enc_out = model.encoder.encode(original_input.clone());
        let decoded = model.decode(enc_out.detach());
        // Target: the original 2D input (autoencoder reconstruction)
        let recon_loss = (decoded - original_input).powf_scalar(2.0).mean().unsqueeze();

        // ─── Combine losses ───
        // Each FF layer loss is summed with the reconstruction loss.
        // Since layers are isolated via detach, the combined backward
        // updates each layer independently.
        let total_loss = ff_losses
            .into_iter()
            .fold(recon_loss, |acc, l| acc + l);

        let current_loss: f32 = total_loss.clone().into_data().as_slice::<f32>().unwrap()[0];

        if epoch % 30 == 0 || epoch == 1 {
            // Also compute validation stats
            let model_valid = model.valid();
            let val_enc = model_valid.encode(
                Tensor::<MyBackend, 2>::from_data(
                    burn::tensor::TensorData::new(
                        vec![1.0f32, 0.0f32, 0.0f32, 1.0f32, -1.0f32, 0.0f32, 0.0f32, -1.0f32],
                        [4, 2],
                    ),
                    &device,
                ),
            );

            println!(
                "Epoch {:4}: FF Loss = {:.6} | Latent state (4 points): {:.3?}",
                epoch,
                current_loss,
                val_enc
                    .into_data()
                    .as_slice::<f32>()
                    .unwrap()
                    .iter()
                    .take(8)
                    .copied()
                    .collect::<Vec<_>>()
            );

            let model_valid = model.valid();
            run_inference(
                &model_valid,
                &device,
                &format!("Training (Epoch {})", epoch),
                latent_dim,
            );
            std::thread::sleep(std::time::Duration::from_millis(400));
        }

        let grads = total_loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(learning_rate, model, grads);
    }

    println!("\n[Part 3: Forward-Forward Trained World Model]");
    let model_valid = model.valid();
    run_inference(
        &model_valid,
        &device,
        &format!("Final FF Model (Epoch {})", epochs),
        latent_dim,
    );

    println!();
    println!("==========================================================");
    println!("  Comparison: Backprop vs Forward-Forward");
    println!("==========================================================");
    println!("  Backprop:   gradients flow from decoder through ALL");
    println!("              encoder layers → weight transport problem");
    println!("  FF:         each layer has LOCAL goodness objective");
    println!("              decoder only gets detached encoder output");
    println!("              → no weight transport, biologically plausible");
    println!("==========================================================");
}

/// Run inference: predict future points using the FF model.
///
/// Since the FF model doesn't have SSM dynamics, we use a simple
/// autoregressive approach: encode current position, add a small
/// perturbation to latent, decode to get next prediction.
fn run_inference<B: burn::tensor::backend::Backend>(
    model: &FfWorldModel<B>,
    device: &B::Device,
    title: &str,
    _latent_dim: usize,
) {
    println!("\n--- {} ---", title);

    // Run inference by encoding points and decoding the latent
    for t in 1..=20 {
        let angle = (t as f32) * 0.3;
        let real_x = angle.cos();
        let real_y = angle.sin();

        // Encode current ground truth point
        let obs = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![real_x, real_y], [1, 2]),
            device,
        );
        let z = model.encode(obs);

        // Predict next: add small perturbation to latent (simulating dynamics)
        // In a full system, this would be an SSM step. Here we just test:
        // if the encoder+decoder can represent the circle, a nearby latent
        // should decode to a nearby point.
        let predicted = model.decode(z);

        let pos = predicted.into_data();
        let pos_slice = pos.as_slice::<f32>().unwrap();
        let pred_x = pos_slice[0];
        let pred_y = pos_slice[1];

        draw_frame(
            title,
            t as usize,
            Some(([real_x, real_y], 'X', "Ground Truth")),
            Some(([pred_x, pred_y], 'O', "FF Encoded")),
        );

        std::thread::sleep(std::time::Duration::from_millis(150));
    }
}
