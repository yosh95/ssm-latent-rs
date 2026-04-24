use burn::backend::{Autodiff, NdArray, Wgpu, ndarray::NdArrayDevice, wgpu::WgpuDevice};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;
use ssm_latent_model::latent::LatentPredictor;
use ssm_latent_model::ssm::SsmConfig;
use std::time::Instant;

fn run_benchmark<B: AutodiffBackend>(device: B::Device, name: &str, epochs: usize) {
    let config = SsmConfig::new(128, 32, 2, 8, 4);
    let input_dim = 64;
    let action_dim = 16;
    let seq_len = 64;
    let batch_size = 16;

    let mut model = LatentPredictor::<B>::new(&config, input_dim, action_dim, &device);
    let mut optim = AdamConfig::new().init::<B, LatentPredictor<B>>();

    println!(
        "Starting benchmark for {}: epochs={}, batch_size={}, seq_len={}",
        name, epochs, batch_size, seq_len
    );

    // Warmup
    for _ in 0..2 {
        let obs_data = Tensor::<B, 3>::random(
            [batch_size, seq_len, input_dim],
            burn::tensor::Distribution::Default,
            &device,
        );
        let action_data = Tensor::<B, 3>::random(
            [batch_size, seq_len, action_dim],
            burn::tensor::Distribution::Default,
            &device,
        );
        let (z, predicted_z) = model.forward(obs_data, action_data);
        let loss = model.loss(z, predicted_z, 1.0);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(2e-3, model, grads);
    }

    let start = Instant::now();

    for _epoch in 1..=epochs {
        let obs_data = Tensor::<B, 3>::random(
            [batch_size, seq_len, input_dim],
            burn::tensor::Distribution::Default,
            &device,
        );
        let action_data = Tensor::<B, 3>::random(
            [batch_size, seq_len, action_dim],
            burn::tensor::Distribution::Default,
            &device,
        );

        let (z, predicted_z) = model.forward(obs_data, action_data);
        let loss = model.loss(z, predicted_z, 1.0);

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(2e-3, model, grads);
    }

    let duration = start.elapsed();
    println!("Benchmark for {} completed in: {:?}", name, duration);
    println!("Average time per epoch: {:?}", duration / epochs as u32);
}

fn main() {
    let epochs = 5;

    println!("--- CPU Benchmark (NdArray) ---");
    run_benchmark::<Autodiff<NdArray<f32>>>(NdArrayDevice::Cpu, "NdArray (CPU)", epochs);

    println!("\n--- GPU Benchmark (Wgpu) ---");
    run_benchmark::<Autodiff<Wgpu>>(WgpuDevice::default(), "Wgpu (GPU)", epochs);
}
