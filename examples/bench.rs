use burn::tensor::backend::AutodiffBackend;
use burn::backend::{ndarray::NdArrayDevice, wgpu::WgpuDevice, Autodiff, NdArray, Wgpu};
use burn::optim::{GradientsParams, Optimizer, AdamConfig};
use burn::tensor::Tensor;
use mamba_jepa_rs::mamba::MambaConfig;
use mamba_jepa_rs::jepa::JepaWorldModel;
use std::time::Instant;

fn run_benchmark<B: AutodiffBackend>(device: B::Device, name: &str, epochs: usize) {
    let config = MambaConfig {
        d_model: 128,      
        d_state: 32,      
        expand: 2,
    };
    let input_dim = 64;    
    let action_dim = 16;   
    let seq_len = 64;
    let batch_size = 16;

    let mut model = JepaWorldModel::<B>::new(&config, input_dim, action_dim, &device);
    let mut optim = AdamConfig::new().init::<B, JepaWorldModel<B>>();

    println!("Starting benchmark for {}: epochs={}, batch_size={}, seq_len={}", name, epochs, batch_size, seq_len);
    
    // Warmup
    for _ in 0..2 {
        let obs_data = Tensor::<B, 3>::random([batch_size, seq_len, input_dim], burn::tensor::Distribution::Default, &device);
        let action_data = Tensor::<B, 3>::random([batch_size, seq_len, action_dim], burn::tensor::Distribution::Default, &device);
        let (z, predicted_z) = model.forward(obs_data, action_data);
        let loss = model.loss(z, predicted_z, 1.0);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(2e-3, model, grads);
    }

    let start = Instant::now();

    for _epoch in 1..=epochs {
        let obs_data = Tensor::<B, 3>::random([batch_size, seq_len, input_dim], burn::tensor::Distribution::Default, &device);
        let action_data = Tensor::<B, 3>::random([batch_size, seq_len, action_dim], burn::tensor::Distribution::Default, &device);

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
    run_benchmark::<Autodiff<Wgpu>>(WgpuDevice::BestAvailable, "Wgpu (GPU)", epochs);
}
