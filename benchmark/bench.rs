use burn::backend::{Autodiff, NdArray, Wgpu, ndarray::NdArrayDevice, wgpu::WgpuDevice};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;
use criterion::{Criterion, BatchSize, criterion_group, criterion_main};
use ssm_latent_model::latent::{LatentPredictor, LatentLossArgs};
use ssm_latent_model::ssm::SsmConfig;

fn benchmark_training_step<B: AutodiffBackend>(
    c: &mut Criterion,
    device: B::Device,
    name: &str,
) {
    let config = SsmConfig::new(128, 32, 2, 8, 4);
    let input_dim = 64;
    let action_dim = 16;
    let seq_len = 64;
    let batch_size = 16;

    let mut group = c.benchmark_group("training_step");
    group.sample_size(10);

    group.bench_function(name, |b| {
        b.iter_batched(
            || {
                let model = LatentPredictor::<B>::new(&config, input_dim, action_dim, &device);
                let optim = AdamConfig::new().init::<B, LatentPredictor<B>>();
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
                (model, optim, obs_data, action_data)
            },
            |(mut model, mut optim, obs_data, action_data)| {
                let (z, predicted_z, reconstructed_x, predicted_x) =
                    model.forward(obs_data.clone(), action_data.clone());
                let loss = model.loss(LatentLossArgs {
                    z,
                    pred_z: predicted_z,
                    reconstructed_x,
                    predicted_x,
                    original_x: obs_data.clone(),
                    stability_weight: 1.0,
                    curvature_weight: 0.5,
                    recon_weight: 1.0,
                });
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optim.step(2e-3, model, grads);
                (model, optim)
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_ndarray(c: &mut Criterion) {
    benchmark_training_step::<Autodiff<NdArray<f32>>>(c, NdArrayDevice::Cpu, "NdArray (CPU)");
}

fn bench_wgpu(c: &mut Criterion) {
    benchmark_training_step::<Autodiff<Wgpu>>(c, WgpuDevice::default(), "Wgpu (GPU)");
}

criterion_group!(benches, bench_ndarray, bench_wgpu);
criterion_main!(benches);
