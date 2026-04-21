use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::Tensor;
use rand::{rngs::StdRng, Rng, SeedableRng};
use ssm_latent_model::latent::{LatentPredictor, LatentState};
use ssm_latent_model::ssm::SsmConfig;

type MyBackend = NdArray<f32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {
    let device = NdArrayDevice::default();
    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility

    let config = SsmConfig {
        d_model: 32,
        d_state: 16,
        expand: 2,
        n_heads: 4,
        mimo_rank: 2,
        use_conv: true,
        conv_kernel: 4,
    };
    let input_dim = 2;
    let action_dim = 2;
    let seq_len = 32;
    let batch_size = 2;
    let epochs = 100;

    let mut model =
        LatentPredictor::<MyAutodiffBackend>::new(&config, input_dim, action_dim, &device);
    let mut optim =
        AdamConfig::new().init::<MyAutodiffBackend, LatentPredictor<MyAutodiffBackend>>();

    println!("==========================================================");
    println!(" Latent SSM Predictor (Reproducible Mode)");
    println!("==========================================================");

    // Training Loop
    for epoch in 1..=epochs {
        let mut obs_vec = Vec::new();
        let mut act_vec = Vec::new();

        for b in 0..batch_size {
            let offset = (b as f32) * 0.5 + (epoch as f32) * 0.01;
            for t in 0..seq_len {
                let angle = (t as f32) * 0.3 + offset;
                let noise_obs: f32 = rng.gen_range(-0.005..0.005);
                let noise_act: f32 = rng.gen_range(-0.005..0.005);

                obs_vec.extend_from_slice(&[angle.cos() + noise_obs, angle.sin() + noise_obs]);
                act_vec.extend_from_slice(&[
                    -(angle.sin()) * 0.1 + noise_act,
                    angle.cos() * 0.1 + noise_act,
                ]);
            }
        }

        let obs_data = Tensor::<MyAutodiffBackend, 3>::from_data(
            burn::tensor::TensorData::new(obs_vec, [batch_size, seq_len, input_dim]),
            &device,
        );
        let action_data = Tensor::<MyAutodiffBackend, 3>::from_data(
            burn::tensor::TensorData::new(act_vec, [batch_size, seq_len, action_dim]),
            &device,
        );

        let (z, predicted_z) = model.forward(obs_data, action_data);
        let loss = model.loss(z.clone(), predicted_z, 1.0);

        if epoch % 50 == 0 || epoch == 1 {
            let z_var = z.clone().var(0).mean().into_data();
            println!(
                "Epoch {:3}: Total Loss = {:?}, Latent Variance = {:?}",
                epoch,
                loss.clone().into_data(),
                z_var
            );
        }

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(2e-3, model, grads);
    }

    println!("\nPhase 2: Sequence Prediction");
    let model_valid = model.valid();

    let initial_obs = Tensor::<MyBackend, 3>::from_data(
        burn::tensor::TensorData::new(vec![1.0, 0.0, 0.8, 0.6], [batch_size, 1, 2]),
        &device,
    );

    let z_start = model_valid.encode(initial_obs);
    let mut current_z = z_start.squeeze::<2>(1);

    let d_inner = config.d_model * config.expand;
    let d_head = d_inner / config.n_heads;

    let mut state = LatentState {
        h: Tensor::zeros(
            [
                batch_size,
                config.n_heads,
                config.d_state,
                d_head / config.mimo_rank,
            ],
            &device,
        ),
        prev_bx: None,
        conv_state: if config.use_conv {
            Some(Tensor::zeros(
                [batch_size, d_inner, config.conv_kernel - 1],
                &device,
            ))
        } else {
            None
        },
    };

    println!("Starting prediction from z[0]...");
    for t in 1..=5 {
        let action = Tensor::<MyBackend, 2>::from_data(
            burn::tensor::TensorData::new(vec![0.0, 0.1, 0.0, 0.1], [batch_size, 2]),
            &device,
        );

        let (next_z, next_state) = model_valid.step(current_z, action, state);

        current_z = next_z;
        state = next_state;

        println!(
            "Step {}: z_hat[0] (first 3 dims): {:?}",
            t,
            current_z.clone().slice([0..1, 0..3]).into_data()
        );
    }
}
