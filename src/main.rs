use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::optim::{GradientsParams, Optimizer, AdamConfig};
use burn::tensor::Tensor;
use mamba_jepa_rs::mamba::MambaConfig;
use mamba_jepa_rs::jepa::{JepaWorldModel, JepaState};

type MyBackend = NdArray<f32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {
    let device = NdArrayDevice::default();

    let config = MambaConfig {
        d_model: 32,      
        d_state: 16,      
        expand: 2,
    };
    let input_dim = 2;    
    let action_dim = 2;   
    let seq_len = 32;
    let epochs = 150;

    let mut model = JepaWorldModel::<MyAutodiffBackend>::new(&config, input_dim, action_dim, &device);
    let mut optim = AdamConfig::new().init::<MyAutodiffBackend, JepaWorldModel<MyAutodiffBackend>>();

    println!("==========================================================");
    println!(" Improved Mamba-JEPA World Model");
    println!("==========================================================");
    
    // Training Loop
    for epoch in 1..=epochs {
        let mut obs_vec = Vec::new();
        let mut act_vec = Vec::new();
        for t in 0..seq_len {
            let angle = (t as f32) * 0.3;
            obs_vec.extend_from_slice(&[angle.cos(), angle.sin()]);
            act_vec.extend_from_slice(&[-(angle.sin()) * 0.1, angle.cos() * 0.1]); 
        }

        let obs_data = Tensor::<MyAutodiffBackend, 3>::from_data(
            burn::tensor::TensorData::new(obs_vec, [1, seq_len, input_dim]), &device
        );
        let action_data = Tensor::<MyAutodiffBackend, 3>::from_data(
            burn::tensor::TensorData::new(act_vec, [1, seq_len, action_dim]), &device
        );

        let (z, predicted_z) = model.forward(obs_data, action_data);
        let loss = model.loss(z, predicted_z, 1.0);

        if epoch % 50 == 0 || epoch == 1 {
            println!("Epoch {:3}: Total Loss = {:?}", epoch, loss.clone().into_data());
        }

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(2e-3, model, grads);
    }

    println!("\nPhase 2: Open-loop Imagination (Stateful)");
    let model_valid = model.valid();
    
    // 1. Initial Observation
    let initial_obs = Tensor::<MyBackend, 3>::from_data(
        burn::tensor::TensorData::new(vec![1.0, 0.0], [1, 1, 2]), &device
    );
    let initial_act = Tensor::<MyBackend, 3>::from_data(
        burn::tensor::TensorData::new(vec![0.0, 0.1], [1, 1, 2]), &device
    );

    // 2. Encode to get initial latent and internal state
    let z_start = model_valid.encoder.forward(initial_obs);
    let mut current_z = z_start.squeeze::<2>(0);
    
    // Initialize SSM state with zeros
    let mut state = JepaState {
        h_re: Tensor::zeros([1, config.d_model * config.expand, config.d_state], &device),
        h_im: Tensor::zeros([1, config.d_model * config.expand, config.d_state], &device),
    };

    println!("Starting imagination from z[0]...");
    for t in 1..=5 {
        let action = Tensor::<MyBackend, 2>::from_data(
            burn::tensor::TensorData::new(vec![0.0, 0.1], [1, 2]), &device
        );

        // Predict next latent using the recurrent 'step' API
        let (next_z, next_state) = model_valid.step(current_z, action, state);
        
        current_z = next_z;
        state = next_state;
        
        println!("Step {}: z_hat (first 3 dims): {:?}", t, current_z.clone().slice([0..1, 0..3]).into_data());
    }
}
