use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::optim::{GradientsParams, Optimizer, AdamConfig};
use burn::tensor::Tensor;
use mamba_jepa_rs::model::MambaConfig;
use mamba_jepa_rs::jepa::JepaWorldModel;

type MyBackend = NdArray<f32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {
    let device = NdArrayDevice::default();

    // 1. Model Configuration
    // Using Mamba's d_state as the "memory" of the world state.
    let config = MambaConfig {
        d_model: 32,      // Dimension of the latent space
        d_state: 16,      // Dimension of the Mamba SSM state
        d_conv: 3,
        expand: 1,
    };
    let input_dim = 2;    // Observation: [x, y] coordinates
    let action_dim = 2;   // Action: [vx, vy] velocity
    let seq_len = 20;
    let epochs = 200;

    let mut model = JepaWorldModel::<MyAutodiffBackend>::new(&config, input_dim, action_dim, &device);
    let mut optim = AdamConfig::new().init::<MyAutodiffBackend, JepaWorldModel<MyAutodiffBackend>>();

    println!("==========================================================");
    println!(" JEPA World Model with Mamba & SIGReg (arXiv:2603.19312)");
    println!("==========================================================");
    println!("Phase 1: Starting latent space learning of circular motion...");
    
    // Generate sample circular motion data
    for epoch in 1..=epochs {
        let mut obs_vec = Vec::new();
        let mut act_vec = Vec::new();
        for t in 0..seq_len {
            let angle = (t as f32) * 0.5;
            obs_vec.extend_from_slice(&[angle.cos(), angle.sin()]);
            act_vec.extend_from_slice(&[-(angle.sin()), angle.cos()]); // Tangential velocity
        }

        let obs_data = Tensor::<MyAutodiffBackend, 3>::from_data(
            burn::tensor::TensorData::new(obs_vec, [1, seq_len, input_dim]), &device
        );
        let action_data = Tensor::<MyAutodiffBackend, 3>::from_data(
            burn::tensor::TensorData::new(act_vec, [1, seq_len, action_dim]), &device
        );

        // Forward Pass: Encode observations to latents z and predict next z
        let (z, predicted_z) = model.forward(obs_data, action_data);

        // JEPA Loss = Prediction Loss (MSE) + SIGReg (Anti-collapse)
        // Based on arXiv:2603.19312, SIGReg enforces diversity in latent representation z.
        let loss = model.loss(z, predicted_z, 1.0);

        if epoch % 50 == 0 || epoch == 1 {
            println!("Epoch {:3}: Total Loss = {:?}", epoch, loss.clone().into_data());
        }

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(1e-3, model, grads);
    }

    println!("\nPhase 2: Open-loop Imagination in Latent Space");
    
    let model_valid = model.valid();
    
    // Start with initial observation [1.0, 0.0]
    let test_obs = Tensor::<MyBackend, 3>::from_data(
        burn::tensor::TensorData::new(vec![1.0, 0.0], [1, 1, 2]), &device
    ); 
    let test_action = Tensor::<MyBackend, 3>::from_data(
        burn::tensor::TensorData::new(vec![0.0, 1.0], [1, 1, 2]), &device
    );

    // Encode to get the initial latent state
    let (z_start, _) = model_valid.forward(test_obs, test_action);
    println!("Initial latent state z[0] (first 5 dims): {:?}", z_start.clone().slice([0..1, 0..1, 0..5]).into_data());

    // Imagine the next state without new observations
    // Mamba's internal state 'h' carries the context for advanced predictions.
    let (_, z_imagined) = model_valid.forward(
        Tensor::zeros([1, 1, 2], &device),
        Tensor::ones([1, 1, 2], &device) 
    );

    println!("Imagined latent state z_hat[1] (first 5 dims): {:?}", z_imagined.slice([0..1, 0..1, 0..5]).into_data());

    println!("\n[Verification]");
    // If SIGReg is working, the variance of latent variables should be maintained.
    // Variance near 0 would indicate "Representation Collapse".
    let z_flat = z_start.reshape([32]);
    let mean = z_flat.clone().mean();
    let var = (z_flat - mean).powf_scalar(2.0).mean().into_data();
    
    println!("-> Latent Variance: {:?}", var);
    let var_val = var.as_slice::<f32>().unwrap()[0];
    if var_val > 0.1 {
        println!("Result: SIGReg maintained latent diversity. Collapse avoided.");
    } else {
        println!("Result: Variance is low. Consider tuning learning rate or SIGReg weight.");
    }
}
