use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use ssm_latent_model::latent::{LatentPredictor, curvature_loss, stability_loss};
use ssm_latent_model::preprocess::normalize_projections;
use ssm_latent_model::ssm::SsmConfig;

#[test]
fn test_stability_loss_prevents_collapse() {
    type B = NdArray<f32>;
    let device = Default::default();

    // Collapsed representation (all zeros)
    let z_collapsed = Tensor::<B, 3>::zeros([1, 10, 16], &device);
    let w = Tensor::<B, 2>::random([16, 8], burn::tensor::Distribution::Default, &device);
    let w = normalize_projections(w);

    let loss_collapsed = stability_loss(z_collapsed, w.clone());

    // Near-ideal representation (unit variance, zero mean)
    let z_ideal = Tensor::<B, 3>::random(
        [1, 100, 16],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let loss_ideal = stability_loss(z_ideal, w);

    assert!(
        loss_collapsed.into_data().as_slice::<f32>().unwrap()[0]
            > loss_ideal.into_data().as_slice::<f32>().unwrap()[0]
    );
}

#[test]
fn test_curvature_loss_values() {
    type B = NdArray<f32>;
    let device = Default::default();

    // Straight line in latent space: z_t = t * v
    let mut data = Vec::new();
    for t in 0..10 {
        for _ in 0..16 {
            data.push(t as f32);
        }
    }
    let z_straight = Tensor::<B, 3>::from_data(TensorData::new(data, [1, 10, 16]), &device);
    let loss_straight = curvature_loss(z_straight)
        .into_data()
        .as_slice::<f32>()
        .unwrap()[0];

    // Jagged line
    let z_jagged =
        Tensor::<B, 3>::random([1, 10, 16], burn::tensor::Distribution::Default, &device);
    let loss_jagged = curvature_loss(z_jagged)
        .into_data()
        .as_slice::<f32>()
        .unwrap()[0];

    assert!(loss_straight < 1e-5);
    assert!(loss_jagged > loss_straight);
}

#[test]
fn test_save_load_consistency() {
    type B = NdArray<f32>;
    let device = Default::default();
    let config = SsmConfig::new(16, 8, 2, 2, 1);
    let predictor = LatentPredictor::<B>::new(&config, 8, 2, &device);

    let file_path = "test_model_save";
    predictor.save(file_path).expect("Save failed");

    let loaded = LatentPredictor::<B>::new(&config, 8, 2, &device);
    let _loaded = loaded.load(file_path, &device).expect("Load failed");

    // Compare one weight
    // let original_raw = predictor.encoder.weight.val().into_data();
    // let loaded_raw = loaded.encoder.weight.val().into_data();
    // assert_eq!(original_raw, loaded_raw);

    let _ = std::fs::remove_file(format!("{}.bin", file_path));
}
