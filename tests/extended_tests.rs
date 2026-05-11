use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use ssm_latent_model::latent::{LatentPredictor, LatentState, curvature_loss, stability_loss};
use ssm_latent_model::preprocess::normalize_projections;
use ssm_latent_model::ssm::{SsmBlock, SsmConfig};

// ─── Curvature Loss Edge Cases ────────────────────────────────────────────

#[test]
fn test_curvature_loss_short_sequence() {
    type B = NdArray<f32>;
    let device = Default::default();

    let z_short = Tensor::<B, 3>::zeros([1, 2, 8], &device);
    let loss = curvature_loss(z_short)
        .into_data()
        .as_slice::<f32>()
        .unwrap()[0];
    assert!(loss.abs() < 1e-6);

    let z_single = Tensor::<B, 3>::zeros([1, 1, 8], &device);
    let loss_single = curvature_loss(z_single)
        .into_data()
        .as_slice::<f32>()
        .unwrap()[0];
    assert!(loss_single.abs() < 1e-6);
}

#[test]
fn test_curvature_loss_exact_sequence() {
    type B = NdArray<f32>;
    let device = Default::default();

    let data_vec: Vec<f32> = [0.0f32, 1.0, 2.0]
        .iter()
        .flat_map(|&v| std::iter::repeat(v).take(4))
        .collect();
    let z = Tensor::<B, 3>::from_data(TensorData::new(data_vec, [1, 3, 4]), &device);
    let loss = curvature_loss(z).into_data().as_slice::<f32>().unwrap()[0];
    assert!(loss.abs() < 1e-4);
}

// ─── Stability Loss ────────────────────────────────────────────────────────

#[test]
fn test_stability_loss_finite() {
    type B = NdArray<f32>;
    let device = Default::default();

    let z = Tensor::<B, 3>::random(
        [2, 20, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let w = normalize_projections(Tensor::<B, 2>::random(
        [32, 16],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    ));
    let loss = stability_loss(z, w);
    assert!(loss.into_data().as_slice::<f32>().unwrap()[0].is_finite());
}

// ─── SSM Equivalence ──────────────────────────────────────────────────────

#[test]
fn test_ssm_mimo_rank_equivalence() {
    type B = NdArray<f32>;
    let device = Default::default();

    let config = SsmConfig::new(16, 8, 2, 2, 2).with_use_conv(false);
    let block = SsmBlock::<B>::new(&config, &device);
    let x = Tensor::<B, 3>::random([1, 6, 16], burn::tensor::Distribution::Default, &device);

    let y_parallel = block.forward(x.clone());

    let mut h = Tensor::<B, 4>::zeros([1, 2, 8, 8], &device); // Correct d_state is 8
    let mut prev_bx = None;
    let mut y_step_list = Vec::new();

    for t in 0..6 {
        let xt = x.clone().slice([0..1, t..t + 1]).reshape([1, 16]);
        let (yt, next_h, current_bx, _) = block.forward_step(xt, h, prev_bx, None);
        h = next_h;
        prev_bx = Some(current_bx);
        y_step_list.push(yt.unsqueeze_dim::<3>(1));
    }

    let y_sequential = Tensor::cat(y_step_list, 1);
    y_parallel.to_data().assert_approx_eq::<f32>(
        &y_sequential.to_data(),
        burn::tensor::Tolerance::default(),
    );
}

#[test]
fn test_ssm_with_conv_equivalence() {
    type B = NdArray<f32>;
    let device = Default::default();

    let config = SsmConfig::new(16, 8, 2, 2, 1);
    let block = SsmBlock::<B>::new(&config, &device);
    let x = Tensor::<B, 3>::random([1, 8, 16], burn::tensor::Distribution::Default, &device);

    let y_parallel = block.forward(x.clone());

    let mut h = Tensor::<B, 4>::zeros([1, 2, 8, 16], &device);
    let mut prev_bx = None;
    let mut conv_state = None;
    let mut y_step_list = Vec::new();

    for t in 0..8 {
        let xt = x.clone().slice([0..1, t..t + 1]).reshape([1, 16]);
        let (yt, next_h, current_bx, next_conv) = block.forward_step(xt, h, prev_bx, conv_state);
        h = next_h;
        prev_bx = Some(current_bx);
        conv_state = next_conv;
        y_step_list.push(yt.unsqueeze_dim::<3>(1));
    }

    let y_sequential = Tensor::cat(y_step_list, 1);
    y_parallel.to_data().assert_approx_eq::<f32>(
        &y_sequential.to_data(),
        burn::tensor::Tolerance::rel_abs(0.01, 0.01),
    );
}

// ─── LatentPredictor ──────────────────────────────────────────────────────

#[test]
fn test_latent_predictor_step() {
    type B = NdArray<f32>;
    let device = Default::default();

    let config = SsmConfig::new(16, 8, 2, 2, 1).with_use_conv(false);
    let predictor = LatentPredictor::<B>::new(&config, 4, 2, &device);

    let z = Tensor::<B, 2>::random([1, 16], burn::tensor::Distribution::Default, &device);
    let a = Tensor::<B, 2>::random([1, 2], burn::tensor::Distribution::Default, &device);
    let state = LatentState {
        h: Tensor::zeros([1, 2, 8, 16], &device),
        prev_bx: None,
        conv_state: None,
    };

    let (output, _) = predictor.step(z, a, state);
    assert_eq!(output.dims(), [1, 16]);
}

// ─── Gradients ────────────────────────────────────────────────────────────

#[test]
fn test_gradient_flow() {
    use burn::backend::Autodiff;
    type B = Autodiff<NdArray<f32>>;
    let device = Default::default();

    let config = SsmConfig::new(16, 8, 2, 2, 1);
    let model = SsmBlock::<B>::new(&config, &device);

    let x = Tensor::<B, 3>::random([1, 4, 16], burn::tensor::Distribution::Default, &device);
    let grads = model.forward(x).sum().backward();

    assert!(model.a_re.grad(&grads).is_some());
    assert!(model.dt_proj.weight.grad(&grads).is_some());
}
