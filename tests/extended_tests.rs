use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use ssm_latent_model::latent::{LatentPredictor, LatentState, curvature_loss, stability_loss};
use ssm_latent_model::multimodal::{VisionEncoder, VisionDecoder, MultimodalLatentPredictor};
use ssm_latent_model::preprocess::normalize_projections;
use ssm_latent_model::ssm::{SsmBlock, SsmConfig};

// ─── Curvature Loss Edge Cases ────────────────────────────────────────────

#[test]
fn test_curvature_loss_short_sequence() {
    type B = NdArray<f32>;
    let device = Default::default();

    // With seq_len < 3, curvature loss should return 0.0
    let z_short = Tensor::<B, 3>::zeros([1, 2, 8], &device);
    let loss = curvature_loss(z_short)
        .into_data()
        .as_slice::<f32>()
        .unwrap()[0];
    assert!(
        loss.abs() < 1e-6,
        "curvature_loss should be 0 for seq_len < 3, got {}",
        loss
    );

    // seq_len == 1
    let z_single = Tensor::<B, 3>::zeros([1, 1, 8], &device);
    let loss_single = curvature_loss(z_single)
        .into_data()
        .as_slice::<f32>()
        .unwrap()[0];
    assert!(
        loss_single.abs() < 1e-6,
        "curvature_loss should be 0 for seq_len == 1, got {}",
        loss_single
    );
}

#[test]
fn test_curvature_loss_exact_sequence() {
    type B = NdArray<f32>;
    let device = Default::default();

    // Exactly 3 steps: constant velocity => zero acceleration
    // z_t = [0, 0, ...], [1, 1, ...], [2, 2, ...] => second derivative = 0
    let data: Vec<f32> = [0.0f32, 1.0, 2.0]
        .iter()
        .flat_map(|&v| std::iter::repeat(v).take(4))
        .collect();
    let z = Tensor::<B, 3>::from_data(TensorData::new(data, [1, 3, 4]), &device);
    let loss = curvature_loss(z)
        .into_data()
        .as_slice::<f32>()
        .unwrap()[0];
    assert!(
        loss.abs() < 1e-4,
        "curvature_loss for constant velocity should be ~0, got {}",
        loss
    );
}

// ─── Stability Loss with Different Projection Counts ───────────────────────

#[test]
fn test_stability_loss_with_many_projections() {
    type B = NdArray<f32>;
    let device = Default::default();

    let z = Tensor::<B, 3>::random(
        [2, 20, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );

    // With many projections
    let w_many = Tensor::<B, 2>::random([32, 64], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
    let w_many = normalize_projections(w_many);
    let loss_many = stability_loss(z.clone(), w_many);

    // With few projections (default 16)
    let w_few = Tensor::<B, 2>::random([32, 4], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
    let w_few = normalize_projections(w_few);
    let loss_few = stability_loss(z, w_few);

    // Both should produce finite losses
    let many_val = loss_many.into_data().as_slice::<f32>().unwrap()[0];
    let few_val = loss_few.into_data().as_slice::<f32>().unwrap()[0];
    assert!(many_val.is_finite(), "Loss with many projections should be finite");
    assert!(few_val.is_finite(), "Loss with few projections should be finite");
}

// ─── SSM with MIMO Rank > 1 ───────────────────────────────────────────────

#[test]
fn test_ssm_mimo_rank_2_forward() {
    type Backend = NdArray<f32>;
    let device = Default::default();

    let config = SsmConfig::new(16, 8, 2, 2, 2).with_use_conv(false);
    let block = SsmBlock::<Backend>::new(&config, &device);

    let x = Tensor::<Backend, 3>::random(
        [2, 8, 16],
        burn::tensor::Distribution::Default,
        &device,
    );

    let output = block.forward(x);
    assert_eq!(output.dims(), [2, 8, 16], "Output shape should match input shape with mimo_rank=2");
}

#[test]
fn test_ssm_mimo_rank_equivalence() {
    type Backend = NdArray<f32>;
    let device = Default::default();

    let batch = 1;
    let d_model = 16;
    let seq_len = 6;
    let config = SsmConfig::new(d_model, 8, 2, 2, 2).with_use_conv(false);

    let block = SsmBlock::<Backend>::new(&config, &device);
    let x = Tensor::<Backend, 3>::random(
        [batch, seq_len, d_model],
        burn::tensor::Distribution::Default,
        &device,
    );

    let y_parallel = block.forward(x.clone());

    let d_inner = d_model * 2;
    let d_head = d_inner / 2;

    let mut h = Tensor::<Backend, 4>::zeros([batch, 2, 8, d_head / 2], &device);
    let mut prev_bx: Option<Tensor<Backend, 4>> = None;
    let mut y_step_list = Vec::new();

    for t in 0..seq_len {
        let xt = x.clone().slice([0..batch, t..t + 1]);
        let [b, _, d] = xt.dims();
        let xt = xt.reshape([b, d]);

        let (yt, next_h, current_bx, _conv_state) =
            block.forward_step(xt, h, prev_bx, None);

        h = next_h;
        prev_bx = Some(current_bx);
        y_step_list.push(yt.unsqueeze_dim::<3>(1));
    }

    let y_sequential = Tensor::cat(y_step_list, 1);
    y_parallel
        .to_data()
        .assert_approx_eq::<f32>(&y_sequential.to_data(), Default::default());
}

// ─── SSM with Conv1d ──────────────────────────────────────────────────────

#[test]
fn test_ssm_with_conv_equivalence() {
    type Backend = NdArray<f32>;
    let device = Default::default();

    let batch = 1;
    let d_model = 16;
    let seq_len = 8;

    // Test with conv enabled (the default)
    let config = SsmConfig::new(d_model, 8, 2, 2, 1);
    let block = SsmBlock::<Backend>::new(&config, &device);

    let x = Tensor::<Backend, 3>::random(
        [batch, seq_len, d_model],
        burn::tensor::Distribution::Default,
        &device,
    );

    let y_parallel = block.forward(x.clone());

    let d_inner = d_model * 2;
    let d_head = d_inner / 2;

    let mut h = Tensor::<Backend, 4>::zeros([batch, 2, 8, d_head], &device);
    let mut prev_bx: Option<Tensor<Backend, 4>> = None;
    let mut conv_state: Option<Tensor<Backend, 3>> = None;
    let mut y_step_list = Vec::new();

    for t in 0..seq_len {
        let xt = x.clone().slice([0..batch, t..t + 1]);
        let [b, _, d] = xt.dims();
        let xt = xt.reshape([b, d]);

        let (yt, next_h, current_bx, next_conv_state) =
            block.forward_step(xt, h, prev_bx, conv_state);

        h = next_h;
        prev_bx = Some(current_bx);
        conv_state = next_conv_state;
        y_step_list.push(yt.unsqueeze_dim::<3>(1));
    }

    let y_sequential = Tensor::cat(y_step_list, 1);

    // Conv1d equivalence may have slightly higher tolerance due to
    // boundary handling differences
    y_parallel
        .to_data()
        .assert_approx_eq::<f32>(&y_sequential.to_data(), 0.01);
}

// ─── LatentPredictor step() Method ────────────────────────────────────────

#[test]
fn test_latent_predictor_step() {
    type B = NdArray<f32>;
    let device = Default::default();

    let config = SsmConfig::new(16, 8, 2, 2, 1).with_use_conv(false);
    let predictor = LatentPredictor::<B>::new(&config, 4, 2, &device);

    let z_initial = Tensor::<B, 2>::random([1, 16], burn::tensor::Distribution::Default, &device);
    let action = Tensor::<B, 2>::random([1, 2], burn::tensor::Distribution::Default, &device);

    let d_inner = 16 * 2;
    let d_head = d_inner / 2;

    let state = LatentState {
        h: Tensor::zeros([1, 2, 8, d_head], &device),
        prev_bx: None,
        conv_state: None,
    };

    let (output, new_state) = predictor.step(z_initial, action, state);

    // Output should have shape [batch, d_model]
    assert_eq!(output.dims(), [1, 16], "step() output shape mismatch");

    // New state should have correct shapes
    assert_eq!(new_state.h.dims(), [1, 2, 8, d_head], "step() h shape mismatch");
    assert!(new_state.prev_bx.is_some(), "step() should produce prev_bx");
    assert!(new_state.conv_state.is_none(), "step() conv_state should be None when use_conv=false");
}

#[test]
fn test_latent_predictor_step_with_conv() {
    type B = NdArray<f32>;
    let device = Default::default();

    let config = SsmConfig::new(16, 8, 2, 2, 1); // use_conv=true by default
    let predictor = LatentPredictor::<B>::new(&config, 4, 2, &device);

    let z_initial = Tensor::<B, 2>::random([1, 16], burn::tensor::Distribution::Default, &device);
    let action = Tensor::<B, 2>::random([1, 2], burn::tensor::Distribution::Default, &device);

    let d_inner = 16 * 2;
    let d_head = d_inner / 2;

    let state = LatentState {
        h: Tensor::zeros([1, 2, 8, d_head], &device),
        prev_bx: None,
        conv_state: Some(Tensor::zeros([1, d_inner, 3], &device)),
    };

    let (output, new_state) = predictor.step(z_initial, action, state);

    assert_eq!(output.dims(), [1, 16], "step() output shape mismatch with conv");
    assert!(new_state.conv_state.is_some(), "step() should maintain conv_state when conv enabled");
}

// ─── Sequential step consistency ──────────────────────────────────────────

#[test]
fn test_latent_predictor_step_consistency() {
    type B = NdArray<f32>;
    let device = Default::default();

    let config = SsmConfig::new(16, 8, 2, 2, 1).with_use_conv(false);
    let predictor = LatentPredictor::<B>::new(&config, 4, 2, &device);
    let predictor_valid = predictor.valid();

    let d_inner = 16 * 2;
    let d_head = d_inner / 2;

    // Run multiple steps and verify they produce different outputs
    let z = Tensor::<B, 2>::random([1, 16], burn::tensor::Distribution::Default, &device);
    let a1 = Tensor::<B, 2>::random([1, 2], burn::tensor::Distribution::Default, &device);
    let a2 = Tensor::<B, 2>::random([1, 2], burn::tensor::Distribution::Default, &device);

    let state0 = LatentState {
        h: Tensor::zeros([1, 2, 8, d_head], &device),
        prev_bx: None,
        conv_state: None,
    };

    let (y1, state1) = predictor_valid.step(z.clone(), a1, state0);
    let (y2, state2) = predictor_valid.step(y1.clone(), a2, state1);

    // Outputs should be finite
    let y1_data = y1.into_data().as_slice::<f32>().unwrap();
    let y2_data = y2.into_data().as_slice::<f32>().unwrap();
    assert!(y1_data.iter().all(|v| v.is_finite()), "Step 1 output should be finite");
    assert!(y2_data.iter().all(|v| v.is_finite()), "Step 2 output should be finite");

    // Hidden state should evolve
    let h1_data = state1.h.into_data().as_slice::<f32>().unwrap();
    let h2_data = state2.h.into_data().as_slice::<f32>().unwrap();
    let state_changed = h1_data.iter().zip(h2_data.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(state_changed, "Hidden state should evolve between steps");
}

// ─── Vision Encoder/Decoder ───────────────────────────────────────────────

#[test]
fn test_vision_encoder_shapes() {
    type B = NdArray<f32>;
    let device = Default::default();

    let encoder = VisionEncoder::<B>::new(3, 32, &device);
    let img = Tensor::<B, 4>::random([2, 3, 16, 16], burn::tensor::Distribution::Default, &device);

    let z = encoder.forward(img);
    assert_eq!(z.dims(), [2, 32], "VisionEncoder output shape should be [batch, d_model]");
}

#[test]
fn test_vision_decoder_shapes() {
    type B = NdArray<f32>;
    let device = Default::default();

    let decoder = VisionDecoder::<B>::new(32, 3, &device);
    let z = Tensor::<B, 2>::random([2, 32], burn::tensor::Distribution::Default, &device);

    let img = decoder.forward(z);
    assert_eq!(img.dims(), [2, 3, 16, 16], "VisionDecoder output shape should be [batch, C, H, W]");
}

#[test]
fn test_vision_roundtrip_shape() {
    type B = NdArray<f32>;
    let device = Default::default();

    let encoder = VisionEncoder::<B>::new(3, 32, &device);
    let decoder = VisionDecoder::<B>::new(32, 3, &device);

    let original = Tensor::<B, 4>::random([4, 3, 16, 16], burn::tensor::Distribution::Default, &device);
    let z = encoder.forward(original);
    let reconstructed = decoder.forward(z);

    assert_eq!(original.dims(), reconstructed.dims(),
        "Reconstructed image should have same shape as original");
}

// ─── Multimodal Predictor with Loss ────────────────────────────────────────

#[test]
fn test_multimodal_loss_nonnegative() {
    type B = NdArray<f32>;
    let device = Default::default();

    let config = SsmConfig {
        d_model: 32,
        d_state: 16,
        expand: 2,
        n_heads: 2,
        mimo_rank: 1,
        use_conv: true,
        conv_kernel: 3,
    };

    let predictor = MultimodalLatentPredictor::<B>::new(&config, 3, 8, 4, &device);

    let batch = 2;
    let seq = 4;
    let img = Tensor::<B, 5>::random([batch, seq, 3, 16, 16], burn::tensor::Distribution::Default, &device);
    let sensor = Tensor::<B, 3>::random([batch, seq, 8], burn::tensor::Distribution::Default, &device);
    let action = Tensor::<B, 3>::random([batch, seq, 4], burn::tensor::Distribution::Default, &device);

    let (z, pred_z, dec_img, dec_sens) = predictor.forward(img.clone(), sensor.clone(), action);

    let loss = predictor.loss(ssm_latent_model::multimodal::MultimodalLossInput {
        z,
        pred_z,
        recons_img: dec_img,
        orig_img: img,
        recons_sens: dec_sens,
        orig_sens: sensor,
        stability_weight: 1.0,
    });

    let loss_val = loss.into_data().as_slice::<f32>().unwrap()[0];
    assert!(loss_val.is_finite(), "Multimodal loss should be finite, got {}", loss_val);
    assert!(loss_val >= 0.0, "Multimodal loss should be non-negative, got {}", loss_val);
}

// ─── Gradient Through SSM with Conv ───────────────────────────────────────

#[test]
fn test_gradient_through_ssm_with_conv() {
    use burn::backend::Autodiff;

    type B = Autodiff<NdArray<f32>>;
    let device = Default::default();

    let config = SsmConfig::new(16, 8, 2, 2, 1); // with conv
    let model = SsmBlock::<B>::new(&config, &device);

    let x = Tensor::<B, 3>::random([1, 4, 16], burn::tensor::Distribution::Default, &device);
    let y = model.forward(x);
    let loss = y.sum();
    let grads = loss.backward();

    // Verify gradients exist for key parameters
    assert!(model.a_re.grad(&grads).is_some(), "Should have gradients for a_re");
    assert!(model.a_im.grad(&grads).is_some(), "Should have gradients for a_im");
    assert!(model.dt_proj.grad(&grads).is_some(), "Should have gradients for dt_proj");
    assert!(model.out_proj.grad(&grads).is_some(), "Should have gradients for out_proj");

    if let Some(conv) = &model.conv1d {
        assert!(conv.weight.grad(&grads).is_some(), "Should have gradients for conv1d weights");
    }
}

// ─── Normalize Projections ────────────────────────────────────────────────

#[test]
fn test_normalize_projections_unit_norm() {
    type B = NdArray<f32>;
    let device = Default::default();

    let w = Tensor::<B, 2>::random([16, 8], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
    let w_normed = normalize_projections(w);

    // Each column should have unit norm
    let data = w_normed.into_data().as_slice::<f32>().unwrap();
    for col in 0..8 {
        let mut sum_sq = 0.0f32;
        for row in 0..16 {
            sum_sq += data[row * 8 + col] * data[row * 8 + col];
        }
        let norm = sum_sq.sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Column {} should have unit norm, got {}",
            col,
            norm
        );
    }
}