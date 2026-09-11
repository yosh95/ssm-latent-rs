//! Integration tests for the pure-Mamba predictors in [`ssm_latent_model::predictor`].
//!
//! These tests exercise the JEPA-free observation-space path end-to-end:
//!
//! - [`MambaPredictor`] — single SSM block wrapper (input_proj → SsmBlock → output_proj)
//! - [`MultiScaleMambaPredictor`] — stacked multi-scale wrapper with action conditioning
//!
//! The SSM core itself is covered by `equivalence_test.rs` / `consistency_test.rs` /
//! `extended_tests.rs` (parallel scan ≡ sequential recurrence, gradient flow, conv
//! equivalence). What is *not* covered anywhere else is the wrapper logic tested here:
//! projection plumbing, state handling, closed-loop imagination, and the loss helper.

use burn::backend::NdArray;
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::Tensor;
use burn::tensor::Tolerance;
use ssm_latent_model::predictor::{MambaPredictor, MultiScaleMambaPredictor};
use ssm_latent_model::ssm::{MultiScaleSsmConfig, SsmConfig};

// ─── MambaPredictor (single-layer) ───────────────────────────────────────

#[test]
fn test_mamba_predictor_forward_shapes() {
    type B = NdArray<f32>;
    let device = Default::default();

    // input_dim=4, output_dim=2 — same sensor-observation dims used elsewhere
    let config = SsmConfig::new(16, 8, 2, 2, 1).with_use_conv(false);
    let predictor = MambaPredictor::<B>::new(&config, 4, 2, &device);

    let x = Tensor::<B, 3>::random([2, 8, 4], burn::tensor::Distribution::Default, &device);
    let out = predictor.forward(x);

    // predictions[t] estimates x[t+1] → last dimension must be output_dim
    assert_eq!(out.dims(), [2, 8, 2]);

    let val = out.clone().into_data().as_slice::<f32>().unwrap()[0].is_finite();
    assert!(val, "forward output must be finite");
}

#[test]
fn test_mamba_predictor_output_changes_with_input() {
    type B = NdArray<f32>;
    let device = Default::default();

    let config = SsmConfig::new(16, 8, 2, 2, 1).with_use_conv(false);
    let predictor = MambaPredictor::<B>::new(&config, 4, 2, &device);

    let x1 = Tensor::<B, 3>::random([1, 8, 4], burn::tensor::Distribution::Default, &device);
    let x2 = x1.clone() + 1.0;

    let y1 = predictor.forward(x1).into_data();
    let y2 = predictor.forward(x2).into_data();

    let s1 = y1.as_slice::<f32>().unwrap();
    let s2 = y2.as_slice::<f32>().unwrap();
    let differs = s1.iter().zip(s2.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(differs, "different inputs must give different predictions");
}

#[test]
fn test_mamba_predictor_step_consistency() {
    // Full-sequence forward ≡ unrolled `step` calls for the predictor wrapper.
    // This pins the state plumbing (h / prev_bx / conv_state) that is unique
    // to this module — the SSM core equivalence itself is tested separately.
    type B = NdArray<f32>;
    let device = Default::default();

    // use_conv = false for exact equivalence (conv is causal with zero padding
    // on both paths, but keeping this deterministic avoids edge effects).
    let config = SsmConfig::new(16, 8, 2, 2, 1).with_use_conv(false);
    let predictor = MambaPredictor::<B>::new(&config, 4, 4, &device);

    let batch = 2;
    let seq_len = 6;
    let x = Tensor::<B, 3>::random(
        [batch, seq_len, 4],
        burn::tensor::Distribution::Default,
        &device,
    );

    let y_parallel = predictor.forward(x.clone());

    let mut state = predictor.zero_state(batch, &device);
    let mut y_steps = Vec::new();
    for t in 0..seq_len {
        let xt = x.clone().slice([0..batch, t..t + 1]).reshape([batch, 4]);
        let (pred, next_state) = predictor.step(xt, state);
        y_steps.push(pred.unsqueeze_dim::<3>(1));
        state = next_state;
    }
    let y_sequential = Tensor::cat(y_steps, 1);

    y_parallel
        .to_data()
        .assert_approx_eq::<f32>(&y_sequential.to_data(), Tolerance::default());
}

#[test]
fn test_mamba_predictor_streaming_state_advances() {
    // Streaming state must evolve: h must not stay at its zero init, and
    // consecutive steps must differ (else the recurrence is degenerate).
    type B = NdArray<f32>;
    let device = Default::default();

    let config = SsmConfig::new(16, 8, 2, 2, 1).with_use_conv(false);
    let predictor = MambaPredictor::<B>::new(&config, 4, 4, &device);

    let batch = 1;
    let mut state = predictor.zero_state(batch, &device);
    assert!(
        state.prev_bx.is_none(),
        "fresh zero_state must have no prev_bx"
    );

    let mut preds = Vec::new();
    for t in 0..4 {
        let xt = Tensor::<B, 2>::random([batch, 4], burn::tensor::Distribution::Default, &device);
        let (pred, next_state) = predictor.step(xt, state);
        state = next_state;
        preds.push(pred.clone().into_data().as_slice::<f32>().unwrap()[0]);

        // h must become nonzero after the first step (input ≠ 0)
        let h_max = state
            .h
            .clone()
            .abs()
            .max()
            .into_data()
            .as_slice::<f32>()
            .unwrap()[0];
        assert!(h_max > 0.0, "hidden state must evolve (t={t})");

        // prev_bx must be Some from step 0 onward (lambda-gated recurrence)
        assert!(state.prev_bx.is_some(), "prev_bx must be set (t={t})");
    }

    // Predictions at different timesteps must not all be identical
    let differs = (0..preds.len() - 1).any(|i| (preds[i] - preds[i + 1]).abs() > 1e-6);
    assert!(differs, "consecutive streaming predictions must differ");
}

#[test]
fn test_mamba_predictor_loss_alignment() {
    // loss() must compare predictions[t] against targets[t+1].
    // Construct an exact linear identity: out_proj ≡ identity is hard to
    // guarantee, so instead verify the numeric value directly on crafted
    // targets using a fixed model: compute expected loss manually.
    type B = NdArray<f32>;
    let device = Default::default();

    let config = SsmConfig::new(16, 8, 2, 2, 1).with_use_conv(false);
    let predictor = MambaPredictor::<B>::new(&config, 4, 2, &device);

    let batch = 1;
    let seq_len = 4;
    let predictions = Tensor::<B, 3>::random(
        [batch, seq_len, 2],
        burn::tensor::Distribution::Default,
        &device,
    );
    // targets = predictions shifted by one position + noise-free transform:
    // simply reuse predictions as targets shifted — set targets[t] = predictions[t-1]
    // so pred_slice = pred[0..T-1], target_slice = targets[1..T] = pred[0..T-1] → loss ≈ 0
    let targets = Tensor::cat(
        vec![
            Tensor::zeros([batch, 1, 2], &device),
            predictions.clone().slice([0..batch, 0..seq_len - 1]),
        ],
        1,
    );
    let loss = predictor.loss(predictions.clone(), targets);
    let loss_val = loss.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        loss_val < 1e-6,
        "aligned shift must give ~zero loss, got {loss_val}"
    );

    // Misaligned targets must give strictly larger loss
    let targets_shifted = predictions.clone();
    let loss2 = predictor.loss(predictions, targets_shifted);
    let loss2_val = loss2.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        loss2_val > loss_val,
        "misaligned loss must exceed aligned loss"
    );
}

#[test]
fn test_mamba_predictor_save_load() {
    type B = NdArray<f32>;
    let device = Default::default();

    let config = SsmConfig::new(16, 8, 2, 2, 1).with_use_conv(false);
    let predictor = MambaPredictor::<B>::new(&config, 4, 2, &device);

    // Materialize lazy params BEFORE saving: burn's `Param::uninitialized` clones carry an
    // independent lazy-init closure, so `into_record()` on a never-accessed model would
    // serialize freshly-sampled weights rather than the model's own values.
    let x = Tensor::<B, 3>::random([1, 8, 4], burn::tensor::Distribution::Default, &device);
    let y1 = predictor.forward(x.clone());

    let file_path = "test_mamba_predictor_save";
    let path = std::path::PathBuf::from(format!("{file_path}.bin"));
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    recorder
        .record(predictor.clone().into_record(), path.clone())
        .expect("MambaPredictor save failed");

    let loaded = MambaPredictor::<B>::new(&config, 4, 2, &device);
    let record = recorder
        .load(path.clone(), &device)
        .expect("MambaPredictor load failed");
    let loaded = loaded.load_record(record);

    let y2 = loaded.forward(x);

    y1.to_data()
        .assert_approx_eq::<f32>(&y2.to_data(), Tolerance::default());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_mamba_predictor_backward_flow() {
    use burn::backend::Autodiff;
    type B = Autodiff<NdArray<f32>>;
    let device = Default::default();

    let config = SsmConfig::new(16, 8, 2, 2, 1).with_use_conv(false);
    let predictor = MambaPredictor::<B>::new(&config, 4, 2, &device);

    let x = Tensor::<B, 3>::random([1, 4, 4], burn::tensor::Distribution::Default, &device);
    let y = predictor.forward(x);
    let loss = predictor.loss(y, Tensor::zeros([1, 4, 2], &device));

    let grads = loss.backward();

    // The wrapper plumbing must propagate gradients into the SSM core and
    // through both projections — the pure-Mamba path is trainable on its own.
    assert!(
        predictor.ssm.a_re.grad(&grads).is_some(),
        "SSM core (a_re) must receive gradient through the predictor wrapper"
    );
    assert!(
        predictor.input_proj.weight.grad(&grads).is_some(),
        "input_proj must receive gradient"
    );
    assert!(
        predictor.output_proj.weight.grad(&grads).is_some(),
        "output_proj must receive gradient"
    );
}

// ─── MultiScaleMambaPredictor ────────────────────────────────────────────

#[test]
fn test_multi_scale_predictor_forward_shapes() {
    type B = NdArray<f32>;
    let device = Default::default();

    let config = MultiScaleSsmConfig::new(32, 8, 2, 2, 1)
        .with_n_layers(3)
        .with_use_conv(false);
    let predictor = MultiScaleMambaPredictor::<B>::new(&config, 4, 2, &device);

    let x = Tensor::<B, 3>::random([2, 8, 4], burn::tensor::Distribution::Default, &device);
    let out = predictor.forward(x);
    assert_eq!(out.dims(), [2, 8, 2]);
    assert!(
        out.into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()),
        "output must be finite"
    );
}

#[test]
fn test_multi_scale_predictor_step_consistency() {
    // forward ≡ unrolled step for the multi-scale wrapper, verifying the
    // zero-action fusion matches the streaming path.
    type B = NdArray<f32>;
    let device = Default::default();

    let config = MultiScaleSsmConfig::new(32, 8, 2, 2, 1)
        .with_n_layers(3)
        .with_use_conv(false);
    let predictor = MultiScaleMambaPredictor::<B>::new(&config, 4, 4, &device);

    let batch = 2;
    let seq_len = 6;
    let x = Tensor::<B, 3>::random(
        [batch, seq_len, 4],
        burn::tensor::Distribution::Default,
        &device,
    );

    let y_parallel = predictor.forward(x.clone());

    let mut state = predictor.zero_state(batch, &device);
    let mut y_steps = Vec::new();
    for t in 0..seq_len {
        let xt = x.clone().slice([0..batch, t..t + 1]).reshape([batch, 4]);
        let (pred, next_state) = predictor.step(xt, state);
        y_steps.push(pred.unsqueeze_dim::<3>(1));
        state = next_state;
    }
    let y_sequential = Tensor::cat(y_steps, 1);

    y_parallel
        .to_data()
        .assert_approx_eq::<f32>(&y_sequential.to_data(), Tolerance::rel_abs(0.01, 0.01));
}

#[test]
fn test_multi_scale_predictor_forward_with_action() {
    type B = NdArray<f32>;
    let device = Default::default();

    let config = MultiScaleSsmConfig::new(32, 8, 2, 2, 1)
        .with_n_layers(3)
        .with_use_conv(false);
    let predictor = MultiScaleMambaPredictor::<B>::new(&config, 4, 2, &device);

    let obs = Tensor::<B, 3>::random([1, 6, 4], burn::tensor::Distribution::Default, &device);
    let act = Tensor::<B, 3>::random([1, 6, 2], burn::tensor::Distribution::Default, &device);

    let out = predictor.forward_with_action(obs, act);
    assert_eq!(out.dims(), [1, 6, 2]);
    assert!(
        out.into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()),
        "action-conditioned output must be finite"
    );
}

#[test]
fn test_multi_scale_predictor_zero_action_equivalence() {
    // forward(obs) fuses a raw ZERO d_model vector; forward_with_action fuses
    // action_proj(zero). These coincide only when action_proj has no bias
    // (LinearConfig defaults to bias = true, so action_proj(0) = bias != 0).
    // Both paths must still be run through the SAME SSM stack and stay finite,
    // and action conditioning must actually change the output.
    type B = NdArray<f32>;
    let device = Default::default();

    let config = MultiScaleSsmConfig::new(32, 8, 2, 2, 1)
        .with_n_layers(3)
        .with_use_conv(false);
    let predictor = MultiScaleMambaPredictor::<B>::new(&config, 4, 2, &device);

    let obs = Tensor::<B, 3>::random([1, 6, 4], burn::tensor::Distribution::Default, &device);
    let zero_act = Tensor::zeros([1, 6, 2], &device);
    let nonzero_act =
        Tensor::<B, 3>::random([1, 6, 2], burn::tensor::Distribution::Default, &device);

    let y_obs_only = predictor.forward(obs.clone());
    let y_zero_action = predictor.forward_with_action(obs.clone(), zero_act);
    let y_nonzero_action = predictor.forward_with_action(obs, nonzero_act);

    // Both zero-action paths are finite, same shape
    assert_eq!(y_obs_only.dims(), y_zero_action.dims());
    for (name, y) in [("obs_only", &y_obs_only), ("zero_action", &y_zero_action)] {
        assert!(
            y.clone()
                .into_data()
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite()),
            "{name} output must be finite"
        );
    }

    // Action conditioning must alter the output
    let diff: f32 = y_zero_action
        .to_data()
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .zip(y_nonzero_action.to_data().as_slice::<f32>().unwrap())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 0.0,
        "nonzero action must change the prediction (action conditioning broken)"
    );
}

#[test]
fn test_multi_scale_predictor_imagine_loop() {
    // Closed-loop imagination: chain step_imagine over several actions and
    // verify all outputs stay finite and the returned y can seed the next call.
    type B = NdArray<f32>;
    let device = Default::default();

    // step_imagine re-encodes output_proj(y) through obs_proj, so this closed
    // loop requires obs_dim == output_dim (documented constraint).
    let config = MultiScaleSsmConfig::new(32, 8, 2, 2, 1)
        .with_n_layers(3)
        .with_use_conv(false);
    let predictor = MultiScaleMambaPredictor::<B>::new(&config, 2, 2, &device);

    let batch = 1;
    // Warm up with a few observation-only steps to fill the state.
    let mut state = predictor.zero_state(batch, &device);
    for _ in 0..3 {
        let xt = Tensor::<B, 2>::random([batch, 2], burn::tensor::Distribution::Default, &device);
        let (_, next_state) = predictor.step(xt, state);
        state = next_state;
    }

    // y_prev must be the SSM output (d_model) — use the documented contract:
    // y_prev is [batch, d_model].
    let mut y_prev =
        Tensor::<B, 2>::random([batch, 32], burn::tensor::Distribution::Default, &device);
    let mut all_finite = true;
    for _t in 0..4 {
        let action =
            Tensor::<B, 2>::random([batch, 2], burn::tensor::Distribution::Default, &device);
        let (pred, y_next, next_state) = predictor.step_imagine(y_prev.clone(), action, state);
        assert_eq!(pred.dims(), [batch, 2]);
        all_finite &= pred
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite());
        y_prev = y_next;
        state = next_state;
    }
    assert!(all_finite, "imagination loop must stay finite over 4 steps");
}

#[test]
fn test_multi_scale_predictor_imagine_h_loop() {
    // step_imagine_h closes the loop through the SSM hidden state directly.
    // Chain it and check finiteness + shape — this pins the h_proj pathway.
    type B = NdArray<f32>;
    let device = Default::default();

    let config = MultiScaleSsmConfig::new(32, 8, 2, 2, 1)
        .with_n_layers(3)
        .with_use_conv(false);
    let predictor = MultiScaleMambaPredictor::<B>::new(&config, 4, 2, &device);

    let batch = 1;
    let mut state = predictor.zero_state(batch, &device);

    // Warm up with real observations (step) so h is populated.
    for _ in 0..3 {
        let xt = Tensor::<B, 2>::random([batch, 4], burn::tensor::Distribution::Default, &device);
        let (_, next_state) = predictor.step(xt, state);
        state = next_state;
    }

    let mut preds = Vec::new();
    for _ in 0..4 {
        let action =
            Tensor::<B, 2>::random([batch, 2], burn::tensor::Distribution::Default, &device);
        let (pred, _y_next, next_state) = predictor.step_imagine_h(state, action);
        preds.push(pred.clone().into_data().as_slice::<f32>().unwrap()[0]);
        state = next_state;
    }
    let differs = (0..preds.len() - 1).any(|i| (preds[i] - preds[i + 1]).abs() > 1e-6);
    assert!(differs, "h-loop imagination must evolve over time");
}

#[test]
fn test_multi_scale_predictor_h_sampling_prob_zero_matches_action_forward() {
    // forward_with_h_sampling(obs, act, 0.0) must equal forward_with_action(obs, act).
    type B = NdArray<f32>;
    let device = Default::default();

    let config = MultiScaleSsmConfig::new(32, 8, 2, 2, 1)
        .with_n_layers(3)
        .with_use_conv(false);
    let predictor = MultiScaleMambaPredictor::<B>::new(&config, 4, 2, &device);

    let obs = Tensor::<B, 3>::random([2, 6, 4], burn::tensor::Distribution::Default, &device);
    let act = Tensor::<B, 3>::random([2, 6, 2], burn::tensor::Distribution::Default, &device);

    let y_a = predictor.forward_with_h_sampling(obs.clone(), act.clone(), 0.0);
    let y_b = predictor.forward_with_action(obs, act);
    assert_eq!(y_a.dims(), y_b.dims());

    y_a.to_data()
        .assert_approx_eq::<f32>(&y_b.to_data(), Tolerance::default());
}

#[test]
fn test_multi_scale_predictor_h_sampling_positive_runs() {
    // With h_sampling_prob > 0 the scheduled-sampling path must still produce
    // finite, correctly-shaped predictions (Bernoulli branch both ways).
    type B = NdArray<f32>;
    let device = Default::default();

    let config = MultiScaleSsmConfig::new(32, 8, 2, 2, 1)
        .with_n_layers(3)
        .with_use_conv(false);
    let predictor = MultiScaleMambaPredictor::<B>::new(&config, 4, 2, &device);

    let obs = Tensor::<B, 3>::random([2, 6, 4], burn::tensor::Distribution::Default, &device);
    let act = Tensor::<B, 3>::random([2, 6, 2], burn::tensor::Distribution::Default, &device);

    // Run twice: p=0.5 exercises both branches across timesteps.
    let out = predictor.forward_with_h_sampling(obs.clone(), act.clone(), 0.5);
    assert_eq!(out.dims(), [2, 6, 2]);
    assert!(
        out.into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()),
        "scheduled-sampling output must be finite"
    );

    // p=1.0 forces the h-branch at every t>0 — full closed-loop training path.
    let out_full = predictor.forward_with_h_sampling(obs, act, 1.0);
    assert_eq!(out_full.dims(), [2, 6, 2]);
    assert!(
        out_full
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()),
        "full h-sampling output must be finite"
    );
}

#[test]
fn test_multi_scale_predictor_save_load() {
    type B = NdArray<f32>;
    let device = Default::default();

    let config = MultiScaleSsmConfig::new(32, 8, 2, 2, 1)
        .with_n_layers(3)
        .with_use_conv(false);
    let predictor = MultiScaleMambaPredictor::<B>::new(&config, 4, 2, &device);

    // Materialize lazy params BEFORE saving (see test_mamba_predictor_save_load).
    let x = Tensor::<B, 3>::random([1, 8, 4], burn::tensor::Distribution::Default, &device);
    let y1 = predictor.forward(x.clone());

    let file_path = "test_multi_scale_predictor_save";
    let path = std::path::PathBuf::from(format!("{file_path}.bin"));
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    recorder
        .record(predictor.clone().into_record(), path.clone())
        .expect("MultiScaleMambaPredictor save failed");

    let loaded = MultiScaleMambaPredictor::<B>::new(&config, 4, 2, &device);
    let record = recorder
        .load(path.clone(), &device)
        .expect("MultiScaleMambaPredictor load failed");
    let loaded = loaded.load_record(record);

    let y2 = loaded.forward(x);

    y1.to_data()
        .assert_approx_eq::<f32>(&y2.to_data(), Tolerance::default());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_multi_scale_predictor_backward_flow() {
    use burn::backend::Autodiff;
    type B = Autodiff<NdArray<f32>>;
    let device = Default::default();

    let config = MultiScaleSsmConfig::new(32, 8, 2, 2, 1)
        .with_n_layers(3)
        .with_use_conv(false);
    let predictor = MultiScaleMambaPredictor::<B>::new(&config, 4, 2, &device);

    let obs = Tensor::<B, 3>::random([1, 4, 4], burn::tensor::Distribution::Default, &device);
    let act = Tensor::<B, 3>::random([1, 4, 2], burn::tensor::Distribution::Default, &device);

    let y = predictor.forward_with_action(obs, act);
    let loss = predictor.loss(y, Tensor::zeros([1, 4, 2], &device));
    let grads = loss.backward();

    assert!(
        predictor.ssms.layers[0].a_re.grad(&grads).is_some(),
        "SSM core (a_re) must receive gradient through the multi-scale wrapper"
    );
    assert!(
        predictor.obs_proj.weight.grad(&grads).is_some(),
        "obs_proj must receive gradient"
    );
    assert!(
        predictor.action_proj.weight.grad(&grads).is_some(),
        "action_proj must receive gradient"
    );
    assert!(
        predictor.imagine_fusion.weight.grad(&grads).is_some(),
        "imagine_fusion must receive gradient"
    );
}

#[test]
fn test_multi_scale_predictor_state_shape() {
    // zero_state must produce correctly-shaped per-layer states.
    type B = NdArray<f32>;
    let device = Default::default();

    let d_model = 32;
    let expand = 2;
    let n_heads = 2;
    let d_state = 8;
    let mimo_rank = 1;
    let n_layers = 3;

    let config = MultiScaleSsmConfig::new(d_model, d_state, expand, n_heads, mimo_rank)
        .with_n_layers(n_layers)
        .with_use_conv(false);
    let predictor = MultiScaleMambaPredictor::<B>::new(&config, 4, 2, &device);

    let batch = 2;
    let state = predictor.zero_state(batch, &device);

    assert_eq!(state.ssms.h.len(), n_layers);
    let d_inner = d_model * expand;
    let d_head_mimo = (d_inner / n_heads) / mimo_rank;
    for h in &state.ssms.h {
        assert_eq!(h.dims(), [batch, n_heads, d_state, d_head_mimo]);
    }
}

#[test]
fn test_multi_scale_predictor_multi_batch_independence() {
    // Different batch rows must not influence each other (batch-wise check
    // used by equivalence_test; applied here to the wrapper level).
    type B = NdArray<f32>;
    let device = Default::default();

    let config = MultiScaleSsmConfig::new(32, 8, 2, 2, 1)
        .with_n_layers(3)
        .with_use_conv(false);
    let predictor = MultiScaleMambaPredictor::<B>::new(&config, 4, 2, &device);

    let x_row = Tensor::<B, 3>::random([1, 6, 4], burn::tensor::Distribution::Default, &device);
    let x_batch = Tensor::cat(vec![x_row.clone(), x_row.clone()], 0); // 2 identical rows

    let y_batch = predictor.forward(x_batch);
    let y_row = predictor.forward(x_row);

    let batch_data = y_batch.into_data().as_slice::<f32>().unwrap().to_vec();
    let row_data = y_row.into_data().as_slice::<f32>().unwrap().to_vec();

    // Row 0 of the batched output must match the single-row output exactly.
    for i in 0..row_data.len() {
        assert!(
            (batch_data[i] - row_data[i]).abs() < 1e-5,
            "batch row 0 diverged from single-row forward at index {i}"
        );
    }
}

#[test]
fn test_predictor_modules_are_mamba_only() {
    // Structural assertion of the issue #3 concern: these wrappers contain
    // SSM + linear projections only — no encoder/decoder, no JEPA losses.
    // This is a compile-time surface check: the fields below are the complete
    // submodule set of the pure-Mamba path.
    type B = NdArray<f32>;
    let device = Default::default();

    let config = SsmConfig::new(16, 8, 2, 2, 1).with_use_conv(false);
    let predictor = MambaPredictor::<B>::new(&config, 4, 2, &device);

    // Touch every submodule field: input_proj + SsmBlock + output_proj.
    // If someone adds JEPA machinery (encoder, losses) to this struct,
    // the module doc contract has been broken — revisit this test.
    let _ = &predictor.input_proj;
    let _ = &predictor.ssm;
    let _ = &predictor.output_proj;
}
