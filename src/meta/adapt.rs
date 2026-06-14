use crate::latent::{LatentLossArgs, MultiScaleLatentPredictor};
use crate::meta::task::Task;

use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, backend::AutodiffBackend};
use std::time::Instant;

/// Strategy for selecting which parameters to update during adaptation.
#[derive(Clone, Debug, PartialEq)]
pub enum AdaptationStrategy {
    EncoderOnly,
    DecoderOnly,
    SlowSSMOnly,
    Full,
    EncoderPlusSlowSSM,
}

/// Result of an adaptation process.
#[derive(Clone, Debug)]
pub struct AdaptationResult {
    pub initial_loss: f64,
    pub final_loss: f64,
    pub loss_curve: Vec<f64>,
    pub steps_taken: usize,
    pub success: bool,
    pub adaptation_time_ms: f64,
}

impl Default for AdaptationResult {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptationResult {
    pub fn new() -> Self {
        Self {
            initial_loss: 0.0,
            final_loss: 0.0,
            loss_curve: Vec::new(),
            steps_taken: 0,
            success: false,
            adaptation_time_ms: 0.0,
        }
    }
}

/// Hyperparameters for the JEPA loss function.
#[derive(Clone, Debug)]
pub struct LossWeights {
    pub recon: f64,
    pub stability: f64,
    pub curvature: f64,
}

impl Default for LossWeights {
    fn default() -> Self {
        Self {
            recon: 1.0,
            stability: 0.01,
            curvature: 0.005,
        }
    }
}

/// Compute the JEPA latent prediction loss for a model on given data.
///
/// This loss drives the inner-loop adaptation: minimizing prediction error
/// on the support set causes the model to specialize to the task dynamics.
pub fn compute_adaptation_loss<B: AutodiffBackend>(
    model: &MultiScaleLatentPredictor<B>,
    observations: &Tensor<B, 3>,
    actions: &Tensor<B, 3>,
    weights: &LossWeights,
    stability_projections: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let (z, predicted_z, reconstructed_x, predicted_x) = if model.has_action {
        model.forward(observations.clone(), actions.clone())
    } else {
        model.forward_no_action(observations.clone())
    };

    let args = LatentLossArgs {
        z,
        pred_z: predicted_z,
        reconstructed_x,
        predicted_x,
        original_x: observations.clone(),
        stability_weight: weights.stability,
        curvature_weight: weights.curvature,
        recon_weight: weights.recon,
    };

    crate::latent::latent_loss(args, stability_projections)
}

/// Inner loop: adapt the model to a specific task using K steps of gradient descent.
///
/// This is the core of meta-learning — the model quickly specializes to a new task.
/// The optimizer is created once (using `AdamConfig::new().init()`) and reused
/// across steps for consistent optimizer state.
pub fn inner_loop<B>(
    model: &MultiScaleLatentPredictor<B>,
    task: &dyn Task<B>,
    inner_lr: f64,
    inner_steps: usize,
    _strategy: &AdaptationStrategy,
    loss_weights: &LossWeights,
    _device: &B::Device,
) -> (MultiScaleLatentPredictor<B>, AdaptationResult)
where
    B: AutodiffBackend,
{
    let start = Instant::now();
    let mut adapted = model.clone();

    // Get task data
    let (support_obs, support_act) = task.support_set();
    let (query_obs, query_act) = task.query_set();

    let stability_proj = adapted.stability_projections.val().clone();

    // Compute initial loss
    let initial_loss = compute_adaptation_loss(
        &adapted,
        &support_obs,
        &support_act,
        loss_weights,
        stability_proj.clone(),
    );
    let initial_loss_val = initial_loss.into_data().as_slice::<f32>().unwrap()[0] as f64;

    let mut loss_curve = Vec::with_capacity(inner_steps);

    // Create optimizer once, reuse across steps
    let mut optim = AdamConfig::new().init();

    for _step in 0..inner_steps {
        let loss = compute_adaptation_loss(
            &adapted,
            &support_obs,
            &support_act,
            loss_weights,
            stability_proj.clone(),
        );

        let loss_val = loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;
        loss_curve.push(loss_val);

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &adapted);
        adapted = optim.step(inner_lr, adapted, grads);
    }

    // Compute final loss on query set
    let final_loss = compute_adaptation_loss(
        &adapted,
        &query_obs,
        &query_act,
        loss_weights,
        stability_proj,
    );
    let final_loss_val = final_loss.into_data().as_slice::<f32>().unwrap()[0] as f64;

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    (
        adapted,
        AdaptationResult {
            initial_loss: initial_loss_val,
            final_loss: final_loss_val,
            loss_curve,
            steps_taken: inner_steps,
            success: final_loss_val < task.success_threshold(),
            adaptation_time_ms: elapsed,
        },
    )
}

/// Description of an adaptation strategy for display.
pub fn adaptation_strategy_description(strategy: &AdaptationStrategy) -> &'static str {
    match strategy {
        AdaptationStrategy::EncoderOnly => "Encoder only",
        AdaptationStrategy::DecoderOnly => "Decoder only",
        AdaptationStrategy::SlowSSMOnly => "Slow SSM layer only",
        AdaptationStrategy::Full => "All parameters",
        AdaptationStrategy::EncoderPlusSlowSSM => "Encoder + Slow SSM",
    }
}
