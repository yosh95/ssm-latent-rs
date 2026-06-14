use crate::latent::MultiScaleLatentPredictor;
use crate::meta::adapt::{LossWeights, inner_loop};
use crate::meta::task::Task;

use burn::tensor::backend::AutodiffBackend;

/// SAI's core metric: a report on how quickly a model adapts to a task.
///
/// This measures **adaptation speed** — the central quantity in SAI:
/// > "The central quantity is not a checklist of skills, but the speed
/// > and efficiency with which new skills are acquired under realistic
/// > resource constraints."
/// > — Goldfeder et al. (2026)
#[derive(Clone, Debug)]
pub struct AdaptationSpeedReport {
    /// Loss at each adaptation step (on the query set)
    pub adaptation_curve: Vec<f64>,

    /// Loss after N steps (key snapshot metrics)
    pub loss_at_step_1: f64,
    pub loss_at_step_5: f64,
    pub loss_at_step_10: f64,
    pub loss_at_step_20: f64,

    /// Number of steps to reach the task's success threshold
    pub steps_to_success: Option<usize>,

    /// Initial improvement rate: (loss_0 - loss_1) / loss_0
    /// Higher = faster initial adaptation
    pub initial_improvement_rate: f64,

    /// Area Under the Curve of loss-vs-steps.
    /// Lower AUC = faster adaptation (less cumulative error during adaptation)
    pub adaptation_auc: f64,

    /// Final loss after max_steps
    pub final_loss: f64,

    /// Whether adaptation was successful
    pub success: bool,
}

impl AdaptationSpeedReport {
    /// Create a report from a loss curve.
    pub fn from_curve(loss_curve: Vec<f64>, threshold: f64) -> Self {
        let len = loss_curve.len();

        // Find steps to success
        let steps_to_success = loss_curve
            .iter()
            .position(|&l| l < threshold)
            .map(|idx| idx + 1); // 1-indexed

        // Initial improvement rate
        let initial_improvement_rate = if len >= 2 && loss_curve[0] > 0.0 {
            (loss_curve[0] - loss_curve[1]) / loss_curve[0]
        } else {
            0.0
        };

        // AUC (trapezoidal integration)
        let adaptation_auc = if len >= 2 {
            let mut auc = 0.0;
            for i in 1..len {
                let avg = (loss_curve[i - 1] + loss_curve[i]) / 2.0;
                auc += avg; // step width = 1
            }
            auc / len as f64 // normalized by number of steps
        } else {
            loss_curve.first().copied().unwrap_or(0.0)
        };

        Self {
            adaptation_curve: loss_curve.clone(),
            loss_at_step_1: loss_curve.first().copied().unwrap_or(0.0),
            loss_at_step_5: loss_curve.get(4).copied().unwrap_or_default(),
            loss_at_step_10: loss_curve.get(9).copied().unwrap_or_default(),
            loss_at_step_20: loss_curve.get(19).copied().unwrap_or_default(),
            steps_to_success,
            initial_improvement_rate,
            adaptation_auc,
            final_loss: loss_curve.last().copied().unwrap_or(0.0),
            success: steps_to_success.is_some(),
        }
    }
}

/// Evaluate adaptation speed of a model on a task.
///
/// This is the core SAI evaluation function. It measures how quickly
/// the model can adapt to a new task by:
/// 1. Running inner-loop adaptation
/// 2. Recording the loss curve on the query set
/// 3. Computing SAI metrics (steps_to_success, AUC, improvement rate)
pub fn evaluate_adaptation_speed<B>(
    model: &MultiScaleLatentPredictor<B>,
    task: &dyn Task<B>,
    inner_lr: f64,
    max_steps: usize,
    loss_weights: &LossWeights,
    device: &B::Device,
) -> AdaptationSpeedReport
where
    B: AutodiffBackend,
{
    let (_, result) = inner_loop::<B>(
        model,
        task,
        inner_lr,
        max_steps,
        &crate::meta::adapt::AdaptationStrategy::Full,
        loss_weights,
        device,
    );

    AdaptationSpeedReport::from_curve(result.loss_curve, task.success_threshold())
}

/// Compare adaptation speed across multiple training methods.
#[derive(Clone, Debug)]
pub struct AdaptationComparison {
    pub task_name: String,
    pub from_scratch: Option<AdaptationSpeedReport>,
    pub pre_trained: Option<AdaptationSpeedReport>,
    pub meta_learned: Option<AdaptationSpeedReport>,
}

impl AdaptationComparison {
    pub fn new(task_name: &str) -> Self {
        Self {
            task_name: task_name.to_string(),
            from_scratch: None,
            pre_trained: None,
            meta_learned: None,
        }
    }

    pub fn print_summary(&self) {
        println!("── Adaptation Comparison: {} ──", self.task_name);
        println!(
            "{:<20} | {:>12} | {:>12} | {:>12}",
            "Method", "Steps→Success", "AUC", "Init Rate"
        );
        println!("{:-<20}-+-{:-<12}-+-{:-<12}-+-{:-<12}", "", "", "", "");

        if let Some(ref r) = self.from_scratch {
            println!(
                "{:<20} | {:>12?} | {:>12.4} | {:>12.4}",
                "From Scratch", r.steps_to_success, r.adaptation_auc, r.initial_improvement_rate
            );
        }
        if let Some(ref r) = self.pre_trained {
            println!(
                "{:<20} | {:>12?} | {:>12.4} | {:>12.4}",
                "Pre-trained", r.steps_to_success, r.adaptation_auc, r.initial_improvement_rate
            );
        }
        if let Some(ref r) = self.meta_learned {
            println!(
                "{:<20} | {:>12?} | {:>12.4} | {:>12.4}",
                "Meta-Learned 🏆", r.steps_to_success, r.adaptation_auc, r.initial_improvement_rate
            );
        }
    }
}
