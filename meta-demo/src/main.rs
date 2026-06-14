//! # Meta-Learning Demo for Superhuman Adaptable Intelligence (SAI)
//!
//! This demo demonstrates the complete SAI meta-learning pipeline:
//! 1. Create a MultiScaleLatentPredictor (SSM × JEPA world model)
//! 2. Define a distribution of CircleWorld tasks with varying angular velocities
//! 3. Run Reptile meta-training to optimize for adaptation speed
//! 4. Evaluate adaptation speed on holdout (unseen) tasks
//!
//! Reference: Goldfeder, Wyder, LeCun & Shwartz-Ziv (2026)
//! "AI Must Embrace Specialization via Superhuman Adaptable Intelligence"

use burn::backend::Autodiff;
use burn::backend::NdArray;
use ssm_latent_model::latent::MultiScaleLatentPredictor;
use ssm_latent_model::meta::{
    CircleWorldDistribution, CircleWorldTask, DistributionInfo, MetaPipelineConfig,
    run_meta_pipeline,
};
use ssm_latent_model::ssm::MultiScaleSsmConfig;

type Backend = Autodiff<NdArray<f32>>;

fn main() {
    let device = Default::default();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   SAI: Superhuman Adaptable Intelligence Meta-Demo       ║");
    println!("║   Goldfeder, Wyder, LeCun & Shwartz-Ziv (2026)           ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // ── Step 1: Create the world model ──
    println!("=== Step 1: Creating World Model ===");
    let obs_dim = 2;
    let action_dim = 2;
    let ssm_config = MultiScaleSsmConfig::new(32, 8, 2, 2, 1)
        .with_n_layers(3)
        .with_use_conv(true)
        .with_conv_kernel(4);

    let model =
        MultiScaleLatentPredictor::<Backend>::new(&ssm_config, obs_dim, action_dim, &device);

    println!(
        "  Model: d_model={}, layers={}, params=~{}",
        ssm_config.d_model, ssm_config.n_layers, 32000
    );
    println!();

    // ── Step 2: Define task distributions ──
    println!("=== Step 2: Defining Task Distributions ===");

    // Training distribution: CircleWorld with ω ∈ [0.5, 2.0]
    let train_dist = Box::new(CircleWorldDistribution {
        min_omega: 0.5,
        max_omega: 2.0,
        noise_level: 0.02,
        n_support: 16,
        n_query: 64,
        batch_size: 4,
    });
    println!("  Train dist: {}", train_dist.description());

    // Holdout evaluation tasks: unseen angular velocities
    let eval_tasks: Vec<Box<dyn ssm_latent_model::meta::Task<Backend>>> = vec![
        Box::new(CircleWorldTask {
            angular_velocity: 0.3, // unseen: slower than training range
            noise_level: 0.02,
            phase_shift: 0.0,
            n_support: 16,
            n_query: 64,
            batch_size: 4,
        }),
        Box::new(CircleWorldTask {
            angular_velocity: 2.5, // unseen: faster than training range
            noise_level: 0.02,
            phase_shift: 1.5,
            n_support: 16,
            n_query: 64,
            batch_size: 4,
        }),
        Box::new(CircleWorldTask {
            angular_velocity: 1.2, // cross-task: different phase
            noise_level: 0.05,
            phase_shift: 3.0,
            n_support: 16,
            n_query: 64,
            batch_size: 4,
        }),
    ];
    println!("  Eval tasks: {} holdout tasks", eval_tasks.len());
    println!();

    // ── Step 3: Run meta-training ──
    println!("=== Step 3: Meta-Training (Reptile) ===");
    println!("  Training for 30 iterations...");
    println!();

    let pipeline_cfg = MetaPipelineConfig {
        num_iterations: 30,
        inner_lr: 1e-3,
        inner_steps: 5,
        outer_lr: 0.1,
        max_eval_steps: 20,
    };

    let (_meta_learner, train_metrics, eval_reports) =
        run_meta_pipeline::<Backend>(model, train_dist, eval_tasks, pipeline_cfg, device);

    println!();
    println!("=== Step 4: Results Summary ===");
    println!();

    if let Some(last) = train_metrics.last() {
        println!("  Final meta-train metrics:");
        println!("    Avg improvement: {:.6}", last.avg_improvement);
        println!("    Avg final loss: {:.6}", last.avg_final_loss);
        println!("    Success rate: {:.2}%", last.success_rate * 100.0);
        println!(
            "    Avg adaptation time: {:.2}ms",
            last.avg_adaptation_time_ms
        );
    }

    println!();
    println!("  Holdout task evaluation:");
    for (i, report) in eval_reports.iter().enumerate() {
        println!(
            "    Task {}: steps_to_success={:?}, AUC={:.4}, success={}",
            i + 1,
            report.steps_to_success,
            report.adaptation_auc,
            report.success,
        );
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   Meta-learning complete!                              ║");
    println!("║   The model is now optimized for adaptation speed.     ║");
    println!("╚══════════════════════════════════════════════════════════╝");
}
