use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use rand::SeedableRng;
use ssm_latent_model::meta::{
    AdaptationStrategy, CircleWorldDistribution, CircleWorldTask, DistributionInfo,
    SineWaveDistribution, SineWaveTask, Task, TaskDistribution, TaskInfo,
};

type Backend = Autodiff<NdArray<f32>>;

#[test]
fn test_circle_world_task_shapes() {
    let task = CircleWorldTask {
        angular_velocity: 1.0,
        noise_level: 0.02,
        phase_shift: 0.0,
        n_support: 16,
        n_query: 64,
        batch_size: 4,
    };

    let (obs, act): (Tensor<Backend, 3>, Tensor<Backend, 3>) = task.support_set();
    assert_eq!(obs.dims(), [4, 16, 2]);
    assert_eq!(act.dims(), [4, 16, 2]);

    let (q_obs, q_act): (Tensor<Backend, 3>, Tensor<Backend, 3>) = task.query_set();
    assert_eq!(q_obs.dims(), [4, 64, 2]);
    assert_eq!(q_act.dims(), [4, 64, 2]);

    assert_eq!(task.obs_dim(), 2);
    assert_eq!(task.action_dim(), 2);
    assert!(task.success_threshold() > 0.0);
}

#[test]
fn test_sine_wave_task_shapes() {
    let task = SineWaveTask {
        frequency: 1.5,
        amplitude: 1.0,
        phase: 0.0,
        noise_level: 0.02,
        n_support: 16,
        n_query: 64,
        batch_size: 4,
    };

    let (obs, act): (Tensor<Backend, 3>, Tensor<Backend, 3>) = task.support_set();
    assert_eq!(obs.dims(), [4, 16, 1]);
    assert_eq!(act.dims(), [4, 16, 1]);

    assert_eq!(task.obs_dim(), 1);
    assert_eq!(task.action_dim(), 1);
}

#[test]
fn test_circle_world_distribution_sampling() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let dist = CircleWorldDistribution::new();
    let task1: Box<dyn ssm_latent_model::meta::Task<Backend>> = dist.sample_task(&mut rng);
    let task2: Box<dyn ssm_latent_model::meta::Task<Backend>> = dist.sample_task(&mut rng);
    assert_ne!(task1.task_id(), task2.task_id());
    assert_eq!(dist.num_task_families(), 1);
}

#[test]
fn test_sine_wave_distribution_sampling() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let dist = SineWaveDistribution::new();
    let task: Box<dyn ssm_latent_model::meta::Task<Backend>> = dist.sample_task(&mut rng);
    let (obs, _act): (Tensor<Backend, 3>, Tensor<Backend, 3>) = task.support_set();
    assert_eq!(obs.dims(), [4, 16, 1]);
}

#[test]
fn test_adaptation_strategy_description() {
    assert_eq!(
        ssm_latent_model::meta::adaptation_strategy_description(&AdaptationStrategy::Full),
        "All parameters"
    );
    assert_eq!(
        ssm_latent_model::meta::adaptation_strategy_description(&AdaptationStrategy::EncoderOnly),
        "Encoder only"
    );
}
