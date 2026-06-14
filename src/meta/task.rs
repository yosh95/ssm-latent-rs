use burn::tensor::{Tensor, backend::AutodiffBackend};
use rand::rngs::StdRng;
use std::fmt::Debug;

/// Static information about a task (not backend-dependent).
pub trait TaskInfo: Debug + Send {
    fn task_id(&self) -> String;
    fn obs_dim(&self) -> usize;
    fn action_dim(&self) -> usize;
    fn success_threshold(&self) -> f64;
}

/// A task defines a specific problem distribution for meta-learning.
pub trait Task<B: AutodiffBackend>: TaskInfo {
    fn support_set(&self) -> (Tensor<B, 3>, Tensor<B, 3>);
    fn query_set(&self) -> (Tensor<B, 3>, Tensor<B, 3>);
}

/// Static information about a task distribution (not backend-dependent).
pub trait DistributionInfo: Debug + Send {
    fn num_task_families(&self) -> usize;
    fn description(&self) -> String;
}

/// A distribution over tasks for meta-training (backend-dependent for sampling).
pub trait TaskDistribution<B: AutodiffBackend>: DistributionInfo {
    fn sample_task(&self, rng: &mut StdRng) -> Box<dyn Task<B>>;
}

// Blanket impls
impl<T: TaskInfo + ?Sized> TaskInfo for Box<T> {
    fn task_id(&self) -> String {
        (**self).task_id()
    }
    fn obs_dim(&self) -> usize {
        (**self).obs_dim()
    }
    fn action_dim(&self) -> usize {
        (**self).action_dim()
    }
    fn success_threshold(&self) -> f64 {
        (**self).success_threshold()
    }
}
