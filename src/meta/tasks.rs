use crate::meta::task::{DistributionInfo, Task, TaskDistribution, TaskInfo};
use burn::tensor::{Tensor, TensorData, backend::AutodiffBackend};
use rand::{RngExt, rngs::StdRng};

// ─── CircleWorld Task ────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct CircleWorldTask {
    pub angular_velocity: f32,
    pub noise_level: f32,
    pub phase_shift: f32,
    pub n_support: usize,
    pub n_query: usize,
    pub batch_size: usize,
}

impl CircleWorldTask {
    pub fn generate_data<B: AutodiffBackend>(
        &self,
        omega: f32,
        n_steps: usize,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let mut obs_data = Vec::with_capacity(self.batch_size * n_steps * 2);
        let mut act_data = Vec::with_capacity(self.batch_size * n_steps * 2);
        for b in 0..self.batch_size {
            let phase = self.phase_shift + (b as f32) * 1.25;
            for t in 0..n_steps {
                let time = (t as f32) * 0.3;
                let angle = omega * time + phase;
                obs_data.push(angle.cos());
                obs_data.push(angle.sin());
                act_data.push(-omega * 0.3 * angle.sin());
                act_data.push(omega * 0.3 * angle.cos());
            }
        }
        let obs = Tensor::<B, 3>::from_data(
            TensorData::new(obs_data, [self.batch_size, n_steps, 2]),
            device,
        );
        let act = Tensor::<B, 3>::from_data(
            TensorData::new(act_data, [self.batch_size, n_steps, 2]),
            device,
        );
        (obs, act)
    }
}

impl TaskInfo for CircleWorldTask {
    fn task_id(&self) -> String {
        format!("CircleWorld(ω={:.3})", self.angular_velocity)
    }
    fn obs_dim(&self) -> usize {
        2
    }
    fn action_dim(&self) -> usize {
        2
    }
    fn success_threshold(&self) -> f64 {
        0.05
    }
}

impl<B: AutodiffBackend> Task<B> for CircleWorldTask {
    fn support_set(&self) -> (Tensor<B, 3>, Tensor<B, 3>) {
        self.generate_data(self.angular_velocity, self.n_support, &Default::default())
    }
    fn query_set(&self) -> (Tensor<B, 3>, Tensor<B, 3>) {
        self.generate_data(self.angular_velocity, self.n_query, &Default::default())
    }
}

// ─── CircleWorld Distribution ────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct CircleWorldDistribution {
    pub min_omega: f32,
    pub max_omega: f32,
    pub noise_level: f32,
    pub n_support: usize,
    pub n_query: usize,
    pub batch_size: usize,
}

impl Default for CircleWorldDistribution {
    fn default() -> Self {
        Self::new()
    }
}

impl CircleWorldDistribution {
    pub fn new() -> Self {
        Self {
            min_omega: 0.5,
            max_omega: 2.0,
            noise_level: 0.02,
            n_support: 16,
            n_query: 64,
            batch_size: 4,
        }
    }
}

impl DistributionInfo for CircleWorldDistribution {
    fn num_task_families(&self) -> usize {
        1
    }
    fn description(&self) -> String {
        format!(
            "CircleWorld(ω=[{:.1}, {:.1}])",
            self.min_omega, self.max_omega
        )
    }
}

impl<B: AutodiffBackend> TaskDistribution<B> for CircleWorldDistribution {
    fn sample_task(&self, rng: &mut StdRng) -> Box<dyn Task<B>> {
        Box::new(CircleWorldTask {
            angular_velocity: rng.random_range(self.min_omega..self.max_omega),
            noise_level: self.noise_level,
            phase_shift: rng.random_range(0.0..std::f32::consts::TAU),
            n_support: self.n_support,
            n_query: self.n_query,
            batch_size: self.batch_size,
        })
    }
}

// ─── SineWave Task ───────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct SineWaveTask {
    pub frequency: f32,
    pub amplitude: f32,
    pub phase: f32,
    pub noise_level: f32,
    pub n_support: usize,
    pub n_query: usize,
    pub batch_size: usize,
}

impl SineWaveTask {
    pub fn generate_data<B: AutodiffBackend>(
        &self,
        freq: f32,
        n_steps: usize,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let mut obs_data = Vec::with_capacity(self.batch_size * n_steps);
        let mut act_data = Vec::with_capacity(self.batch_size * n_steps);
        for b in 0..self.batch_size {
            let phase = self.phase + (b as f32) * 0.5;
            for t in 0..n_steps {
                let time = (t as f32) * 0.1;
                obs_data.push(self.amplitude * (freq * time + phase).sin());
                act_data.push(self.amplitude * freq * (freq * time + phase).cos());
            }
        }
        let obs = Tensor::<B, 3>::from_data(
            TensorData::new(obs_data, [self.batch_size, n_steps, 1]),
            device,
        );
        let act = Tensor::<B, 3>::from_data(
            TensorData::new(act_data, [self.batch_size, n_steps, 1]),
            device,
        );
        (obs, act)
    }
}

impl TaskInfo for SineWaveTask {
    fn task_id(&self) -> String {
        format!("SineWave(f={:.2})", self.frequency)
    }
    fn obs_dim(&self) -> usize {
        1
    }
    fn action_dim(&self) -> usize {
        1
    }
    fn success_threshold(&self) -> f64 {
        0.05
    }
}

impl<B: AutodiffBackend> Task<B> for SineWaveTask {
    fn support_set(&self) -> (Tensor<B, 3>, Tensor<B, 3>) {
        self.generate_data(self.frequency, self.n_support, &Default::default())
    }
    fn query_set(&self) -> (Tensor<B, 3>, Tensor<B, 3>) {
        self.generate_data(self.frequency, self.n_query, &Default::default())
    }
}

// ─── SineWave Distribution ───────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct SineWaveDistribution {
    pub min_freq: f32,
    pub max_freq: f32,
    pub min_amplitude: f32,
    pub max_amplitude: f32,
    pub noise_level: f32,
    pub n_support: usize,
    pub n_query: usize,
    pub batch_size: usize,
}

impl Default for SineWaveDistribution {
    fn default() -> Self {
        Self::new()
    }
}

impl SineWaveDistribution {
    pub fn new() -> Self {
        Self {
            min_freq: 0.5,
            max_freq: 3.0,
            min_amplitude: 0.5,
            max_amplitude: 2.0,
            noise_level: 0.02,
            n_support: 16,
            n_query: 64,
            batch_size: 4,
        }
    }
}

impl DistributionInfo for SineWaveDistribution {
    fn num_task_families(&self) -> usize {
        1
    }
    fn description(&self) -> String {
        format!("SineWave(f=[{:.1}, {:.1}])", self.min_freq, self.max_freq)
    }
}

impl<B: AutodiffBackend> TaskDistribution<B> for SineWaveDistribution {
    fn sample_task(&self, rng: &mut StdRng) -> Box<dyn Task<B>> {
        Box::new(SineWaveTask {
            frequency: rng.random_range(self.min_freq..self.max_freq),
            amplitude: rng.random_range(self.min_amplitude..self.max_amplitude),
            phase: rng.random_range(0.0..std::f32::consts::TAU),
            noise_level: self.noise_level,
            n_support: self.n_support,
            n_query: self.n_query,
            batch_size: self.batch_size,
        })
    }
}

// ─── Composite Distribution ──────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct CompositeDistribution<B: AutodiffBackend> {
    pub circle_dist: CircleWorldDistribution,
    pub sine_dist: SineWaveDistribution,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> Default for CompositeDistribution<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: AutodiffBackend> CompositeDistribution<B> {
    pub fn new() -> Self {
        Self {
            circle_dist: CircleWorldDistribution::new(),
            sine_dist: SineWaveDistribution::new(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: AutodiffBackend> DistributionInfo for CompositeDistribution<B> {
    fn num_task_families(&self) -> usize {
        2
    }
    fn description(&self) -> String {
        format!(
            "Composite[{} + {}]",
            self.circle_dist.description(),
            self.sine_dist.description()
        )
    }
}

impl<B: AutodiffBackend> TaskDistribution<B> for CompositeDistribution<B> {
    fn sample_task(&self, rng: &mut StdRng) -> Box<dyn Task<B>> {
        if rng.random_bool(0.5) {
            self.circle_dist.sample_task(rng)
        } else {
            self.sine_dist.sample_task(rng)
        }
    }
}
