use crate::model::JepaLanguageModel;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, Optimizer};
use burn::prelude::ToElement;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor};
use serde::Deserialize;
use ssm_latent_model::ssm::SsmConfig;

pub struct StoryBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct StoryBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> StoryBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub fn batch_stories(&self, items: Vec<(String, Vec<usize>)>) -> StoryBatch<B> {
        let batch_size = items.len();
        let min_len = items.iter().map(|(_, ids)| ids.len()).min().unwrap_or(0);

        if min_len < 2 {
            panic!("Sequences in batch are too short.");
        }

        let seq_len = min_len.min(64); // Limit sequence length for stability
        let mut inputs_flat = Vec::with_capacity(batch_size * (seq_len - 1));
        let mut targets_flat = Vec::with_capacity(batch_size * (seq_len - 1));

        for (_, ids) in items {
            for i in 0..seq_len - 1 {
                inputs_flat.push(ids[i] as i32);
                targets_flat.push(ids[i + 1] as i32);
            }
        }

        let inputs = Tensor::<B, 1, Int>::from_ints(inputs_flat.as_slice(), &self.device)
            .reshape([batch_size, seq_len - 1]);
        let targets = Tensor::<B, 1, Int>::from_ints(targets_flat.as_slice(), &self.device)
            .reshape([batch_size, seq_len - 1]);

        StoryBatch { inputs, targets }
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct ModelConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub expand: usize,
    pub n_heads: usize,
    pub mimo_rank: usize,
    pub use_conv: bool,
    pub conv_kernel: usize,
}

impl From<ModelConfig> for SsmConfig {
    fn from(config: ModelConfig) -> Self {
        Self {
            d_model: config.d_model,
            d_state: config.d_state,
            expand: config.expand,
            n_heads: config.n_heads,
            mimo_rank: config.mimo_rank,
            use_conv: config.use_conv,
            conv_kernel: config.conv_kernel,
        }
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct TrainingConfig {
    pub num_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub dataset_samples: usize,
    #[serde(skip, default = "default_ssm_config")]
    pub model_config: Option<SsmConfig>,
    #[serde(skip, default = "AdamConfig::new")]
    pub optimizer: AdamConfig,
}

fn default_ssm_config() -> Option<SsmConfig> {
    None
}

impl TrainingConfig {
    #[allow(dead_code)]
    pub fn new(ssm_config: SsmConfig) -> Self {
        Self {
            model_config: Some(ssm_config),
            optimizer: AdamConfig::new(),
            num_epochs: 30,
            batch_size: 8,
            learning_rate: 1e-4,
            dataset_samples: 300,
        }
    }
}

pub fn train<B: AutodiffBackend>(
    _artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
    vocab_size: usize,
    dataset: Vec<(String, Vec<usize>)>,
) -> JepaLanguageModel<B> {
    let model_config = config.model_config.expect("Model config must be set");
    let mut model = JepaLanguageModel::<B>::new(&model_config, vocab_size, &device);
    let mut optim = config.optimizer.init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let batcher = StoryBatcher::<B>::new(device.clone());

    println!("Starting manual training loop with Burn Embeddings...");

    for epoch in 1..=config.num_epochs {
        let mut total_loss = 0.0;
        let mut count = 0;

        for chunk in dataset.chunks(config.batch_size) {
            if chunk.len() < config.batch_size {
                continue;
            }

            let batch = batcher.batch_stories(chunk.to_vec());

            let logits = model.forward(batch.inputs);
            let [b, t, v] = logits.dims();

            let logits_flat = logits.reshape([b * t, v]);
            let targets_flat = batch.targets.reshape([b * t]);

            let loss = loss_fn.forward(logits_flat, targets_flat);

            total_loss += loss.clone().into_scalar().to_f64();
            count += 1;

            let grads = loss.backward();
            let grads = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optim.step(config.learning_rate, model, grads);
        }

        if count > 0 {
            println!(
                "Epoch {}/{} - Loss: {:.4}",
                epoch,
                config.num_epochs,
                total_loss / count as f64
            );
        }
    }

    model
}
