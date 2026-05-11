use burn::data::dataloader::batcher::Batcher;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, Optimizer};
use burn::prelude::ToElement;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, Int};
use crate::model::JepaLanguageModel;
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
}

impl<B: Backend> Batcher<B, Vec<usize>, StoryBatch<B>> for StoryBatcher<B> {
    fn batch(&self, items: Vec<Vec<usize>>, _device: &B::Device) -> StoryBatch<B> {
        let batch_size = items.len();
        // Align all sequences to the same length (match the minimum length)
        let min_len = items.iter().map(|it| it.len()).min().unwrap_or(0);
        
        if min_len < 2 {
            panic!("Sequences in batch are too short. Minimum length must be 2.");
        }

        let seq_len = min_len;
        let mut inputs_flat = Vec::with_capacity(batch_size * (seq_len - 1));
        let mut targets_flat = Vec::with_capacity(batch_size * (seq_len - 1));

        for seq in items {
            for i in 0..seq_len - 1 {
                inputs_flat.push(seq[i] as i32);
                targets_flat.push(seq[i + 1] as i32);
            }
        }

        let inputs = Tensor::<B, 1, Int>::from_ints(inputs_flat.as_slice(), &self.device)
            .reshape([batch_size, seq_len - 1]);
        let targets = Tensor::<B, 1, Int>::from_ints(targets_flat.as_slice(), &self.device)
            .reshape([batch_size, seq_len - 1]);

        StoryBatch { inputs, targets }
    }
}

pub struct TrainingConfig {
    pub model_config: SsmConfig,
    pub optimizer: AdamConfig,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
}

impl TrainingConfig {
    pub fn new(ssm_config: SsmConfig) -> Self {
        Self {
            model_config: ssm_config,
            optimizer: AdamConfig::new(),
            num_epochs: 5,
            batch_size: 16,
            learning_rate: 1e-4,
        }
    }
}

pub fn train<B: AutodiffBackend>(
    _artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
    vocab_size: usize,
    dataset: Vec<Vec<usize>>,
) -> JepaLanguageModel<B> {
    let mut model = JepaLanguageModel::<B>::new(&config.model_config, vocab_size, &device);
    let mut optim = config.optimizer.init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let batcher = StoryBatcher::<B>::new(device.clone());

    println!("Starting manual training loop...");

    for epoch in 1..=config.num_epochs {
        let mut total_loss = 0.0;
        let mut count = 0;

        for chunk in dataset.chunks(config.batch_size) {
            if chunk.len() < config.batch_size { continue; }
            
            let batch = batcher.batch(chunk.to_vec(), &device);
            
            let logits = model.forward(batch.inputs);
            let targets = batch.targets;

            // logits: [B, T, V], targets: [B, T]
            let [b, t, v] = logits.dims();
            let logits_flat = logits.reshape([b * t, v]);
            let targets_flat = targets.reshape([b * t]);

            let loss = loss_fn.forward(logits_flat, targets_flat);
            
            let loss_val = loss.clone().into_scalar().to_f64();
            if loss_val.is_nan() {
                println!("Warning: Loss is NaN at Epoch {}, Chunk {}", epoch, count);
                continue;
            }
            total_loss += loss_val;
            count += 1;

            let grads = loss.backward();
            let grads = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optim.step(config.learning_rate, model, grads);
        }
        
        if count > 0 {
            println!("Epoch {}/{} - Loss: {:.4}", epoch, config.num_epochs, total_loss / count as f64);
        } else {
            println!("Epoch {}/{} - No valid batches", epoch, config.num_epochs);
        }
    }

    model
}
