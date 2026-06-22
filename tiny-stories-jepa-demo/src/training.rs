use crate::model::JepaLanguageModel;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, Optimizer};
use burn::prelude::ToElement;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor};
use serde::Deserialize;
use ssm_latent_model::latent::lejepa_loss;
use ssm_latent_model::ssm::SsmConfig;

/// Batcher for story data.
///
/// Converts (text, token_ids) pairs into training batches.
/// Each batch produces (inputs, targets) where targets[t] = inputs[t+1].
pub struct StoryBatcher<B: Backend> {
    device: B::Device,
}

/// A batch of training data.
///
/// - `inputs`: token IDs for the model [batch_size, seq_len]
/// - `targets`: shifted token IDs (next-token prediction target) [batch_size, seq_len]
pub struct StoryBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> StoryBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    /// Create a batch from a list of tokenized stories.
    ///
    /// All sequences are truncated to the shortest length (to avoid padding logic),
    /// and further limited to `max_seq_len` for memory stability.
    pub fn batch_stories(
        &self,
        items: Vec<(String, Vec<u32>)>,
        max_seq_len: usize,
    ) -> StoryBatch<B> {
        let batch_size = items.len();
        let min_len = items.iter().map(|(_, ids)| ids.len()).min().unwrap_or(0);

        if min_len < 2 {
            panic!("Sequences in batch are too short (min_len={})", min_len);
        }

        let seq_len = min_len.min(max_seq_len);
        let mut inputs_flat = Vec::with_capacity(batch_size * seq_len);
        let mut targets_flat = Vec::with_capacity(batch_size * seq_len);

        for (_, ids) in items {
            for i in 0..seq_len {
                // inputs[t] = ids[t], targets[t] = ids[t+1] (last target is 0/pad)
                inputs_flat.push(ids[i] as i32);
                if i + 1 < ids.len() {
                    targets_flat.push(ids[i + 1] as i32);
                } else {
                    targets_flat.push(0); // pad for last position
                }
            }
        }

        let inputs = Tensor::<B, 1, Int>::from_ints(inputs_flat.as_slice(), &self.device)
            .reshape([batch_size, seq_len]);
        let targets = Tensor::<B, 1, Int>::from_ints(targets_flat.as_slice(), &self.device)
            .reshape([batch_size, seq_len]);

        StoryBatch { inputs, targets }
    }
}

/// JEPA-specific configuration (from config.toml [model.jepa] section).
#[derive(Deserialize, Debug, Clone)]
pub struct JepaConfig {
    pub sigreg_weight: f64,
    pub gen_weight: f64,
    pub n_projections: usize,
    pub cf_freqs: Vec<f64>,
}

impl Default for JepaConfig {
    fn default() -> Self {
        Self {
            sigreg_weight: 0.1,
            gen_weight: 1.0,
            n_projections: 16,
            cf_freqs: vec![0.5, 1.0, 1.5, 2.0],
        }
    }
}

/// Model configuration (from config.toml [model] section).
#[derive(Deserialize, Debug, Clone)]
pub struct ModelConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub expand: usize,
    pub n_heads: usize,
    pub mimo_rank: usize,
    pub use_conv: bool,
    pub conv_kernel: usize,
    pub jepa: Option<JepaConfig>,
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

/// Training configuration (from config.toml [training] section).
#[derive(Deserialize, Debug, Clone)]
pub struct TrainingConfig {
    pub num_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub dataset_samples: usize,
    #[serde(skip, default = "default_ssm_config")]
    pub model_config: Option<SsmConfig>,
    #[serde(skip, default = "default_jepa_config")]
    pub jepa_config: Option<JepaConfig>,
    #[serde(skip, default = "AdamConfig::new")]
    pub optimizer: AdamConfig,
}

fn default_ssm_config() -> Option<SsmConfig> {
    None
}

fn default_jepa_config() -> Option<JepaConfig> {
    None
}

impl TrainingConfig {
    #[allow(dead_code)]
    pub fn new(ssm_config: SsmConfig, jepa_config: JepaConfig) -> Self {
        Self {
            model_config: Some(ssm_config),
            jepa_config: Some(jepa_config),
            optimizer: AdamConfig::new(),
            num_epochs: 30,
            batch_size: 4,
            learning_rate: 1e-4,
            dataset_samples: 300,
        }
    }
}

/// Train the JEPA language model.
///
/// The training objective is a **hybrid** of:
/// 1. **JEPA loss** (lejepa_loss): latent prediction MSE + SIGReg
/// 2. **Generation loss** (cross-entropy): next-token prediction
///
/// Total loss = gen_weight * L_gen + sigreg_weight * L_jepa
///
/// This follows the LLM-JEPA formulation (Huang, LeCun, Balestriero 2025):
/// ```text
/// L_total = γ · L_gen + λ · L_jepa
/// ```
/// where L_jepa = MSE(z', z_{t+1}) + SIGReg(z)
pub fn train<B: AutodiffBackend>(
    config: TrainingConfig,
    device: B::Device,
    vocab_size: usize,
    dataset: Vec<(String, Vec<u32>)>,
) -> JepaLanguageModel<B> {
    let ssm_config = config.model_config.expect("Model config (SSM) must be set");
    let jepa_config = config.jepa_config.unwrap_or_default();

    let n_projections = jepa_config.n_projections;
    let sigreg_weight = jepa_config.sigreg_weight;
    let gen_weight = jepa_config.gen_weight;
    let cf_freqs = jepa_config.cf_freqs;

    let mut model = JepaLanguageModel::<B>::new(&ssm_config, vocab_size, n_projections, &device);
    let mut optim = config.optimizer.init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let batcher = StoryBatcher::<B>::new(device.clone());

    let max_seq_len = 64; // Limit sequence length for training stability

    println!("\n========= JEPA Language Model Training =========");
    println!("  Architecture: Embed → Encoder → SSM(×2) → Decoder → Head");
    println!(
        "  d_model={}, d_state={}, n_heads={}",
        ssm_config.d_model, ssm_config.d_state, ssm_config.n_heads
    );
    println!(
        "  JEPA: sigreg_weight={}, gen_weight={}, n_projections={}",
        sigreg_weight, gen_weight, n_projections
    );
    println!("  CF freqs: {:?}", cf_freqs);
    println!("  Optimizer: Adam, lr={}", config.learning_rate);
    println!(
        "  Vocab: o200k_base (byte-level BPE, {} tokens)",
        vocab_size
    );
    println!(
        "  Dataset: {} samples, max_seq_len={}",
        dataset.len(),
        max_seq_len
    );
    println!("================================================\n");

    for epoch in 1..=config.num_epochs {
        let mut total_loss = 0.0;
        let mut total_gen_loss = 0.0;
        let mut total_jepa_loss = 0.0;
        let mut count = 0;

        for chunk in dataset.chunks(config.batch_size) {
            if chunk.len() < config.batch_size {
                continue;
            }

            let batch = batcher.batch_stories(chunk.to_vec(), max_seq_len);

            // Forward: get (z, pred_z, logits)
            let (z, pred_z, logits) = model.forward(batch.inputs);
            let [b, t, v] = logits.dims();

            // --- Generation loss (next-token prediction) ---
            let logits_flat = logits.reshape([b * t, v]);
            let targets_flat = batch.targets.reshape([b * t]);
            let gen_loss = loss_fn.forward(logits_flat, targets_flat);

            // --- JEPA loss (latent prediction + SIGReg) ---
            // This is the core of JEPA: predict in representation space.
            // z.shape: [b, t, d_model], pred_z.shape: [b, t, d_model]
            let projections = model.projections.val(); // frozen random projections
            let jepa_loss = lejepa_loss(z, pred_z, projections, sigreg_weight, &cf_freqs);

            // --- Combined loss ---
            let loss = gen_loss.clone().mul_scalar(gen_weight) + jepa_loss.clone();

            let loss_val: f64 = loss.clone().into_scalar().to_f64();
            let gen_val: f64 = gen_loss.into_scalar().to_f64();
            let jepa_val: f64 = jepa_loss.into_scalar().to_f64();

            total_loss += loss_val;
            total_gen_loss += gen_val;
            total_jepa_loss += jepa_val;
            count += 1;

            // Backprop
            let grads = loss.backward();
            let grads = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optim.step(config.learning_rate, model, grads);
        }

        if count > 0 {
            println!(
                "Epoch {:2}/{} | Total: {:.4} | Gen: {:.4} | JEPA: {:.4} | JEPA/Total: {:.2}%",
                epoch,
                config.num_epochs,
                total_loss / count as f64,
                total_gen_loss / count as f64,
                total_jepa_loss / count as f64,
                total_jepa_loss / total_loss.max(1e-8) * 100.0,
            );
        }
    }

    println!("\n✅ Training complete.");
    model
}
