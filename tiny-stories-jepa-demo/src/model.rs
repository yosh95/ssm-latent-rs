use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use ssm_latent_model::ssm::{SsmBlock, SsmConfig};

/// JEPA-style Language Model using SSM dynamics.
/// This model learns to predict the next token's representation in latent space.
#[derive(Module, Debug)]
pub struct JepaLanguageModel<B: Backend> {
    /// Token embedding layer
    pub embedding: Embedding<B>,
    /// Projection layer to map embeddings to SSM latent space
    pub input_projection: Linear<B>,
    /// SSM layers for latent dynamics prediction
    pub ssm_layers: Vec<SsmBlock<B>>,
    /// Output head (Decoder)
    pub output_head: Linear<B>,
    pub d_model: usize,
}

impl<B: Backend> JepaLanguageModel<B> {
    pub fn new(config: &SsmConfig, vocab_size: usize, device: &B::Device) -> Self {
        let embedding = EmbeddingConfig::new(vocab_size, config.d_model).init(device);
        let input_projection = LinearConfig::new(config.d_model, config.d_model).init(device);

        // Stack multiple SSM layers for better latent representation
        let mut ssm_layers = Vec::new();
        for _ in 0..2 {
            ssm_layers.push(SsmBlock::new(config, device));
        }

        let output_head = LinearConfig::new(config.d_model, vocab_size).init(device);

        Self {
            embedding,
            input_projection,
            ssm_layers,
            output_head,
            d_model: config.d_model,
        }
    }

    /// Forward pass
    /// x = Embedding(tokens)
    /// z = Projection(x)
    /// z_next = SSM(z)
    /// logits = Decoder(z_next)
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let x = self.embedding.forward(input_ids);
        let mut z = self.input_projection.forward(x);

        for ssm in &self.ssm_layers {
            z = ssm.forward(z);
        }

        self.output_head.forward(z)
    }

    /// Autoregressive step for story generation
    pub fn step(&self, input_ids: Tensor<B, 2, Int>, top_k: usize) -> Tensor<B, 1, Int> {
        let logits = self.forward(input_ids);
        let [batch, seq_len, vocab_size] = logits.dims();

        let last_logits = logits
            .slice([0..batch, (seq_len - 1)..seq_len])
            .reshape([batch, vocab_size]);

        if top_k <= 1 {
            return last_logits.argmax(1).reshape([batch]);
        }

        let (values, indices) = last_logits.topk_with_indices(top_k, 1);
        let probs = burn::tensor::activation::softmax(values, 1);

        let sample_idx = probs.argmax(1).reshape([batch, 1]);
        indices.gather(1, sample_idx).reshape([batch])
    }
}
