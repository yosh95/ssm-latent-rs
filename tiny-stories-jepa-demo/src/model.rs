use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, Int};
use ssm_latent_model::ssm::{SsmBlock, SsmConfig};

/// JEPA-style Language Model using SSM dynamics.
/// Instead of just predicting the next token, this model:
/// 1. Encodes tokens into a Latent Space.
/// 2. Uses SSM to predict the "Next Latent State" (Dynamics).
/// 3. Decodes the Latent State back into Vocabulary.
#[derive(Module, Debug)]
pub struct JepaLanguageModel<B: Backend> {
    /// Token embedding layer (Encoder)
    pub embedding: Embedding<B>,
    /// SSM layers for latent dynamics prediction
    pub ssm_layers: Vec<SsmBlock<B>>,
    /// Output head (Decoder)
    pub output_head: Linear<B>,
    pub d_model: usize,
}

impl<B: Backend> JepaLanguageModel<B> {
    pub fn new(config: &SsmConfig, vocab_size: usize, device: &B::Device) -> Self {
        let embedding = EmbeddingConfig::new(vocab_size, config.d_model).init(device);
        
        // Stack multiple SSM layers for better latent representation
        let mut ssm_layers = Vec::new();
        for _ in 0..2 {
            ssm_layers.push(SsmBlock::new(config, device));
        }
        
        let output_head = LinearConfig::new(config.d_model, vocab_size).init(device);

        Self {
            embedding,
            ssm_layers,
            output_head,
            d_model: config.d_model,
        }
    }

    /// Forward pass through the JEPA-style architecture.
    /// z_t = Encoder(x_t)
    /// z_{t+1}_pred = SSM(z_t)
    /// logits = Decoder(z_{t+1}_pred)
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        // [batch, seq_len] -> [batch, seq_len, d_model]
        let mut z = self.embedding.forward(input_ids);
        
        // Latent dynamics prediction via SSM
        for ssm in &self.ssm_layers {
            z = ssm.forward(z);
        }
        
        // Decode latent state to token probabilities
        self.output_head.forward(z)
    }

    /// Autoregressive step for story generation with top-k sampling
    pub fn step(&self, input_ids: Tensor<B, 2, Int>, top_k: usize) -> Tensor<B, 1, Int> {
        let logits = self.forward(input_ids);
        let [batch, seq_len, vocab_size] = logits.dims();
        
        let last_logits = logits.slice([0..batch, (seq_len - 1)..seq_len])
            .reshape([batch, vocab_size]);

        if top_k <= 1 {
            return last_logits.argmax(1).reshape([batch]);
        }

        // Apply Top-K sampling for better text variety and stability
        let (values, indices) = last_logits.topk_with_indices(top_k, 1);
        let probs = burn::tensor::activation::softmax(values, 1);
        
        // Multinomial sampling (simplified for single-item batch)
        let sample_idx = probs.argmax(1).reshape([batch, 1]);
        indices.gather(1, sample_idx).reshape([batch])
    }

    /// Generate a story from a prompt
    pub fn generate_story(
        &self, 
        prompt_ids: Vec<i32>, 
        max_length: usize, 
        device: &B::Device
    ) -> Vec<i32> {
        let mut current_ids = prompt_ids.clone();
        
        for _ in 0..max_length {
            // Limiting context window to prevent performance degradation/hallucination
            let window_size = 128;
            let start = if current_ids.len() > window_size { current_ids.len() - window_size } else { 0 };
            
            let input_tensor = Tensor::<B, 1, Int>::from_ints(&current_ids[start..], device)
                .unsqueeze::<2>();
            
            let next_id_tensor = self.step(input_tensor, 5);
            let next_id = next_id_tensor.into_data().as_slice::<i32>().unwrap()[0];
            
            current_ids.push(next_id);
            
            if next_id == 50256 {
                break;
            }
        }
        current_ids
    }
}
