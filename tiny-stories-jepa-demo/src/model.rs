use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor};
use ssm_latent_model::preprocess::normalize_projections;
use ssm_latent_model::ssm::{SsmBlock, SsmConfig};

/// JEPA-style Language Model using SSM dynamics in latent space.
///
/// # Architecture
///
/// ```text
/// tokens → Embedding → Encoder(z) → SSM(z→z') → Decoder(z'→logits) → next token
///                                     ↓
///                              lejepa_loss(z, z', projections)
///                              + cross_entropy(logits, targets)
/// ```
///
/// The key insight is that **prediction happens in latent space** (z), not token space.
/// The decoder converts latents to logits for generation, but the SSM learns
/// to predict the *representation* of the next token, not the token itself.
/// This follows LeCun's JEPA philosophy: predict in representation space,
/// then (optionally) decode for generation.
///
/// # Why this matters
///
/// Standard LLMs predict:  P(token_t | tokens_{<t})  ← token space
/// JEPA Language Model:    z_t = Encoder(tokens_{<t})   ← latent space
///                         z'_t = SSM(z_t)              ← latent prediction
///                         P(token_t | z'_t)            ← generation (decoder)
///
/// The JEPA loss (latent MSE + SIGReg) ensures the latent space captures
/// **semantic structure** rather than surface statistics — the model must
/// understand what the text *means*, not just what word comes next.
#[derive(Module, Debug)]
pub struct JepaLanguageModel<B: Backend> {
    /// Token embedding layer: token IDs → d_model
    pub embedding: Embedding<B>,
    /// Encoder: maps embedding to latent space z
    /// (can be deeper; single Linear is sufficient for initial experiments)
    pub encoder: Linear<B>,
    /// SSM layers for latent dynamics prediction (z → z')
    pub ssm_layers: Vec<SsmBlock<B>>,
    /// Decoder: maps latent z back to vocabulary logits for generation
    pub decoder: Linear<B>,
    /// Output head: logits → vocabulary distribution (optional; decoder can do this)
    pub output_head: Linear<B>,
    /// Random projection matrix for SIGReg (collapse prevention)
    /// Shape: [d_model, n_projections]
    /// Frozen — not trained, only used for loss computation
    pub projections: burn::module::Param<Tensor<B, 2>>,
    /// Model dimension
    pub d_model: usize,
    /// Number of random projections for SIGReg
    pub n_projections: usize,
}

impl<B: Backend> JepaLanguageModel<B> {
    /// Create a new JEPA language model.
    ///
    /// # Arguments
    /// * `config` - SSM configuration (d_model, d_state, etc.)
    /// * `vocab_size` - Size of the token vocabulary
    /// * `n_projections` - Number of random projections for SIGReg loss
    /// * `device` - Device to place parameters on
    pub fn new(
        config: &SsmConfig,
        vocab_size: usize,
        n_projections: usize,
        device: &B::Device,
    ) -> Self {
        let embedding = EmbeddingConfig::new(vocab_size, config.d_model).init(device);
        // Encoder: d_model → d_model (could be deeper, but single Linear is
        // sufficient to demonstrate the JEPA principle)
        let encoder = LinearConfig::new(config.d_model, config.d_model).init(device);
        // SSM layers for latent dynamics (2 layers for depth without overfitting)
        let mut ssm_layers = Vec::new();
        for _ in 0..2 {
            ssm_layers.push(SsmBlock::new(config, device));
        }
        // Decoder: latent z → d_model → logits
        let decoder = LinearConfig::new(config.d_model, config.d_model).init(device);
        let output_head = LinearConfig::new(config.d_model, vocab_size).init(device);

        // Random projection matrix for SIGReg (LeJEPA collapse prevention).
        // These projections are normalized to unit length and DETACHED from
        // gradients — they serve as fixed probes, not learnable parameters.
        // The Param wrapper is only for serialization, not gradient tracking.
        let raw_projections = Tensor::<B, 2>::random(
            [config.d_model, n_projections],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        let projections = normalize_projections(raw_projections).detach();

        Self {
            embedding,
            encoder,
            ssm_layers,
            decoder,
            output_head,
            projections: burn::module::Param::from_tensor(projections),
            d_model: config.d_model,
            n_projections,
        }
    }

    /// Full forward pass: tokens → (z, pred_z, logits)
    ///
    /// Returns a tuple of:
    /// - `z` — current latent representations `[batch, seq_len, d_model]`
    /// - `pred_z` — predicted next latent states `[batch, seq_len, d_model]`
    /// - `logits` — vocabulary logits for generation `[batch, seq_len, vocab_size]`
    ///
    /// The caller can then compute:
    /// - JEPA loss from `(z, pred_z)`: `lejepa_loss(z, pred_z, projections, weight, freqs)`
    /// - Generation loss from `(logits, targets)`: `cross_entropy(logits, targets)`
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        // [batch, seq_len] → [batch, seq_len, d_model]
        let x = self.embedding.forward(input_ids);

        // Encode to latent space: embedding → z
        // This is the "joint embedding" step — observations become latents
        let z = self.encoder.forward(x);

        // SSM dynamics: predict z' from z
        // This is the "predictive" step — latents predict future latents
        let mut pred_z = z.clone();
        for ssm in &self.ssm_layers {
            pred_z = ssm.forward(pred_z);
        }

        // Decode: z' → logits (for generation)
        // The decoder is an auxiliary path — it allows the model to generate
        // text, but the SSM learns entirely in latent space.
        let decoded = self.decoder.forward(pred_z.clone());
        let logits = self.output_head.forward(decoded);

        (z, pred_z, logits)
    }

    /// Autoregressive generation step.
    ///
    /// Given input_ids (context), predicts the next token.
    /// Uses top-k sampling to balance diversity and quality.
    pub fn step(&self, input_ids: Tensor<B, 2, Int>, top_k: usize) -> Tensor<B, 1, Int> {
        let (_z, _pred_z, logits) = self.forward(input_ids);
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
