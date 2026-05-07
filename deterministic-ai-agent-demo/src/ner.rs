use crate::encoder::EmbeddingEncoder;
use crate::model::MasterMatcher;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use std::collections::HashMap;

pub struct NERExtractor<B: Backend> {
    matcher: MasterMatcher<B>,
}

impl<B: Backend> NERExtractor<B> {
    pub fn new(_dim: usize, threshold: f32) -> anyhow::Result<Self> {
        Ok(Self {
            matcher: MasterMatcher::new(threshold),
        })
    }

    pub fn register_device(
        &mut self,
        name: &str,
        encoder: &EmbeddingEncoder<B>,
    ) -> anyhow::Result<()> {
        let emb = encoder.encode(name)?;
        self.matcher.add_template(name, emb);
        Ok(())
    }

    pub fn extract(
        &self,
        text: &str,
        encoder: &EmbeddingEncoder<B>,
    ) -> anyhow::Result<HashMap<String, String>> {
        let tokens = encoder.get_tokens(text)?;
        let hidden_states = encoder.get_hidden_states(text)?;
        let [seq_len, _dim] = hidden_states.dims();

        let mut results = HashMap::new();

        for (i, token) in tokens.iter().enumerate().take(seq_len) {
            #[allow(clippy::single_range_in_vec_init)]
            // slice([i..i+1]) → [1, dim]; drop the singleton row axis → [dim]
            let token_emb: Tensor<B, 1> = hidden_states.clone().slice([i..i + 1]).squeeze_dims(&[0]);
            if let Some((name, _sim)) = self.matcher.match_entity(token, &token_emb) {
                // In this demo, we use "device" as the only parameter key
                results.insert("device".to_string(), name);
            }
        }

        Ok(results)
    }
}

pub fn align_labels_with_tokens(
    tokens: &[String],
    params: &HashMap<String, serde_json::Value>,
) -> Vec<i32> {
    let mut labels = vec![0; tokens.len()];
    if let Some(device) = params.get("device").and_then(|v| v.as_str()) {
        for (i, token) in tokens.iter().enumerate() {
            if token.contains(device) || device.contains(token) {
                labels[i] = 1; // 1 for DEVICE
            }
        }
    }
    labels
}
