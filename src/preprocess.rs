use anyhow::{Context, Result};
use ndarray::{Array2, Axis};
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

pub struct LogEmbedder {
    session: Session,
    tokenizer: Tokenizer,
}

impl LogEmbedder {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
            anyhow::anyhow!("Failed to load tokenizer from {}: {}", tokenizer_path, e)
        })?;

        let model_bytes = std::fs::read(model_path)
            .with_context(|| format!("Failed to read model file from {}", model_path))?;

        let session = Session::builder()?
            .commit_from_memory(&model_bytes)
            .with_context(|| "Failed to create ORT session from memory")?;

        Ok(Self { session, tokenizer })
    }

    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();

        let seq_len = input_ids.len();
        let input_ids_array = Array2::from_shape_vec((1, seq_len), input_ids)
            .context("Failed to create input_ids array")?;
        let attention_mask_array = Array2::from_shape_vec((1, seq_len), attention_mask)
            .context("Failed to create attention_mask array")?;

        let input_ids_tensor = Tensor::from_array(input_ids_array)?;
        let attention_mask_tensor = Tensor::from_array(attention_mask_array)?;

        let inputs = ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
        ];

        let outputs = self.session.run(inputs).context("Session run failed")?;

        let embeddings = outputs["last_hidden_state"]
            .try_extract_array::<f32>()
            .context("Failed to extract embeddings array")?;

        let mean = embeddings
            .mean_axis(Axis(1))
            .context("Mean pooling failed")?;

        Ok(mean.to_owned().into_raw_vec_and_offset().0)
    }
}
