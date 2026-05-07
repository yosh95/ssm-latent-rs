use anyhow::{Result, anyhow};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use hf_hub::api::sync::Api;
use ndarray::{Array2, Axis};
use ort::session::Session;
use ort::value::Tensor as OrtTensor;
use tokenizers::Tokenizer;

pub struct EmbeddingEncoder<B: Backend> {
    session: std::sync::Mutex<Session>,
    tokenizer: Tokenizer,
    device: B::Device,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> EmbeddingEncoder<B> {
    pub fn new(model_id: &str, device: B::Device) -> Result<Self> {
        let api = Api::new()?;
        let repo = api.model(model_id.to_string());

        let model_path = repo
            .get("onnx/model.onnx")
            .or_else(|_| repo.get("model.onnx"))
            .map_err(|e| anyhow!("Failed to find ONNX model: {}", e))?;

        let tokenizer_path = repo.get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        let session = Session::builder()
            .and_then(|b| Ok(b.with_intra_threads(1)?))
            .map_err(|e| anyhow!("Failed to configure session: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow!("Failed to load model: {}", e))?;

        Ok(Self {
            session: std::sync::Mutex::new(session),
            tokenizer,
            device,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Tensor<B, 1>> {
        let prefixed = if text.starts_with("query: ") {
            text.to_string()
        } else {
            format!("query: {}", text)
        };

        let encoding = self
            .tokenizer
            .encode(prefixed, true)
            .map_err(|e| anyhow!("Tokenization error: {}", e))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();
        let seq = ids.len();

        let input_ids = Array2::from_shape_vec((1, seq), ids).unwrap();
        let attention_mask = Array2::from_shape_vec((1, seq), mask).unwrap();

        let mut inputs = ort::inputs![
            "input_ids" => OrtTensor::from_array(input_ids)?,
            "attention_mask" => OrtTensor::from_array(attention_mask)?,
        ];

        // Some models need token_type_ids
        let mut session_guard = self.session.lock().unwrap();
        let input_names: Vec<String> = session_guard
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        if input_names.iter().any(|n| n == "token_type_ids") {
            let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();
            let type_ids_array = Array2::from_shape_vec((1, seq), type_ids).unwrap();
            inputs.push((
                "token_type_ids".into(),
                OrtTensor::from_array(type_ids_array)?.into(),
            ));
        }

        let outputs = session_guard.run(inputs)?;
        let embeddings = outputs[0].try_extract_array::<f32>()?;

        // Mean pooling: [Batch, Seq, Dim] -> [Dim]
        let mean_pooled = embeddings.mean_axis(Axis(1)).unwrap();
        let vec = mean_pooled.to_owned().into_raw_vec_and_offset().0;

        let tensor = Tensor::<B, 1>::from_data(vec.as_slice(), &self.device);
        let norm = tensor
            .clone()
            .powf_scalar(2.0)
            .sum()
            .sqrt()
            .add_scalar(1e-9);
        Ok(tensor.div(norm))
    }

    pub fn get_hidden_states(&self, text: &str) -> Result<Tensor<B, 2>> {
        let prefixed = if text.starts_with("query: ") {
            text.to_string()
        } else {
            format!("query: {}", text)
        };

        let encoding = self
            .tokenizer
            .encode(prefixed, true)
            .map_err(|e| anyhow!("Tokenization error: {}", e))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();
        let seq = ids.len();

        let input_ids = Array2::from_shape_vec((1, seq), ids).unwrap();
        let attention_mask = Array2::from_shape_vec((1, seq), mask).unwrap();

        let mut inputs = ort::inputs![
            "input_ids" => OrtTensor::from_array(input_ids)?,
            "attention_mask" => OrtTensor::from_array(attention_mask)?,
        ];

        let mut session_guard = self.session.lock().unwrap();
        let input_names: Vec<String> = session_guard
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        if input_names.iter().any(|n| n == "token_type_ids") {
            let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();
            let type_ids_array = Array2::from_shape_vec((1, seq), type_ids).unwrap();
            inputs.push((
                "token_type_ids".into(),
                OrtTensor::from_array(type_ids_array)?.into(),
            ));
        }

        let outputs = session_guard.run(inputs)?;
        let embeddings = outputs[0].try_extract_array::<f32>()?;
        let shape = embeddings.dim();
        let seq_len = shape[1];
        let dim = shape[2];
        let vec = embeddings.to_owned().into_raw_vec_and_offset().0;

        Ok(Tensor::<B, 1>::from_data(vec.as_slice(), &self.device).reshape([seq_len, dim]))
    }

    pub fn get_tokens(&self, text: &str) -> Result<Vec<String>> {
        let prefixed = if text.starts_with("query: ") {
            text.to_string()
        } else {
            format!("query: {}", text)
        };
        let tokens = self
            .tokenizer
            .encode(prefixed, true)
            .map_err(|e| anyhow!("Tokenization error: {}", e))?;
        Ok(tokens.get_tokens().to_vec())
    }

    pub fn get_tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}
