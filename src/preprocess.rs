
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;
use ndarray::{Array2, Axis};

pub struct LogEmbedder {
    session: Session,
    tokenizer: Tokenizer,
}

impl LogEmbedder {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Self {
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
        let model_bytes = std::fs::read(model_path).expect("Failed to read model file");
        let session = Session::builder().unwrap()
            .commit_from_memory(&model_bytes).unwrap();
        
        Self { session, tokenizer }
    }

    pub fn embed(&mut self, text: &str) -> Vec<f32> {
        let encoding = self.tokenizer.encode(text, true).unwrap();
        let input_ids = encoding.get_ids().iter().map(|&x| x as i64).collect::<Vec<_>>();
        let attention_mask = encoding.get_attention_mask().iter().map(|&x| x as i64).collect::<Vec<_>>();
        
        let seq_len = input_ids.len();
        let input_ids_array = Array2::from_shape_vec((1, seq_len), input_ids).unwrap();
        let attention_mask_array = Array2::from_shape_vec((1, seq_len), attention_mask).unwrap();

        // Convert ndarray to ort Tensors
        let input_ids_tensor = Tensor::from_array(input_ids_array).unwrap();
        let attention_mask_tensor = Tensor::from_array(attention_mask_array).unwrap();

        let inputs = ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
        ];

        let outputs = self.session.run(inputs).unwrap();
        let embeddings = outputs["last_hidden_state"].try_extract_array::<f32>().unwrap();
        
        // Mean pooling: embeddings is an ArrayViewD
        // Expecting [batch, seq, dim] -> [1, seq, dim]
        let mean = embeddings.mean_axis(Axis(1)).unwrap();
        mean.to_owned().into_raw_vec_and_offset().0
    }
}
