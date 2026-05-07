use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::ToElement;
use burn::tensor::Tensor;
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;

/// Intent classification neural network
#[derive(Module, Debug)]
pub struct IntentClassifier<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    relu: Relu,
    #[module(retain = false)]
    centroids: Option<Tensor<B, 2>>,
}

#[derive(Config, Debug)]
pub struct IntentClassifierConfig {
    pub input_dim: usize,
    pub num_intents: usize,
    #[config(default = 128)]
    pub hidden_dim: usize,
}

impl IntentClassifierConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> IntentClassifier<B> {
        IntentClassifier {
            fc1: LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            fc2: LinearConfig::new(self.hidden_dim, self.num_intents).init(device),
            relu: Relu::new(),
            centroids: None,
        }
    }
}

impl<B: Backend> IntentClassifier<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = self.relu.forward(x);
        self.fc2.forward(x)
    }

    pub fn predict_with_confidence(&self, x: Tensor<B, 1>) -> (u32, f32) {
        let x_batched = x.unsqueeze::<2>();
        // forward() returns [1, num_intents]; squeeze axis 0 (the batch singleton) → [num_intents]
        let logits: Tensor<B, 1> = self.forward(x_batched).squeeze_dims(&[0]);
        let probs = softmax(logits, 0);

        let probs_vec: Vec<f32> = probs.into_data().convert::<f32>().iter::<f32>().collect();
        let (intent_id, confidence) = probs_vec
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, v)| (i as u32, *v))
            .unwrap_or((0, 0.0));

        (intent_id, confidence)
    }

    pub fn get_in_dist_similarity(&self, x: Tensor<B, 1>) -> f32 {
        let centroids = match &self.centroids {
            Some(c) => c,
            None => return 0.0,
        };
        // centroids: [num_intents, dim], x: [dim]
        // matmul centroids with x -> [num_intents]
        let x_unsqueezed = x.unsqueeze::<2>().transpose(); // [dim, 1]
        // matmul: [num_intents, dim] × [dim, 1] → [num_intents, 1]; drop trailing singleton → [num_intents]
        let similarities: Tensor<B, 1> = centroids.clone().matmul(x_unsqueezed).squeeze_dims(&[1]); // [num_intents]

        let sim_vec: Vec<f32> = similarities
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        sim_vec.into_iter().fold(f32::MIN, f32::max)
    }

    pub fn set_centroids(&mut self, centroids: Tensor<B, 2>) {
        self.centroids = Some(centroids);
    }
}

/// Token classification (NER) neural network
#[derive(Module, Debug)]
pub struct NERClassifier<B: Backend> {
    head: Linear<B>,
}

#[derive(Config, Debug)]
pub struct NERClassifierConfig {
    pub input_dim: usize,
    pub num_labels: usize,
}

impl NERClassifierConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> NERClassifier<B> {
        NERClassifier {
            head: LinearConfig::new(self.input_dim, self.num_labels).init(device),
        }
    }
}

impl<B: Backend> NERClassifier<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.head.forward(x)
    }
}

/// Deterministic matching engine (Exact match + Embedding similarity)
pub struct MasterMatcher<B: Backend> {
    device_templates: std::collections::HashMap<String, Tensor<B, 1>>,
    threshold: f32,
}

impl<B: Backend> MasterMatcher<B> {
    pub fn new(threshold: f32) -> Self {
        Self {
            device_templates: std::collections::HashMap::new(),
            threshold,
        }
    }

    pub fn add_template(&mut self, name: &str, embedding: Tensor<B, 1>) {
        self.device_templates.insert(name.to_string(), embedding);
    }

    pub fn match_entity(&self, token_str: &str, embedding: &Tensor<B, 1>) -> Option<(String, f32)> {
        let clean_token = token_str.replace("##", "");
        for name in self.device_templates.keys() {
            if name.eq_ignore_ascii_case(&clean_token) {
                return Some((name.clone(), 1.0));
            }
        }

        let mut best_name = None;
        let mut max_sim = -1.0f32;

        // Normalize embedding
        let norm = embedding
            .clone()
            .powf_scalar(2.0)
            .sum()
            .sqrt()
            .add_scalar(1e-9);
        let norm_emb = embedding.clone().div(norm);

        for (name, template) in &self.device_templates {
            // template should be normalized
            let sim_tensor = norm_emb.clone().mul(template.clone()).sum();
            let sim = sim_tensor.into_scalar().to_f32();

            if sim > max_sim && sim > self.threshold {
                max_sim = sim;
                best_name = Some((name.clone(), sim));
            }
        }
        best_name
    }
}
