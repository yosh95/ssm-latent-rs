use crate::model::{IntentClassifier, NERClassifier};

use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams};
use burn::optim::Optimizer;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;

pub struct Trainer;

impl Default for Trainer {
    fn default() -> Self {
        Self::new()
    }
}

impl Trainer {
    pub fn new() -> Self {
        Self
    }

    /// Train the NER classifier
    pub fn train_ner<B: AutodiffBackend>(
        &self,
        classifier: NERClassifier<B>,
        token_embeddings: Tensor<B, 3>, // [N, Seq_Len, Hidden_Dim]
        labels: Tensor<B, 2>,          // [N, Seq_Len]
        epochs: usize,
        learning_rate: f64,
    ) -> NERClassifier<B> {
        let [n, seq_len, dim] = token_embeddings.dims();
        let mut optim = AdamConfig::new().init();

        let x_flat = token_embeddings.reshape([n * seq_len, dim]);
        let y_flat = labels.reshape([n * seq_len]).int();

        let mut classifier = classifier;

        for epoch in 1..=epochs {
            let logits = classifier.forward(x_flat.clone());
            let loss = CrossEntropyLossConfig::new()
                .init(&logits.device())
                .forward(logits, y_flat.clone());

            if epoch % 10 == 0 || epoch == 1 {
                println!(
                    "NER Epoch: {:>3}, Loss: {:.6}",
                    epoch,
                    loss.clone().into_scalar()
                );
            }

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &classifier);
            classifier = optim.step(learning_rate, classifier, grads);
        }
        classifier
    }

    /// Train the Intent classifier
    pub fn train_intent<B: AutodiffBackend>(
        &self,
        classifier: IntentClassifier<B>,
        embeddings: Tensor<B, 2>,
        labels: Tensor<B, 1>,
        epochs: usize,
        learning_rate: f64,
    ) -> IntentClassifier<B> {
        let mut optim = AdamConfig::new().init();
        let labels_int = labels.clone().int();

        let mut classifier = classifier;

        for epoch in 1..=epochs {
            let logits = classifier.forward(embeddings.clone());
            let loss = CrossEntropyLossConfig::new()
                .init(&logits.device())
                .forward(logits, labels_int.clone());

            if epoch % 50 == 0 || epoch == 1 {
                println!(
                    "Intent Epoch: {:>3}, Loss: {:.6}",
                    epoch,
                    loss.clone().into_scalar()
                );
            }

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &classifier);
            classifier = optim.step(learning_rate, classifier, grads);
        }
        classifier
    }

    /// Calculate class centroids in O(n) for OOD detection
    pub fn calculate_centroids<B: AutodiffBackend>(
        &self,
        embeddings: Tensor<B, 2>,
        labels_vec: &[u32],
        num_classes: usize,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let [_num_samples, dim] = embeddings.dims();
        let mut sums = vec![vec![0.0f32; dim]; num_classes];
        let mut counts = vec![0.0f32; num_classes];

        let emb_data: Vec<f32> = embeddings
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        for (i, &label) in labels_vec.iter().enumerate() {
            let l = label as usize;
            if l < num_classes {
                for j in 0..dim {
                    sums[l][j] += emb_data[i * dim + j];
                }
                counts[l] += 1.0;
            }
        }

        let mut centroid_data = Vec::with_capacity(num_classes * dim);
        for i in 0..num_classes {
            let mut row = sums[i].clone();
            if counts[i] > 0.0 {
                for val in row.iter_mut() {
                    *val /= counts[i];
                }
                let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-9;
                for val in row.iter_mut() {
                    *val /= norm;
                }
            }
            centroid_data.extend(row);
        }

        Tensor::<B, 1>::from_data(centroid_data.as_slice(), device).reshape([num_classes, dim])
    }
}
