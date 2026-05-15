/// Forward-Forward Layer: a single linear layer + normalization trained with FF.
///
/// This is a single layer in the Forward-Forward network. It is trained
/// independently: gradients stop at this layer's output (via `.detach()` from
/// the previous layer), so each layer learns a local goodness objective
/// without backpropagation through the network.
///
/// The layer composition:
/// ```text
/// x → Linear → L2-Normalize → y
/// ```
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use crate::forward_forward::l2_normalize;

/// A single Forward-Forward layer.
///
/// Architecture: `Linear(d_in, d_out)` followed by L2 normalization.
/// The normalization prevents the layer from trivially increasing goodness
/// by scaling up its weights.
#[derive(Module, Debug)]
pub struct FfLayer<B: Backend> {
    pub linear: Linear<B>,
    pub d_in: usize,
    pub d_out: usize,
}

impl<B: Backend> FfLayer<B> {
    pub fn new(d_in: usize, d_out: usize, device: &B::Device) -> Self {
        let linear = LinearConfig::new(d_in, d_out).init(device);
        Self {
            linear,
            d_in,
            d_out,
        }
    }

    /// Forward pass with L2 normalization.
    ///
    /// Returns the normalized activations. The goodness for this layer
    /// is computed as `mean(activations²)`.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let y = self.linear.forward(x);
        l2_normalize(y)
    }
}

/// A stack of Forward-Forward layers, trained independently.
///
/// Each layer is trained via its own local goodness objective.
/// During the forward pass, gradients do NOT flow between layers
/// (each layer's input is `.detach()` from the previous layer's output).
/// This is the key property that makes FF biologically plausible and
/// backprop-free.
///
/// # Training flow
/// 1. Feed positive data through all layers (with detach between layers)
/// 2. Record activations at each layer for positive data
/// 3. Feed negative data through all layers (same weights, with detach)
/// 4. Record activations at each layer for negative data
/// 5. Update each layer independently with `ff_loss_threshold(pos, neg)`
///
/// # Inference flow
/// Forward pass with detach. The final layer's output is the latent representation.
#[derive(Module, Debug)]
pub struct FfEncoder<B: Backend> {
    pub layers: Vec<FfLayer<B>>,
}

impl<B: Backend> FfEncoder<B> {
    /// Create a new FF encoder with the given layer sizes.
    ///
    /// Each element in `layer_sizes` defines the input→output dimension
    /// of one layer. For example, `[2, 16, 8]` creates two layers:
    /// a 2→16 layer followed by a 16→8 layer.
    pub fn new(layer_sizes: &[usize], device: &B::Device) -> Self {
        assert!(
            layer_sizes.len() >= 2,
            "Need at least 2 sizes to define one layer"
        );
        let layers: Vec<_> = layer_sizes
            .windows(2)
            .map(|w| FfLayer::new(w[0], w[1], device))
            .collect();
        Self { layers }
    }

    /// Feed positive data through all layers, recording activations at each.
    ///
    /// Returns a vector of (layer_index, activations) for each layer.
    /// Called during the "positive pass" of FF training.
    pub fn positive_pass(&self, x: Tensor<B, 2>) -> Vec<Tensor<B, 2>> {
        let mut activations = Vec::with_capacity(self.layers.len());
        let mut current = x;
        for layer in &self.layers {
            let y = layer.forward(current);
            activations.push(y.clone());
            // Detach to prevent gradients flowing backwards
            current = y.detach();
        }
        activations
    }

    /// Feed negative data through all layers, recording activations at each.
    ///
    /// Uses the SAME weights as positive_pass. Called during the "negative pass".
    pub fn negative_pass(&self, x: Tensor<B, 2>) -> Vec<Tensor<B, 2>> {
        let mut activations = Vec::with_capacity(self.layers.len());
        let mut current = x;
        for layer in &self.layers {
            let y = layer.forward(current);
            activations.push(y.clone());
            current = y.detach();
        }
        activations
    }

    /// Forward pass for inference (no gradient tracking needed).
    ///
    /// Returns the encoded latent representation (output of the last layer).
    pub fn encode(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut current = x;
        for layer in &self.layers {
            let y = layer.forward(current);
            current = y.detach();
        }
        current
    }

    /// Compute per-layer FF losses.
    ///
    /// Returns a vector of losses, one per layer. Each loss compares the
    /// positive and negative activations for that layer.
    ///
    /// The optimizer should sum these losses and call `.backward()` on the sum
    /// to update all layers simultaneously (since each layer's gradients are
    /// already isolated via detach).
    pub fn compute_layer_losses(
        &self,
        pos_activations: &[Tensor<B, 2>],
        neg_activations: &[Tensor<B, 2>],
        threshold: f64,
    ) -> Vec<Tensor<B, 1>> {
        pos_activations
            .iter()
            .zip(neg_activations.iter())
            .map(|(pos, neg)| crate::forward_forward::ff_loss_threshold(pos, neg, threshold))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn test_ff_encoder_builds() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let encoder = FfEncoder::<NdArray>::new(&[2, 16, 8], &device);
        assert_eq!(encoder.layers.len(), 2);
    }

    #[test]
    fn test_ff_encoder_forward() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let encoder = FfEncoder::<NdArray>::new(&[4, 8, 4], &device);
        let x = Tensor::<NdArray, 2>::from_data(
            burn::tensor::TensorData::new(vec![1.0f32; 8], [2, 4]),
            &device,
        );
        let encoded = encoder.encode(x);
        assert_eq!(encoded.dims(), [2, 4]);
    }
}
