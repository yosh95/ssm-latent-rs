use burn::tensor::Tensor;
/// Forward-Forward Algorithm utilities.
///
/// Implements the Forward-Forward algorithm (Hinton, 2022) where each layer
/// learns via a local "goodness" objective rather than backpropagation of
/// a global error signal.
///
/// The key principle:
/// - **Positive data** (real) should produce high goodness (large squared activations)
/// - **Negative data** (fake/corrupted) should produce low goodness
/// - Each layer optimizes its own goodness independently
/// - Gradients do NOT flow between layers (achieved via `.detach()`)
///
/// This is more biologically plausible than backpropagation and can be more
/// energy-efficient since each layer's update is purely local.
use burn::tensor::backend::Backend;

/// L2-normalize along the last dimension.
///
/// The Forward-Forward algorithm normalizes activations at each layer to prevent
/// the goodness from being trivially maximized by scaling up weights.
///
/// Burn's `sum_dim` keeps the reduced dimension as size 1, so no explicit
/// unsqueeze is needed — the norm already has the same rank as the input
/// and broadcasts correctly.
pub fn l2_normalize<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let ndim = D;
    let norm = x.clone().powf_scalar(2.0).sum_dim(ndim - 1).sqrt() + 1e-8;
    // `sum_dim` keeps the dim as size 1, so norm has rank D with last dim = 1.
    // Division broadcasts: [batch, d] / [batch, 1] → [batch, d]
    x / norm
}

/// Compute Forward-Forward goodness: mean squared activation.
///
/// The goodness is the objective each layer maximizes for positive data
/// and minimizes for negative data.
///
/// ```text
/// goodness(x) = mean(x²)
/// ```
pub fn goodness<B: Backend, const D: usize>(x: &Tensor<B, D>) -> Tensor<B, 1> {
    x.clone().powf_scalar(2.0).mean().unsqueeze()
}

/// Forward-Forward loss: encourages high goodness for positive, low for negative.
///
/// ```text
/// L = -goodness(pos) + goodness(neg)
/// ```
///
/// This is minimized when positive activations are large and negative
/// activations are small.
pub fn ff_loss<B: Backend, const D: usize>(
    pos_activations: &Tensor<B, D>,
    neg_activations: &Tensor<B, D>,
) -> Tensor<B, 1> {
    let pos_g = goodness(pos_activations);
    let neg_g = goodness(neg_activations);
    neg_g - pos_g
}

/// Forward-Forward loss with threshold (more stable variant).
///
/// ```text
/// L = ReLU(threshold - goodness(pos))² + goodness(neg)²
/// ```
///
/// This only penalizes positive goodness below the threshold, and always
/// penalizes negative goodness. This is more stable than the simple version
/// because it doesn't encourage unbounded growth of positive goodness.
pub fn ff_loss_threshold<B: Backend, const D: usize>(
    pos_activations: &Tensor<B, D>,
    neg_activations: &Tensor<B, D>,
    threshold: f64,
) -> Tensor<B, 1> {
    let pos_g = goodness(pos_activations);
    let neg_g = goodness(neg_activations);

    // Penalize when positive goodness is below threshold
    let pos_penalty = (pos_g.clone() - threshold).clamp_min(0.0).powf_scalar(2.0);
    // Always penalize negative goodness (want it near 0)
    let neg_penalty = neg_g.powf_scalar(2.0);

    pos_penalty + neg_penalty
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn test_l2_normalize() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let x = Tensor::<NdArray, 2>::from_data(
            burn::tensor::TensorData::new(vec![3.0f32, 4.0f32], [1, 2]),
            &device,
        );
        let normalized = l2_normalize(x);
        let values = normalized.into_data().as_slice::<f32>().unwrap().to_vec();
        // 3/5 = 0.6, 4/5 = 0.8
        assert!((values[0] - 0.6).abs() < 1e-5);
        assert!((values[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_goodness() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let x = Tensor::<NdArray, 2>::from_data(
            burn::tensor::TensorData::new(vec![3.0f32, 4.0f32], [1, 2]),
            &device,
        );
        let g = goodness(&x);
        let val = g.into_data().as_slice::<f32>().unwrap()[0];
        // mean(9, 16) = 12.5
        assert!((val - 12.5).abs() < 1e-5);
    }
}
