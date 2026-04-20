use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use mamba_rs::model::{MambaBlock, MambaConfig};

#[test]
fn test_gradient_calculable() {
    type Backend = Autodiff<NdArray<f32>>;
    let device = Default::default();

    let config = MambaConfig {
        d_model: 4,
        d_state: 4,
        d_conv: 3,
        expand: 1,
    };

    let model = MambaBlock::<Backend>::new(&config, &device);

    let u = Tensor::<Backend, 3>::random([1, 4, 4], burn::tensor::Distribution::Default, &device);
    let delta =
        Tensor::<Backend, 3>::random([1, 4, 4], burn::tensor::Distribution::Default, &device);
    let b_re =
        Tensor::<Backend, 3>::random([1, 4, 4], burn::tensor::Distribution::Default, &device);
    let b_im =
        Tensor::<Backend, 3>::random([1, 4, 4], burn::tensor::Distribution::Default, &device);
    let c_re =
        Tensor::<Backend, 3>::random([1, 4, 4], burn::tensor::Distribution::Default, &device);
    let c_im =
        Tensor::<Backend, 3>::random([1, 4, 4], burn::tensor::Distribution::Default, &device);

    let (y, _, _) = model.selective_scan(u, delta, b_re, b_im, c_re, c_im);
    let loss = y.sum();
    let grads = loss.backward();

    // Verify gradients for some parameters
    let grad_a_re = model.a_re.grad(&grads);
    assert!(grad_a_re.is_some());
}
