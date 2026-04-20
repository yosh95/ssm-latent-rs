use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;
use mamba_jepa_rs::mamba::{MambaBlock, MambaConfig};

#[test]
fn test_gradient_calculable() {
    type Backend = Autodiff<NdArray<f32>>;
    let device = Default::default();

    let config = MambaConfig {
        d_model: 4,
        d_state: 4,
        expand: 1,
    };

    let model = MambaBlock::<Backend>::new(&config, &device);

    let x = Tensor::<Backend, 3>::random([1, 4, 4], burn::tensor::Distribution::Default, &device);
    
    let y = model.forward(x);
    let loss = y.sum();
    let grads = loss.backward();

    let grad_a_re = model.a_re.grad(&grads);
    assert!(grad_a_re.is_some());
}
