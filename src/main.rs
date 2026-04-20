use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::tensor::{Tensor, TensorData};
use mamba_rs::model::{ComplexTensor, MambaBlock, MambaConfig};

type MyBackend = NdArray<f32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {
    let device = NdArrayDevice::default();

    let d_model = 64;
    let seq_len = 32;
    let d_inner = 64;
    let d_state = 16;
    let lr = 0.001;
    let epochs = 50;

    let config = MambaConfig {
        d_model,
        d_state,
        d_conv: 3,
        expand: 1,
    };

    let mut model = MambaBlock::<MyAutodiffBackend>::new(&config, &device);

    println!("Starting Training Loop (Associative Scan)...");

    let mut u_data = vec![0.0f32; seq_len * d_inner];
    for t in 0..seq_len {
        let val = (t as f32 * 0.5).sin();
        for d in 0..d_inner {
            u_data[t * d_inner + d] = val;
        }
    }

    let u = Tensor::<MyAutodiffBackend, 3>::from_data(
        TensorData::new(u_data, [1, seq_len, d_inner]),
        &device,
    );
    let target = u.clone();

    let delta =
        Tensor::<MyAutodiffBackend, 3>::zeros([1, seq_len, d_inner], &device).add_scalar(0.1);
    let b_re =
        Tensor::<MyAutodiffBackend, 3>::zeros([1, seq_len, d_state], &device).add_scalar(0.1);
    let b_im = Tensor::<MyAutodiffBackend, 3>::zeros([1, seq_len, d_state], &device);
    let c_re =
        Tensor::<MyAutodiffBackend, 3>::zeros([1, seq_len, d_state], &device).add_scalar(0.1);
    let c_im = Tensor::<MyAutodiffBackend, 3>::zeros([1, seq_len, d_state], &device);

    let mut optim = SgdConfig::new().init::<MyAutodiffBackend, MambaBlock<MyAutodiffBackend>>();

    for epoch in 1..=epochs {
        // Fast forward using parallel scan. Internal states h_re, h_im are also available.
        let (y, _h_re, _h_im) = model.selective_scan(
            u.clone(),
            delta.clone(),
            b_re.clone(),
            b_im.clone(),
            c_re.clone(),
            c_im.clone(),
        );

        let loss = (y - target.clone()).powf_scalar(2.0).mean();

        if epoch % 10 == 0 || epoch == 1 {
            println!("Epoch {}: Loss = {:?}", epoch, loss.clone().into_data());
        }

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);
    }

    println!("\nStep (Inference/JEPA) Demo with State Management:");
    // Inject initial state from outside.
    let h_re = Tensor::<MyBackend, 3>::zeros([1, d_inner, d_state], &device);
    let h_im = Tensor::<MyBackend, 3>::zeros([1, d_inner, d_state], &device);

    let u_step = Tensor::<MyBackend, 2>::ones([1, d_inner], &device);
    let dt_step = Tensor::<MyBackend, 2>::zeros([1, d_inner], &device).add_scalar(0.1);
    let br_step = Tensor::<MyBackend, 2>::zeros([1, d_state], &device).add_scalar(0.1);
    let bi_step = Tensor::<MyBackend, 2>::zeros([1, d_state], &device);
    let cr_step = Tensor::<MyBackend, 2>::zeros([1, d_state], &device).add_scalar(0.1);
    let ci_step = Tensor::<MyBackend, 2>::zeros([1, d_state], &device);

    let model_inf = model.valid();
    // The step returns state, allowing it to be passed to the next time step, similar to JEPA.
    let (y_step, next_h) = model_inf.step(
        u_step,
        dt_step,
        ComplexTensor {
            re: br_step,
            im: bi_step,
        },
        ComplexTensor {
            re: cr_step,
            im: ci_step,
        },
        ComplexTensor { re: h_re, im: h_im },
    );
    let next_h_re = next_h.re;

    let mean_data = y_step.mean().into_data();
    println!("Step output mean: {:?}", mean_data);
    println!("Updated state h_re shape: {:?}", next_h_re.dims());
    println!("State is now ready to be used as latent representation for JEPA.");
}
