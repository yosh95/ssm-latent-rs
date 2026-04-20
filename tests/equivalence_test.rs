use burn::backend::NdArray;
use burn::tensor::Tensor;
use mamba_rs::model::{ComplexTensor, MambaBlock, MambaConfig};

#[test]
fn test_selective_scan_vs_step_equivalence() {
    type Backend = NdArray<f32>;
    let device = Default::default();

    let d_model = 4;
    let seq_len = 8;
    let d_inner = 4;
    let d_state = 4;

    let config = MambaConfig {
        d_model,
        d_state,
        d_conv: 3,
        expand: 1,
    };

    let model = MambaBlock::<Backend>::new(&config, &device);

    // Generate random input data
    let u = Tensor::<Backend, 3>::random(
        [1, seq_len, d_inner],
        burn::tensor::Distribution::Default,
        &device,
    );
    let delta = burn::tensor::activation::softplus(
        Tensor::<Backend, 3>::random(
            [1, seq_len, d_inner],
            burn::tensor::Distribution::Default,
            &device,
        ),
        1.0,
    ); // delta must be positive
    let b_re = Tensor::<Backend, 3>::random(
        [1, seq_len, d_state],
        burn::tensor::Distribution::Default,
        &device,
    );
    let b_im = Tensor::<Backend, 3>::random(
        [1, seq_len, d_state],
        burn::tensor::Distribution::Default,
        &device,
    );
    let c_re = Tensor::<Backend, 3>::random(
        [1, seq_len, d_state],
        burn::tensor::Distribution::Default,
        &device,
    );
    let c_im = Tensor::<Backend, 3>::random(
        [1, seq_len, d_state],
        burn::tensor::Distribution::Default,
        &device,
    );

    // 1. Execute selective_scan (parallel scan)
    let (y_parallel, h_re_parallel, h_im_parallel) = model.selective_scan(
        u.clone(),
        delta.clone(),
        b_re.clone(),
        b_im.clone(),
        c_re.clone(),
        c_im.clone(),
    );

    // 2. Execute step (sequential processing) in a loop
    let mut h_re_step = Tensor::<Backend, 3>::zeros([1, d_inner, d_state], &device);
    let mut h_im_step = Tensor::<Backend, 3>::zeros([1, d_inner, d_state], &device);
    let mut y_step_list = Vec::new();

    for t in 0..seq_len {
        let ut = u.clone().slice([0..1, t..t + 1]).squeeze::<2>(1);
        let dt = delta.clone().slice([0..1, t..t + 1]).squeeze::<2>(1);
        let br = b_re.clone().slice([0..1, t..t + 1]).squeeze::<2>(1);
        let bi = b_im.clone().slice([0..1, t..t + 1]).squeeze::<2>(1);
        let cr = c_re.clone().slice([0..1, t..t + 1]).squeeze::<2>(1);
        let ci = c_im.clone().slice([0..1, t..t + 1]).squeeze::<2>(1);

        let (yt, next_h) = model.step(
            ut,
            dt,
            ComplexTensor { re: br, im: bi },
            ComplexTensor { re: cr, im: ci },
            ComplexTensor {
                re: h_re_step,
                im: h_im_step,
            },
        );

        h_re_step = next_h.re.clone();
        h_im_step = next_h.im.clone();
        y_step_list.push(yt.unsqueeze_dim::<3>(1));

        // Compare internal states at each step (h_parallel is [batch, seq, d_inner, d_state])
        let hr_p = h_re_parallel
            .clone()
            .slice([0..1, t..t + 1])
            .squeeze::<3>(1);
        let hi_p = h_im_parallel
            .clone()
            .slice([0..1, t..t + 1])
            .squeeze::<3>(1);

        // Check error in internal states
        hr_p.to_data().assert_approx_eq(&h_re_step.to_data(), 3);
        hi_p.to_data().assert_approx_eq(&h_im_step.to_data(), 3);
    }

    let y_sequential = Tensor::cat(y_step_list, 1);

    // Check equivalence of the final output
    // Confirm that they match with about 3 decimal places (1e-3) of precision
    y_parallel
        .to_data()
        .assert_approx_eq(&y_sequential.to_data(), 3);

    println!("Equivalence test passed! Parallel scan matches sequential steps.");
}
