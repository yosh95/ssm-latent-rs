use burn::backend::NdArray;
use burn::tensor::Tensor;
use ssm_latent_model::ssm::{SsmBlock, SsmConfig};

#[test]
fn test_ssm_block_forward_equivalence_multi_batch() {
    type Backend = NdArray<f32>;
    let device = Default::default();

    let batch = 2;
    let d_model = 16;
    let seq_len = 8;
    let d_state = 8;
    let expand = 2;
    let n_heads = 2;
    let mimo_rank = 1;

    let config = SsmConfig::new(d_model, d_state, expand, n_heads, mimo_rank).with_use_conv(false);

    let block = SsmBlock::<Backend>::new(&config, &device);

    let x = Tensor::<Backend, 3>::random(
        [batch, seq_len, d_model],
        burn::tensor::Distribution::Default,
        &device,
    );

    let y_parallel = block.forward(x.clone());

    let d_inner = d_model * expand;
    let d_head = d_inner / n_heads;

    let mut h = Tensor::<Backend, 4>::zeros([batch, n_heads, d_state, d_head / mimo_rank], &device);
    let mut prev_bx: Option<Tensor<Backend, 4>> = None;
    let mut conv_state: Option<Tensor<Backend, 3>> = None;
    let mut y_step_list = Vec::new();

    for t in 0..seq_len {
        let xt = x.clone().slice([0..batch, t..t + 1]);
        let [batch, _seq, d_model] = xt.dims();
        let xt = xt.reshape([batch, d_model]);

        let (yt, next_h, current_bx, next_conv_state) =
            block.forward_step(xt, h, prev_bx, conv_state);

        h = next_h;
        prev_bx = Some(current_bx);
        conv_state = next_conv_state;
        y_step_list.push(yt.unsqueeze_dim::<3>(1));
    }

    let y_sequential = Tensor::cat(y_step_list, 1);

    y_parallel
        .to_data()
        .assert_approx_eq::<f32>(&y_sequential.to_data(), Default::default());
}
