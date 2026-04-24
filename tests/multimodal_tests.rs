use burn::backend::NdArray;
use burn::tensor::Tensor;
use ssm_latent_model::multimodal::MultimodalLatentPredictor;
use ssm_latent_model::ssm::SsmConfig;

#[test]
fn test_multimodal_forward_logic() {
    let device = Default::default();
    let config = SsmConfig {
        d_model: 32,
        d_state: 16,
        expand: 2,
        n_heads: 2,
        mimo_rank: 1,
        use_conv: true,
        conv_kernel: 3,
    };

    let predictor = MultimodalLatentPredictor::<NdArray>::new(&config, 3, 8, 4, &device);

    let batch = 2;
    let seq = 4;

    // Mock data
    // images: [B, T, C, H, W] = [2, 4, 3, 16, 16]
    let img = Tensor::<NdArray, 5>::zeros([batch, seq, 3, 16, 16], &device);
    // sensors: [B, T, SensorDim] = [2, 4, 8]
    let sensor = Tensor::<NdArray, 3>::zeros([batch, seq, 8], &device);
    // actions: [B, T, ActionDim] = [2, 4, 4]
    let action = Tensor::<NdArray, 3>::zeros([batch, seq, 4], &device);

    let (pred_z, dec_img, dec_sens) = predictor.forward(img, sensor, action);

    // Verify expected output shapes
    assert_eq!(pred_z.dims(), [batch, seq, 32]);
    assert_eq!(dec_img.dims(), [batch, seq, 3, 16, 16]);
    assert_eq!(dec_sens.dims(), [batch, seq, 8]);
}
