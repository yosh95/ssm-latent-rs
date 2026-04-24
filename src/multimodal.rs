use crate::ssm::{SsmBlock, SsmConfig};
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct VisionEncoder<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    linear: Linear<B>,
}

impl<B: Backend> VisionEncoder<B> {
    pub fn new(channels: usize, d_model: usize, device: &B::Device) -> Self {
        let conv1 = Conv2dConfig::new([channels, 16], [3, 3])
            .with_stride([2, 2])
            .init(device);
        let conv2 = Conv2dConfig::new([16, 32], [3, 3])
            .with_stride([2, 2])
            .init(device);
        // Input 16x16 -> 7x7 (conv1) -> 3x3 (conv2)
        let linear = LinearConfig::new(32 * 3 * 3, d_model).init(device);

        Self {
            conv1,
            conv2,
            linear,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.conv2.forward(x);
        let x = burn::tensor::activation::relu(x);
        let [batch, channels, h, w] = x.dims();
        let x = x.reshape([batch, channels * h * w]);
        self.linear.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct MultimodalLatentPredictor<B: Backend> {
    vision_encoder: VisionEncoder<B>,
    sensor_encoder: Linear<B>,
    action_encoder: Linear<B>,
    fusion: Linear<B>,
    ssm: SsmBlock<B>,
    // Decoders
    vision_decoder: Linear<B>,
    sensor_decoder: Linear<B>,
    d_model: usize,
    img_size: [usize; 3], // C, H, W
}

impl<B: Backend> MultimodalLatentPredictor<B> {
    pub fn new(
        config: &SsmConfig,
        img_channels: usize,
        sensor_dim: usize,
        action_dim: usize,
        device: &B::Device,
    ) -> Self {
        let vision_encoder = VisionEncoder::new(img_channels, config.d_model, device);
        let sensor_encoder = LinearConfig::new(sensor_dim, config.d_model).init(device);
        let action_encoder = LinearConfig::new(action_dim, config.d_model).init(device);

        let fusion = LinearConfig::new(config.d_model * 3, config.d_model).init(device);
        let ssm = SsmBlock::new(config, device);

        let vision_decoder = LinearConfig::new(config.d_model, img_channels * 16 * 16).init(device);
        let sensor_decoder = LinearConfig::new(config.d_model, sensor_dim).init(device);

        Self {
            vision_encoder,
            sensor_encoder,
            action_encoder,
            fusion,
            ssm,
            vision_decoder,
            sensor_decoder,
            d_model: config.d_model,
            img_size: [img_channels, 16, 16],
        }
    }

    pub fn forward(
        &self,
        images: Tensor<B, 5>,
        sensors: Tensor<B, 3>,
        actions: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 5>, Tensor<B, 3>) {
        let [batch, seq_len, channels, h, w] = images.dims();

        let img_flat = images.reshape([batch * seq_len, channels, h, w]);
        let z_vis = self
            .vision_encoder
            .forward(img_flat)
            .reshape([batch, seq_len, self.d_model]);
        let z_sens = self.sensor_encoder.forward(sensors);
        let z_act = self.action_encoder.forward(actions);

        let fused = Tensor::cat(vec![z_vis, z_sens, z_act], 2);
        let u = self.fusion.forward(fused);
        let predicted_z = self.ssm.forward(u);

        let decoded_img = self.vision_decoder.forward(predicted_z.clone()).reshape([
            batch,
            seq_len,
            self.img_size[0],
            self.img_size[1],
            self.img_size[2],
        ]);
        let decoded_sensor = self.sensor_decoder.forward(predicted_z.clone());

        (predicted_z, decoded_img, decoded_sensor)
    }
}
