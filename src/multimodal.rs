use crate::latent::stability_loss;
use crate::ssm::{SsmBlock, SsmConfig};
use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
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
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let conv2 = Conv2dConfig::new([16, 32], [3, 3])
            .with_stride([2, 2])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);
        // Input 16x16 -> 8x8 (conv1) -> 4x4 (conv2)
        let linear = LinearConfig::new(32 * 4 * 4, d_model).init(device);

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
pub struct VisionDecoder<B: Backend> {
    linear: Linear<B>,
    deconv1: ConvTranspose2d<B>,
    deconv2: ConvTranspose2d<B>,
    final_conv: Conv2d<B>,
}

impl<B: Backend> VisionDecoder<B> {
    pub fn new(d_model: usize, channels: usize, device: &B::Device) -> Self {
        let linear = LinearConfig::new(d_model, 32 * 4 * 4).init(device);
        let deconv1 = ConvTranspose2dConfig::new([32, 16], [3, 3])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .with_padding_out([1, 1])
            .init(device);
        let deconv2 = ConvTranspose2dConfig::new([16, 8], [3, 3])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .with_padding_out([1, 1])
            .init(device);
        let final_conv = Conv2dConfig::new([8, channels], [1, 1]).init(device);

        Self {
            linear,
            deconv1,
            deconv2,
            final_conv,
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 4> {
        let x = self.linear.forward(x);
        let [batch, _] = x.dims();
        let x = x.reshape([batch, 32, 4, 4]);
        let x = self.deconv1.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.deconv2.forward(x);
        let x = burn::tensor::activation::relu(x);
        self.final_conv.forward(x)
    }
}

pub struct MultimodalLossInput<B: Backend> {
    pub z: Tensor<B, 3>,
    pub pred_z: Tensor<B, 3>,
    pub recons_img: Tensor<B, 5>,
    pub orig_img: Tensor<B, 5>,
    pub recons_sens: Tensor<B, 3>,
    pub orig_sens: Tensor<B, 3>,
    pub stability_weight: f64,
}

#[derive(Module, Debug)]
pub struct MultimodalLatentPredictor<B: Backend> {
    vision_encoder: VisionEncoder<B>,
    sensor_encoder: Linear<B>,
    action_encoder: Linear<B>,
    fusion: Linear<B>,
    ssm: SsmBlock<B>,
    // Decoders
    vision_decoder: VisionDecoder<B>,
    sensor_decoder: Linear<B>,
    // Stability projections
    stability_projections: Param<Tensor<B, 2>>,
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

        let vision_decoder = VisionDecoder::new(config.d_model, img_channels, device);
        let sensor_decoder = LinearConfig::new(config.d_model, sensor_dim).init(device);

        let stability_projections = Tensor::<B, 2>::random(
            [config.d_model, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        let norm = stability_projections
            .clone()
            .powf_scalar(2.0)
            .sum_dim(0)
            .sqrt()
            + 1e-6;

        Self {
            vision_encoder,
            sensor_encoder,
            action_encoder,
            fusion,
            ssm,
            vision_decoder,
            sensor_decoder,
            stability_projections: Param::from_tensor((stability_projections / norm).detach()),
            d_model: config.d_model,
            img_size: [img_channels, 16, 16],
        }
    }

    pub fn forward(
        &self,
        images: Tensor<B, 5>,
        sensors: Tensor<B, 3>,
        actions: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 5>, Tensor<B, 3>) {
        let [batch, seq_len, channels, h, w] = images.dims();

        let img_flat = images.reshape([batch * seq_len, channels, h, w]);
        let z_vis = self
            .vision_encoder
            .forward(img_flat)
            .reshape([batch, seq_len, self.d_model]);
        let z_sens = self.sensor_encoder.forward(sensors);
        let z_act = self.action_encoder.forward(actions);

        let fused = Tensor::cat(vec![z_vis.clone(), z_sens, z_act], 2);
        let u = self.fusion.forward(fused);
        let predicted_z = self.ssm.forward(u);

        let pred_z_flat = predicted_z.clone().reshape([batch * seq_len, self.d_model]);
        let decoded_img = self
            .vision_decoder
            .forward(pred_z_flat)
            .reshape([batch, seq_len, channels, h, w]);
        let decoded_sensor = self.sensor_decoder.forward(predicted_z.clone());

        (z_vis, predicted_z, decoded_img, decoded_sensor)
    }

    pub fn loss(&self, input: MultimodalLossInput<B>) -> Tensor<B, 1> {
        let [batch, seq_len, _] = input.z.dims();
        let target_z = input.z.clone().detach().slice([0..batch, 1..seq_len]);
        let pred_slice = input.pred_z.slice([0..batch, 0..seq_len - 1]);

        let mse_latent = (target_z - pred_slice).powf_scalar(2.0).mean();
        let mse_img = (input.orig_img - input.recons_img).powf_scalar(2.0).mean();
        let mse_sens = (input.orig_sens - input.recons_sens)
            .powf_scalar(2.0)
            .mean();

        let reg_loss = stability_loss(input.z, self.stability_projections.val());

        mse_latent + mse_img + mse_sens + reg_loss.mul_scalar(input.stability_weight)
    }
}
