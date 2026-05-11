mod data;
mod model;
mod training;

use crate::data::StoryDataPipeline;
use crate::model::JepaLanguageModel;
use crate::training::{train, ModelConfig, TrainingConfig};
use burn::backend::Wgpu;
use burn::module::AutodiffModule;
use burn::tensor::{Int, Tensor};
use serde::Deserialize;
use ssm_latent_model::ssm::SsmConfig;
use std::fs;

#[derive(Deserialize, Debug)]
struct FullConfig {
    model: ModelConfig,
    training: TrainingConfig,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("================================================");
    println!("   🚀 TinyStories JEPA (Pretrained Encoder) 🚀   ");
    println!("================================================\n");

    // Load configuration
    let config_str = fs::read_to_string("config.toml").expect("Failed to read config.toml");
    let mut full_config: FullConfig =
        toml::from_str(&config_str).expect("Failed to parse config.toml");

    let ssm_config: SsmConfig = full_config.model.into();
    full_config.training.model_config = Some(ssm_config.clone());

    type Backend = Wgpu;
    let device = burn::backend::wgpu::WgpuDevice::default();

    let pipeline = StoryDataPipeline::new()?;
    let vocab_size = pipeline.tokenizer.get_vocab_size(true);

    println!("Fetching TinyStories dataset...");
    let full_text = pipeline.fetch_tiny_stories()?;

    let mut dataset = Vec::new();
    for text in full_text.lines().take(full_config.training.dataset_samples) {
        if text.trim().is_empty() {
            continue;
        }
        let encoding = pipeline.tokenizer.encode(text, true).unwrap();
        let ids: Vec<usize> = encoding.get_ids().iter().map(|&x| x as usize).collect();
        if ids.len() > 5 {
            dataset.push((text.to_string(), ids));
        }
    }
    println!("Prepared {} story samples for training.", dataset.len());

    println!("\n[Phase 1] Training SSM dynamics on Embeddings...");
    let model = train::<burn::backend::Autodiff<Backend>>(
        "/tmp/jepa-ssm",
        full_config.training,
        device.clone(),
        vocab_size,
        dataset,
    );

    // Convert to inference model
    let model_valid = JepaLanguageModel::<Backend> {
        embedding: model.embedding.valid(),
        input_projection: model.input_projection.valid(),
        ssm_layers: model.ssm_layers.into_iter().map(|s| s.valid()).collect(),
        output_head: model.output_head.valid(),
        d_model: model.d_model,
    };

    println!("\n[Phase 2] Generation Mode (Latent Prediction)");
    let prompt = "Once upon a time, a small bird";
    println!("\nPrompt: \x1b[36m{}\x1b[0m", prompt);

    println!("\nGenerating story...");
    println!("--- Story Output ---");
    print!("{}", prompt);

    let mut current_ids: Vec<usize> = pipeline
        .tokenizer
        .encode(prompt, true)
        .unwrap()
        .get_ids()
        .iter()
        .map(|&x| x as usize)
        .collect();

    for _ in 0..40 {
        let input_tensor = Tensor::<Backend, 1, Int>::from_ints(
            current_ids
                .iter()
                .map(|&x| x as i32)
                .collect::<Vec<i32>>()
                .as_slice(),
            &device,
        )
        .reshape([1, current_ids.len()]);

        let next_id_tensor = model_valid.step(input_tensor, 5);
        let next_id = next_id_tensor.into_data().as_slice::<i32>().unwrap()[0];

        let word = pipeline
            .tokenizer
            .decode(&[next_id as u32], true)
            .map_err(|e| anyhow::anyhow!(e))?;

        print!("{}", word);
        std::io::Write::flush(&mut std::io::stdout())?;

        current_ids.push(next_id as usize);
        if next_id == 50256 || current_ids.len() > 100 {
            break;
        } // EOS or limit
    }

    println!("\n\n--- Done ---");
    Ok(())
}
