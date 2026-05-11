mod model;
mod data;
mod training;

use burn::backend::Wgpu;
use burn::module::AutodiffModule;
use crate::model::JepaLanguageModel;
use crate::data::StoryDataPipeline;
use crate::training::{train, TrainingConfig};
use ssm_latent_model::ssm::SsmConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("========================================");
    println!("   🚀 TinyStories JEPA-SSM Chat AI 🚀   ");
    println!("========================================\n");

    type Backend = Wgpu;
    let device = burn::backend::wgpu::WgpuDevice::default();
    
    let pipeline = StoryDataPipeline::new()?;
    println!("Loading tokenizer (GPT-2)... Success.");
    
    let config = SsmConfig {
        d_model: 256, 
        d_state: 16,
        expand: 2,
        n_heads: 4,
        mimo_rank: 1,
        use_conv: true,
        conv_kernel: 4,
    };

    let vocab_size = pipeline.tokenizer.get_vocab_size(true);
    
    // --- Real Training Data ---
    println!("Fetching TinyStories dataset from Hugging Face...");
    let full_text = pipeline.fetch_tiny_stories()?;
    println!("Dataset loaded. Tokenizing (Top 500 lines)...");
    
    let mut dataset = Vec::new();
    for text in full_text.lines().take(500) {
        if text.trim().is_empty() { continue; }
        let encoding = pipeline.tokenizer.encode(text, true).unwrap();
        let ids: Vec<usize> = encoding.get_ids().iter().map(|&x| x as usize).collect();
        if ids.len() > 1 {
            dataset.push(ids);
        }
    }
    println!("Prepared {} story samples for training.", dataset.len());

    let mut train_config = TrainingConfig::new(config.clone());
    train_config.batch_size = 8;
    
    println!("\n[Phase 1] Training on latent dynamics...");
    let model = train::<burn::backend::Autodiff<Backend>>(
        "/tmp/jepa-ssm",
        train_config,
        device.clone(),
        vocab_size,
        dataset,
    );
    
    // Convert to non-autodiff for inference
    let model_valid = JepaLanguageModel::<Backend> {
        embedding: model.embedding.valid(),
        ssm_layers: model.ssm_layers.into_iter().map(|s| s.valid()).collect(),
        output_head: model.output_head.valid(),
        d_model: model.d_model,
    };

    println!("\n[Phase 2] Generation Mode");
    let prompt = "A brave little mouse decided to climb";
    println!("\nPrompt: \x1b[36m{}\x1b[0m", prompt);

    let encoding = pipeline.tokenizer.encode(prompt, true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let prompt_ids: Vec<i32> = encoding.get_ids().iter().map(|&x| x as i32).collect();

    println!("\nGenerating story (JEPA Latent Prediction)...");
    println!("--- Story Output ---");
    
    let generated_ids = model_valid.generate_story(prompt_ids, 30, &device);
    let result = pipeline.tokenizer.decode(&generated_ids.iter().map(|&x| x as u32).collect::<Vec<_>>(), true)
        .map_err(|e| anyhow::anyhow!(e))?;

    for word in result.split_whitespace() {
        print!("{} ", word);
        std::io::Write::flush(&mut std::io::stdout())?;
        std::thread::sleep(std::time::Duration::from_millis(40));
    }

    println!("\n\n--- Done ---");
    Ok(())
}
