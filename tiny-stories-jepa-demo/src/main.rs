mod data;
mod model;
mod training;

use crate::data::{ByteLevelTokenizer, download_tiny_stories, prepare_dataset};
use crate::model::JepaLanguageModel;
use crate::training::{train, ModelConfig, TrainingConfig};
use burn::backend::Wgpu;
use burn::module::AutodiffModule;
use burn::tensor::{Int, Tensor};
use serde::Deserialize;
use ssm_latent_model::ssm::SsmConfig;
use std::fs;
use std::path::Path;

#[derive(Deserialize, Debug)]
struct FullConfig {
    model: ModelConfig,
    training: TrainingConfig,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("═══════════════════════════════════════════════════");
    println!("   🚀  Language JEPA (Byte-Level BPE)  🚀        ");
    println!("   SSM × LeJEPA — Rust-based JEPA Language Model ");
    println!("═══════════════════════════════════════════════════\n");

    // ── Load configuration ──────────────────────────────────
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let config_path = std::path::Path::new(manifest_dir).join("config.toml");
    let config_str = fs::read_to_string(config_path).expect("Failed to read config.toml");
    let mut full_config: FullConfig =
        toml::from_str(&config_str).expect("Failed to parse config.toml");

    let jepa_config = full_config.model.jepa.take().unwrap_or_default();
    let ssm_config: SsmConfig = full_config.model.clone().into();
    full_config.training.model_config = Some(ssm_config.clone());
    full_config.training.jepa_config = Some(jepa_config.clone());

    type Backend = Wgpu;
    let device = burn::backend::wgpu::WgpuDevice::default();

    // ── Initialize byte-level BPE tokenizer ─────────────────
    println!("[1/4] Initializing byte-level BPE tokenizer (o200k_base)...");
    let tokenizer = ByteLevelTokenizer::new();
    let vocab_size = tokenizer.vocab_size();
    println!("      ✓ o200k_base loaded | vocab_size = {}", vocab_size);

    // ── Download dataset ────────────────────────────────────
    println!("\n[2/4] Loading TinyStories dataset...");
    let cache_path = Path::new(manifest_dir).join("data").join("TinyStoriesV2-GPT4-train.txt");
    let full_text = download_tiny_stories(&cache_path)?;

    let dataset = prepare_dataset(
        &full_text,
        &tokenizer,
        full_config.training.dataset_samples,
        6, // minimum token length
    );
    println!("      ✓ {} samples prepared", dataset.len());

    // Show a sample encoding
    if let Some((text, ids)) = dataset.first() {
        println!("\n      Sample encoding:");
        println!("      Text:  {}", &text[..text.len().min(80)]);
        println!("      IDs:   {:?}", &ids[..ids.len().min(12)]);
        let decoded = tokenizer.decode(&ids[..ids.len().min(12)])?;
        println!("      Decoded: {}", decoded);
    }

    // ── Train ───────────────────────────────────────────────
    println!("\n[3/4] Training JEPA Language Model...");
    let model = train::<burn::backend::Autodiff<Backend>>(
        full_config.training,
        device.clone(),
        vocab_size,
        dataset,
    );

    // ── Generate ────────────────────────────────────────────
    println!("\n[4/4] Generation Mode (Latent Prediction)\n");

    // Convert to inference model (no autodiff)
    let model_valid = JepaLanguageModel::<Backend> {
        embedding: model.embedding.valid(),
        encoder: model.encoder.valid(),
        ssm_layers: model.ssm_layers.into_iter().map(|s| s.valid()).collect(),
        decoder: model.decoder.valid(),
        output_head: model.output_head.valid(),
        projections: model.projections.valid(),
        d_model: model.d_model,
        n_projections: model.n_projections,
    };

    let prompt = "Once upon a time, a small bird";
    println!("Prompt: \x1b[36m{}\x1b[0m\n", prompt);
    println!("--- Generated Story ---");
    print!("{}", prompt);

    let mut current_ids: Vec<u32> = tokenizer.encode(prompt);
    
    for _ in 0..40 {
        let seq_len = current_ids.len();
        let input_tensor = Tensor::<Backend, 1, Int>::from_ints(
            current_ids.iter().map(|&x| x as i32).collect::<Vec<i32>>().as_slice(),
            &device,
        )
        .reshape([1, seq_len]);

        let next_id_tensor = model_valid.step(input_tensor, 10);
        let next_id = next_id_tensor.into_data().as_slice::<i32>().unwrap()[0] as u32;

        let word = tokenizer.decode(&[next_id])?;
        print!("{}", word);
        std::io::Write::flush(&mut std::io::stdout())?;

        current_ids.push(next_id);
        // Stop at EOS token (o200k_base uses <|endoftext|> = token 199999)
        if next_id >= 199999 || current_ids.len() > 100 {
            break;
        }
    }

    println!("\n\n═══════════════════════════════════════════════════");
    println!("   ✅ Generation complete");
    println!("═══════════════════════════════════════════════════\n");

    Ok(())
}
