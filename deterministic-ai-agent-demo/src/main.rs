use anyhow::{Context, Result};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::tensor::Tensor;
use deterministic_ai_agent_demo::AgentEngine;
use deterministic_ai_agent_demo::encoder::EmbeddingEncoder;
use deterministic_ai_agent_demo::model::{IntentClassifierConfig, NERClassifierConfig};
use deterministic_ai_agent_demo::ner::NERExtractor;
use deterministic_ai_agent_demo::train::Trainer;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct TrainingItem {
    input: String,
    intent_id: u32,
    #[serde(rename = "parameters")]
    _parameters: HashMap<String, serde_json::Value>,
}

fn main() -> Result<()> {
    type Backend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();

    let model_dir = "models";
    fs::create_dir_all(model_dir)?;

    // 1. Initialize Encoder
    println!("--- Step 1: Initialize Encoder ---");
    let encoder = std::sync::Arc::new(EmbeddingEncoder::<Backend>::new(
        "intfloat/multilingual-e5-small",
        device.clone(),
    )?);
    let dim = 384;

    // 2. Load Data
    println!("\n--- Step 2: Load Training Data ---");
    let data_path = "data/sample_data.json";
    if !Path::new(data_path).exists() {
        fs::create_dir_all("data")?;
        fs::write(
            data_path,
            r#"[
                {"input": "Warning: The Motor_B on line 5 show excessive vibration", "intent_id": 0, "parameters": {"device": "Motor_B"}},
                {"input": "Conveyor_A のベルト異常振動を検出。", "intent_id": 0, "parameters": {"device": "Conveyor_A"}},
                {"input": "Temperature of Motor_A is too high", "intent_id": 1, "parameters": {"device": "Motor_A"}},
                {"input": "Conveyor_B stopped unexpectedly", "intent_id": 0, "parameters": {"device": "Conveyor_B"}}
            ]"#,
        )?;
    }
    let content =
        fs::read_to_string(data_path).with_context(|| format!("Failed to read {}", data_path))?;
    let items: Vec<TrainingItem> = serde_json::from_str(&content)?;
    println!("Loaded {} training items.", items.len());

    // 3. Train Intent Classifier
    println!("\n--- Step 3: Training Intent Classifier ---");
    let mut intent_embeddings = Vec::new();
    let mut intent_labels_vec = Vec::new();
    for item in &items {
        // encoder.encode returns [Dim], make it [1, Dim] for stack
        intent_embeddings.push(encoder.encode(&item.input)?.unsqueeze());
        intent_labels_vec.push(item.intent_id);
    }

    let train_embeddings = Tensor::<Backend, 2>::cat(intent_embeddings, 0);
    let train_labels = Tensor::<Backend, 1>::from_data(
        intent_labels_vec
            .iter()
            .map(|&i| i as f32)
            .collect::<Vec<_>>()
            .as_slice(),
        &device,
    );

    let num_intents = (intent_labels_vec.iter().max().unwrap_or(&0) + 1) as usize;
    let intent_config = IntentClassifierConfig::new(dim, num_intents);
    let intent_classifier = intent_config.init::<Backend>(&device);

    let trainer = Trainer::new();
    let mut intent_classifier = trainer.train_intent::<Backend>(
        intent_classifier,
        train_embeddings.clone(),
        train_labels,
        200,
        1e-3,
    );

    let centroids = trainer.calculate_centroids::<Backend>(
        train_embeddings,
        &intent_labels_vec,
        num_intents,
        &device,
    );
    intent_classifier.set_centroids(centroids);
    println!("Intent Training completed.");

    // 4. Train NER Classifier
    println!("\n--- Step 4: Training NER Classifier ---");
    let mut ner_embeddings_list = Vec::new();
    for item in &items {
        // [Seq_Len, Dim]
        let states = encoder.get_hidden_states(&item.input)?;
        ner_embeddings_list.push(states);
    }
    
    // Simplification: use the first item's sequence length to stack for demo
    // In real NER we'd handle variable lengths
    let first_seq_len = ner_embeddings_list[0].dims()[0];
    let mut truncated_ner_embs = Vec::new();
    for emb in ner_embeddings_list {
        if emb.dims()[0] >= first_seq_len {
            // [Seq_Len, Dim] -> [1, Seq_Len, Dim]
            truncated_ner_embs.push(emb.slice([0..first_seq_len]).unsqueeze());
        }
    }
    let train_ner_embeddings = Tensor::<Backend, 3>::cat(truncated_ner_embs, 0);
    let [n_ner, s_ner, _] = train_ner_embeddings.dims();
    let ner_labels = Tensor::<Backend, 2>::zeros([n_ner, s_ner], &device); // Dummy labels

    let ner_config = NERClassifierConfig::new(dim, 3);
    let ner_classifier = ner_config.init::<Backend>(&device);
    let _ner_classifier = trainer.train_ner::<Backend>(
        ner_classifier,
        train_ner_embeddings,
        ner_labels,
        50,
        1e-3,
    );
    println!("NER Training completed.");

    // 5. Initialize Engine and Run Inference
    println!("\n--- Step 5: Initializing Agent Engine ---");
    let mut ner_extractor = NERExtractor::new(dim, 0.8)?;
    ner_extractor.register_device("Motor_A", &encoder)?;
    ner_extractor.register_device("Motor_B", &encoder)?;
    ner_extractor.register_device("Conveyor_A", &encoder)?;
    ner_extractor.register_device("Conveyor_B", &encoder)?;

    // Use Inner Backend for inference engine to avoid overhead if needed, 
    // but here we can just keep using Backend.
    let engine = AgentEngine::new(encoder, intent_classifier, ner_extractor, None)?;

    // 6. Final Demo Inference
    let test_inputs = vec![
        "Warning: The Motor_B on line 5 show excessive vibration",
        "Conveyor_A のベルト異常振動を検出。",
        "Someone is eating an apple in the cafeteria", // OOD example
    ];

    println!("\n--- Final Demo ---");
    for input in test_inputs {
        let result = engine.run_step(input)?;
        println!("\nInput: '{}'", input);
        println!(
            "  Intent ID: {}, Confidence: {:.4}, OOD Score (Sim): {:.4}",
            result.intent_id, result.confidence, result.in_dist_similarity
        );
        println!("  Status: {}", result.status);
        if !result.parameters.is_empty() {
            println!("  Entities: {:?}", result.parameters);
        }
    }

    Ok(())
}
