use anyhow::Result;
use hf_hub::api::sync::Api;
use std::fs;
use tokenizers::Tokenizer;

pub struct StoryDataPipeline {
    pub tokenizer: Tokenizer,
}

impl StoryDataPipeline {
    pub fn new() -> Result<Self> {
        let api = Api::new()?;

        // Use a standard tokenizer from HF Hub
        let repo = api.model("gpt2".to_string());
        let tokenizer_path = repo.get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self { tokenizer })
    }

    /// Fetch TinyStories dataset using hf-hub
    #[allow(dead_code)]
    pub fn fetch_tiny_stories(&self) -> Result<String> {
        let api = Api::new()?;
        let repo = api.dataset("roneneldan/TinyStories".to_string());
        let train_file = repo.get("TinyStoriesV2-GPT4-train.txt")?;

        let content = fs::read_to_string(train_file)?;
        Ok(content)
    }
}
