use anyhow::Result;
use std::fs;
use std::path::Path;

/// Byte-level BPE tokenizer (o200k_base — GPT-4o tokenizer).
///
/// OpenAI's `o200k_base` is a **byte-level** BPE tokenizer:
///   - Every byte sequence can be encoded (no unknown tokens)
///   - Vocabulary size: ~200,000 tokens
///   - Language-agnostic: works for any UTF-8 text
///   - No lower-casing, no unicode normalization — raw bytes only
///
/// This replaces the previous GPT-2 tokenizer (which required `hf-hub` and
/// downloaded model files). `tiktoken` is a pure-Rust implementation with
/// zero external dependencies and instant loading.
pub struct ByteLevelTokenizer {
    enc: &'static tiktoken::CoreBpe,
}

impl ByteLevelTokenizer {
    /// Create a new tokenizer using o200k_base (byte-level BPE).
    ///
    /// `o200k_base` is the encoding used by GPT-4o, o1, o3, and o4-mini.
    /// It is a byte-level BPE with ~200K tokens, supporting any UTF-8 text.
    pub fn new() -> Self {
        let enc = tiktoken::get_encoding("o200k_base")
            .expect("o200k_base encoding must be available");
        Self { enc }
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.enc.encode(text)
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.enc
            .decode_to_string(tokens)
            .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))
    }

    /// Get the vocabulary size (safe upper bound for o200k_base).
    pub fn vocab_size(&self) -> usize {
        200000
    }

}

/// Download TinyStories dataset from HuggingFace datasets via direct URL.
///
/// This avoids the `hf-hub` crate dependency (which pulls in git-lfs logic).
/// TinyStories is a collection of short stories (~2M tokens) written by GPT-4
/// to be understandable by 3-4 year olds.
///
/// URL: https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
pub fn download_tiny_stories(cache_path: &Path) -> Result<String> {
    if cache_path.exists() {
        let content = fs::read_to_string(cache_path)?;
        println!("  Loaded cached dataset ({} bytes)", content.len());
        return Ok(content);
    }

    let url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt";
    println!("  Downloading TinyStories dataset...");

    let response = ureq::get(url)
        .call()
        .map_err(|e| anyhow::anyhow!("Failed to download TinyStories: {}", e))?;

    let content = response
        .into_body()
        .read_to_string()
        .map_err(|e| anyhow::anyhow!("Failed to read response body: {}", e))?;

    // Cache to disk for future runs
    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(cache_path, &content)?;

    println!("  Downloaded and cached {} bytes", content.len());
    Ok(content)
}

/// Prepare the dataset: tokenize stories and return (text, token_ids) pairs.
pub fn prepare_dataset(
    full_text: &str,
    tokenizer: &ByteLevelTokenizer,
    max_samples: usize,
    min_length: usize,
) -> Vec<(String, Vec<u32>)> {
    let mut dataset = Vec::new();
    for line in full_text.lines().take(max_samples) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let ids = tokenizer.encode(trimmed);
        if ids.len() >= min_length {
            dataset.push((trimmed.to_string(), ids));
        }
    }
    dataset
}
