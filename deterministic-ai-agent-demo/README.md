# Deterministic AI Agent (Rust)

A high-performance, deterministic AI agent implemented in Rust using the **Candle** framework. This agent is designed for industrial OT environments, providing intent classification, out-of-distribution (OOD) detection, and Named Entity Recognition (NER) without any Python runtime or ONNX dependencies.

## Key Features

- **Pure Rust Implementation**: Zero dependency on Python at runtime.
- **Embedded Model Management**: Automatically downloads and manages models (e.g., `multilingual-e5-small`) from Hugging Face Hub using `hf-hub`.
- **Candle Framework**: Utilizes Hugging Face's `candle` for efficient tensor operations and model inference.
- **Deterministic Logic**: Combines neural intent classification with a hybrid (Neural + Exact Match) engine for reliable NER.
- **Reliable OOD Detection**: Uses class centroids to reject inputs falling outside the training distribution.

## Architecture

- **Encoder**: Uses BERT-based models (`multilingual-e5-small`) to generate 384-dimensional text embeddings.
- **Classifier**: A lightweight neural network for intent classification with integrated centroid-based similarity scoring (higher similarity = in-distribution).
- **NER Extractor**: A hybrid engine combining a neural token classifier and an exact-match master data matcher.

## Usage

### Prerequisites

- Rust (latest stable version)
- OpenSSL (required for `hf-hub` to download models)

### Building

```bash
cargo build --release
```

### Running

The agent can be integrated as a library or used via a CLI. The core logic ensures that any input falling outside the trained distribution or below confidence thresholds is rejected for safety.

## Development

### Training

Training is performed directly in Rust using Candle's autograd features. The agent learns intent centroids during the training phase to enable robust OOD detection.

```rust
// Example snippet
let encoder = EmbeddingEncoder::new("intfloat/multilingual-e5-small")?;
let vector = encoder.encode("Sensor A overheated")?;
```

## License

MIT License
