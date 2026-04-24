# ssm-latent-model

Rust implementation of a latent state predictor leveraging State Space Models (SSM).

![World Model Demo](images/world_model.gif)

## Features

- **SSM Dynamics**: Efficient sequence modeling using state space principles.
- **Latent Prediction**: Predicts future states in an embedding space.
- **Stability Regularizer**: Prevents representation collapse during training without contrastive samples.
- **Rust + Burn**: High-performance implementation supporting multiple backends (WGPU, NdArray, LibTorch, etc.).

## Installation

```bash
git clone <repository-url>
cd ssm-latent-model
cargo build
```

## Usage

Run the demonstration script:

```bash
cargo run --release
```

The demo consists of three parts:
1.  **Observation**: Visualize the raw signal (a noisy circular motion).
2.  **Dreaming (Training)**: The model learns the underlying laws of the world. Every 20 epochs, it runs a "mental simulation" side-by-side with the ground truth to show how its understanding improves.
3.  **Pure Imagination**: The model predicts future states without any external observations, relying solely on its internal "World Model".

## Testing

Run equivalence tests (Parallel vs Sequential):
```bash
cargo test
```

## License

MIT License
