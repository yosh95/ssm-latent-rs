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

### Native Demo
Run the command-line demonstration script:
```bash
cargo run --release
```

The native demo consists of three parts:
1.  **Observation**: Visualize the raw signal (a noisy circular motion).
2.  **Dreaming (Training)**: The model learns the underlying laws of the world. Every 20 epochs, it runs a "mental simulation" side-by-side with the ground truth to show how its understanding improves.
3.  **Pure Imagination**: The model predicts future states without any external observations, relying solely on its internal "World Model".

### Log Anomaly Detection Demo (SenseTransformer + SSM)
This demo showcases how to detect semantic anomalies in system logs using a pre-trained SentenceTransformer and the Latent SSM model.

![Log Anomaly Demo](images/log_demo.gif)

```bash
cargo run -p log-anomaly-demo --release
```

### WASM Metronome Demo

The metronome learning demo runs entirely in the browser using WebAssembly and [Trunk](https://trunkrs.dev/). **Training and prediction are performed locally in your browser.**

![WASM Metronome Demo](images/wasm_demo.gif)

1. Add wasm32 target:
   ```bash
   rustup target add wasm32-unknown-unknown
   ```

2. Install trunk:
   ```bash
   cargo install trunk
   ```
3. Navigate to the demo directory:
   ```bash
   cd wasm-demo
   ```
4. Run the development server:
   ```bash
   trunk serve --release
   ```
5. Open your browser to `http://localhost:8080`.

The WASM demo consists of:
- **Blue Line**: Reality (The physics-driven ground truth).
- **Orange Line**: Imagination (The model's prediction).
- **In-Browser Training**: The model learns the metronome's dynamics in real-time within the browser. After about 100 epochs, the "Imagination" will sync smoothly with "Reality".

## Testing

Run equivalence tests (Parallel vs Sequential):
```bash
cargo test
```

## Configuration

The project uses a `config.toml` file to control model architecture and training hyperparameters:

```toml
[model]
d_model = 64       # Latent dimension
d_state = 16       # State dimension for SSM
expand = 2         # Expansion factor for inner dimension
n_heads = 4        # Number of heads (MIMO)
mimo_rank = 1      # MIMO rank (d_head must be divisible by this)
use_conv = true    # Enable conv1d before SSM
conv_kernel = 4    # Kernel size for conv1d

[train]
learning_rate = 1e-3
epochs = 120
batch_size = 4
seq_len = 32
stability_weight = 1.0   # Weight for stability regularizer
curvature_weight = 0.5   # Weight for temporal straightening
recon_weight = 1.0       # Weight for reconstruction loss

[anomaly]
k_mad = 3.0              # MAD sensitivity for anomaly threshold
alpha_ewma = 0.1         # EWMA smoothing factor
k_ewma = 3.0             # EWMA sensitivity for anomaly threshold
```

### Parameter Descriptions

- **`[model]`**: Architecture of the SSM block. `d_model` is the core latent dimension; `d_state` controls the state size of the SSM dynamics.
- **`[train]`**: Training hyperparameters. The `stability_weight` and `curvature_weight` control the strength of the representation collapse prevention and temporal straightening regularizers, respectively.
- **`[anomaly]`**: (Log anomaly demo only) Parameters for the hybrid adaptive anomaly threshold using MAD calibration and EWMA online tracking.

## References

- Lahoti, A., Li, K. Y., Chen, B., Wang, C., Bick, A., Kolter, J. Z., Dao, T., & Gu, A. (2026). Mamba-3: Improved Sequence Modeling using State Space Principles. *arXiv preprint arXiv:2603.15569*. [https://arxiv.org/abs/2603.15569](https://arxiv.org/abs/2603.15569)
- Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., & Balestriero, R. (2026). LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels. *arXiv preprint arXiv:2603.19312*. [https://arxiv.org/abs/2603.19312](https://arxiv.org/abs/2603.19312)

## License

MIT License
