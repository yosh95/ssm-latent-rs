# Edge Mamba: Framework-Free Implementation

This is a pure Rust implementation of Mamba-3 architecture focusing on **manual backpropagation** and **ONNX compatibility**.

## Features

- **No Deep Learning Framework**: Built only with `ndarray` and `rayon`. No Burn, PyTorch, or TensorFlow dependency.
- **Manual Backpropagation**: Fully implemented `backward` pass for the Selective Scan kernel.
- **Complex Numbers as Pairs**: Real and Imaginary parts are handled as separate tensors to ensure compatibility with standard ONNX opset (which lacks native complex support).
- **Parallelized**: Uses Rayon to parallelize channel-wise computations.

## Architecture: Complex SSM via Real Pairs

The state $h$ is updated as:
$$h_{re} = (A_{re} \cdot h_{re} - A_{im} \cdot h_{im}) + B_{re} \cdot u$$
$$h_{im} = (A_{re} \cdot h_{im} + A_{im} \cdot h_{re}) + B_{im} \cdot u$$

This allows the model to be exported to any inference engine that supports basic floating-point arithmetic.

## Quick Start

```bash
cargo run --release
```

## Running Tests
```bash
cargo test
```
