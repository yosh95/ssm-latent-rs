# NAB Anomaly Detection Demo

This directory contains a demo of using the SSM-Latent world model for anomaly detection on the Numenta Anomaly Benchmark (NAB) dataset.

## Structure

- `src/main.rs`: The main demo program that loads NAB data, trains a model, and detects anomalies.
- `data/`: The NAB dataset (CSV files).
- `Cargo.toml`: Package configuration.

## How it works

1.  **Data Loading**: Loads a standard NAB CSV file (e.g., `machine_temperature_system_failure.csv`).
2.  **Normalization**: Normalizes the values to [0, 1].
3.  **Probationary Training**: The first 15% of the data is used as a "probationary period" where the model learns the normal patterns of the time series.
4.  **SSM-Latent Model**: USes a State Space Model (SSM) within a Latent World Model framework to predict the next steps in the sequence.
5.  **Anomaly Detection**: Reconstruction error (actual vs predicted/reconstructed value) is used as an anomaly score.
6.  **Adaptive Threshold**: Combines Median Absolute Deviation (MAD) for baseline calibration and Exponential Weighted Moving Average (EWMA) for online threshold tracking.

## Data Acquisition

To perform a full benchmark, the NAB dataset is required. You can obtain it from the official Numenta repository:

1. **Clone the NAB repository** (anywhere outside this project):
   ```bash
   git clone https://github.com/numenta/NAB.git
   ```
2. **Copy the data and labels**:
   Copy the `data` and `labels` directories into this `nab-demo/` directory:
   ```bash
   cp -r /path/to/NAB/data ./nab-demo/
   ```

Note: A subset of official data may already be included in this directory for immediate testing.

## Running the Demo

To run the demo:

```bash
cargo run --bin nab-anomaly-demo
```

## Evaluation

The demo outputs detected anomalies to the console. In a full NAB evaluation, these would be scored against `labels/combined_labels.json` (not provided in this repository but part of the official NAB benchmark) using the scoring window methodology.

The SSM-Latent model is particularly effective for NAB because it can capture multi-scale temporal dependencies, making it robust to seasonal shifts while remaining sensitive to sudden structural breaks.
