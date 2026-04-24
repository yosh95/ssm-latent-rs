# Default to showing help
default: help

# List all available commands
help:
    @just --list

# --- Native Execution (GPU by default) ---

# Run with native GPU (default features)
run:
    cargo run --release

# Run with CPU (ndarray) on native
run-cpu:
    cargo run --release --no-default-features --features ndarray

# --- Wasm / WASI (Portable) ---

# Run as WASI using wasmtime (CPU)
# Requires: rustup target add wasm32-wasi && brew install wasmtime
wasi:
    cargo build --target wasm32-wasi --no-default-features --features ndarray
    wasmtime run target/wasm32-wasi/debug/ssm-latent-model.wasm

# Build for WebBrowser (WebGPU)
# Requires: cargo install wasm-pack
wasm-web:
    wasm-pack build --target web --out-dir pkg -- --features wgpu

# --- Quality Assurance ---

# Run linter
lint:
    cargo fmt --all -- --check
    cargo clippy --all-targets --all-features -- -D warnings

# Run tests
test:
    cargo test --all-targets --all-features

# Check for vulnerabilities
audit:
    cargo audit

# Install native binary
install:
    cargo install --path .
