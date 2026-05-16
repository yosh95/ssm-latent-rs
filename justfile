# ─── ssm-latent-model · justfile ───────────────────────────────
# Default to showing help
default: help

# List all available commands
help:
    @just --list --unsorted

# ─── 🎮 Demos (Native, GPU default) ──────────────────────────────

# BP Baseline: SSM + JEPA with full backprop (circle world)
bp:
    cargo run -p circle-world-demo --release

# BP Baseline (CPU fallback)
bp-cpu:
    cargo run -p circle-world-demo --release --no-default-features --features ndarray

# Log anomaly detection demo (MAD + EWMA hybrid threshold)
log-anomaly:
    cargo run -p log-anomaly-demo --release

# Deterministic AI Agent demo (industrial OT)
agent:
    cargo run -p deterministic-ai-agent-demo --release

# NAB benchmark anomaly detection demo
nab:
    cargo run -p nab-demo --release

# TinyStories JEPA language model demo
jepa:
    cargo run -p tiny-stories-jepa-demo --release

# ─── 🌐 WASM Demos (requires trunk) ──────────────────────────────

# Ball Catch game (in-browser, training + inference)
wasm-ball:
    cd game-playing-wasm && trunk serve --release

# Metronome demo (in-browser, synchronize with periodic signals)
wasm-metronome:
    cd metronome-demo && trunk serve --release

# Build WebAssembly packages (requires wasm-pack)
wasm-build:
    wasm-pack build game-playing-wasm --target web --out-dir pkg -- --features wgpu
    wasm-pack build metronome-demo --target web --out-dir pkg -- --features wgpu

# ─── 🧪 Quality Assurance ────────────────────────────────────────

# Run linter (format check + clippy)
lint:
    cargo fmt --all -- --check
    cargo clippy --all-targets --all-features -- -D warnings

# Run all tests
test:
    cargo test --all-targets --all-features

# Run only core tests
test-core:
    cargo test --test core_tests

# Run only equivalence tests (parallel scan ≡ sequential step)
test-equiv:
    cargo test --test equivalence_test

# Run extended tests (edge cases, MIMO rank, vision, gradients)
test-extended:
    cargo test --test extended_tests

# Run benchmarks
bench:
    cargo bench

# Check for vulnerabilities
audit:
    cargo audit

# Fix formatting
fmt:
    cargo fmt --all

# ─── 📦 Build ────────────────────────────────────────────────────

# Build all targets (release)
build:
    cargo build --workspace --release

# Build all targets (release, CPU only)
build-cpu:
    cargo build --workspace --release --no-default-features --features ndarray,autodiff

# Install native binary
install:
    cargo install --path .