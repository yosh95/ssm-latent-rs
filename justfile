# ─── ssm-latent-rs Justfile ───────────────────────────────────────────────
# Run `just` or `just --list` to see all available recipes.

default: list

# ─── Development ──────────────────────────────────────────────────────────

# Run all checks (format, lint, test) in sequence
@check: fmt-check clippy test

# Format all source code
@fmt:
    cargo fmt --all

# Check formatting (CI-friendly, exits non-zero if unformatted)
@fmt-check:
    cargo fmt --all -- --check

# Run clippy with strict lints
@clippy:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

# Run clippy with auto-fix (allow dirty working tree)
@clippy-fix:
    cargo clippy --workspace --all-targets --all-features --fix --allow-dirty -- -D warnings

# Run all tests
@test:
    cargo test --workspace --all-targets --all-features

# Run tests with output (no capture)
@test-verbose:
    cargo test --workspace --all-targets --all-features -- --nocapture

# Run only unit tests (library only)
@test-unit:
    cargo test --workspace --lib --all-features

# Run only integration tests
@test-integration:
    cargo test --workspace --test '*' --all-features

# Run a specific test by name (e.g., `just test-filter equivalence`)
@test-filter filter:
    cargo test --workspace --all-targets --all-features -- {{filter}}

# ─── SSM-specific Tests ────────────────────────────────────────────────────

# Run only core tests
@test-core:
    cargo test --workspace --test core_tests

# Run only equivalence tests (parallel scan ≡ sequential step)
@test-equiv:
    cargo test --workspace --test equivalence_test

# Run extended tests (edge cases, MIMO rank, vision, gradients)
@test-extended:
    cargo test --workspace --test extended_tests

# ─── Build & Install ───────────────────────────────────────────────────────

# Debug build (full workspace)
@build:
    cargo build --workspace

# Release build (optimized, full workspace)
@build-release:
    cargo build --workspace --release

# Release build (CPU only)
@build-cpu:
    cargo build --workspace --release --no-default-features --features ndarray,autodiff

# Install the native binary locally
@install: build-release
    cargo install --path .

# ─── Run ──────────────────────────────────────────────────────────────────

# Run the main application
@run *args:
    cargo run -- {{args}}

# Run with release optimizations
@run-release *args:
    cargo run --release -- {{args}}

# ─── 🎮 Demos (Native, GPU default) ───────────────────────────────────────

# Circle Baseline: SSM + JEPA with full backprop (circle world)
@circle:
    cargo run -p circle-world-demo --release

# Circle Baseline (CPU fallback)
@circle-cpu:
    cargo run -p circle-world-demo --release --no-default-features --features ndarray

# Log anomaly detection demo (MAD + EWMA hybrid threshold)
@log-anomaly:
    cargo run -p log-anomaly-demo --release

# Deterministic AI Agent demo (industrial OT)
@agent:
    cargo run -p deterministic-ai-agent-demo --release

# NAB benchmark anomaly detection demo
@nab:
    cargo run -p nab-demo --release

# TinyStories JEPA language model demo
@jepa:
    cargo run -p tiny-stories-jepa-demo --release

# ─── 🌐 WASM Demos (requires trunk) ────────────────────────────────────────

# Ball Catch game (in-browser, training + inference)
@wasm-ball:
    cd game-playing-wasm && trunk serve --release

# Metronome demo (in-browser, synchronize with periodic signals)
@wasm-metronome:
    cd metronome-demo && trunk serve --release

# Build WebAssembly packages (requires wasm-pack)
@wasm-build:
    wasm-pack build game-playing-wasm --target web --out-dir pkg -- --features wgpu
    wasm-pack build metronome-demo --target web --out-dir pkg -- --features wgpu

# ─── Benchmarks ───────────────────────────────────────────────────────────

# Run benchmarks
@bench:
    cargo bench

# ─── Documentation ────────────────────────────────────────────────────────

# Build and open documentation
@docs:
    cargo doc --workspace --open --no-deps

# Build documentation without opening
@docs-build:
    cargo doc --workspace --no-deps

# ─── Security & Auditing ─────────────────────────────────────────────────

# Run cargo-audit (requires `cargo install cargo-audit`)
@audit:
    cargo audit

# Run cargo-deny (requires `cargo install cargo-deny`)
@deny:
    cargo deny check

# Update dependencies
@update:
    cargo update --workspace

# Check for outdated dependencies (requires `cargo install cargo-outdated`)
@outdated:
    cargo outdated --workspace

# ─── Cleanup ──────────────────────────────────────────────────────────────

# Remove build artifacts
@clean:
    cargo clean

# Full clean including target directory
@clean-all: clean
    rm -rf target

# ─── CI Pipeline (run before pushing) ─────────────────────────────────────

# Full CI check (format, lint, test, build)
@ci: fmt-check clippy test build-release

# ─── Utility ─────────────────────────────────────────────────────────────

# List all available recipes
@list:
    @just --list
