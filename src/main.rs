//! Binary entry point for `ssm-latent-model`.
//!
//! This crate is primarily a library. Running the binary directly will
//! display available demo commands.

fn main() {
    println!("ssm-latent-model v{}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("This is a library crate. To run demos, use one of the following:");
    println!();
    println!("  Native demos:");
    println!("    cargo run -p circle-world-demo --release");
    println!("    cargo run -p tiny-stories-jepa-demo --release");
    println!();
    println!("  WASM demos (requires trunk):");
    println!("    cd game-playing-wasm && trunk serve --release");
    println!();
    println!("  Benchmarks:");
    println!("    cargo bench");
    println!();
    println!("  Tests:");
    println!("    cargo test --all-targets --all-features");
}
