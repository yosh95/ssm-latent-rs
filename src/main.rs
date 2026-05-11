fn main() {
    eprintln!("Note: This is the library crate root. To run a demo, use one of:");
    eprintln!("  cargo run -p circle-world-demo --release");
    eprintln!("  cargo run -p log-anomaly-demo --release");
    eprintln!("  cargo run -p deterministic-ai-agent-demo --release");
    eprintln!();
    eprintln!("For WASM demos, see the game-playing-wasm/ or metronome-demo/ directories.");
    std::process::exit(1);
}
