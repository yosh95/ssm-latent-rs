//! # ssm-latent-model
//!
//! A Rust-based exploration of **Latent World Models**, integrating Mamba-style
//! State Space Models (SSM) with JEPA (Joint-Embedding Predictive Architecture)
//! for efficient future state prediction in latent space.
//!
//! # Module Organization
//!
//! - [`ssm`]: Core SSM block with complex rotation dynamics and parallel scan
//! - [`latent`]: Latent world model predictor, loss functions, and stability mechanisms
//! - [`multimodal`]: Multimodal (vision + sensor + action) latent predictor
//! - [`preprocess`]: ONNX-based text embedding utilities (behind `ort` feature flag)
//!
//! # Key Features
//!
//! - **O(L log L) parallel scan** for training, **O(1) state update** for inference
//! - **JEPA-style latent prediction** with stability and curvature losses
//! - **Cross-platform**: CPU (NdArray), GPU (WGPU), and WASM backends via Burn

pub mod error;
pub mod ff_model;
pub mod forward_forward;
pub mod latent;
pub mod multimodal;
pub mod preprocess;
pub mod rtrl;
pub mod rtrl_world;
pub mod ssm;
