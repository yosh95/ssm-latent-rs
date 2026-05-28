//! Preprocessing utilities for embedding text data.
//!
//! This module provides the \[`LogEmbedder`\] struct (behind the `ort` feature flag)
//! for generating sentence-level embeddings using ONNX Runtime, and shared
//! utility functions for projection normalization.

#[cfg(feature = "ort")]
use anyhow::{Context, Result};
use burn::tensor::{Tensor, backend::Backend};

#[cfg(feature = "ort")]
use ndarray::{Array2, Axis};
#[cfg(feature = "ort")]
use ort::session::Session;
#[cfg(feature = "ort")]
use ort::value::Tensor as OrtTensor;
#[cfg(feature = "ort")]
use tokenizers::Tokenizer;

#[cfg(feature = "ort")]
use std::sync::{Arc, Mutex};

/// Text embedder using ONNX Runtime and HuggingFace tokenizers.
///
/// Loads a SentenceTransformer-compatible ONNX model and generates
/// sentence-level embeddings via mean pooling over token representations.
/// This is primarily used by the `log-anomaly-demo` for semantic log analysis.
///
/// # Feature Flag
/// Only available when the `ort` feature is enabled.
#[cfg(feature = "ort")]
#[derive(Clone)]
pub struct LogEmbedder {
    session: Arc<Mutex<Session>>,
    tokenizer: Tokenizer,
}

#[cfg(feature = "ort")]
impl LogEmbedder {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
            anyhow::anyhow!("Failed to load tokenizer from {}: {}", tokenizer_path, e)
        })?;

        let session = Session::builder()?
            .commit_from_file(model_path)
            .with_context(|| "Failed to create ORT session")?;

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer,
        })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();
        let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();
        let seq = ids.len();

        let input_ids = Array2::from_shape_vec((1, seq), ids)?;
        let attention_mask = Array2::from_shape_vec((1, seq), mask)?;
        let token_type_ids = Array2::from_shape_vec((1, seq), type_ids)?;

        let inputs = ort::inputs![
            "input_ids" => OrtTensor::from_array(input_ids)?,
            "attention_mask" => OrtTensor::from_array(attention_mask)?,
            "token_type_ids" => OrtTensor::from_array(token_type_ids)?,
        ];

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock session"))?;
        let outputs = session.run(inputs)?;
        let embeddings = outputs[0].try_extract_array::<f32>()?;

        let mean = embeddings.mean_axis(Axis(1)).context("Pooling failed")?;
        Ok(mean.to_owned().into_raw_vec_and_offset().0)
    }
}

/// Normalize stability projections matrix column-wise to unit length.
///
/// Each column of the projection matrix is divided by its L2 norm,
/// ensuring that the random projections are properly scaled for
/// the [`stability_loss`](crate::latent::stability_loss) computation.
///
/// # Arguments
/// * `projections` - Matrix of shape `[d_model, n_projections]`
///
/// # Returns
/// Column-normalized matrix of the same shape.
pub fn normalize_projections<B: Backend>(projections: Tensor<B, 2>) -> Tensor<B, 2> {
    let norm = projections.clone().powf_scalar(2.0).sum_dim(0).sqrt() + 1e-6;
    projections / norm
}
