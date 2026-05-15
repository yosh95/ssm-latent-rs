use burn::tensor::Tensor;
/// Truncated Real-Time Recurrent Learning (RTRL) SSM.
///
/// This module implements a training method for the SSM that does NOT require
/// backpropagation through time (BPTT). Instead, it uses **1-step truncated RTRL**:
///
/// At each timestep t:
/// ```text
/// h_{t-1}, bx_{t-1}, conv_state  ← detach (no gradient from future)
/// h_t = f(h_{t-1}, x_t, θ)        ← forward_step with detached past
/// loss_t = MSE(y_t, target_t)      ← local loss
/// loss_t.backward()                ← gradient only through current step
/// ```
///
/// This is equivalent to Truncated RTRL with horizon 1. Gradients flow
/// through the "direct" path (∂h_t/∂θ) but NOT through ∂h_t/∂h_{t-1}.
///
/// Key properties:
/// - **O(1) memory**: no computation graph grows with sequence length
/// - **O(1) gradient computation per step**: no BPTT
/// - **Biologically plausible**: like dopamine-based learning (credit for
///   immediate prediction, not long-range consequences)
/// - **No weight transport**: gradients don't flow across SSM layer boundaries
use burn::tensor::backend::Backend;

use crate::ssm::{MultiScaleSsmBlock, MultiScaleState, SsmBlock};

/// Run one SSM step with all recurrent state detached.
///
/// This prevents gradient flow through time: the loss at step t+1 cannot
/// affect parameters through the hidden state path from step t.
///
/// Returns (output, new_state, step_loss_candidates).
pub fn ssm_step_detached<B: Backend>(
    ssm: &MultiScaleSsmBlock<B>,
    x: Tensor<B, 2>, // [batch, d_model]
    prev_state: &MultiScaleState<B>,
) -> (Tensor<B, 2>, MultiScaleState<B>) {
    let num_layers_recip = 1.0f64 / ssm.n_layers as f64;
    let mut out = x;
    let mut new_hs = Vec::with_capacity(ssm.n_layers);
    let mut new_prev_bxs = Vec::with_capacity(ssm.n_layers);
    let mut new_conv_states = Vec::with_capacity(ssm.n_layers);

    let use_conv = ssm.layers[0].conv1d.is_some();

    for (i, (layer, norm)) in ssm.layers.iter().zip(ssm.norms.iter()).enumerate() {
        let residual = out.clone();
        let normalized = norm.forward(out);

        // CRITICAL: detach all recurrent state before passing to forward_step.
        // This is what makes it "truncated RTRL" — no gradient through h_{t-1}.
        let prev_h_detached = prev_state.h[i].clone().detach();
        let prev_bx_detached = prev_state.prev_bx[i].clone().map(|bx| bx.detach());
        let conv_state_detached = if i == 0 && use_conv {
            prev_state.conv_state[i].clone().map(|cs| cs.detach())
        } else {
            None
        };

        let (y, next_h, current_bx, next_conv_state) = layer.forward_step(
            normalized,
            prev_h_detached,
            prev_bx_detached,
            conv_state_detached,
        );

        out = (y.mul_scalar(num_layers_recip) + residual).clamp(-100.0, 100.0);
        new_hs.push(next_h);
        new_prev_bxs.push(Some(current_bx));
        new_conv_states.push(next_conv_state);
    }

    (
        out,
        MultiScaleState {
            h: new_hs,
            prev_bx: new_prev_bxs,
            conv_state: new_conv_states,
        },
    )
}

/// Run one single-layer SSM step with all recurrent state detached.
///
/// Same as `ssm_step_detached` but for a single `SsmBlock` (non-multi-scale).
pub fn ssm_single_step_detached<B: Backend>(
    ssm: &SsmBlock<B>,
    x: Tensor<B, 2>,
    prev_h: Tensor<B, 4>,
    prev_bx: Option<Tensor<B, 4>>,
    conv_state: Option<Tensor<B, 3>>,
) -> (
    Tensor<B, 2>,
    Tensor<B, 4>,
    Tensor<B, 4>,
    Option<Tensor<B, 3>>,
) {
    ssm.forward_step(
        x,
        prev_h.detach(),
        prev_bx.map(|bx| bx.detach()),
        conv_state.map(|cs| cs.detach()),
    )
}

/// Accumulator for no-BP training gradients.
///
/// Since we call `.backward()` at each timestep, Burn's autodiff
/// accumulates gradients in the parameter tensors. We don't need
/// to manually sum them.
///
/// However, we need to be careful to call `optimizer.step()` only
/// after all timesteps in a batch have been processed to get the
/// full gradient estimate.
pub struct RtrlAccumulator {
    pub step_count: usize,
    pub total_loss: f32,
}

impl Default for RtrlAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl RtrlAccumulator {
    pub fn new() -> Self {
        Self {
            step_count: 0,
            total_loss: 0.0,
        }
    }

    pub fn record(&mut self, loss: f32) {
        self.step_count += 1;
        self.total_loss += loss;
    }

    pub fn avg_loss(&self) -> f32 {
        if self.step_count == 0 {
            0.0
        } else {
            self.total_loss / self.step_count as f32
        }
    }
}
