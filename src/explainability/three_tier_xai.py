"""
Three-tier explainability layer used to validate FL-MMT predictions:

  Tier 1 — Integrated Gradients (Sundararajan et al., 2017): pixel/voxel-level
           attribution over CT and histopathology inputs, validated against
           radiologist/pathologist annotation masks.
  Tier 2 — Attention flow (Abnar & Zuidema, 2020): traces information flow
           through the HCMA cross-attention stack to show which modality
           dominated a given prediction.
  Tier 3 — SHAP (Lundberg & Lee, 2017): feature-level attribution over the
           structured clinical and genomic inputs.
"""

import torch
from captum.attr import IntegratedGradients
import shap


def integrated_gradients_attribution(model, inputs, target_index, baseline=None, n_steps=50):
    """
    Tier 1: pixel/voxel-level attribution for image modalities (CT, histopathology).

    Args:
        model: forward function mapping input tensor -> logits for the task of interest
        inputs: input tensor (B, C, H, W) or (B, C, D, H, W) for CT volumes
        target_index: class index to attribute
        baseline: optional baseline tensor; defaults to zeros
    Returns:
        attributions: same shape as inputs
    """
    ig = IntegratedGradients(model)
    if baseline is None:
        baseline = torch.zeros_like(inputs)
    attributions, _ = ig.attribute(inputs, baselines=baseline, target=target_index,
                                    n_steps=n_steps, return_convergence_delta=True)
    return attributions


def attention_flow(attention_matrices):
    """
    Tier 2: computes attention rollout ("flow") across the stacked HCMA
    cross-attention layers, following Abnar & Zuidema (2020).

    Args:
        attention_matrices: list of (B, heads, T, T) tensors, one per layer,
                             collected via forward hooks on CrossModalAttentionBlock.
    Returns:
        rollout: (B, T, T) tensor representing cumulative attention flow
    """
    # Average over heads, add residual identity (accounts for skip connections),
    # then re-normalize and chain-multiply across layers.
    rollout = None
    for attn in attention_matrices:
        attn_avg = attn.mean(dim=1)  # average over heads -> (B, T, T)
        identity = torch.eye(attn_avg.size(-1), device=attn_avg.device).unsqueeze(0)
        attn_res = 0.5 * attn_avg + 0.5 * identity
        attn_res = attn_res / attn_res.sum(dim=-1, keepdim=True)

        rollout = attn_res if rollout is None else torch.bmm(attn_res, rollout)

    return rollout


def shap_attribution(predict_fn, background_data, sample_data):
    """
    Tier 3: SHAP attribution over structured clinical/genomic features.

    Args:
        predict_fn: callable mapping (N, num_features) numpy array -> (N,) predictions
        background_data: (K, num_features) reference/background samples
        sample_data: (M, num_features) samples to explain
    Returns:
        shap_values: SHAP values, shape (M, num_features)
    """
    explainer = shap.KernelExplainer(predict_fn, background_data)
    shap_values = explainer.shap_values(sample_data)
    return shap_values
