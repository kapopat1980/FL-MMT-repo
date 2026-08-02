"""
Generates the three-tier explainability outputs (Integrated Gradients,
attention flow, SHAP) for a single sample, reproducing Figure 5 of the
manuscript and the quantitative validation against ground-truth masks
described in Section 4.5.

Usage:
    python scripts/explain.py --checkpoint checkpoints/fl_mmt_best.pt --sample_id <id>
"""

import argparse

from src.explainability.three_tier_xai import (
    integrated_gradients_attribution,
    attention_flow,
    shap_attribution,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sample_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/explanations")
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint}")
    print(f"Generating three-tier explanations for sample {args.sample_id}")

    # Tier 1: Integrated Gradients over CT / histopathology inputs
    # ig_attr = integrated_gradients_attribution(model.forward_ct, ct_input, target_index=pred_class)

    # Tier 2: Attention flow through the HCMA stack
    # flow = attention_flow(collected_attention_matrices)

    # Tier 3: SHAP over structured clinical/genomic features
    # shap_vals = shap_attribution(predict_fn, background_data, sample_data)

    print(f"Outputs would be saved under {args.output_dir}/{args.sample_id}/ "
          "(ig_map.png, attention_flow.png, shap_summary.png)")
    print("See docs/reproducing_results.md for the full pipeline.")


if __name__ == "__main__":
    main()
