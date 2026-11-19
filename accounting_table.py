#!/usr/bin/env python3
"""Build the Table 4 / 4b accounting from one machine-readable manifest.

Manifest CSV columns (one row per model-input unit):
  dataset, patient_id (or image-group id for LC25000), source_image_id, unit_id,
  is_provider_augmented (0/1), is_synthetic (0/1; train only), split (train/val/test),
  tasks (semicolon list, e.g. "subtype;survival")
Usage: python accounting_table.py manifest.csv > accounting.txt
"""
import sys
import pandas as pd

def main(path):
    m = pd.read_csv(path)
    assert not ((m.is_synthetic == 1) & (m.split != "train")).any(), "synthetic samples outside train"
    # leakage check: no patient/group in more than one split
    leak = m.groupby("patient_id").split.nunique()
    assert (leak == 1).all(), f"{(leak > 1).sum()} patients/groups straddle partitions"
    real = m[m.is_synthetic == 0]
    t = real.groupby(["dataset", "split"]).agg(
        patients=("patient_id", "nunique"), source_images=("source_image_id", "nunique"),
        input_units=("unit_id", "nunique"), provider_aug=("is_provider_augmented", "sum")).reset_index()
    print("== Table 4 (unique counts per dataset and split) =="); print(t.to_string(index=False))
    ex = real.assign(task=real.tasks.str.split(";")).explode("task")
    inst = ex.groupby(["task", "split"]).size().unstack(fill_value=0)
    uniq = real.groupby("split").unit_id.nunique()
    inst.loc["SUM task instances"] = inst.sum()
    inst.loc["UNIQUE units"] = uniq
    inst.loc["SURPLUS (sum - unique)"] = inst.loc["SUM task instances"] - uniq
    print("\n== Table 4b (task instances) =="); print(inst.to_string())

if __name__ == "__main__":
    main(sys.argv[1])
