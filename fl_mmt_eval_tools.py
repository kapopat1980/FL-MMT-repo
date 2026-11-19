#!/usr/bin/env python3
"""Evaluation helpers for FL-MMT.
  cindex-compare  paired bootstrap of delta Harrell-C between two models on the SAME patients
                  CSV: time,event,risk_a,risk_b (higher risk = shorter survival)
  stratify        metric per modality-availability pattern
                  CSV: pattern,y_true,y_score  (binary AUC per pattern, bootstrap CI)
"""
import argparse, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

def harrell_c(time, event, risk):
    t, e, r = map(np.asarray, (time, event, risk))
    num = den = 0.0
    for i in np.where(e == 1)[0]:
        m = t > t[i]
        den += m.sum(); num += (r[i] > r[m]).sum() + 0.5 * (r[i] == r[m]).sum()
    return num / den if den else np.nan

def cindex_compare(df, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(df); d = []
    for _ in range(B):
        s = df.iloc[rng.integers(0, n, n)]
        d.append(harrell_c(s.time, s.event, s.risk_a) - harrell_c(s.time, s.event, s.risk_b))
    d = np.array(d); ca, cb = harrell_c(df.time, df.event, df.risk_a), harrell_c(df.time, df.event, df.risk_b)
    return dict(C_a=ca, C_b=cb, delta=ca - cb, ci95=(np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)))

def stratify(df, B=1000, seed=0):
    rng = np.random.default_rng(seed); out = {}
    for k, g in df.groupby("pattern"):
        a = roc_auc_score(g.y_true, g.y_score); bs = []
        for _ in range(B):
            s = g.iloc[rng.integers(0, len(g), len(g))]
            if s.y_true.nunique() == 2: bs.append(roc_auc_score(s.y_true, s.y_score))
        out[k] = (len(g), a, np.percentile(bs, 2.5), np.percentile(bs, 97.5))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["cindex-compare", "stratify"]); ap.add_argument("csv")
    a = ap.parse_args(); df = pd.read_csv(a.csv)
    print(cindex_compare(df) if a.cmd == "cindex-compare" else stratify(df))
