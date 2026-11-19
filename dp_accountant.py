#!/usr/bin/env python3
"""
Reproducible per-client DP-SGD privacy accounting for FL-MMT (Section 4.6).

Mechanism : Poisson-subsampled Gaussian mechanism (DP-SGD, per-example clipping C).
Accountant: Renyi DP (Mironov 2017; Mironov, Talwar & Zhang 2019; Wang, Balle &
            Kasiviswanathan 2019) via Google's `dp_accounting` library
            (pip install dp-accounting), converted to (epsilon, delta)-DP.
Unit      : example-level DP for each client's local training set, assuming the
            adversary observes every per-step update of every round (no
            amplification from client sampling; all K=8 clients take part in all R rounds).

Usage:
    python dp_accountant.py                     # default FL-MMT configuration
    python dp_accountant.py --train-frac 0.70 --batch 32 --rounds 120 --epochs 3 \
                            --eps 2.1 --delta 1e-5 --out dp_accounting.csv
Edit CLIENTS below (or pass --clients-csv with columns client,n_total) to change sizes.
"""
import argparse, csv, math
import dp_accounting
from dp_accounting import rdp

# Client sample counts from Table 3 (n_total per client, before the 70/15/15 split).
CLIENTS = {1: 4812, 2: 3961, 3: 3214, 4: 2788, 5: 4103, 6: 3587, 7: 2311, 8: 4335}
ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 513))


def epsilon(sigma, q, steps, delta):
    acct = rdp.RdpAccountant(ORDERS)
    acct.compose(dp_accounting.PoissonSampledDpEvent(q, dp_accounting.GaussianDpEvent(sigma)), steps)
    return acct.get_epsilon(delta)


def calibrate_sigma(q, steps, delta, eps_target, lo=0.3, hi=50.0, tol=1e-4):
    """Smallest sigma (bisection) with epsilon(sigma) <= eps_target."""
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if epsilon(mid, q, steps, delta) > eps_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eps", type=float, default=2.1)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--clients-csv", default=None)
    ap.add_argument("--out", default="dp_accounting.csv")
    a = ap.parse_args()

    clients = CLIENTS
    if a.clients_csv:
        with open(a.clients_csv) as f:
            clients = {int(r["client"]): int(r["n_total"]) for r in csv.DictReader(f)}

    rows = []
    print(f"{'client':>6} {'n_train':>8} {'q':>8} {'steps/ep':>9} {'T_total':>8} {'sigma':>7} {'eps':>6}")
    for k, n in clients.items():
        n_train = int(round(a.train_frac * n))
        q = a.batch / n_train
        steps_per_epoch = math.ceil(n_train / a.batch)
        T = a.rounds * a.epochs * steps_per_epoch
        sigma = calibrate_sigma(q, T, a.delta, a.eps)
        eps = epsilon(sigma, q, T, a.delta)
        # Sensitivity check: what epsilon results if sigma had been calibrated for the
        # (incorrect) count T = R*E = 360 steps but 'T' steps were actually executed?
        sigma_360 = calibrate_sigma(q, a.rounds * a.epochs, a.delta, a.eps)
        eps_if_360 = epsilon(sigma_360, q, T, a.delta)
        rows.append(dict(client=k, n_total=n, n_train=n_train, q=round(q, 5),
                         steps_per_epoch=steps_per_epoch, total_steps=T,
                         sigma=round(sigma, 3), eps=round(eps, 3),
                         sigma_if_T360=round(sigma_360, 3), eps_if_T360_miscalibrated=round(eps_if_360, 1)))
        print(f"{k:>6} {n_train:>8} {q:>8.5f} {steps_per_epoch:>9} {T:>8} {sigma:>7.3f} {eps:>6.3f}")

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
