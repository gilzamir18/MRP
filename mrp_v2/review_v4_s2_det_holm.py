"""Re-adjust the 10M deterministic OR-vs-DQ contrasts (study S2) over the pre-registered
conditions only (F1, F2, F5, F6), as for the stochastic policy; F3 and F4 come from the
earlier long-training run and keep their own family (Holm over F3, F4; REVIEW_F34_10M_contrasts.csv).
No training: reads existing raw p-values and writes a new CSV.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(HERE, "..", "docs", "results_v2", "analysis")


def holm(p):
    p = list(p)
    order = sorted(range(len(p)), key=lambda i: p[i])
    m = len(p)
    adj = [0.0] * m
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj


def main():
    s2 = pd.read_csv(os.path.join(AN, "CLARIFY_S2_contrasts.csv"))
    det = s2[(s2.family == "S2 — 10M det") & (s2.cond.isin(["F1", "F2", "F5", "F6"]))].copy()
    det["p_holm_prereg"] = holm(det.p_raw)
    det["family_v4"] = "S2 10M det, Holm over F1, F2, F5, F6"
    f34 = pd.read_csv(os.path.join(AN, "REVIEW_F34_10M_contrasts.csv"))
    f34 = f34[(f34.kind == "eval_det") & (f34.v2 == "DQ")].copy()
    f34["p_holm_prereg"] = f34.p_holm
    f34["family_v4"] = "earlier 10M run, det, Holm over F3, F4 (not pre-specified)"
    cols = ["family_v4", "kind", "cond", "v1", "v2", "mean1", "mean2", "diff", "ci_lo", "ci_hi",
            "rbc", "p_raw", "p_holm_prereg"]
    out = pd.concat([det[cols], f34[cols]], ignore_index=True)
    path = os.path.join(AN, "REVIEW_v4_S2_det_holm.csv")
    out.to_csv(path, index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
