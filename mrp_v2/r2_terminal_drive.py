"""
r2_terminal_drive.py — R2-C (prereg/prereg_2026-11-20.md): drive at death vs mean drive visited.

    python r2_terminal_drive.py random     # random policy, 2000 episodes per condition (seeds 50000-51999)
    python r2_terminal_drive.py trained    # trained DQ / OR agents of R2_term_diag (diag field)

Delta = D_L - Dbar (Dbar = discounted, renormalised mean of D_1..D_{L-1}), per episode that ends
in death.  Trained agents: mean over the death episodes of each seed, then over seeds; Wilcoxon
one-sample (seed-level Delta vs 0) for DQ, Holm over F1-F6.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from forage_env import policy_random  # noqa: E402
from forage_ppo import eval_diagnostics  # noqa: E402
from forage_run import FCONDS  # noqa: E402

RES = os.path.join(os.path.dirname(HERE), "docs", "results_v2")
OUT = os.path.join(RES, "analysis")
CONDS = ["F1", "F2", "F3", "F4", "F5", "F6"]


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def summarise(eps):
    d = [e for e in eps if e["event"] and e["length"] > 1]
    if not d:
        return dict(n_death=0, delta=np.nan, DL=np.nan, Dbar=np.nan, L=np.nan)
    DL = np.array([e["DL"] for e in d])
    Db = np.array([e["Dbar_disc"] for e in d])
    return dict(n_death=len(d), delta=float(np.mean(DL - Db)), DL=float(DL.mean()), Dbar=float(Db.mean()),
                L=float(np.mean([e["length"] for e in d])))


def random_policy():
    rows = []
    for c in CONDS:
        kw = dict(W=9, decay_mu=FCONDS[c]["decay_mu"])
        rng = np.random.default_rng(0)
        eps = eval_diagnostics(lambda env: policy_random(env, rng), None, FCONDS[c]["k"], kw,
                               list(range(50_000, 52_000)), terminate=True)
        s = summarise(eps)
        rows.append(dict(cond=c, k=FCONDS[c]["k"], policy="random", **s))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "R2C_terminal_drive_random.csv"), index=False)
    return df


def trained():
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "R2_term_diag", "*.json"))):
        r = json.load(open(f))
        j = r["job"]
        for pol in ("stoch", "det"):
            rows.append(dict(cond=j["cond"], variant=j["variant"], seed=j["seed"], policy=pol,
                             **summarise(r["diag"][pol])))
    sl = pd.DataFrame(rows)
    sl.to_csv(os.path.join(OUT, "R2C_terminal_drive_seed_level.csv"), index=False)
    out = []
    for pol in ("stoch", "det"):
        for v in ("DQ", "OR_fixed"):
            sub = sl[(sl.policy == pol) & (sl.variant == v)]
            ps, recs = [], []
            for c in CONDS:
                x = sub[sub.cond == c].delta.dropna().values
                p = wilcoxon(x, method="exact").pvalue if len(x) >= 5 and np.any(x != 0) else np.nan
                recs.append(dict(policy=pol, variant=v, cond=c, n_seeds=len(x), delta_mean=x.mean() if len(x) else np.nan,
                                 delta_sd=x.std(ddof=1) if len(x) > 1 else np.nan,
                                 DL=sub[sub.cond == c].DL.mean(), Dbar=sub[sub.cond == c].Dbar.mean(),
                                 deaths_per_seed=sub[sub.cond == c].n_death.mean(), p=p))
                ps.append(p)
            pv = np.array([q if np.isfinite(q) else 1.0 for q in ps])
            for rr, a in zip(recs, holm(pv)):
                rr["p_holm"] = a
            out += recs
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(OUT, "R2C_terminal_drive_trained.csv"), index=False)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    {"random": random_policy, "trained": trained}[sys.argv[1] if len(sys.argv) > 1 else "random"]()
