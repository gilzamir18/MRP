"""
r2_aggregate.py — R2-F (prereg/prereg_2026-11-20.md): one test across conditions per main claim.

Statistic T = sum_c mean(d_c) / sd(d_c), d_c = seed-level paired differences in condition c.
Null: sign flips of d within each condition (conditions are independent training runs).
1e5 Monte-Carlo permutations, two-sided.  Output: R2F_aggregate_tests.csv.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(os.path.dirname(HERE), "docs")
V2, V3 = os.path.join(DOCS, "results_v2"), os.path.join(DOCS, "results_v3_mrp")
N_PERM = 100_000


def strat_test(diffs, rng):
    diffs = [np.asarray(d, float) for d in diffs]
    sds = [d.std(ddof=1) for d in diffs]
    T = sum(d.mean() / s for d, s in zip(diffs, sds))
    null = np.zeros(N_PERM)
    for d, s in zip(diffs, sds):
        sg = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
        null += (sg * d).mean(1) / s       # sd is invariant to sign flips
    p = (1 + np.sum(np.abs(null) >= abs(T) - 1e-12)) / (N_PERM + 1)
    return T, p


def forage(study_tag, conds, a, b, policy="eval", metric="frac_viable"):
    """seed-level metric of variant a and b from forage JSONs; study_tag = list of (study, tag, conds)."""
    out = {}
    for study, tag, cs in study_tag:
        for c in cs:
            if c not in conds:
                continue
            vals = {}
            for v in (a, b):
                for f in glob.glob(os.path.join(V2, study, f"{c}__{v}__{tag}__s*.json")):
                    r = json.load(open(f))
                    vals.setdefault(v, {})[r["job"]["seed"]] = np.mean([e[metric] for e in r[policy]])
            seeds = sorted(set(vals.get(a, {})) & set(vals.get(b, {})))
            out[c] = np.array([vals[a][s] - vals[b][s] for s in seeds])
    return out


def mrp(csv, tag, conds, a, b, metric):
    df = pd.read_csv(csv)
    if "tag" in df.columns:
        df = df[df.tag == tag]
    out = {}
    for c in conds:
        x = df[(df.cond == c) & (df.variant == a)].set_index("seed")[metric]
        y = df[(df.cond == c) & (df.variant == b)].set_index("seed")[metric]
        s = sorted(set(x.index) & set(y.index))
        out[c] = (x.loc[s] - y.loc[s]).values
    return out


def main():
    rng = np.random.default_rng(2026_11_20)
    claims = []
    C38 = ["C3", "C4", "C5", "C6", "C7", "C8"]
    claims.append(("MRP term: OR - Alive (RMST)", mrp(os.path.join(V3, "analysis", "L_seed_level.csv"),
                                                    "shared", C38, "OR_fixed", "Alive", "rmst")))
    claims.append(("MRP noterm: OR - DQ (viable fraction)", mrp(os.path.join(V3, "analysis", "NT_seed_level.csv"),
                                                              "shared_noterm", C38, "OR_fixed", "DQ", "frac_viable")))
    F36 = ["F3", "F4", "F5", "F6"]
    base = [("F_noterm", "noterm", ["F3", "F4"]), ("F_ksweep", "noterm", ["F5", "F6"])]
    for pol in ("eval", "eval_det"):
        claims.append((f"forage noterm 3M: OR - DQ ({pol})", forage(base, F36, "OR_fixed", "DQ", pol)))
    for tag, lab in (("noterm", "replication"), ("noterm_nr", "return-normalised")):
        if glob.glob(os.path.join(V2, "R2_nt_norm", f"*__{tag}__*.json")):
            for pol in ("eval", "eval_det"):
                claims.append((f"forage noterm 3M {lab}: OR - DQ ({pol})",
                               forage([("R2_nt_norm", tag, F36)], F36, "OR_fixed", "DQ", pol)))
    rows = []
    for name, d in claims:
        d = {c: x for c, x in d.items() if len(x) >= 2}
        if not d:
            continue
        T, p = strat_test(list(d.values()), rng)
        rows.append(dict(claim=name, conditions=",".join(d), n_seeds=",".join(str(len(x)) for x in d.values()),
                         mean_diff_by_cond=",".join(f"{x.mean():+.4g}" for x in d.values()), T=T, p_perm=p))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(V2, "analysis", "R2F_aggregate_tests.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
