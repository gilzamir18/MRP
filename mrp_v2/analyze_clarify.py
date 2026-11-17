"""
analyze_clarify.py — analysis of the clarification studies pre-registered in
docs/prereg_2026-11-16.md (S1 sign controls, S2 10M k-sweep, S3 30M F4, S4 F6 grid,
S5 deterministic evaluation).  No adaptive OR anywhere.

Holm is applied across the conditions of each family (as pre-registered).
Usage: MRP_RES=.../docs/results_v2 python analyze_clarify.py
Output: docs/results_v2/analysis/CLARIFY_*.csv and CLARIFY_RESULTS.md
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze_nt import paired_test, holm  # noqa: E402
from forage_env import run_reference, policy_random, GreedyPolicy  # noqa: E402
from forage_run import FCONDS, EVAL_SEEDS, Z_GRID, B_GRID  # noqa: E402
from forage_grid_sign import transitions  # noqa: E402
from forage_env import or_per_variable  # noqa: E402

RES = os.environ.get("MRP_RES", os.path.join(os.path.dirname(HERE), "docs", "results_v2"))
OUT = os.path.join(RES, "analysis")
TAU = 500
ALLC = ["F1", "F2", "F3", "F4", "F5", "F6"]


def load(study, tag=None):
    rows = []
    for p in glob.glob(os.path.join(RES, study, "*.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        j = r["job"]
        if tag is not None and j["tag"] != tag:
            continue
        for kind in ("eval", "eval_det"):
            ev = r[kind]
            rows.append(dict(study=study, cond=j["cond"], variant=j["variant"], seed=j["seed"],
                             kind=kind, steps=j.get("timesteps", 3_000_000),
                             rmst=float(np.mean([min(e["length"], TAU) for e in ev])),
                             in_zone=float(np.mean([e["frac_in_zone"] for e in ev])),
                             over=float(np.mean([e["over_share"] for e in ev]))))
    return pd.DataFrame(rows)


def refs():
    rows = []
    for c in ALLC:
        cc = FCONDS[c]
        kw = dict(W=9, decay_mu=cc["decay_mu"])
        for name, pol in (("random", policy_random), ("greedy", GreedyPolicy())):
            o = pd.DataFrame(run_reference(pol, cc["k"], kw, EVAL_SEEDS, terminate=True))
            rows.append(dict(cond=c, policy=name, rmst=float(np.minimum(o.length, TAU).mean())))
    return pd.DataFrame(rows).pivot(index="cond", columns="policy", values="rmst")


def contrast(S, pairs, conds, kind="eval", family=""):
    rows = []
    for c in conds:
        sub = S[(S.cond == c) & (S.kind == kind)]
        for v1, v2 in pairs:
            a = sub[sub.variant == v1].set_index("seed").rmst
            b = sub[sub.variant == v2].set_index("seed").rmst
            com = a.index.intersection(b.index)
            if len(com) < 3:
                continue
            t = paired_test(a.loc[com].values, b.loc[com].values)
            rows.append(dict(family=family, kind=kind, cond=c, v1=v1, v2=v2,
                             mean1=a.loc[com].mean(), sd1=a.loc[com].std(), mean2=b.loc[com].mean(),
                             sd2=b.loc[com].std(), wins1=int((a.loc[com] > b.loc[com]).sum()), **t))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_holm"] = np.nan
        for (_, _), g in df.groupby(["v1", "v2"]):
            df.loc[g.index, "p_holm"] = holm(g.p_raw.values)
    return df


def fmt(df, cols=("cond", "v1", "v2", "n", "mean1", "mean2", "diff", "wins1", "p_raw", "p_holm")):
    if df.empty:
        return "(no data yet)\n"
    d = df.copy()
    d["diff"] = d.mean1 - d.mean2
    d = d[[c for c in cols if c in d]]
    return d.round(3).to_markdown(index=False) + "\n"


def main():
    md = ["# Clarification studies (pre-registered 2026-11-16)\n"]
    R = refs()
    R.to_csv(os.path.join(OUT, "CLARIFY_refs.csv"))
    md += ["## References (RMST, τ = 500)\n", R.round(1).to_markdown() + "\n"]

    base3 = pd.concat([load("F_term"), load("F_ksweep", tag="term"), load("F_sign2")], ignore_index=True)
    base3 = base3[base3.variant.isin(["OR_fixed", "DQ", "Alive", "DQ_c001", "DQ_c01", "OR_m1", "DR", "DR_alive"])]
    long10 = load("F_long", tag="term10M")
    long10 = long10[~long10.variant.str.startswith("OR_adapt")]
    long30 = load("F_long30")

    # means table (3M)
    mt = base3[base3.kind == "eval"].groupby(["cond", "variant"]).rmst.agg(["mean", "std", "count"]).round(1)
    mt.to_csv(os.path.join(OUT, "CLARIFY_S1_means.csv"))
    md += ["## S1 — sign controls (3M, termination): RMST mean ± SD (n)\n",
           mt.reset_index().to_markdown(index=False) + "\n"]
    s1 = pd.concat([contrast(base3, [("DQ_c001", "DQ")], ALLC, family="a"),
                    contrast(base3, [("OR_fixed", "DQ_c001")], ALLC, family="b"),
                    contrast(base3, [("OR_m1", "OR_fixed")], ALLC, family="c"),
                    contrast(base3, [("DQ_c01", "Alive")], ALLC, family="d (descriptive)"),
                    contrast(base3, [("OR_fixed", "DQ")], ALLC, family="ref: OR vs DQ")], ignore_index=True)
    s1.to_csv(os.path.join(OUT, "CLARIFY_S1_contrasts.csv"), index=False)
    md += ["### S1 contrasts (Holm across the six conditions within each pair)\n", fmt(s1)]
    # share of the OR-DQ gap explained by sign
    rows = []
    for c in ALLC:
        m = base3[(base3.kind == "eval") & (base3.cond == c)].groupby("variant").rmst.mean()
        if {"OR_fixed", "DQ", "DQ_c001"} <= set(m.index):
            gap = m.OR_fixed - m.DQ
            rows.append(dict(cond=c, OR=m.OR_fixed, DQ=m.DQ, DQ_c001=m.DQ_c001, gap=gap,
                             share_sign=(m.DQ_c001 - m.DQ) / gap if abs(gap) > 1e-9 else np.nan,
                             random=R.loc[c, "random"]))
    sh = pd.DataFrame(rows)
    md += ["### Share of the OR − DQ gap closed by the shift (DQ+0.01 − DQ)/(OR − DQ)\n",
           (sh.round(2).to_markdown(index=False) if not sh.empty else "(no data)") + "\n"]

    # S2/S3: long training
    for name, D, conds in (("S2 — 10M", long10, ALLC), ("S3 — 30M (F4)", long30, ["F4"])):
        if D.empty:
            md += [f"## {name}\n(no data yet)\n"]
            continue
        mt = D.groupby(["kind", "cond", "variant"]).rmst.agg(["mean", "std", "count"]).round(1)
        md += [f"## {name}: RMST mean ± SD (n)\n", mt.reset_index().to_markdown(index=False) + "\n"]
        pairs = [("OR_fixed", "DQ")]
        cs = [c for c in conds if c not in ("F3", "F4")] if name.startswith("S2") else conds
        c1 = contrast(D, pairs, cs, "eval", family=name)
        c2 = contrast(D, pairs, conds, "eval_det", family=name + " det")
        extra = contrast(D, [("DQ_c001", "DQ"), ("OR_fixed", "DQ_c001"), ("OR_fixed", "Alive")], ["F5", "F6"], "eval",
                         family=name + " extra")
        md += ["### OR vs DQ, stochastic (Holm over the pre-registered conditions)\n", fmt(c1),
               "### OR vs DQ, deterministic (secondary)\n", fmt(c2), "### Other (F5/F6)\n", fmt(extra)]
        pd.concat([c1, c2, extra]).to_csv(os.path.join(OUT, f"CLARIFY_{name[:2]}_contrasts.csv"), index=False)
        oz = D[D.kind == "eval"].groupby(["cond", "variant"])[["in_zone", "over"]].mean().round(2)
        md += ["### In-zone fraction and over-consumption share (stochastic)\n", oz.reset_index().to_markdown(index=False) + "\n"]

    # S4: F6 grid
    G = load("F_zbgrid")
    G = G[(G.cond == "F6") & (G.kind == "eval")]
    if G.empty:
        md += ["## S4 — F6 grid\n(no data yet)\n"]
    else:
        G["z"] = G.variant.str.extract(r"_z([0-9.]+)_").astype(float)
        G["beta"] = G.variant.str.extract(r"_b([0-9.]+)$").astype(float)
        d0, d1 = transitions("F6")
        mr = {(z, b): float(or_per_variable(d0, d1, z, b).mean(1).mean()) for z in Z_GRID for b in B_GRID}
        M = G.groupby(["z", "beta"]).rmst.agg(["mean", "std", "count"]).reset_index()
        M["mean_r"] = [mr[(z, b)] for z, b in zip(M.z, M.beta)]
        M.to_csv(os.path.join(OUT, "CLARIFY_S4_F6grid.csv"), index=False)
        rnd = R.loc["F6", "random"]
        alive = base3[(base3.cond == "F6") & (base3.variant == "Alive") & (base3.kind == "eval")].rmst.mean()
        best = M.loc[M["mean"].idxmax()]
        dflt = M[(M.z == 0.5) & (M.beta == 1.0)].iloc[0]
        p1 = M[M.mean_r <= -0.9]
        top5 = M.nlargest(5, "mean")
        sat = M[M.z == 0.9]
        md += ["## S4 — F6 grid (k = 8): RMST mean over 5 seeds\n",
               M.pivot(index="z", columns="beta", values="mean").round(0).to_markdown() + "\n",
               f"Random policy {rnd:.1f}; Alive (3M, 10 seeds) {alive:.1f}.\n\n",
               f"- P1 (cells with mean r ≤ −0.9 below random): {int((p1['mean'] < rnd).sum())}/{len(p1)} "
               f"({p1['mean'].round(0).tolist()})\n",
               f"- P2 (top-5 cells have −0.5 ≤ mean r ≤ 0.5): {int(top5.mean_r.between(-0.5, 0.5).sum())}/5 "
               f"(mean r = {top5.mean_r.round(2).tolist()}, cells {list(zip(top5.z, top5.beta))})\n",
               f"- P3 (default within 1 SD of best): best ({best.z}, {best.beta}) = {best['mean']:.1f} ± {best['std']:.1f}; "
               f"default = {dflt['mean']:.1f} ± {dflt['std']:.1f}; gap = {best['mean'] - dflt['mean']:.1f} "
               f"→ {'yes' if best['mean'] - dflt['mean'] < min(best['std'], dflt['std']) else 'no'}\n",
               f"- P4 (z = 0.9 within ±25% of Alive): {int(((sat['mean'] - alive).abs() <= 0.25 * alive).sum())}/{len(sat)} "
               f"({sat['mean'].round(0).tolist()})\n"]

    # S5: deterministic evaluation for all OR vs DQ comparisons at 3M
    c5 = pd.concat([contrast(base3, [("OR_fixed", "DQ")], ALLC, "eval", "S5 stoch"),
                    contrast(base3, [("OR_fixed", "DQ")], ALLC, "eval_det", "S5 det")])
    c5.to_csv(os.path.join(OUT, "CLARIFY_S5_contrasts.csv"), index=False)
    md += ["## S5 — OR vs DQ at 3M, stochastic vs deterministic policy\n", fmt(c5.assign(cond=c5["family"] + " " + c5["cond"]))]
    det_all = base3[base3.kind == "eval_det"].groupby(["cond", "variant"]).rmst.mean().unstack().round(1)
    md += ["### Deterministic RMST, 3M\n", det_all.to_markdown() + "\n"]

    with open(os.path.join(OUT, "CLARIFY_RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
