"""
review_r1_reanalysis.py — re-analysis without training, for the revision (round 1 of the review team).

(1) Foraging with termination, 10^7 steps, F3 and F4 (study F_long, tag term10M): OR vs DQ and
    OR vs Alive, stochastic and deterministic evaluation; exact Wilcoxon, Holm over {F3, F4}
    within each pair and evaluation.  -> REVIEW_F34_10M_contrasts.csv
(2) Foraging without termination, 10^7 steps (F_long_nt) and 3x10^6 steps (F_noterm, F_ksweep):
    fraction of variable-steps within |v| <= 0.5 (in-zone), OR vs DQ, Holm over F3-F6.
    -> REVIEW_noterm_inzone_contrasts.csv
(3) MRP: absorbing state under the random-logit policy (logits ~ N(0,1) each step), same episodes
    and generator as reward_scale_probe.py (20 episodes, env seeds 5000+ep, rng seed 0).
    -> REVIEW_mrp_absorption.csv
(4) F6 grid: mean early reward recalculated (CLARIFY_S4_F6grid.csv) against the values written
    before the runs (docs/prereg_2026-11-16.md, S4).  -> REVIEW_F6grid_prereg_vs_recalc.csv

Only reads saved results and simulates the random policy; writes new CSVs to
docs/results_v2/analysis.  Usage: python review_r1_reanalysis.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze_nt import paired_test, holm  # noqa: E402

RES = os.path.join(os.path.dirname(HERE), "docs", "results_v2")
OUT = os.path.join(RES, "analysis")
TAU = 500


def load(study, tag):
    rows = []
    for p in glob.glob(os.path.join(RES, study, "*.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        j = r["job"]
        if j["tag"] != tag:
            continue
        for kind in ("eval", "eval_det"):
            ev = r[kind]
            rows.append(dict(cond=j["cond"], variant=j["variant"], seed=j["seed"], kind=kind,
                             rmst=float(np.mean([min(e["length"], TAU) for e in ev])),
                             in_zone=float(np.mean([e["frac_in_zone"] for e in ev])),
                             viable=float(np.mean([e["frac_viable"] for e in ev]))))
    return pd.DataFrame(rows)


def contrasts(D, metric, pairs, conds, label):
    out = []
    for kind in ("eval", "eval_det"):
        for v1, v2 in pairs:
            rows = []
            for c in conds:
                g = D[(D.cond == c) & (D.kind == kind)]
                a = g[g.variant == v1].set_index("seed")[metric]
                b = g[g.variant == v2].set_index("seed")[metric]
                com = a.index.intersection(b.index)
                if len(com) < 3:
                    continue
                t = paired_test(a.loc[com].values, b.loc[com].values)
                rows.append(dict(study=label, metric=metric, kind=kind, cond=c, v1=v1, v2=v2,
                                 mean1=a.loc[com].mean(), sd1=a.loc[com].std(), mean2=b.loc[com].mean(),
                                 sd2=b.loc[com].std(), wins1=int((a.loc[com] > b.loc[com]).sum()), **t))
            if rows:
                df = pd.DataFrame(rows)
                df["p_holm"] = holm(df.p_raw.values)
                out.append(df)
    return pd.concat(out, ignore_index=True)


def part1():
    D = load("F_long", "term10M")
    D = D[D.variant.isin(["OR_fixed", "DQ", "Alive"])]
    C = contrasts(D, "rmst", [("OR_fixed", "DQ"), ("OR_fixed", "Alive")], ["F3", "F4"],
                  "F_long 1e7 steps, termination (Holm over F3, F4)")
    C.to_csv(os.path.join(OUT, "REVIEW_F34_10M_contrasts.csv"), index=False)
    print(C[["kind", "cond", "v1", "v2", "mean1", "mean2", "wins1", "p_raw", "p_holm"]].round(4).to_string())


def part2():
    D10 = load("F_long_nt", "noterm10M")
    D3 = pd.concat([load("F_noterm", "noterm"), load("F_ksweep", "noterm")], ignore_index=True)
    out = []
    for D, lab in ((D3, "3e6 steps, no termination"), (D10, "1e7 steps, no termination")):
        D = D[D.variant.isin(["OR_fixed", "DQ"]) & D.cond.isin(["F3", "F4", "F5", "F6"])]
        D = D.drop_duplicates(["cond", "variant", "seed", "kind"])
        out.append(contrasts(D, "in_zone", [("OR_fixed", "DQ")], ["F3", "F4", "F5", "F6"],
                             lab + " (Holm over F3-F6)"))
    C = pd.concat(out, ignore_index=True)
    C.to_csv(os.path.join(OUT, "REVIEW_noterm_inzone_contrasts.csv"), index=False)
    print(C[["study", "kind", "cond", "mean1", "mean2", "wins1", "p_raw", "p_holm"]].round(4).to_string())


def part3():
    from mrp_env import MRPEnv, make_reward_fn, MIN_SIGNIFICANT
    from runner import CONDITIONS
    rows = []
    for episodes_label, seeds in (("probe 5000-5019", [5000 + e for e in range(20)]),
                                  ("evaluation 1000-1019", [1000 + e for e in range(20)])):
        for c, p in CONDITIONS.items():
            env = MRPEnv(p["n"], p["budget"], p["noise"], make_reward_fn("DR"))
            rng = np.random.default_rng(0)
            end_frac, step_frac, dq_zero, all_abs, lengths, pooled = [], [], [], [], [], []
            for s in seeds:
                env.reset(s)
                done = False
                fr = []
                while not done:
                    logits = rng.normal(0, 1, p["n"])
                    _, _, term, trunc, info = env.step(logits)
                    fr.append(np.mean(np.abs(env.v) <= MIN_SIGNIFICANT))
                    all_abs.append(float(fr[-1] == 1.0))
                    pooled.append(fr[-1])
                    d0, d1 = info["dev_prev"], info["dev"]
                    dq_zero.append(float(np.mean(d0 ** 2) - np.mean(d1 ** 2) == 0.0))
                    done = term or trunc
                end_frac.append(fr[-1])
                step_frac.append(np.mean(fr))
                lengths.append(len(fr))
            rows.append(dict(episodes=episodes_label, cond=c, n=p["n"], budget=p["budget"], noise=p["noise"],
                             mean_length=np.mean(lengths),
                             frac_variable_steps_absorbed_pooled=np.mean(pooled),
                             frac_absorbed_at_end=np.mean(end_frac),
                             frac_absorbed_mean_over_steps=np.mean(step_frac),
                             frac_steps_all_absorbed=np.mean(all_abs),
                             frac_steps_dq_exactly_zero=np.mean(dq_zero)))
    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(OUT, "REVIEW_mrp_absorption.csv"), index=False)
    print(A.round(3).to_string())


def part4():
    pre = {0.1: [0.14, -0.29, -0.72, -1.58, -3.30], 0.3: [0.35, 0.03, -0.30, -0.95, -2.25],
           0.5: [0.62, 0.43, 0.24, -0.14, -0.90], 0.7: [0.85, 0.78, 0.70, 0.55, 0.25],
           0.9: [0.97, 0.95, 0.94, 0.91, 0.85]}
    betas = [0.0, 0.5, 1.0, 2.0, 4.0]
    M = pd.read_csv(os.path.join(OUT, "CLARIFY_S4_F6grid.csv"))
    M["mean_r_prereg"] = [pre[round(z, 1)][betas.index(b)] for z, b in zip(M.z, M.beta)]
    M["abs_diff_2dp"] = (M.mean_r.round(2) - M.mean_r_prereg).abs().round(2)
    M.to_csv(os.path.join(OUT, "REVIEW_F6grid_prereg_vs_recalc.csv"), index=False)
    print(M[["z", "beta", "mean_r", "mean_r_prereg", "abs_diff_2dp"]].round(3).to_string())
    print("max abs diff (2 dp):", M.abs_diff_2dp.max(), " cells differing:", int((M.abs_diff_2dp > 0).sum()))


if __name__ == "__main__":
    part1()
    part2()
    part3()
    part4()
