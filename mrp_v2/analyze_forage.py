"""
analyze_forage.py — analysis of the foraging gridworld (stages FT and FN of forage_run.py).

Pre-specified plan (see forage_run.py docstring): unit = training seed; 20 paired
evaluation episodes; stochastic evaluation is primary.
  FT (with termination):    primary = RMST of episode length (tau = 500)
  FN (no termination):      primary = viable fraction (|v| <= 1);
                            secondary = RMST of the first violation
  both: in-zone fraction (|v| <= 0.5), mean |v| (lower = better), over-consumption share.
Families: F1 every variant vs Alive; F2 OR fixed vs DR / DQ / Dead-zone / no-penalty / no-zone,
OR adaptive vs fixed, DR + alive vs DR.  Exact Wilcoxon, Holm within condition x family x metric.

Usage: python analyze_forage.py   -> docs/results_v2/analysis/FORAGE_RESULTS.md, FORAGE_*.csv, figs/FORAGE_*.png
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import RES, OUT, FIG, LABEL, COLOR, fmt_p, md_table  # noqa: E402
from analyze_nt import contrasts, friedman, fig_forest  # noqa: E402
from forage_env import run_reference, policy_random, GreedyPolicy  # noqa: E402
from forage_run import FCONDS, EVAL_SEEDS, VARIANTS as RUN_VARIANTS  # noqa: E402

TAU = 500
CONDS = ["F1", "F2", "F3", "F4"]
VARIANTS = ["OR_adapt", "OR_fixed", "DQ", "DR", "Alive", "DR_alive", "DeadZone", "OR_noPen", "OR_noZone"]
assert set(VARIANTS) == set(RUN_VARIANTS)
F1 = [(v, "Alive") for v in VARIANTS if v != "Alive"]
F2 = [("OR_fixed", "DR"), ("OR_fixed", "DQ"), ("OR_fixed", "DeadZone"), ("OR_fixed", "OR_noPen"),
      ("OR_fixed", "OR_noZone"), ("OR_adapt", "OR_fixed"), ("DR_alive", "DR")]
STUDIES = {
    "F_term": dict(title="With termination (death ends the episode)", metrics={
        "rmst": ("RMST survival, τ = 500 (primary)", True),
        "frac_in_zone": ("In-zone fraction (|v| ≤ 0.5)", True),
        "mean_dev": ("Mean |v|", False),
        "over_share": ("Over-consumption share", False)}),
    "F_noterm": dict(title="No termination (v clipped at ±1, 500-step episodes)", metrics={
        "frac_viable": ("Viable fraction (primary)", True),
        "t_first_viol": ("RMST first violation, τ = 500", True),
        "frac_in_zone": ("In-zone fraction (|v| ≤ 0.5)", True),
        "mean_dev": ("Mean |v|", False),
        "over_share": ("Over-consumption share", False)}),
}


def load(study):
    rows, curves = [], []
    for p in glob.glob(os.path.join(RES, study, "*.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        j = r["job"]
        for kind in ("eval", "eval_det"):
            for e in r[kind]:
                rows.append(dict(cond=j["cond"], variant=j["variant"], seed=j["seed"], kind=kind, **e))
        tc = r["train_curve"]
        curves.append(dict(cond=j["cond"], variant=j["variant"], seed=j["seed"],
                           step=np.array(tc["step"]), len=np.array(tc["len"])))
    return pd.DataFrame(rows), curves


def seed_level(ep):
    ep = ep.copy()
    ep["rmst"] = np.minimum(ep["length"], TAU)
    ep["t_first_viol"] = np.minimum(ep["first_viol"], TAU)
    g = ep.groupby(["kind", "cond", "variant", "seed"])
    return g.agg(rmst=("rmst", "mean"), frac_viable=("frac_viable", "mean"),
                 t_first_viol=("t_first_viol", "mean"), frac_in_zone=("frac_in_zone", "mean"),
                 mean_dev=("mean_dev", "mean"), over_share=("over_share", "mean"),
                 eat_rate=("eat_rate", "mean"), death=("event", "mean")).reset_index()


def references(terminate):
    rows = []
    for c in CONDS:
        cc = FCONDS[c]
        kw = dict(W=9, decay_mu=cc["decay_mu"])
        for name, pol in (("random", policy_random), ("greedy", GreedyPolicy())):
            o = pd.DataFrame(run_reference(pol, cc["k"], kw, EVAL_SEEDS, terminate=terminate))
            rows.append(dict(cond=c, policy=name, rmst=np.minimum(o.length, TAU).mean(),
                             frac_viable=o.frac_viable.mean(), t_first_viol=np.minimum(o.first_viol, TAU).mean(),
                             frac_in_zone=o.frac_in_zone.mean(), mean_dev=o.mean_dev.mean(),
                             over_share=o.over_share.mean()))
    return pd.DataFrame(rows)


def fig_bars(S, R, metric, lab, hib, fname, title):
    fig, axes = plt.subplots(1, len(CONDS), figsize=(3.6 * len(CONDS), 3.8))
    for ax, c in zip(axes, CONDS):
        sub = S[S.cond == c]
        for i, v in enumerate(VARIANTS):
            x = sub[sub.variant == v][metric].values
            if len(x):
                ax.bar(i, x.mean(), color=COLOR.get(v, "#999"), alpha=.85)
                ax.scatter(i + np.random.default_rng(i).uniform(-.2, .2, len(x)), x, s=6, color="k", zorder=3)
        ref = R[R.cond == c].set_index("policy")[metric]
        for pol, ls in (("random", ":"), ("greedy", "-.")):
            ax.axhline(ref[pol], color="k", ls=ls, lw=1, label=pol)
        ax.set_xticks(range(len(VARIANTS)))
        ax.set_xticklabels([LABEL.get(v, v) for v in VARIANTS], rotation=70, fontsize=7)
        k, d = FCONDS[c]["k"], FCONDS[c]["decay_mu"]
        ax.set_title(f"{c} (k={k}, decay={d})", fontsize=9)
        ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel(lab + ("" if hib else " (lower = better)"))
    axes[-1].legend(fontsize=7)
    fig.suptitle(f"Foraging gridworld — {title} — {lab} (dots: training seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=160)
    plt.close(fig)


def fig_learning(curves, fname, title, total=3_000_000, nb=30):
    edges = np.linspace(0, total, nb + 1)
    mid = (edges[1:] + edges[:-1]) / 2
    fig, axes = plt.subplots(1, len(CONDS), figsize=(3.6 * len(CONDS), 3.4))
    for ax, c in zip(axes, CONDS):
        for v in VARIANTS:
            cs = [cv for cv in curves if cv["cond"] == c and cv["variant"] == v]
            if not cs:
                continue
            M = []
            for cv in cs:
                idx = np.digitize(cv["step"], edges) - 1
                M.append([cv["len"][idx == b].mean() if (idx == b).any() else np.nan for b in range(nb)])
            ax.plot(mid / 1e6, np.nanmean(M, 0), color=COLOR.get(v, "#999"), label=LABEL.get(v, v), lw=1.2)
        ax.set_title(c, fontsize=9)
        ax.set_xlabel("training steps (M)")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("training episode length")
    axes[-1].legend(fontsize=6)
    fig.suptitle(f"Foraging gridworld — {title} — learning curves (mean of seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=160)
    plt.close(fig)


def posthoc_and_params():
    """Post-hoc (not pre-specified) sign control: shaped rewards vs DR + alive (with termination);
    and the final values of the learned OR parameters."""
    from analyze import paired_test, holm
    out = ["## Post-hoc: shaped rewards vs DR + alive (with termination, RMST)\n",
           "Not in the pre-specified families; Holm within condition over these three contrasts.\n"]
    p = os.path.join(OUT, "FORAGE_F_term_seed_level.csv")
    if os.path.exists(p):
        S = pd.read_csv(p)
        S = S[S.kind == "eval"]
        rows = []
        for c in CONDS:
            s = S[S.cond == c]
            for v1 in ("OR_fixed", "DQ", "OR_noPen"):
                a = s[s.variant == v1].set_index("seed").rmst
                b = s[s.variant == "DR_alive"].set_index("seed").rmst
                common = a.index.intersection(b.index)
                if len(common) < 3:
                    continue
                rows.append(dict(cond=c, contrast=f"{LABEL.get(v1, v1)} vs DR + alive",
                                 mean1=a[common].mean(), mean2=b[common].mean(),
                                 **paired_test(a[common].values, b[common].values)))
        D = pd.DataFrame(rows)
        if not D.empty:
            D["p_holm"] = np.nan
            for c, g in D.groupby("cond"):
                D.loc[g.index, "p_holm"] = holm(g.p_raw.values)
            D.to_csv(os.path.join(OUT, "FORAGE_F_term_posthoc_sign.csv"), index=False)
            out.append(md_table(D, ["cond", "contrast", lambda r: f"{r['mean1']:.1f}", lambda r: f"{r['mean2']:.1f}",
                                    lambda r: f"{r['diff']:+.1f} [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]",
                                    lambda r: fmt_p(r["p_raw"]), lambda r: fmt_p(r["p_holm"])],
                                ["Cond", "Contrast", "Mean 1", "Mean 2", "Δ [95% CI]", "p raw", "p Holm"]) + "\n")
    out.append("## Learned OR parameters (Adaptive OR; init z = 0.50, β = 1.00)\n")
    rows = []
    for study in STUDIES:
        for f in glob.glob(os.path.join(RES, study, "*__OR_adapt__*.json")):
            with open(f, encoding="utf-8") as fh:
                r = json.load(fh)
            rows.append(dict(study=study, cond=r["job"]["cond"], z=r["or_final"]["z_of_mean"],
                             beta=r["or_final"]["beta_of_mean"]))
    if rows:
        P = pd.DataFrame(rows).groupby(["study", "cond"]).agg(
            z=("z", "mean"), z_sd=("z", "std"), beta=("beta", "mean"), beta_sd=("beta", "std"), n=("z", "size"))
        P.to_csv(os.path.join(OUT, "FORAGE_learned_params.csv"))
        out.append(P.round(3).to_markdown() + "\n")
    return out


def main():
    os.makedirs(FIG, exist_ok=True)
    rep = ["# Foraging gridworld (Crafter replacement) — results\n",
           "Generated by `mrp_v2/analyze_forage.py`. 9×9 grid, k internal variables (set point 0, viability "
           "|v| ≤ 1) depleting at per-episode rates; one station per resource; actions = 4 moves, stay, consume "
           "(+0.2 to the station's variable; over-consumption possible). Categorical PPO (2×64 MLP), 3×10⁶ steps, "
           "10 training seeds, 20 paired evaluation episodes (stochastic policy with fixed action RNG; "
           "deterministic results in the CSV). Holm within condition × family × endpoint.\n",
           "| Cond | k | decay/step |\n|---|---|---|\n" +
           "\n".join(f"| {c} | {FCONDS[c]['k']} | {FCONDS[c]['decay_mu']} |" for c in CONDS) + "\n"]
    for study, spec in STUDIES.items():
        ep, curves = load(study)
        if ep.empty:
            rep.append(f"## {spec['title']}\n\n(no results)\n")
            continue
        Sall = seed_level(ep)
        Sall.to_csv(os.path.join(OUT, f"FORAGE_{study}_seed_level.csv"), index=False)
        S = Sall[Sall.kind == "eval"]
        n = S.groupby(["cond", "variant"]).size()
        R = references(terminate=(study == "F_term"))
        R.to_csv(os.path.join(OUT, f"FORAGE_{study}_references.csv"), index=False)
        rep.append(f"## {spec['title']}\n\nSeeds per cell: min {n.min()}, max {n.max()}.\n")
        all_ct = []
        for metric, (lab, hib) in spec["metrics"].items():
            rep.append(f"### {lab}\n")
            rows = []
            fmt = (lambda z: f"{z:.1f}") if metric in ("rmst", "t_first_viol") else (lambda z: f"{z:.3f}")
            for c in CONDS:
                sub = S[S.cond == c]
                row = {"cond": c}
                for v in VARIANTS:
                    x = sub[sub.variant == v][metric].values
                    row[v] = f"{fmt(x.mean())} ± {fmt(x.std(ddof=1))}" if len(x) > 1 else "—"
                ref = R[R.cond == c].set_index("policy")[metric]
                row["random"], row["greedy"] = fmt(ref["random"]), fmt(ref["greedy"])
                p, W = friedman(S, c, metric)
                row["Friedman"] = f"p = {fmt_p(p)}, W = {W:.2f}" if not np.isnan(p) else "—"
                rows.append(row)
            T = pd.DataFrame(rows)
            rep.append(md_table(T, ["cond"] + VARIANTS + ["random", "greedy", "Friedman"],
                                ["Cond"] + [LABEL.get(v, v) for v in VARIANTS] + ["Random", "Greedy", "Friedman"]) + "\n")
            for fam, pairs in (("F1 vs Alive", F1), ("F2 geometry", F2)):
                ct = contrasts(S, CONDS, pairs, fam, metric)
                if ct.empty:
                    continue
                all_ct.append(ct)
                rep.append(f"**{fam}** — {lab}\n")
                rep.append(md_table(ct, ["cond", "contrast", lambda r: f"{r['mean1']:.4g}", lambda r: f"{r['mean2']:.4g}",
                                         lambda r: f"{r['diff']:+.4g} [{r['ci_lo']:+.3g}, {r['ci_hi']:+.3g}]",
                                         lambda r: f"{r['rbc']:+.2f}", lambda r: fmt_p(r["p_raw"]),
                                         lambda r: fmt_p(r["p_holm"])],
                                    ["Cond", "Contrast", "Mean 1", "Mean 2", "Δ [95% CI]", "r_rb", "p raw", "p Holm"]) + "\n")
            fig_bars(S, R, metric, lab, hib, f"FORAGE_{study}_{metric}.png", spec["title"])
        prim = list(spec["metrics"])[0]
        if all_ct:
            CT = pd.concat(all_ct, ignore_index=True)
            CT.to_csv(os.path.join(OUT, f"FORAGE_{study}_contrasts.csv"), index=False)
            fig_forest(CT[(CT.metric == prim) & (CT.family == "F2 geometry")], f"FORAGE_{study}_forest.png",
                       f"Foraging — {spec['title']} — geometry contrasts, {spec['metrics'][prim][0]}")
        if study == "F_term":
            fig_learning(curves, f"FORAGE_{study}_learning.png", spec["title"])
        # deterministic evaluation, primary endpoint (secondary analysis)
        D = Sall[Sall.kind == "eval_det"].groupby(["cond", "variant"])[prim].mean().unstack()[VARIANTS]
        rep.append(f"### Deterministic (argmax) evaluation — {spec['metrics'][prim][0]}, mean of seeds\n")
        rep.append(D.round(3).to_markdown() + "\n")
    rep += posthoc_and_params()
    with open(os.path.join(OUT, "FORAGE_RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep))
    print("wrote", os.path.join(OUT, "FORAGE_RESULTS.md"))


if __name__ == "__main__":
    main()
