"""
analyze_nt.py — analysis of the no-termination control (stages NT / NT2).

In the NT environment a variable that crosses the viability bound is clipped to it and
episodes always last 1000 steps, so a constant per-step reward ("Alive") carries no
information and a constant offset (DR vs DR_alive) cannot change the optimal policy.
The question is whether the OR regime geometry carries useful information *beyond* its
sign, i.e. whether OR still beats the continuous drives when sign cannot matter.

Endpoints (seed level = mean over the 20 paired evaluation episodes), fixed before running:
  primary    frac_viable   fraction of variable-steps within the viability bound |v| <= 1
  secondary  t_first_viol  RMST (tau = 1000) of the first viability violation
                           (= survival time had the episode terminated, as in the main study)
             frac_in_zone  fraction of variable-steps with |v| <= 0.5  (aligned with OR, z = 0.5)
             mean_dev      mean |v| (aligned with DR / DQ; lower is better)
Families (Holm within condition x family x endpoint):
  F1 "learns"   every variant vs Alive (no-information control)
  F2 "geometry" OR fixed vs DR, DQ, Dead-zone, OR no-penalty, OR no-zone;
                OR adaptive vs OR fixed; DR + alive vs DR (offset sanity check)
Reference policies (no training): uniform, proportional (alloc ∝ |v|), greedy (all to max |v|).

Usage: python analyze_nt.py            (reads docs/results_v2/NT_noterm, writes docs/results_v2/analysis/NT_*)
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import RES, OUT, FIG, TAU, LABEL, COLOR, load, holm, paired_test, boot_ci, fmt_p, md_table  # noqa: E402
from mrp_env import MRPEnv, make_reward_fn  # noqa: E402
from runner import CONDITIONS, DEFAULTS  # noqa: E402

STUDY = os.environ.get("NT_STUDY", "NT_noterm")
VARIANTS = ["OR_adapt", "OR_fixed", "DQ", "DR", "Alive", "DR_alive", "DeadZone", "OR_noPen", "OR_noZone"]
METRICS = {  # name -> (label, higher_is_better)
    "frac_viable": ("Viable fraction (primary)", True),
    "t_first_viol": ("RMST first violation", True),
    "frac_in_zone": ("In-zone fraction (|v|<=0.5)", True),
    "mean_dev": ("Mean |v|", False),
}
F1 = [(v, "Alive") for v in VARIANTS if v != "Alive"]
F2 = [("OR_fixed", "DR"), ("OR_fixed", "DQ"), ("OR_fixed", "DeadZone"), ("OR_fixed", "OR_noPen"),
      ("OR_fixed", "OR_noZone"), ("OR_adapt", "OR_fixed"), ("DR_alive", "DR")]


def seed_level(ep):
    ep = ep.copy()
    ep["t_first_viol"] = np.minimum(ep["first_viol"], TAU)
    g = ep.groupby(["cond", "variant", "seed", "n", "noise", "budget"])
    return g.agg(frac_viable=("frac_viable", "mean"), t_first_viol=("t_first_viol", "mean"),
                 viol_rate=("viol_event", "mean"), frac_in_zone=("frac_in_zone", "mean"),
                 mean_dev=("mean_dev", "mean"), frac_all_viable=("frac_all_viable", "mean"),
                 alloc_dev_corr=("alloc_dev_corr", "mean"), top10_share=("top10_share", "mean"),
                 alloc_entropy=("alloc_entropy", "mean")).reset_index()


def contrasts(S, conds, pairs, family, metric):
    rows = []
    for c in conds:
        sub = S[S.cond == c]
        for v1, v2 in pairs:
            a = sub[sub.variant == v1].set_index("seed")[metric]
            b = sub[sub.variant == v2].set_index("seed")[metric]
            common = a.index.intersection(b.index)
            if len(common) < 3:
                continue
            t = paired_test(a.loc[common].values, b.loc[common].values)
            rows.append(dict(family=family, metric=metric, cond=c,
                             contrast=f"{LABEL.get(v1, v1)} vs {LABEL.get(v2, v2)}", v1=v1, v2=v2,
                             mean1=a.loc[common].mean(), mean2=b.loc[common].mean(), **t))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["p_holm"] = np.nan
    for c, g in df.groupby("cond"):
        df.loc[g.index, "p_holm"] = holm(g.p_raw.values)
    return df


def friedman(S, c, metric):
    piv = S[S.cond == c].pivot_table(index="seed", columns="variant", values=metric).dropna()
    vs = [v for v in VARIANTS if v in piv]
    if len(vs) < 3 or piv.shape[0] < 3:
        return np.nan, np.nan
    st, p = stats.friedmanchisquare(*[piv[v].values for v in vs])
    return p, st / (piv.shape[0] * (len(vs) - 1))


def reference_policies(conds, seeds):
    rf = make_reward_fn("Alive")
    rows = []
    for c in conds:
        cc = CONDITIONS[c]
        for pol in ("uniform", "proportional", "greedy"):
            vals = []
            for sd in seeds:
                e = MRPEnv(cc["n"], cc["budget"], cc["noise"], rf, terminate=False)
                o = e.reset(int(sd))
                done, t, fv, via, inz, md = False, 0, None, [], [], []
                while not done:
                    v = np.abs(o[:cc["n"]].astype(np.float64))
                    lg = (np.zeros(cc["n"]) if pol == "uniform" else
                          np.log(v + 1e-6) if pol == "proportional" else 50.0 * v)
                    o, _, te, tr, info = e.step(lg)
                    t += 1
                    done = te or tr
                    via.append(1 - info["viol"].mean())
                    inz.append((info["dev"] <= 0.5).mean())
                    md.append(info["dev"].mean())
                    if fv is None and info["viol"].any():
                        fv = t
                vals.append((np.mean(via), min(fv or t, TAU), np.mean(inz), np.mean(md)))
            m = np.mean(vals, 0)
            rows.append(dict(cond=c, policy=pol, frac_viable=m[0], t_first_viol=m[1],
                             frac_in_zone=m[2], mean_dev=m[3]))
    return pd.DataFrame(rows)


def fig_metric_bars(S, R, conds, metric, fname):
    lab, hib = METRICS[metric]
    fig, axes = plt.subplots(1, len(conds), figsize=(3.6 * len(conds), 3.8), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, conds):
        sub = S[S.cond == c]
        xs = np.arange(len(VARIANTS))
        for i, v in enumerate(VARIANTS):
            x = sub[sub.variant == v][metric].values
            if not len(x):
                continue
            ax.bar(i, x.mean(), color=COLOR.get(v, "#999"), alpha=.85)
            ax.scatter(np.full(len(x), i) + np.random.default_rng(i).uniform(-.2, .2, len(x)), x,
                       s=6, color="k", zorder=3)
        ref = R[R.cond == c].set_index("policy")[metric]
        for pol, ls in (("uniform", ":"), ("proportional", "--"), ("greedy", "-.")):
            if pol in ref:
                ax.axhline(ref[pol], color="k", ls=ls, lw=1, label=pol)
        allv = np.concatenate([sub[metric].values, ref.values])
        pad = (allv.max() - allv.min()) * 0.15 + 1e-9
        ax.set_ylim(allv.min() - pad, allv.max() + pad)
        ax.set_xticks(xs)
        ax.set_xticklabels([LABEL.get(v, v) for v in VARIANTS], rotation=70, fontsize=7)
        ax.set_title(c, fontsize=10)
        ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel(lab + ("" if hib else " (lower = better)"))
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle(f"No-termination control — {lab} (dots: training seeds; lines: reference policies)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=160)
    plt.close(fig)


def fig_forest(ct, fname, title):
    if ct.empty:
        return
    conds = list(dict.fromkeys(ct.cond))
    labels = list(dict.fromkeys(ct.contrast))
    fig, axes = plt.subplots(1, len(conds), figsize=(3.3 * len(conds), 0.42 * len(labels) + 1.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, conds):
        g = ct[ct.cond == c].set_index("contrast").reindex(labels)
        y = np.arange(len(labels))[::-1]
        for yi, (_, r) in zip(y, g.iterrows()):
            if pd.isna(r["diff"]):
                continue
            ax.errorbar(r["diff"], yi, xerr=[[r["diff"] - r["ci_lo"]], [r["ci_hi"] - r["diff"]]], fmt="o",
                        color="#c0392b" if r["p_holm"] < 0.05 else "#555", ms=5, capsize=2)
        ax.axvline(0, color="k", lw=.8)
        ax.set_title(c, fontsize=10)
        ax.grid(alpha=.3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
    fig.suptitle(title + " (red: Holm p < 0.05)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=160)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    ep, meta = load(STUDY)
    if ep.empty:
        print("no results in", os.path.join(RES, STUDY))
        return
    S = seed_level(ep)
    # analyse only complete conditions (every variant with 10 seeds); others are still running
    kk = S.groupby(["cond", "variant"]).size().unstack()
    complete = [c for c in kk.index if kk.loc[c].reindex(VARIANTS).min() >= 10]
    skipped = sorted(set(S.cond) - set(complete))
    if skipped:
        print("incomplete (skipped):", skipped)
    S = S[S.cond.isin(complete)]
    conds = [c for c in CONDITIONS if c in complete]
    S.to_csv(os.path.join(OUT, "NT_seed_level.csv"), index=False)
    R = reference_policies(conds, DEFAULTS["eval_seeds"])
    R.to_csv(os.path.join(OUT, "NT_reference_policies.csv"), index=False)

    rep = ["# No-termination control (NT) — results\n",
           "Generated by `mrp_v2/analyze_nt.py`. Shared (permutation-equivariant) actor, 100k steps, "
           "10 training seeds, 20 paired evaluation episodes of exactly 1000 steps. A variable that crosses "
           "the viability bound is clipped to it; episodes never terminate, so the Alive reward carries no "
           "information and a constant offset cannot change the optimal policy.\n",
           "Primary endpoint (fixed before running): **viable fraction** = fraction of variable-steps with "
           "|v| ≤ 1. Secondary: RMST of the first violation (the survival time the agent would have had "
           "with termination), in-zone fraction (|v| ≤ 0.5; aligned with OR's z = 0.5) and mean |v| "
           "(aligned with DR/DQ; lower is better). Holm adjustment within condition × family × endpoint.\n"]
    k = S.groupby(["cond", "variant"]).size()
    rep.append(f"Seeds per cell: min {k.min()}, max {k.max()}.\n")

    all_ct = []
    for metric, (lab, hib) in METRICS.items():
        rep.append(f"## {lab}\n")
        rows = []
        for c in conds:
            sub = S[S.cond == c]
            row = {"cond": c}
            for v in VARIANTS:
                x = sub[sub.variant == v][metric].values
                row[v] = (f"{x.mean():.3f} ± {x.std(ddof=1):.3f}" if metric != "t_first_viol" and len(x) > 1 else
                          f"{x.mean():.0f} ± {x.std(ddof=1):.0f}" if len(x) > 1 else
                          (f"{x.mean():.3f}" if len(x) else "—"))
            ref = R[R.cond == c].set_index("policy")[metric]
            fmt = (lambda z: f"{z:.0f}") if metric == "t_first_viol" else (lambda z: f"{z:.3f}")
            row["uniform"], row["prop."], row["greedy"] = (fmt(ref["uniform"]), fmt(ref["proportional"]),
                                                          fmt(ref["greedy"]))
            p, W = friedman(S, c, metric)
            row["Friedman"] = f"p = {fmt_p(p)}, W = {W:.2f}" if not np.isnan(p) else "—"
            rows.append(row)
        T = pd.DataFrame(rows)
        cols = ["cond"] + VARIANTS + ["uniform", "prop.", "greedy", "Friedman"]
        rep.append(md_table(T, cols, ["Cond"] + [LABEL.get(v, v) for v in VARIANTS] +
                            ["Uniform", "Proportional", "Greedy", "Friedman (9 rewards)"]) + "\n")
        for fam, pairs in (("F1 vs no-information control", F1), ("F2 geometry", F2)):
            ct = contrasts(S, conds, pairs, fam, metric)
            if ct.empty:
                continue
            all_ct.append(ct)
            rep.append(f"**{fam}** — {lab}\n")
            rep.append(md_table(ct, ["cond", "contrast", lambda r: f"{r['mean1']:.4g}", lambda r: f"{r['mean2']:.4g}",
                                     lambda r: f"{r['diff']:+.4g} [{r['ci_lo']:+.3g}, {r['ci_hi']:+.3g}]",
                                     lambda r: f"{r['rbc']:+.2f}", lambda r: fmt_p(r["p_raw"]),
                                     lambda r: fmt_p(r["p_holm"])],
                                ["Cond", "Contrast", "Mean 1", "Mean 2", "Δ [95% CI]", "r_rb", "p raw",
                                 "p Holm"]) + "\n")
        fig_metric_bars(S, R, conds, metric, f"NT_{metric}.png")
    CT = pd.concat(all_ct, ignore_index=True) if all_ct else pd.DataFrame()
    if not CT.empty:
        CT.to_csv(os.path.join(OUT, "NT_contrasts.csv"), index=False)
        for metric in METRICS:
            fig_forest(CT[(CT.metric == metric) & (CT.family == "F2 geometry")], f"NT_forest_{metric}.png",
                       f"No-termination, geometry contrasts — {METRICS[metric][0]}")

    rep.append("## Behaviour\n")
    B = S.groupby(["cond", "variant"])[["alloc_dev_corr", "top10_share", "alloc_entropy", "frac_all_viable",
                                         "viol_rate"]].mean().reset_index()
    B["variant"] = pd.Categorical(B["variant"], VARIANTS)
    B = B.sort_values(["cond", "variant"])
    rep.append(md_table(B, ["cond", lambda r: LABEL.get(r["variant"], r["variant"]),
                            lambda r: f"{r['alloc_dev_corr']:+.2f}", lambda r: f"{r['top10_share']:.3f}",
                            lambda r: f"{r['alloc_entropy']:.3f}", lambda r: f"{r['frac_all_viable']:.3f}",
                            lambda r: f"{r['viol_rate']:.2f}"],
                        ["Cond", "Variant", "corr(alloc, |v|)", "top-10% share", "alloc entropy",
                         "P(all viable)", "P(any violation)"]) + "\n")
    with open(os.path.join(OUT, "NT_RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep))
    print("wrote", os.path.join(OUT, "NT_RESULTS.md"))


if __name__ == "__main__":
    main()
