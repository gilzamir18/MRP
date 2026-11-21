"""
r2_summary_figure.py — summary figure requested in review r2 ('one figure with the paired
differences per condition').  Grid of 2 x 4 panels: rows = comparison (OR - DQ, OR - Alive);
columns = MRP with termination, MRP without, foraging with, foraging without.  With termination the
endpoint is the RMST (steps; tau = 1000 in the MRP, 500 in foraging), without termination the
viable fraction.  Each panel has its own y scale (the effects differ by orders of magnitude).
In the foraging panels without termination the deterministic evaluation policy is shown with hollow
markers.  Points = mean paired difference over the 10 training seeds; bars = 95% percentile
bootstrap CI of the mean (10^4 resamples).
Output: output/SUMMARY_paired_differences.{pdf,png}, docs/results_v2/analysis/R2_summary_figure.csv
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(os.path.dirname(HERE), "docs")
V2, V3 = os.path.join(DOCS, "results_v2"), os.path.join(DOCS, "results_v3_mrp")
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "output")
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#2b2b29", "#6b6a63", "#e4e3de"
rng = np.random.default_rng(2026_11_20)


def boot_ci(d, n=10_000):
    d = np.asarray(d, float)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return np.percentile(m, [2.5, 97.5])


def mrp_diffs(csv, tag, metric, a, b):
    df = pd.read_csv(csv)
    if tag is not None and "tag" in df.columns:
        df = df[df.tag == tag]
    out = {}
    for c in sorted(df.cond.unique(), key=lambda s: int(s[1:])):
        x = df[(df.cond == c) & (df.variant == a)].set_index("seed")[metric]
        y = df[(df.cond == c) & (df.variant == b)].set_index("seed")[metric]
        s = sorted(set(x.index) & set(y.index))
        if len(s) >= 2:
            out[c] = (x.loc[s] - y.loc[s]).values
    return out


def forage_diffs(sources, a, b, metric, policy="eval"):
    out = {}
    for study, tag in sources:
        for c in ["F1", "F2", "F3", "F4", "F5", "F6"]:
            vals = {}
            for v in (a, b):
                for f in glob.glob(os.path.join(V2, study, f"{c}__{v}__{tag}__s*.json")):
                    r = json.load(open(f))
                    vals.setdefault(v, {})[r["job"]["seed"]] = np.mean([e[metric] for e in r[policy]])
            s = sorted(set(vals.get(a, {})) & set(vals.get(b, {})))
            if len(s) >= 2:
                out[c] = np.array([vals[a][k] - vals[b][k] for k in s])
    return out


def main():
    Lcsv = os.path.join(V3, "analysis", "L_seed_level.csv")
    NTcsv = os.path.join(V3, "analysis", "NT_seed_level.csv")
    fterm = [("F_term", "term"), ("F_ksweep", "term")]
    fnt = [("F_noterm", "noterm"), ("F_ksweep", "noterm")]
    cols = ["MRP\nwith termination", "MRP\nwithout termination", "Foraging\nwith termination",
            "Foraging\nwithout termination"]
    ylabs = ["Δ RMST (steps)", "Δ viable fraction", "Δ RMST (steps)", "Δ viable fraction"]
    grid = {
        ("OR − DQ", 0): [("stochastic", True, mrp_diffs(Lcsv, "shared", "rmst", "OR_fixed", "DQ"))],
        ("OR − DQ", 1): [("stochastic", True, mrp_diffs(NTcsv, None, "frac_viable", "OR_fixed", "DQ"))],
        ("OR − DQ", 2): [("stochastic", True, forage_diffs(fterm, "OR_fixed", "DQ", "length"))],
        ("OR − DQ", 3): [("stochastic policy", True, forage_diffs(fnt, "OR_fixed", "DQ", "frac_viable")),
                         ("deterministic policy", False,
                          forage_diffs(fnt, "OR_fixed", "DQ", "frac_viable", "eval_det")),
                         ("stochastic, return normalisation (separate runs)", "sq",
                          forage_diffs([("R2_nt_norm", "noterm_nr")], "OR_fixed", "DQ", "frac_viable"))],
        ("OR − Alive", 0): [("stochastic", True, mrp_diffs(Lcsv, "shared", "rmst", "OR_fixed", "Alive"))],
        ("OR − Alive", 1): [("stochastic", True, mrp_diffs(NTcsv, None, "frac_viable", "OR_fixed", "Alive"))],
        ("OR − Alive", 2): [("stochastic", True, forage_diffs(fterm, "OR_fixed", "Alive", "length"))],
        ("OR − Alive", 3): [("stochastic", True, forage_diffs(fnt, "OR_fixed", "Alive", "frac_viable"))],
    }
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "serif"})
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.0), constrained_layout=True)
    rows = []
    for ri, (comp, col) in enumerate((("OR − DQ", BLUE), ("OR − Alive", ORANGE))):
        for ci in range(4):
            ax = axes[ri, ci]
            series = grid[(comp, ci)]
            conds = sorted({c for *_, d in series for c in d}, key=lambda s: int(s[1:]))
            xs = {c: i for i, c in enumerate(conds)}
            off = [0.0] if len(series) == 1 else list(np.linspace(-0.27, 0.27, len(series)))
            ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
            for (lab, filled, d), o in zip(series, off):
                for j, (c, x) in enumerate(d.items()):
                    lo, hi = boot_ci(x)
                    m = x.mean()
                    ax.plot([xs[c] + o] * 2, [lo, hi], color=col, lw=1.4, solid_capstyle="round", zorder=2)
                    ax.plot(xs[c] + o, m, "s" if filled == "sq" else "o", ms=4, mec=col,
                            mfc="white" if filled is False else col, mew=1.3,
                            zorder=3, label=lab if (j == 0 and len(series) > 1) else None)
                    rows.append(dict(comparison=comp, panel=cols[ci], policy=lab, cond=c, n=len(x),
                                     mean=m, ci_lo=lo, ci_hi=hi))
            ax.set_xticks(range(len(conds)), conds)
            ax.tick_params(length=2)
            if ri == 0:
                ax.set_title(cols[ci], fontsize=7.5, color=INK)
            ax.set_ylabel(f"{comp}\n{ylabs[ci]}" if ci == 0 else ylabs[ci], fontsize=7)
            ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            if len(series) > 1:
                handles = ax.get_legend_handles_labels()
    fig.legend(*handles, frameon=False, fontsize=7, loc="outside lower center", ncol=3,
               title="Foraging without termination, OR − DQ:", title_fontsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"SUMMARY_paired_differences.{ext}"), dpi=300)
    pd.DataFrame(rows).to_csv(os.path.join(V2, "analysis", "R2_summary_figure.csv"), index=False)
    print("ok", len(rows))


if __name__ == "__main__":
    main()
