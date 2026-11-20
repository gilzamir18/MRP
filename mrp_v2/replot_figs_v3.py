"""
replot_figs_v3.py — re-draw six paper figures from saved results with the final labels
("OR" instead of "OR (Fixed)", "permutation-equivariant actor" instead of "Shared actor",
"Uniform (≈ untrained net)", no "Fixed OR" in titles).  No training and no new statistics:
every figure is drawn from the seed-level CSVs / JSON already in docs/results_v2_noadapt.
Existing figures and CSVs are not touched; the new files get the suffix _v3 and are written
to output/figs_v2.

Usage:  python replot_figs_v3.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["MRP_RES"] = os.path.abspath(os.path.join(HERE, "..", "docs", "results_v2_noadapt"))
sys.path.insert(0, HERE)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import experiments  # noqa: E402
for lst in (experiments.MAIN, experiments.SIGN, experiments.GEOM, experiments.ALL_VARIANTS):
    while "OR_adapt" in lst:
        lst.remove("OR_adapt")
import analyze  # noqa: E402
import analyze_forage  # noqa: E402
from forage_run import FCONDS, Z_GRID, B_GRID  # noqa: E402

OUTFIG = os.path.abspath(os.path.join(HERE, "..", "..", "output", "figs_v2"))
AN = os.path.join(os.environ["MRP_RES"], "analysis")

# labels used in the paper
analyze.LABEL["OR_fixed"] = "OR"
analyze.REF_LABEL["zero"] = "Uniform (≈ untrained net)"
analyze.FIG = OUTFIG
analyze_forage.FIG = OUTFIG
analyze_forage.VARIANTS = [v for v in analyze_forage.VARIANTS if v != "OR_adapt"]
VAR_MRP = [v for v in experiments.MAIN if v != "OR_adapt"] + ["Alive"]


def mrp_bounds():
    for st, fname, title in (("L", "L_bounds_v3.png",
                              "Permutation-equivariant actor (dots = training seeds, bar = mean) vs reference policies"),
                             ("A", "A_bounds_v3.png",
                              "Standard MLP agent (dots = training seeds, bar = mean) vs reference policies")):
        S = pd.read_csv(os.path.join(AN, f"{st}_seed_level.csv"))
        if st == "L":
            S = S[S.tag == "shared"]
        B = pd.read_csv(os.path.join(AN, f"{st}_bounds.csv"))
        analyze.fig_bounds(S, B, VAR_MRP, fname, title)


def mrp_learning():
    _, meta = analyze.load("L_learn")
    analyze.fig_learning(meta, VAR_MRP, "L_learning_main_v3.png", tag="shared")


def forage_term():
    S = pd.read_csv(os.path.join(AN, "FORAGE_F_term_seed_level.csv"))
    S = S[S.kind == "eval"]
    R = pd.read_csv(os.path.join(AN, "FORAGE_F_term_references.csv"))
    analyze_forage.fig_bars(S, R, "rmst", "RMST survival, τ = 500", True, "FORAGE_F_term_rmst_v3.png",
                            "With termination")
    _, curves = analyze_forage.load("F_term")
    curves = [c for c in curves if c["variant"] != "OR_adapt"]
    analyze_forage.fig_learning(curves, "FORAGE_F_term_learning_v3.png", "With termination")


def landscape():
    J = pd.read_csv(os.path.join(AN, "FORAGE_Q4_grid_sign.csv"))
    conds = ["F1", "F2", "F3", "F4"]
    fig, axes = plt.subplots(1, len(conds), figsize=(4.3 * len(conds), 3.9))
    for ax, cond in zip(axes, conds):
        j = J[J.cond == cond]
        piv = j.pivot(index="z", columns="beta", values="mean")
        mr = j.pivot(index="z", columns="beta", values="mean_r")
        im = ax.imshow(piv.values, origin="lower", cmap="viridis", aspect="auto")
        ax.contour(mr.values, levels=[0.0], colors="w", linewidths=2, origin="lower")
        for (i, k), val in np.ndenumerate(piv.values):
            ax.text(k, i, f"{val:.0f}", ha="center", va="center", fontsize=7,
                    color="w" if val < np.nanmax(piv.values) * 0.6 else "k")
        ax.plot([B_GRID.index(1.0)], [Z_GRID.index(0.5)], marker="s", mfc="none", mec="r", ms=16, mew=1.5)
        ax.set_xticks(range(len(B_GRID)))
        ax.set_xticklabels([f"{b:g}" for b in B_GRID])
        ax.set_yticks(range(len(Z_GRID)))
        ax.set_yticklabels([f"{z:g}" for z in Z_GRID])
        ax.set_xlabel(r"$\beta$")
        k, d = FCONDS[cond]["k"], FCONDS[cond]["decay_mu"]
        ax.set_title(f"{cond} (k={k}, depletion {d})", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel(r"$z_{\mathrm{safe}}$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTFIG, "FORAGE_Q4_zbgrid_v3.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    mrp_bounds()
    forage_term()
    landscape()
    mrp_learning()
    print("written to", OUTFIG)
