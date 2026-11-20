"""
replot_figs_v4.py — re-draw the three MRP figures of the paper from the v3 re-run (corrected
dynamics, docs/results_v3_mrp), with the same labels as replot_figs_v3.py ("OR" instead of
"OR (Fixed)", "permutation-equivariant actor", "Uniform (≈ untrained net)").

Copy of replot_figs_v3.py restricted to the MRP figures (L_bounds, A_bounds, L_learning_main);
the foraging figures are unchanged and are not redrawn.  No training and no new statistics: every
figure is drawn from the seed-level CSVs / JSON already in docs/results_v3_mrp.  Existing files are
not touched; the new files get the suffix _v4 and are written to output/figs_v2.

Usage:  python replot_figs_v4.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["MRP_RES"] = os.path.abspath(os.path.join(HERE, "..", "docs", "results_v3_mrp"))
sys.path.insert(0, HERE)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import pandas as pd  # noqa: E402

import experiments  # noqa: E402
for lst in (experiments.MAIN, experiments.SIGN, experiments.GEOM, experiments.ALL_VARIANTS):
    while "OR_adapt" in lst:
        lst.remove("OR_adapt")
import analyze  # noqa: E402

OUTFIG = os.path.abspath(os.path.join(HERE, "..", "..", "output", "figs_v2"))
AN = os.path.join(os.environ["MRP_RES"], "analysis")
assert analyze.RES == os.environ["MRP_RES"], analyze.RES

# labels used in the paper
analyze.LABEL["OR_fixed"] = "OR"
analyze.REF_LABEL["zero"] = "Uniform (≈ untrained net)"
analyze.FIG = OUTFIG
VAR_MRP = [v for v in experiments.MAIN if v != "OR_adapt"] + ["Alive"]

OUT_NAMES = ("L_bounds_v4.png", "A_bounds_v4.png", "L_learning_main_v4.png")


def mrp_bounds():
    for st, fname, title in (("L", "L_bounds_v4.png",
                              "Permutation-equivariant actor (dots = training seeds, bar = mean) vs reference policies"),
                             ("A", "A_bounds_v4.png",
                              "Standard MLP agent (dots = training seeds, bar = mean) vs reference policies")):
        S = pd.read_csv(os.path.join(AN, f"{st}_seed_level.csv"))
        if st == "L":
            S = S[S.tag == "shared"]
        B = pd.read_csv(os.path.join(AN, f"{st}_bounds.csv"))
        analyze.fig_bounds(S, B, VAR_MRP, fname, title)


def mrp_learning():
    _, meta = analyze.load("L_learn")
    analyze.fig_learning(meta, VAR_MRP, "L_learning_main_v4.png", tag="shared")


if __name__ == "__main__":
    for f in OUT_NAMES:
        if os.path.exists(os.path.join(OUTFIG, f)):
            raise SystemExit(f"{f} already exists in {OUTFIG}; refusing to overwrite")
    mrp_bounds()
    mrp_learning()
    print("written to", OUTFIG, OUT_NAMES)
