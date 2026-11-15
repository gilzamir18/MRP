"""
analyze_outer.py — Q4 follow-up studies in the foraging task.

  * F_freeze: Adaptive OR with the actor frozen (pure circular drift of z, beta)
  * OR_outer: (z, beta) adapted by an outer loop on external fitness (in F_term / F_noterm)
  * F_zbgrid: fixed OR on a 5 x 5 (z, beta) grid, scored by survival (fitness landscape)

Writes docs/results_v2/analysis/FORAGE_Q4.md, CSVs and figures figs/FORAGE_Q4_*.png
"""
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
from analyze import RES, OUT, FIG, paired_test, holm, fmt_p  # noqa: E402

CONDS = ["F1", "F2", "F3", "F4"]
TAU = 500


def seed_metric(study, variants):
    rows = []
    for p in glob.glob(os.path.join(RES, study, "*.json")):
        r = json.load(open(p, encoding="utf-8"))
        j = r["job"]
        if j["variant"] not in variants:
            continue
        ev = r["eval"]
        f = r.get("or_final") or {}
        rows.append(dict(cond=j["cond"], variant=j["variant"], seed=j["seed"],
                         rmst=np.mean([min(e["length"], TAU) for e in ev]),
                         frac_viable=np.mean([e["frac_viable"] for e in ev]),
                         z=f.get("z_of_mean"), beta=f.get("beta_of_mean")))
    return pd.DataFrame(rows)


def contrasts(S, metric, pairs):
    rows = []
    for c in CONDS:
        s = S[S.cond == c]
        for v1, v2 in pairs:
            a = s[s.variant == v1].set_index("seed")[metric]
            b = s[s.variant == v2].set_index("seed")[metric]
            com = a.index.intersection(b.index)
            if len(com) < 3:
                continue
            rows.append(dict(cond=c, contrast=f"{v1} vs {v2}", mean1=a[com].mean(), mean2=b[com].mean(),
                             **paired_test(a[com].values, b[com].values)))
    D = pd.DataFrame(rows)
    if not D.empty:
        D["p_holm"] = np.nan
        for c, g in D.groupby("cond"):
            D.loc[g.index, "p_holm"] = holm(g.p_raw.values)
    return D


def main():
    rep = ["# Q4 follow-up: learned zone and penalty (foraging task)\n"]
    # ---------- frozen-policy drift ----------
    fr = []
    for p in glob.glob(os.path.join(RES, "F_freeze", "*.json")):
        r = json.load(open(p, encoding="utf-8"))
        fr.append(dict(term=r["terminate"], cond=r["cond"], seed=r["seed"],
                       z=r["or_final"]["z_of_mean"], beta=r["or_final"]["beta_of_mean"]))
    if fr:
        F = pd.DataFrame(fr).groupby(["term", "cond"]).agg(z=("z", "mean"), z_sd=("z", "std"),
                                                          beta=("beta", "mean"), beta_sd=("beta", "std"),
                                                          n=("z", "size"))
        F.to_csv(os.path.join(OUT, "FORAGE_freeze_params.csv"))
        rep += ["## Adaptive OR with a frozen actor (init z = 0.50, beta = 1.00)\n", F.round(3).to_markdown(), "\n"]
    # ---------- outer loop ----------
    V = ["OR_outer", "OR_adapt", "OR_fixed", "DQ"]
    for study, metric in (("F_term", "rmst"), ("F_noterm", "frac_viable")):
        S = seed_metric(study, V)
        if S.empty or "OR_outer" not in set(S.variant):
            continue
        S.to_csv(os.path.join(OUT, f"FORAGE_Q4_{study}_outer_seed_level.csv"), index=False)
        T = S.groupby(["cond", "variant"])[metric].mean().unstack()[[v for v in V if v in set(S.variant)]]
        P = S[S.variant.isin(["OR_outer", "OR_adapt"])].groupby(["variant", "cond"])[["z", "beta"]].agg(["mean", "std"])
        D = contrasts(S, metric, [("OR_outer", "OR_fixed"), ("OR_outer", "OR_adapt"), ("OR_outer", "DQ")])
        D.to_csv(os.path.join(OUT, f"FORAGE_Q4_{study}_outer_contrasts.csv"), index=False)
        rep += [f"## Outer-loop OR — {study} ({metric})\n", T.round(3).to_markdown(), "\n",
                "Final parameters (transform of the mean):\n", P.round(3).to_markdown(), "\n",
                D[["cond", "contrast", "mean1", "mean2", "diff", "ci_lo", "ci_hi", "p_raw", "p_holm"]].round(3).to_markdown(index=False), "\n"]
    # ---------- (z, beta) landscape ----------
    G = []
    for p in glob.glob(os.path.join(RES, "F_zbgrid", "*.json")):
        r = json.load(open(p, encoding="utf-8"))
        j = r["job"]
        _, zs, bs = j["variant"].split("_")
        G.append(dict(cond=j["cond"], z=float(zs[1:]), beta=float(bs[1:]), seed=j["seed"],
                      rmst=np.mean([min(e["length"], TAU) for e in r["eval"]])))
    if G:
        G = pd.DataFrame(G)
        G.to_csv(os.path.join(OUT, "FORAGE_Q4_zbgrid_seed_level.csv"), index=False)
        M = G.groupby(["cond", "z", "beta"]).rmst.agg(["mean", "std", "size"]).reset_index()
        M.to_csv(os.path.join(OUT, "FORAGE_Q4_zbgrid_means.csv"), index=False)
        rep.append("## Fitness landscape of fixed OR (RMST, mean over seeds)\n")
        fig, axes = plt.subplots(1, len(CONDS), figsize=(4.2 * len(CONDS), 3.8))
        for ax, c in zip(axes, CONDS):
            m = M[M.cond == c]
            if m.empty:
                continue
            piv = m.pivot(index="z", columns="beta", values="mean")
            rep += [f"**{c}** (n per cell: {int(m['size'].min())}-{int(m['size'].max())})\n", piv.round(0).to_markdown(), "\n"]
            im = ax.imshow(piv.values, origin="lower", cmap="viridis", aspect="auto")
            ax.set_xticks(range(piv.shape[1]))
            ax.set_xticklabels([f"{b:g}" for b in piv.columns])
            ax.set_yticks(range(piv.shape[0]))
            ax.set_yticklabels([f"{z:g}" for z in piv.index])
            for (i, jj), val in np.ndenumerate(piv.values):
                if not np.isnan(val):
                    ax.text(jj, i, f"{val:.0f}", ha="center", va="center", fontsize=7,
                            color="w" if val < np.nanmax(piv.values) * 0.6 else "k")
            ax.set_xlabel(r"$\beta$")
            ax.set_title(c, fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046)
        axes[0].set_ylabel(r"$z_{\mathrm{safe}}$")
        fig.suptitle("Fixed OR: survival (RMST, τ = 500) over the (z_safe, β) grid, foraging with termination", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "FORAGE_Q4_zbgrid.png"), dpi=160)
        plt.close(fig)
    with open(os.path.join(OUT, "FORAGE_Q4.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep))
    print("wrote", os.path.join(OUT, "FORAGE_Q4.md"))


if __name__ == "__main__":
    main()
