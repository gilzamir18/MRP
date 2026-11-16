"""
analyze_followup.py — follow-up studies of 2026-11-15 (foraging task).

  * F_init  : Adaptive (joint) and outer-loop OR started from BAD calibrations
              (punitive (0.30, 2), (0.30, 4); saturated (0.88, 0.10)), vs Fixed OR at the same point
              (F_zbgrid for the punitive points, F_init for the saturated one) and vs default Fixed OR.
  * F_ksweep: OR_fixed / DQ / Alive at k = 6, 8 (+ k = 2, 4 from F_term / F_noterm), both protocols.
  * F_long  : Adaptive vs Fixed OR at 1e7 steps (F1, F3), parameter drift.

Writes docs/results_v2/analysis/FORAGE_FOLLOWUP.md, CSVs, figs/FORAGE_init_traj.png, figs/FORAGE_ksweep.png
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
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze import RES, OUT, FIG, paired_test, holm  # noqa: E402

TAU = 500
CONDS = ["F1", "F2", "F3", "F4"]
INITS = [(0.30, 2.0), (0.30, 4.0), (0.88, 0.10)]
KOF = {"F1": 2, "F3": 4, "F5": 6, "F6": 8}


def rows(study, keep=None):
    out = []
    for p in glob.glob(os.path.join(RES, study, "*.json")):
        r = json.load(open(p, encoding="utf-8"))
        j = r["job"]
        if keep is not None and not keep(j):
            continue
        ev = r["eval"]
        f = r.get("or_final") or {}
        traj = r.get("or_traj") or []
        out.append(dict(study=study, cond=j["cond"], variant=j["variant"], seed=j["seed"], tag=j.get("tag"),
                        terminate=j["terminate"], timesteps=j["timesteps"],
                        rmst=np.mean([min(e["length"], TAU) for e in ev]),
                        frac_viable=np.mean([e["frac_viable"] for e in ev]),
                        z=f.get("z_of_mean"), beta=f.get("beta_of_mean"),
                        traj=[(s, d["z_of_mean"], d["beta_of_mean"]) for s, d in traj]))
    return pd.DataFrame(out)


def pair(A, B, metric="rmst", by=("cond",)):
    """paired by (cond, seed); returns per-cond tests + pooled test over all (cond, seed) pairs."""
    a = A.set_index(list(by) + ["seed"])[metric]
    b = B.set_index(list(by) + ["seed"])[metric]
    com = a.index.intersection(b.index)
    res = []
    for c in sorted(set(i[0] for i in com)):
        idx = [i for i in com if i[0] == c]
        res.append(dict(cond=c, mean1=a[idx].mean(), mean2=b[idx].mean(), **paired_test(a[idx].values, b[idx].values)))
    pooled = dict(cond="pooled", mean1=a[com].mean(), mean2=b[com].mean(), **paired_test(a[com].values, b[com].values))
    return res, pooled


def md(D, cols=None, nd=3):
    D = D if cols is None else D[cols]
    return D.round(nd).to_markdown(index=False)


def study_init(rep):
    I = rows("F_init")
    G = rows("F_zbgrid")
    T = rows("F_term", keep=lambda j: j["variant"] in ("OR_fixed", "OR_adapt", "OR_outer"))
    T = T[T.seed.isin(range(42, 47))]
    I.drop(columns="traj").to_csv(os.path.join(OUT, "FORAGE_init_seed_level.csv"), index=False)
    rep.append("## F_init — adaptation started from bad calibrations (termination, RMST, 5 seeds)\n")
    summ, tests = [], []
    for (z, b) in INITS:
        tag = f"init_z{z:.2f}_b{b:.2f}"
        fixed_name = f"ORg_z{z:.2f}_b{b:.2f}"
        fx = I[(I.tag == tag) & (I.variant == fixed_name)]
        if fx.empty:
            fx = G[G.variant == fixed_name]
        fx = fx[fx.seed.isin(range(42, 47))]
        ad = I[(I.tag == tag) & (I.variant == "OR_adapt")]
        ou = I[(I.tag == tag) & (I.variant == "OR_outer")]
        dflt = T[T.variant == "OR_fixed"]
        for c in CONDS:
            g = lambda D: D[D.cond == c]
            summ.append(dict(init=f"({z:.2f}, {b:.2f})", cond=c,
                             fixed_at_init=g(fx).rmst.mean(), adapt=g(ad).rmst.mean(), outer=g(ou).rmst.mean(),
                             default_fixed=g(dflt).rmst.mean(),
                             adapt_z=g(ad).z.mean(), adapt_beta=g(ad).beta.mean(),
                             outer_z=g(ou).z.mean(), outer_beta=g(ou).beta.mean()))
        for nm, A, B in (("adapt vs fixed@init", ad, fx), ("outer vs fixed@init", ou, fx),
                         ("adapt vs outer", ad, ou), ("adapt vs default fixed", ad, dflt),
                         ("outer vs default fixed", ou, dflt)):
            per, pooled = pair(A, B)
            for d in per + [pooled]:
                tests.append(dict(init=f"({z:.2f}, {b:.2f})", contrast=nm, **d))
    S = pd.DataFrame(summ)
    D = pd.DataFrame(tests)
    pooledD = D[D.cond == "pooled"].copy()
    pooledD["p_holm"] = np.nan
    for i, g in pooledD.groupby("init"):
        pooledD.loc[g.index, "p_holm"] = holm(g.p_raw.values)
    S.to_csv(os.path.join(OUT, "FORAGE_init_summary.csv"), index=False)
    D.to_csv(os.path.join(OUT, "FORAGE_init_contrasts.csv"), index=False)
    rep += ["Mean RMST per cell and final parameters (transform of the mean):\n", md(S, nd=2), "\n",
            "Pooled paired contrasts (20 cond x seed pairs per init; Holm within init):\n",
            md(pooledD, ["init", "contrast", "mean1", "mean2", "diff", "ci_lo", "ci_hi", "rbc", "p_raw", "p_holm"]), "\n",
            "Per-condition contrasts (n = 5; exact Wilcoxon minimum two-sided p = 0.0625):\n",
            md(D[D.cond != "pooled"], ["init", "contrast", "cond", "mean1", "mean2", "diff", "ci_lo", "ci_hi"]), "\n"]
    # trajectories
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    cols = {"OR_adapt": "C0", "OR_outer": "C3"}
    for (z, b), ls in zip(INITS + [(0.5, 1.0)], ["-", "--", ":", "-."]):
        if (z, b) == (0.5, 1.0):
            src = T[T.variant.isin(["OR_adapt", "OR_outer"])]
        else:
            src = I[(I.tag == f"init_z{z:.2f}_b{b:.2f}") & I.variant.isin(["OR_adapt", "OR_outer"])]
        for v, g in src.groupby("variant"):
            tr = [np.array(t) for t in g.traj if len(t)]
            n = min(len(t) for t in tr)
            M = np.mean([t[:n] for t in tr], axis=0)
            axes[0].plot(M[:, 0] / 1e6, M[:, 1], color=cols[v], ls=ls, lw=1.3,
                         label=f"{'joint' if v == 'OR_adapt' else 'outer'} from ({z:g}, {b:g})")
            axes[1].plot(M[:, 0] / 1e6, M[:, 2], color=cols[v], ls=ls, lw=1.3)
    axes[0].set_ylabel(r"$z_{\mathrm{safe}}$")
    axes[1].set_ylabel(r"$\beta$")
    for ax in axes:
        ax.set_xlabel("environment steps (M)")
    axes[0].legend(fontsize=6.5, ncol=2)
    fig.suptitle("Learned OR parameters from four initialisations (mean over F1-F4 x 5 seeds)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "FORAGE_init_traj.png"), dpi=160)
    plt.close(fig)


def study_ksweep(rep):
    K = rows("F_ksweep")
    base_t = rows("F_term", keep=lambda j: j["cond"] in ("F1", "F3") and j["variant"] in ("OR_fixed", "DQ", "Alive"))
    base_n = rows("F_noterm", keep=lambda j: j["cond"] in ("F1", "F3") and j["variant"] in ("OR_fixed", "DQ", "Alive"))
    A = pd.concat([K, base_t, base_n], ignore_index=True)
    A["k"] = A.cond.map(KOF)
    A.drop(columns="traj").to_csv(os.path.join(OUT, "FORAGE_ksweep_seed_level.csv"), index=False)
    rep.append("## F_ksweep — number of needs (decay 0.008; k = 2, 4 from the main study), 10 seeds\n")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (term, metric) in zip(axes, ((True, "rmst"), (False, "frac_viable"))):
        S = A[A.terminate == term]
        M = S.groupby(["k", "variant"])[metric].agg(["mean", "std"]).unstack()
        rep += [f"### {'termination (RMST)' if term else 'no termination (viable fraction)'}\n",
                M.round(3).to_markdown(), "\n"]
        tests = []
        for v1, v2 in (("OR_fixed", "DQ"), ("OR_fixed", "Alive"), ("DQ", "Alive")):
            for k in sorted(S.k.unique()):
                a = S[(S.k == k) & (S.variant == v1)].set_index("seed")[metric]
                b = S[(S.k == k) & (S.variant == v2)].set_index("seed")[metric]
                com = a.index.intersection(b.index)
                tests.append(dict(k=k, contrast=f"{v1} vs {v2}", mean1=a[com].mean(), mean2=b[com].mean(),
                                  **paired_test(a[com].values, b[com].values)))
        D = pd.DataFrame(tests)
        D["p_holm"] = np.nan
        for k, g in D.groupby("k"):
            D.loc[g.index, "p_holm"] = holm(g.p_raw.values)
        D.to_csv(os.path.join(OUT, f"FORAGE_ksweep_{'term' if term else 'noterm'}_contrasts.csv"), index=False)
        rep += [md(D, ["k", "contrast", "mean1", "mean2", "diff", "ci_lo", "ci_hi", "rbc", "p_raw", "p_holm"]), "\n"]
        # trend of the OR - DQ gap with k (Spearman over seed-level differences)
        gap = []
        for k in sorted(S.k.unique()):
            a = S[(S.k == k) & (S.variant == "OR_fixed")].set_index("seed")[metric]
            b = S[(S.k == k) & (S.variant == "DQ")].set_index("seed")[metric]
            com = a.index.intersection(b.index)
            gap += [(k, d) for d in (a[com] - b[com]).values]
        rho, p = stats.spearmanr([g[0] for g in gap], [g[1] for g in gap])
        rep.append(f"OR - DQ gap vs k: Spearman rho = {rho:.3f}, p = {p:.3g} (n = {len(gap)} seed-level differences)\n")
        for v, c in (("OR_fixed", "C0"), ("DQ", "C1"), ("Alive", "C2")):
            m = M[("mean", v)]
            s = M[("std", v)]
            ax.errorbar(m.index, m.values, yerr=s.values, marker="o", capsize=3, color=c, label=v.replace("_fixed", ""))
        ax.set_xlabel("number of needs k")
        ax.set_xticks([2, 4, 6, 8])
        ax.set_ylabel("RMST (τ = 500)" if term else "viable fraction")
        ax.set_title("with termination" if term else "without termination", fontsize=9)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "FORAGE_ksweep.png"), dpi=160)
    plt.close(fig)


def study_long(rep):
    L = rows("F_long")
    T = rows("F_term", keep=lambda j: j["cond"] in ("F1", "F3") and j["variant"] in ("OR_fixed", "OR_adapt"))
    T = T[T.seed.isin(range(42, 47))]
    rep.append("## F_long — 1e7 steps vs 3e6 (termination, RMST, 5 seeds)\n")
    rowsL = []
    for c in ("F1", "F3"):
        for v in ("OR_fixed", "OR_adapt"):
            l = L[(L.cond == c) & (L.variant == v)]
            t = T[(T.cond == c) & (T.variant == v)]
            rowsL.append(dict(cond=c, variant=v, rmst_3M=t.rmst.mean(), rmst_3M_sd=t.rmst.std(),
                              rmst_10M=l.rmst.mean(), rmst_10M_sd=l.rmst.std(),
                              z_3M=t.z.mean(), beta_3M=t.beta.mean(), z_10M=l.z.mean(), beta_10M=l.beta.mean()))
    R = pd.DataFrame(rowsL)
    R.to_csv(os.path.join(OUT, "FORAGE_long_summary.csv"), index=False)
    per, pooled = pair(L[L.variant == "OR_adapt"], L[L.variant == "OR_fixed"])
    D = pd.DataFrame(per + [pooled])
    per2, pooled2 = pair(L[L.variant == "OR_adapt"], T[T.variant == "OR_adapt"])
    D2 = pd.DataFrame(per2 + [pooled2])
    rep += [md(R, nd=3), "\n", "Adaptive vs Fixed at 1e7:\n",
            md(D, ["cond", "mean1", "mean2", "diff", "ci_lo", "ci_hi", "rbc", "p_raw"]), "\n",
            "Adaptive 1e7 vs Adaptive 3e6 (same seeds):\n",
            md(D2, ["cond", "mean1", "mean2", "diff", "ci_lo", "ci_hi", "rbc", "p_raw"]), "\n"]
    # drift over time
    for c in ("F1", "F3"):
        g = L[(L.cond == c) & (L.variant == "OR_adapt")]
        tr = [np.array(t) for t in g.traj]
        n = min(len(t) for t in tr)
        Mt = np.mean([t[:n] for t in tr], axis=0)
        pts = [Mt[np.argmin(np.abs(Mt[:, 0] - s))] for s in (1e6, 3e6, 5e6, 1e7)]
        rep.append(f"{c} adaptive drift (step, z, beta): " + "; ".join(f"{p[0]/1e6:.0f}M: {p[1]:.3f}, {p[2]:.3f}" for p in pts) + "\n")


def main():
    rep = ["# Follow-up studies (2026-11-15): bad initialisation, number of needs, long training\n"]
    study_init(rep)
    study_ksweep(rep)
    study_long(rep)
    with open(os.path.join(OUT, "FORAGE_FOLLOWUP.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep))
    print("wrote", os.path.join(OUT, "FORAGE_FOLLOWUP.md"))


if __name__ == "__main__":
    main()
