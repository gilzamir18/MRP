"""
analyze.py — statistics, tables and figures for the revised MRP experiments.

Inferential unit = TRAINING SEED.  For each (condition, variant, seed) the
outcome is the restricted mean survival time at tau = T_max = 1000,
RMST = mean_k min(L_k, tau) over the 20 paired evaluation episodes.  Because
the only censoring is administrative at tau, the Kaplan-Meier RMST equals this
truncated mean exactly; KM curves are reported descriptively.

Tests (all paired by training seed):
  * Friedman omnibus across the 4 main treatments.
  * Two-sided exact Wilcoxon signed-rank tests for pre-specified contrasts.
    Every table reports p_raw, p_holm and p_bonf (within-condition family), and
    p_holm_global (Holm across all conditions and contrasts of that table).
  * Effect size: mean paired difference with 95% percentile bootstrap CI over
    seeds (10 000 resamples), and matched-pairs rank-biserial correlation.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from experiments import MAIN, SIGN, GEOM, ALL_VARIANTS, INITS  # noqa: E402
from runner import CONDITIONS  # noqa: E402

RES = os.path.abspath(os.environ.get("MRP_RES", os.path.join(HERE, "..", "docs", "results_v2")))
OUT = os.path.join(RES, "analysis")
FIG = os.path.join(OUT, "figs")
TAU = 1000
LABEL = {"OR_adapt": "OR (Adaptive)", "OR_fixed": "OR (Fixed)", "DQ": "DQ", "DR": "DR",
         "Alive": "Alive (+1)", "DR_alive": "DR + alive", "DQ_alive": "DQ + alive",
         "DeadZone": "Dead-zone", "OR_noZone": "OR no zone", "OR_noPen": "OR no penalty",
         "OR_min": "OR min-agg", "OR_softmin": "OR soft-min"}
COLOR = {"OR_adapt": "#c0392b", "OR_fixed": "#e67e22", "DQ": "#2e6fd8", "DR": "#27a05a",
         "Alive": "#7f8c8d", "DR_alive": "#16a085", "DQ_alive": "#5dade2",
         "DeadZone": "#8e44ad", "OR_noZone": "#d35400", "OR_noPen": "#f1c40f",
         "OR_min": "#6d4c41", "OR_softmin": "#ad1457"}


# ─────────────────────────────────────────────────────────────────────────────
def load(study):
    rows, meta = [], []
    for p in glob.glob(os.path.join(RES, study, "*.json")):
        r = json.load(open(p))
        j = r["job"]
        base = dict(study=study, cond=j["cond"], variant=j["variant"], tag=j.get("tag", "base"),
                    seed=j["seed"], n=r["cond_params"]["n"], noise=r["cond_params"]["noise"],
                    budget=r["cond_params"]["budget"])
        for e in r["eval"]:
            # NB: the eval episode's "seed" must not overwrite the TRAINING seed
            rows.append({**base, **{("eval_seed" if k == "seed" else k): v for k, v in e.items()}})
        m = dict(base)
        m["train_step"] = r["train_curve"]["step"]
        m["train_len"] = r["train_curve"]["len"]
        m["or_final"] = r["or_final"]
        m["or_traj"] = r["or_traj"]
        m["wall_s"] = r["wall_s"]
        meta.append(m)
    return pd.DataFrame(rows), meta


def seed_level(ep):
    g = ep.groupby(["study", "cond", "variant", "tag", "seed", "n", "noise", "budget"])
    s = g.agg(rmst=("length", lambda x: np.mean(np.minimum(x, TAU))),
              event_rate=("event", "mean"),
              frac_in_zone=("frac_in_zone", "mean"),
              frac_nonrec=("frac_nonrecovery", "mean"),
              alloc_entropy=("alloc_entropy", "mean"),
              alloc_dev_corr=("alloc_dev_corr", "mean"),
              top10_share=("top10_share", "mean")).reset_index()
    s["surv_tau"] = 1.0 - s["event_rate"]
    return s


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    order = np.argsort(pv)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pv[i])
        adj[i] = min(1.0, run)
    out[ok] = adj
    return out


def boot_ci(d, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    if len(d) < 2:
        return np.nan, np.nan
    bs = rng.choice(d, (n_boot, len(d)), replace=True).mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired_test(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    lo, hi = boot_ci(d)
    nz = d[d != 0]
    if len(nz) == 0:
        return dict(W=np.nan, p_raw=1.0, diff=0.0, ci_lo=0.0, ci_hi=0.0, rbc=0.0, n=len(d))
    res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided",
                         method="exact" if len(nz) == len(d) and len(nz) <= 25 else "auto")
    r = stats.rankdata(np.abs(nz))
    rbc = float((r[nz > 0].sum() - r[nz < 0].sum()) / r.sum())
    return dict(W=float(res.statistic), p_raw=float(res.pvalue), diff=float(d.mean()),
                ci_lo=lo, ci_hi=hi, rbc=rbc, n=len(d))


def contrast_table(S, conds, pairs, family_name):
    rows = []
    for c in conds:
        sub = S[S.cond == c]
        for (v1, v2) in pairs:
            a = sub[sub.variant == v1].set_index("seed")["rmst"]
            b = sub[sub.variant == v2].set_index("seed")["rmst"]
            common = a.index.intersection(b.index)
            if len(common) < 3:
                continue
            t = paired_test(a.loc[common].values, b.loc[common].values)
            rows.append(dict(family=family_name, cond=c, contrast=f"{LABEL.get(v1, v1)} vs {LABEL.get(v2, v2)}",
                             v1=v1, v2=v2, mean1=a.loc[common].mean(), mean2=b.loc[common].mean(), **t))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["p_holm"] = np.nan
    df["p_bonf"] = np.nan
    for c, g in df.groupby("cond"):
        df.loc[g.index, "p_holm"] = holm(g.p_raw.values)
        df.loc[g.index, "p_bonf"] = np.minimum(1.0, g.p_raw.values * len(g))
    df["p_holm_global"] = holm(df.p_raw.values)
    return df


def fmt_p(p):
    if p is None or np.isnan(p):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def summary_table(S, conds, variants, friedman_variants=None):
    rows = []
    for c in conds:
        sub = S[S.cond == c]
        row = dict(cond=c, n=int(sub.n.iloc[0]), noise=float(sub.noise.iloc[0]),
                   budget=float(sub.budget.iloc[0]))
        for v in variants:
            x = sub[sub.variant == v]["rmst"].values
            if len(x):
                lo, hi = boot_ci(x)
                row[f"{v}_mean"] = x.mean()
                row[f"{v}_sd"] = x.std(ddof=1) if len(x) > 1 else np.nan
                row[f"{v}_lo"], row[f"{v}_hi"] = lo, hi
                row[f"{v}_k"] = len(x)
                row[f"{v}_surv"] = sub[sub.variant == v]["surv_tau"].mean()
        fv = friedman_variants or variants
        piv = sub[sub.variant.isin(fv)].pivot_table(index="seed", columns="variant", values="rmst").dropna()
        if piv.shape[1] >= 3 and piv.shape[0] >= 3:
            st, p = stats.friedmanchisquare(*[piv[v].values for v in fv if v in piv])
            row["friedman_chi2"], row["friedman_p"] = st, p
            # Kendall's W
            row["kendall_W"] = st / (piv.shape[0] * (piv.shape[1] - 1))
        rows.append(row)
    return pd.DataFrame(rows)


def md_table(df, cols, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(f(r)) if callable(f) else str(r[f]) for f in cols) + " |")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
def km_curve(lengths, events, tau=TAU):
    t = np.arange(0, tau + 1)
    L = np.asarray(lengths)
    E = np.asarray(events, bool)
    surv = np.ones_like(t, dtype=float)
    s = 1.0
    for i, tt in enumerate(t):
        at_risk = (L >= tt).sum()
        d = ((L == tt) & E).sum()
        if at_risk > 0 and d > 0:
            s *= 1 - d / at_risk
        surv[i] = s
    return t, surv


def fig_km(ep, variants, fname, title):
    conds = list(CONDITIONS)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
    for ax, c in zip(axes.flat, conds):
        sub = ep[ep.cond == c]
        for v in variants:
            x = sub[sub.variant == v]
            if x.empty:
                continue
            t, s = km_curve(x.length, x.event)
            ax.step(t, s, where="post", color=COLOR[v], label=LABEL[v], lw=1.8,
                    ls="--" if v in ("OR_fixed",) or v in GEOM or v in SIGN else "-")
        p = CONDITIONS[c]
        ax.set_title(f"{c}: n={p['n']}, σ={p['noise']}, b={p['budget']}", fontsize=10)
        ax.set_xlabel("step")
        mx = sub.length.max() if not sub.empty else TAU
        ax.set_xlim(0, min(TAU, max(50, mx * 1.05)))
        ax.grid(alpha=.3)
    axes[0, 0].set_ylabel("survival probability")
    axes[1, 0].set_ylabel("survival probability")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=160)
    plt.close(fig)


def fig_learning(meta, variants, fname, total=100_000, study="A_main", tag="base"):
    grid = np.linspace(0, total, 51)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, c in zip(axes.flat, CONDITIONS):
        for v in variants:
            curves = []
            for m in meta:
                if m["cond"] != c or m["variant"] != v or m["tag"] != tag:
                    continue
                st, ln = np.array(m["train_step"]), np.array(m["train_len"])
                if len(st) < 2:
                    continue
                # mean episode length of episodes ending in each bin
                y = np.full(len(grid) - 1, np.nan)
                for i in range(len(grid) - 1):
                    msk = (st > grid[i]) & (st <= grid[i + 1])
                    if msk.any():
                        y[i] = ln[msk].mean()
                y = pd.Series(y).ffill().bfill().values
                curves.append(y)
            if not curves:
                continue
            C = np.array(curves)
            mu = C.mean(0)
            se = C.std(0, ddof=1) / np.sqrt(len(C))
            x = (grid[1:] + grid[:-1]) / 2
            ax.plot(x, mu, color=COLOR[v], label=LABEL[v], lw=1.8)
            ax.fill_between(x, mu - 1.96 * se, mu + 1.96 * se, color=COLOR[v], alpha=.15)
        p = CONDITIONS[c]
        ax.set_title(f"{c}: n={p['n']}, σ={p['noise']}, b={p['budget']}", fontsize=10)
        ax.set_xlabel("training steps")
        ax.grid(alpha=.3)
    axes[0, 0].set_ylabel("training episode length")
    axes[1, 0].set_ylabel("training episode length")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Learning curves (mean ± 95% CI over 10 training seeds)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=160)
    plt.close(fig)


def fig_forest(ct, fname, title):
    if ct.empty:
        return
    conds = list(dict.fromkeys(ct.cond))
    contrasts = list(dict.fromkeys(ct.contrast))
    fig, axes = plt.subplots(1, len(conds), figsize=(3.2 * len(conds), 0.45 * len(contrasts) + 1.8),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, conds):
        g = ct[ct.cond == c].set_index("contrast").reindex(contrasts)
        y = np.arange(len(contrasts))[::-1]
        for yi, (_, r) in zip(y, g.iterrows()):
            if np.isnan(r["diff"]):
                continue
            sig = r["p_holm"] < 0.05
            ax.errorbar(r["diff"], yi, xerr=[[r["diff"] - r["ci_lo"]], [r["ci_hi"] - r["diff"]]],
                        fmt="o", color="#c0392b" if sig else "#555", ms=5, capsize=2)
        ax.axvline(0, color="k", lw=.8)
        ax.set_title(c, fontsize=10)
        ax.grid(alpha=.3)
        ax.set_yticks(y)
        ax.set_yticklabels(contrasts, fontsize=8)
    fig.suptitle(title + "  (red: Holm-adjusted p < 0.05 within condition)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=160)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def analyze_main(report):
    ep, meta = load("A_main")
    if ep.empty:
        return None, None
    S = seed_level(ep)
    S.to_csv(os.path.join(OUT, "A_seed_level.csv"), index=False)
    conds = list(CONDITIONS)

    # Table 5 replacement (main treatments)
    T5 = summary_table(S, conds, ALL_VARIANTS, friedman_variants=MAIN)
    T5.to_csv(os.path.join(OUT, "A_table_rmst_all_variants.csv"), index=False)
    main_pairs = [("OR_adapt", "OR_fixed"), ("OR_adapt", "DQ"), ("OR_adapt", "DR"),
                  ("OR_fixed", "DQ"), ("OR_fixed", "DR"), ("DQ", "DR")]
    T6 = contrast_table(S, conds, main_pairs, "main")
    T6.to_csv(os.path.join(OUT, "A_table_main_contrasts.csv"), index=False)

    # controls: sign/offset & geometry
    sign_pairs = [("OR_adapt", "Alive"), ("OR_fixed", "Alive"), ("DR_alive", "DR"),
                  ("DQ_alive", "DQ"), ("OR_fixed", "DR_alive"), ("OR_fixed", "DQ_alive")]
    Tsign = contrast_table(S, conds, sign_pairs, "sign")
    Tsign.to_csv(os.path.join(OUT, "A_table_sign_controls.csv"), index=False)
    geom_pairs = [("OR_fixed", "DeadZone"), ("OR_fixed", "OR_noZone"), ("OR_fixed", "OR_noPen"),
                  ("OR_fixed", "OR_min"), ("OR_fixed", "OR_softmin"), ("OR_noPen", "Alive")]
    Tgeom = contrast_table(S, conds, geom_pairs, "geometry")
    Tgeom.to_csv(os.path.join(OUT, "A_table_geometry_controls.csv"), index=False)

    # figures
    fig_km(ep, MAIN, "A_km_main.png", "Kaplan–Meier survival, main treatments (20 eval episodes × 10 seeds)")
    fig_km(ep, ["OR_adapt", "OR_fixed", "Alive", "DR_alive", "DQ_alive", "DR", "DQ"], "A_km_sign.png",
           "Kaplan–Meier survival: reward sign/offset controls")
    fig_km(ep, ["OR_fixed"] + GEOM, "A_km_geometry.png", "Kaplan–Meier survival: regime-geometry controls")
    fig_learning(meta, MAIN, "A_learning_main.png")
    fig_learning(meta, ["OR_adapt", "OR_fixed", "Alive", "DR_alive", "DQ_alive"], "A_learning_sign.png")
    fig_forest(T6, "A_forest_main.png", "Main contrasts: paired RMST difference (95% bootstrap CI)")
    fig_forest(Tsign, "A_forest_sign.png", "Sign/offset controls")
    fig_forest(Tgeom, "A_forest_geometry.png", "Regime-geometry controls")

    # bar overview of all variants
    fig, ax = plt.subplots(figsize=(18, 5.5))
    x = np.arange(len(conds))
    w = 0.8 / len(ALL_VARIANTS)
    for i, v in enumerate(ALL_VARIANTS):
        mu = [T5.loc[T5.cond == c, f"{v}_mean"].values[0] for c in conds]
        lo = [T5.loc[T5.cond == c, f"{v}_lo"].values[0] for c in conds]
        hi = [T5.loc[T5.cond == c, f"{v}_hi"].values[0] for c in conds]
        ax.bar(x + (i - len(ALL_VARIANTS) / 2 + .5) * w, mu, w, color=COLOR[v], label=LABEL[v],
               yerr=[np.array(mu) - lo, np.array(hi) - mu], capsize=1.5, error_kw=dict(lw=.7))
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\nn={CONDITIONS[c]['n']} σ={CONDITIONS[c]['noise']} b={CONDITIONS[c]['budget']}"
                        for c in conds], fontsize=8)
    ax.set_ylabel("RMST (τ=1000), seed-level mean, log scale")
    ax.legend(ncol=6, fontsize=8, loc="upper right")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "A_bar_all_variants.png"), dpi=160)
    plt.close(fig)

    # behaviour
    B = S.groupby(["cond", "variant"])[["frac_in_zone", "frac_nonrec", "alloc_entropy", "alloc_dev_corr",
                                        "top10_share"]].mean().reset_index()
    B.to_csv(os.path.join(OUT, "A_behaviour.csv"), index=False)

    # ---------- report text ----------
    report.append("## A. Main study (10 training seeds × 20 paired evaluation episodes)\n")
    report.append("Outcome: seed-level RMST at τ = 1000 (mean ± SD across seeds; [95% bootstrap CI]). "
                  "Friedman across the four main treatments, blocked by training seed.\n")
    cols = ["cond", lambda r: f"{r['n']:.0f}", lambda r: r["noise"], lambda r: r["budget"]]
    hdr = ["Cond", "n", "σ", "b"]
    for v in MAIN:
        cols.append(lambda r, v=v: f"{r[f'{v}_mean']:.1f} ± {r[f'{v}_sd']:.1f} [{r[f'{v}_lo']:.0f}, {r[f'{v}_hi']:.0f}]")
        hdr.append(LABEL[v])
    cols.append(lambda r: f"χ²={r['friedman_chi2']:.2f}, p={fmt_p(r['friedman_p'])}, W={r['kendall_W']:.2f}")
    hdr.append("Friedman")
    report.append(md_table(T5, cols, hdr) + "\n")
    report.append("\nSurvival to τ (fraction of evaluation episodes reaching T_max):\n")
    cols = ["cond"] + [lambda r, v=v: f"{r.get(f'{v}_surv', np.nan):.2f}" for v in ALL_VARIANTS]
    report.append(md_table(T5, cols, ["Cond"] + [LABEL[v] for v in ALL_VARIANTS]) + "\n")
    report.append("\nAll 12 variants, RMST mean ± SD:\n")
    cols = ["cond"] + [lambda r, v=v: f"{r[f'{v}_mean']:.1f} ± {r[f'{v}_sd']:.1f}" for v in ALL_VARIANTS]
    report.append(md_table(T5, cols, ["Cond"] + [LABEL[v] for v in ALL_VARIANTS]) + "\n")

    def ctab(df, title):
        report.append(f"\n### {title}\n")
        report.append("p_raw = unadjusted exact Wilcoxon signed-rank (two-sided, n = 10 seeds); "
                      "p_holm / p_bonf = adjusted within condition (family = rows of this table for that "
                      "condition); p_holm_glob = Holm across all rows of the table.\n")
        cols = ["cond", "contrast", lambda r: f"{r['diff']:+.1f} [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]",
                lambda r: f"{r['rbc']:+.2f}", lambda r: f"{r['W']:.1f}", lambda r: fmt_p(r["p_raw"]),
                lambda r: fmt_p(r["p_holm"]), lambda r: fmt_p(r["p_bonf"]), lambda r: fmt_p(r["p_holm_global"])]
        report.append(md_table(df, cols, ["Cond", "Contrast", "Δ RMST [95% CI]", "r_rb", "W", "p_raw",
                                          "p_holm", "p_bonf", "p_holm_glob"]) + "\n")
    ctab(T6, "A.1 Main pairwise contrasts")
    ctab(Tsign, "A.2 Reward sign / offset controls")
    ctab(Tgeom, "A.3 Regime-geometry controls")
    report.append("\n### A.4 Behavioural diagnostics (evaluation, mean over seeds)\n")
    report.append("frac_in_zone = fraction of variables with |v| ≤ 0.5; alloc_dev_corr = correlation between "
                  "allocation and current deviation; top10_share = budget share given to the 10% most deviated "
                  "variables (uniform = 0.10); alloc_entropy normalised to [0,1].\n")
    Bm = B[B.variant.isin(MAIN + ["Alive"])]
    report.append(md_table(Bm, ["cond", lambda r: LABEL[r["variant"]], lambda r: f"{r['frac_in_zone']:.2f}",
                                lambda r: f"{r['alloc_dev_corr']:.2f}", lambda r: f"{r['top10_share']:.3f}",
                                lambda r: f"{r['alloc_entropy']:.3f}"],
                           ["Cond", "Variant", "in zone", "corr(alloc, dev)", "top-10% share", "entropy"]) + "\n")
    Bd = bounds_table(S, MAIN + ["Alive"], "A")
    if Bd is not None:
        report_bounds(report, Bd, MAIN + ["Alive"], "A.5 Trained agents vs non-learning reference policies")
        fig_bounds(S, Bd, MAIN + ["Alive"], "A_bounds.png",
                   "Trained agents (dots = seeds, bar = mean) vs reference policies (lines)")
    return S, meta


REF_POL = ["zero", "random", "prop", "greedy", "oracle"]
REF_LABEL = {"zero": "Uniform (= untrained net)", "random": "Random logits", "prop": "Proportional |v|",
             "greedy": "Greedy max |v|", "oracle": "Water-filling oracle"}


def bounds_table(S, variants, study_label):
    """Trained agents vs non-learning reference policies (same evaluation episodes)."""
    path = os.path.join(OUT, "policy_bounds_episodes.csv")
    if not os.path.exists(path):
        return None
    PB = pd.read_csv(path)
    PB["rmst"] = np.minimum(PB.length, TAU)
    ref = PB.groupby(["cond", "policy"]).rmst.mean().unstack()
    rows = []
    for c in ref.index:
        sub = S[S.cond == c]
        if sub.empty:
            continue
        row = dict(cond=c, **{p: ref.loc[c, p] for p in REF_POL})
        for v in variants:
            x = sub[sub.variant == v].rmst.values
            if len(x) < 3:
                continue
            d = x - ref.loc[c, "zero"]
            p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            row[f"{v}_mean"] = x.mean()
            row[f"{v}_vs_zero"] = d.mean()
            row[f"{v}_p_raw"] = p
            row[f"{v}_frac_oracle"] = x.mean() / ref.loc[c, "oracle"]
        rows.append(row)
    B = pd.DataFrame(rows)
    pc = [k for k in B.columns if k.endswith("_p_raw")]
    within = np.vstack([holm(row) for row in B[pc].values])      # family = variants within a condition
    flat = holm(B[pc].values.ravel()).reshape(B[pc].shape)       # family = whole table
    for i, k in enumerate(pc):
        B[k.replace("_p_raw", "_p_holm")] = within[:, i]
        B[k.replace("_p_raw", "_p_holm_glob")] = flat[:, i]
    B.to_csv(os.path.join(OUT, f"{study_label}_bounds.csv"), index=False)
    return B


def report_bounds(report, B, variants, title):
    report.append(f"\n### {title}\n")
    report.append("Reference policies are evaluated on the same 20 evaluation episodes. `Uniform` (logits = 0) is "
                  "what the untrained, deterministically evaluated network does (output-layer gain 0.01), so it "
                  "is the no-learning baseline. Δ0 = trained RMST − uniform RMST (mean over seeds); "
                  "p = one-sample exact Wilcoxon over seeds (n = 10, so the smallest attainable two-sided p is "
                  "0.002); p_holm = Holm within condition (family = the variants in the row); the Holm "
                  "adjustment over the whole table is in the CSV (column *_p_holm_glob).\n")
    cols = ["cond"] + [lambda r, p=p: f"{r[p]:.0f}" for p in REF_POL]
    hdr = ["Cond"] + [REF_LABEL[p] for p in REF_POL]
    for v in variants:
        if f"{v}_mean" not in B:
            continue
        cols.append(lambda r, v=v: f"{r[f'{v}_mean']:.0f} (Δ0 {r[f'{v}_vs_zero']:+.0f}; "
                                   f"p={fmt_p(r[f'{v}_p_raw'])}/{fmt_p(r[f'{v}_p_holm'])})")
        hdr.append(LABEL[v] + " (p_raw/p_holm)")
    report.append(md_table(B, cols, hdr) + "\n")


def fig_bounds(S, B, variants, fname, title):
    conds = list(B.cond)
    fig, axes = plt.subplots(1, len(conds), figsize=(2.3 * len(conds), 4.2))
    axes = np.atleast_1d(axes)
    mk = {"zero": ("k", "_"), "random": ("0.5", "_"), "prop": ("#1f77b4", "^"),
          "greedy": ("#9467bd", "s"), "oracle": ("#d62728", "*")}
    for ax, c in zip(axes, conds):
        for i, v in enumerate(variants):
            x = S[(S.cond == c) & (S.variant == v)].rmst.values
            ax.scatter(np.full(len(x), i) + np.random.default_rng(i).uniform(-.15, .15, len(x)), x,
                       s=10, color=COLOR[v], alpha=.8)
            ax.plot([i - .3, i + .3], [x.mean()] * 2, color=COLOR[v], lw=2)
        r = B[B.cond == c].iloc[0]
        for p in REF_POL:
            col, m = mk[p]
            ax.axhline(r[p], color=col, lw=.9, ls="--" if p in ("zero", "random") else ":")
            ax.plot(len(variants) + .2, r[p], marker=m, color=col, ms=8, ls="none",
                    label=REF_LABEL[p] if c == conds[0] else None)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([LABEL[v] for v in variants], rotation=60, fontsize=7)
        ax.set_title(c, fontsize=9)
        ax.set_yscale("log")
        ax.grid(alpha=.3, axis="y")
    axes[0].set_ylabel("RMST (τ = 1000), per training seed")
    fig.legend(loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(os.path.join(FIG, fname), dpi=160)
    plt.close(fig)


def analyze_learn(report, S_main):
    """Stage L: same comparisons with the permutation-equivariant (shared) actor."""
    ep, meta = load("L_learn")
    if ep.empty:
        return None
    S_all = seed_level(ep)
    S_all.to_csv(os.path.join(OUT, "L_seed_level.csv"), index=False)
    S = S_all[S_all.tag == "shared"]
    epS = ep[ep.tag == "shared"]
    conds = [c for c in CONDITIONS if c in set(S.cond)]
    VL = MAIN + SIGN + [v for v in GEOM if v in set(S.variant)]

    T = summary_table(S, conds, VL, friedman_variants=MAIN)
    T.to_csv(os.path.join(OUT, "L_table_rmst.csv"), index=False)
    main_pairs = [("OR_adapt", "OR_fixed"), ("OR_adapt", "DQ"), ("OR_adapt", "DR"),
                  ("OR_fixed", "DQ"), ("OR_fixed", "DR"), ("DQ", "DR")]
    sign_pairs = [("OR_adapt", "Alive"), ("OR_fixed", "Alive"), ("DR_alive", "DR"),
                  ("DQ_alive", "DQ"), ("OR_fixed", "DR_alive"), ("OR_fixed", "DQ_alive")]
    CM = contrast_table(S, conds, main_pairs, "L_main")
    CS = contrast_table(S, conds, sign_pairs, "L_sign")
    CM.to_csv(os.path.join(OUT, "L_table_main_contrasts.csv"), index=False)
    CS.to_csv(os.path.join(OUT, "L_table_sign_controls.csv"), index=False)

    # architecture effect: shared actor vs paper MLP (stage A), same reward, same seed
    rows = []
    for c in conds:
        for v in VL:
            a = S[(S.cond == c) & (S.variant == v)].set_index("seed").rmst
            b = S_main[(S_main.cond == c) & (S_main.variant == v)].set_index("seed").rmst
            com = a.index.intersection(b.index)
            if len(com) >= 3:
                rows.append(dict(cond=c, variant=v, contrast=f"{LABEL[v]}: shared − MLP",
                                 mean_shared=a.loc[com].mean(), mean_mlp=b.loc[com].mean(),
                                 **paired_test(a.loc[com].values, b.loc[com].values)))
    CA = pd.DataFrame(rows)
    if not CA.empty:
        CA["p_holm"] = CA.groupby("cond").p_raw.transform(lambda p: holm(p.values))
        CA["p_holm_global"] = holm(CA.p_raw.values)
        CA.to_csv(os.path.join(OUT, "L_table_arch_effect.csv"), index=False)

    # learned OR parameters with a learning agent
    P = pd.DataFrame([dict(cond=m["cond"], seed=m["seed"], **m["or_final"]) for m in meta
                      if m["variant"] == "OR_adapt" and m["tag"] == "shared" and m["or_final"]])
    if not P.empty:
        P.to_csv(os.path.join(OUT, "L_learned_params.csv"), index=False)

    # behaviour
    B = S.groupby(["cond", "variant"])[["frac_in_zone", "alloc_dev_corr", "top10_share",
                                        "alloc_entropy"]].mean().reset_index()
    B.to_csv(os.path.join(OUT, "L_behaviour.csv"), index=False)

    # long training subset
    SL = S_all[S_all.tag == "shared_long300k"]

    # figures
    fig_km(epS, MAIN, "L_km_main.png", "Kaplan–Meier survival, shared actor (20 eval episodes × 10 seeds)")
    fig_km(epS, ["OR_adapt", "OR_fixed", "Alive", "DR_alive", "DQ_alive", "DR", "DQ"], "L_km_sign.png",
           "Kaplan–Meier survival, shared actor: reward sign/offset controls")
    fig_learning(meta, MAIN + ["Alive"], "L_learning_main.png", tag="shared")
    fig_forest(CM, "L_forest_main.png", "Shared actor — main contrasts: paired RMST difference (95% bootstrap CI)")
    fig_forest(CS, "L_forest_sign.png", "Shared actor — sign/offset controls")
    Bd = bounds_table(S, MAIN + ["Alive"], "L")
    if Bd is not None:
        fig_bounds(S, Bd, MAIN + ["Alive"], "L_bounds.png",
                   "Shared actor (dots = seeds, bar = mean) vs reference policies (lines)")

    # ---------- report ----------
    report.append("\n## L. Learnability: the same comparison with an agent that learns\n")
    report.append("Stage A agents (paper architecture: one 2×64 tanh MLP over the 4n+1 observation) do not learn: "
                  "training-episode length does not increase and the deterministic policy ends at or below the "
                  "untrained network (= uniform allocation; Section A.5). Stage L repeats the main and sign "
                  "comparisons with a permutation-equivariant actor: one small MLP (5→64→64→1, tanh) applied to "
                  "each variable's features (v_i, |v_i|, 100·decay_i, b, mean|v|), giving that variable's logit; "
                  "critic, PPO settings, budget (100k steps), seeds and evaluation episodes are identical to A.\n")
    cols = ["cond", lambda r: f"{r['n']:.0f}", lambda r: r["noise"], lambda r: r["budget"]]
    hdr = ["Cond", "n", "σ", "b"]
    for v in MAIN:
        cols.append(lambda r, v=v: f"{r[f'{v}_mean']:.1f} ± {r[f'{v}_sd']:.1f} [{r[f'{v}_lo']:.0f}, {r[f'{v}_hi']:.0f}]")
        hdr.append(LABEL[v])
    cols.append(lambda r: f"χ²={r['friedman_chi2']:.2f}, p={fmt_p(r['friedman_p'])}, W={r['kendall_W']:.2f}"
                if "friedman_chi2" in r and not pd.isna(r["friedman_chi2"]) else "–")
    hdr.append("Friedman")
    report.append(md_table(T, cols, hdr) + "\n")
    report.append("\nAll variants, RMST mean ± SD:\n")
    cols = ["cond"] + [lambda r, v=v: (f"{r[f'{v}_mean']:.1f} ± {r[f'{v}_sd']:.1f}"
                                       if not pd.isna(r.get(f"{v}_mean", np.nan)) else "–") for v in VL]
    report.append(md_table(T, cols, ["Cond"] + [LABEL[v] for v in VL]) + "\n")
    if any(v in set(S.variant) for v in GEOM):
        report.append("(Geometry controls were run only in C3–C6, where the learning agent separates the rewards; "
                      "in C1/C2 all rewards are near the 1000-step ceiling and in C7/C8 all are within ~1 step "
                      "of the floor.)\n")

    def ctab(df, title):
        report.append(f"\n### {title}\n")
        report.append("p_raw = exact Wilcoxon signed-rank (two-sided, n = 10 seeds); p_holm / p_bonf within "
                      "condition; p_holm_glob = Holm across all rows of the table.\n")
        c_ = ["cond", "contrast", lambda r: f"{r['diff']:+.1f} [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]",
              lambda r: f"{r['rbc']:+.2f}", lambda r: f"{r['W']:.1f}", lambda r: fmt_p(r["p_raw"]),
              lambda r: fmt_p(r["p_holm"]), lambda r: fmt_p(r["p_bonf"]), lambda r: fmt_p(r["p_holm_global"])]
        report.append(md_table(df, c_, ["Cond", "Contrast", "Δ RMST [95% CI]", "r_rb", "W", "p_raw",
                                        "p_holm", "p_bonf", "p_holm_glob"]) + "\n")
    if not CM.empty:
        ctab(CM, "L.1 Main pairwise contrasts (shared actor)")
    if not CS.empty:
        ctab(CS, "L.2 Reward sign / offset controls (shared actor)")
    gconds = [c for c in conds if S[(S.cond == c) & S.variant.isin(GEOM)].shape[0] > 0]
    if gconds:
        geom_pairs = [("OR_fixed", "DeadZone"), ("OR_fixed", "OR_noZone"), ("OR_fixed", "OR_noPen"),
                      ("OR_fixed", "OR_min"), ("OR_fixed", "OR_softmin"), ("OR_noPen", "Alive"),
                      ("DeadZone", "DR")]
        CG = contrast_table(S, gconds, geom_pairs, "L_geometry")
        CG.to_csv(os.path.join(OUT, "L_table_geometry_controls.csv"), index=False)
        fig_forest(CG, "L_forest_geometry.png", "Shared actor — regime-geometry controls")
        ctab(CG, "L.2b Regime-geometry controls (shared actor; C3–C6)")
    if Bd is not None:
        report_bounds(report, Bd, MAIN + ["Alive"], "L.3 Shared actor vs non-learning reference policies")
    if not CA.empty:
        report.append("\n### L.4 Architecture effect (shared actor − paper MLP, same reward and training seed)\n")
        report.append("p_holm within condition (family = the 7 rewards); p_holm_glob across the table.\n")
        report.append(md_table(CA, ["cond", "contrast", lambda r: f"{r['mean_mlp']:.0f}",
                                    lambda r: f"{r['mean_shared']:.0f}",
                                    lambda r: f"{r['diff']:+.1f} [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]",
                                    lambda r: fmt_p(r["p_raw"]), lambda r: fmt_p(r["p_holm"]),
                                    lambda r: fmt_p(r["p_holm_global"])],
                               ["Cond", "Reward", "MLP", "Shared", "Δ [95% CI]", "p_raw", "p_holm",
                                "p_holm_glob"]) + "\n")
    if not P.empty:
        report.append("\n### L.5 Adaptive OR parameters when the agent learns (init z = 0.5, β = 1.0)\n")
        report.append("Transform of the learned mean (Eqs. 18–19), mean ± SD over seeds; p = one-sample exact "
                      "Wilcoxon of the final value against the initial value (Holm across the 16 tests).\n")
        rows = []
        for c, g in P.groupby("cond"):
            pz = stats.wilcoxon(g.z_of_mean - 0.5).pvalue if np.any(g.z_of_mean != 0.5) else 1.0
            pb = stats.wilcoxon(g.beta_of_mean - 1.0).pvalue if np.any(g.beta_of_mean != 1.0) else 1.0
            rows.append(dict(cond=c, z=f"{g.z_of_mean.mean():.3f} ± {g.z_of_mean.std(ddof=1):.3f}",
                             beta=f"{g.beta_of_mean.mean():.3f} ± {g.beta_of_mean.std(ddof=1):.3f}",
                             sz=f"{g.sigma_sz.mean():.3f}", sb=f"{g.sigma_beta.mean():.3f}", pz=pz, pb=pb))
        R = pd.DataFrame(rows)
        adj = holm(np.r_[R.pz.values, R.pb.values])
        R["pz_h"], R["pb_h"] = adj[:len(R)], adj[len(R):]
        report.append(md_table(R, ["cond", "z", "beta", "sz", "sb",
                                   lambda r: f"{fmt_p(r['pz'])} / {fmt_p(r['pz_h'])}",
                                   lambda r: f"{fmt_p(r['pb'])} / {fmt_p(r['pb_h'])}"],
                               ["Cond", "z_safe", "β", "σ_z (raw)", "σ_β (raw)", "z vs 0.5: p_raw / p_holm",
                                "β vs 1.0: p_raw / p_holm"]) + "\n")
    report.append("\n### L.6 Behaviour (shared actor vs MLP)\n")
    report.append("corr(alloc, dev) > 0 and top-10% share > 0.10 indicate allocation directed at the most "
                  "deviated variables.\n")
    BA = S_main.groupby(["cond", "variant"])[["alloc_dev_corr", "top10_share"]].mean().reset_index()
    BB = B.merge(BA, on=["cond", "variant"], suffixes=("", "_mlp"))
    BB = BB[BB.variant.isin(MAIN + ["Alive"])]
    report.append(md_table(BB, ["cond", lambda r: LABEL[r["variant"]],
                                lambda r: f"{r['alloc_dev_corr_mlp']:.2f} → {r['alloc_dev_corr']:.2f}",
                                lambda r: f"{r['top10_share_mlp']:.3f} → {r['top10_share']:.3f}"],
                           ["Cond", "Variant", "corr(alloc, dev): MLP → shared",
                            "top-10% share: MLP → shared"]) + "\n")
    if not SL.empty:
        VLong = [v for v in MAIN + ["Alive", "DR_alive"] if v in set(SL.variant)]
        report.append("\n### L.7 Longer training (300k steps)\n")
        report.append("RMST at 100k (stage L, same seeds) → 300k, mean ± SD at 300k; k = number of seeds. "
                      "C3 and C5 have 10 seeds and include the survival-bonus controls; C1 has 5 seeds "
                      "(descriptive only: with 5 seeds the smallest attainable exact two-sided p is 0.0625).\n")
        rows = []
        for c in sorted(set(SL.cond)):
            g = SL[SL.cond == c].pivot_table(index="seed", columns="variant", values="rmst")
            g100 = S[S.cond == c].pivot_table(index="seed", columns="variant", values="rmst")
            r = dict(cond=c)
            for v in VLong:
                if v in g:
                    x = g[v].dropna()
                    r[v] = f"{g100.loc[x.index, v].mean():.0f} → {x.mean():.0f} ± {x.std(ddof=1):.0f} (k={len(x)})"
            rows.append(r)
        R = pd.DataFrame(rows).fillna("–")
        report.append(md_table(R, ["cond"] + [v for v in VLong if v in R],
                               ["Cond"] + [f"{LABEL[v]}" for v in VLong if v in R]) + "\n")
        SLt = SL[SL.cond.isin([c for c in set(SL.cond)
                               if SL[SL.cond == c].groupby("variant").seed.nunique().min() >= 6])]
        if not SLt.empty:
            long_pairs = main_pairs + [("OR_adapt", "Alive"), ("OR_fixed", "Alive"), ("DR_alive", "DR"),
                                       ("OR_fixed", "DR_alive")]
            CL = contrast_table(SLt, sorted(set(SLt.cond)), long_pairs, "L_long")
            CL.to_csv(os.path.join(OUT, "L_table_long300k_contrasts.csv"), index=False)
            fig_forest(CL, "L_forest_long300k.png", "Shared actor, 300k steps: paired RMST differences")
            T3 = summary_table(SLt, sorted(set(SLt.cond)), [v for v in VLong], friedman_variants=MAIN)
            report.append("\nFriedman (4 main treatments, 300k): " + "; ".join(
                f"{r['cond']}: χ²={r['friedman_chi2']:.2f}, p={fmt_p(r['friedman_p'])}, W={r['kendall_W']:.2f}"
                for _, r in T3.iterrows() if not pd.isna(r.get("friedman_chi2", np.nan))) + "\n")
            ctab(CL, "L.7b Pairwise contrasts at 300k steps (10 seeds)")

    # ---- L.8 reward normalisation ----
    SN = S_all[S_all.tag == "shared_nr1"]
    if not SN.empty:
        nconds = sorted(set(SN.cond))
        VN = [v for v in MAIN + ["Alive", "DR_alive"] if v in set(SN.variant)]
        report.append("\n### L.8 Reward-scale control: running-return reward normalisation (shared actor; C3–C6)\n")
        report.append("Same runs as L but with the reward divided by a running estimate of the discounted-return "
                      "SD (as SB3 VecNormalize), which removes differences in reward scale between methods. "
                      "RMST mean ± SD (unnormalised → normalised, same seeds):\n")
        rows = []
        for c in nconds:
            r = dict(cond=c)
            for v in VN:
                a = SN[(SN.cond == c) & (SN.variant == v)].set_index("seed").rmst
                b = S[(S.cond == c) & (S.variant == v)].set_index("seed").rmst
                if len(a):
                    r[v] = f"{b.loc[b.index.intersection(a.index)].mean():.1f} → {a.mean():.1f} ± {a.std(ddof=1):.1f}"
            rows.append(r)
        R = pd.DataFrame(rows).fillna("–")
        report.append(md_table(R, ["cond"] + VN, ["Cond"] + [LABEL[v] for v in VN]) + "\n")
        TN = summary_table(SN, nconds, VN, friedman_variants=MAIN)
        report.append("\nFriedman (4 main treatments, normalised): " + "; ".join(
            f"{r['cond']}: χ²={r['friedman_chi2']:.2f}, p={fmt_p(r['friedman_p'])}, W={r['kendall_W']:.2f}"
            for _, r in TN.iterrows() if not pd.isna(r.get("friedman_chi2", np.nan))) + "\n")
        CN = contrast_table(SN, nconds, main_pairs + [("OR_adapt", "Alive"), ("OR_fixed", "Alive"),
                                                      ("DR_alive", "DR"), ("OR_fixed", "DR_alive")], "L_norm")
        CN.to_csv(os.path.join(OUT, "L_table_rewnorm_contrasts.csv"), index=False)
        fig_forest(CN, "L_forest_rewnorm.png", "Shared actor with reward normalisation: paired RMST differences")
        ctab(CN, "L.8b Pairwise contrasts with reward normalisation")

    # ---- L.9 observation noise with a learning agent ----
    pairs_noise = [("C1", "C2"), ("C3", "C4"), ("C5", "C6"), ("C7", "C8")]
    rows = []
    for c0, c1 in pairs_noise:
        for v in MAIN + ["Alive"]:
            a = S[(S.cond == c1) & (S.variant == v)].set_index("seed").rmst
            b = S[(S.cond == c0) & (S.variant == v)].set_index("seed").rmst
            com = a.index.intersection(b.index)
            if len(com) >= 3:
                rows.append(dict(pair=f"{c0}→{c1}", variant=v, sigma0=b.loc[com].mean(), sigma02=a.loc[com].mean(),
                                 **paired_test(a.loc[com].values, b.loc[com].values)))
    NZ = pd.DataFrame(rows)
    if not NZ.empty:
        NZ["p_holm"] = NZ.groupby("pair").p_raw.transform(lambda p: holm(p.values))
        NZ.to_csv(os.path.join(OUT, "L_table_noise_effect.csv"), index=False)
        report.append("\n### L.9 Effect of observation noise (σ = 0 → 0.2) with the learning agent\n")
        report.append("Same training seed in both conditions (different environment, so the pairing is by seed "
                      "only); p_raw = exact Wilcoxon; p_holm within each condition pair.\n")
        report.append(md_table(NZ, ["pair", lambda r: LABEL[r["variant"]], lambda r: f"{r['sigma0']:.1f}",
                                    lambda r: f"{r['sigma02']:.1f}",
                                    lambda r: f"{r['diff']:+.1f} [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]",
                                    lambda r: fmt_p(r["p_raw"]), lambda r: fmt_p(r["p_holm"])],
                               ["Conds", "Variant", "σ=0", "σ=0.2", "Δ [95% CI]", "p_raw", "p_holm"]) + "\n")
    return S


def analyze_params(report, S_main, meta_main):
    report.append("\n## P. Adaptive parameters: initialisation, calibration vs drift\n")
    # A: learned values with raw + transformed
    rows = []
    for m in meta_main:
        if m["variant"] == "OR_adapt" and m["or_final"]:
            rows.append(dict(cond=m["cond"], seed=m["seed"], **m["or_final"]))
    if not rows:
        return
    L = pd.DataFrame(rows)
    agg = L.groupby("cond").agg(['mean', 'std']).drop(columns="seed")
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(os.path.join(OUT, "P_learned_params_main.csv"), index=False)
    report.append("Initialisation (all main Adaptive-OR runs): μ_sz = logit((0.5−0.1)/0.8) = 0 → z̄_safe = 0.50; "
                  "μ_β = softplus⁻¹(1) = 0.5413 → β̄ = 1.00; log σ_sz = log σ_β = −1 (σ = 0.368, unconstrained "
                  "space). OR-Fixed uses exactly these initial values, so the ablation starts from the same point.\n")
    report.append("Table values: raw μ (unconstrained) and the transformed means σ(μ_sz)·0.8+0.1 and softplus(μ_β) "
                  "(Eqs. 18–19), plus E[transformed sample]; mean (SD) across 10 seeds after 100k steps.\n")
    cols = ["cond", lambda r: f"{r['mu_sz_raw_mean']:+.3f} ({r['mu_sz_raw_std']:.3f})",
            lambda r: f"{r['z_of_mean_mean']:.3f} ({r['z_of_mean_std']:.3f})",
            lambda r: f"{r['z_mean_of_samples_mean']:.3f}",
            lambda r: f"{r['mu_beta_raw_mean']:+.3f} ({r['mu_beta_raw_std']:.3f})",
            lambda r: f"{r['beta_of_mean_mean']:.3f} ({r['beta_of_mean_std']:.3f})",
            lambda r: f"{r['beta_mean_of_samples_mean']:.3f}",
            lambda r: f"{r['sigma_sz_mean']:.3f}", lambda r: f"{r['sigma_beta_mean']:.3f}"]
    report.append(md_table(agg, cols, ["Cond", "μ_sz raw", "z̄_safe", "E[z]", "μ_β raw", "β̄", "E[β]",
                                       "σ_sz", "σ_β"]) + "\n")

    # trajectories figure (main)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for c in CONDITIONS:
        Z, Bt, st = [], [], None
        for m in meta_main:
            if m["variant"] == "OR_adapt" and m["cond"] == c and m["or_traj"]:
                st = m["or_traj"]["step"]
                Z.append([p["z_of_mean"] for p in m["or_traj"]["params"]])
                Bt.append([p["beta_of_mean"] for p in m["or_traj"]["params"]])
        if Z:
            k = min(map(len, Z))
            axes[0].plot(st[:k], np.mean([z[:k] for z in Z], 0), label=c)
            axes[1].plot(st[:k], np.mean([b[:k] for b in Bt], 0), label=c)
    axes[0].set_title("z̄_safe trajectory (mean over seeds)")
    axes[1].set_title("β̄ trajectory (mean over seeds)")
    for ax in axes:
        ax.set_xlabel("training steps")
        ax.grid(alpha=.3)
        ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "P_param_trajectories_main.png"), dpi=160)
    plt.close(fig)

    ep, meta = load("P_params")
    if ep.empty:
        return
    SP = seed_level(ep)
    SP.to_csv(os.path.join(OUT, "P_seed_level.csv"), index=False)

    # P1: inits
    rows = []
    for m in meta + meta_main:
        if m["variant"] != "OR_adapt" or not m["or_final"]:
            continue
        if m["study"] == "A_main" and m["seed"] not in range(42, 47):
            continue
        tag = m["tag"] if m["study"] == "P_params" else "init_z0.5_b1.0"
        if not tag.startswith("init"):
            continue
        z0 = float(tag.split("_")[1][1:])
        b0 = float(tag.split("_")[2][1:])
        rows.append(dict(cond=m["cond"], seed=m["seed"], z0=z0, b0=b0,
                         z=m["or_final"]["z_of_mean"], beta=m["or_final"]["beta_of_mean"]))
    I = pd.DataFrame(rows)
    if not I.empty:
        I["dz"] = I.z - I.z0
        I["dbeta"] = I.beta - I.b0
        I.to_csv(os.path.join(OUT, "P1_inits_final_params.csv"), index=False)
        Ig = I.groupby(["cond", "z0", "b0"])[["z", "beta", "dz", "dbeta"]].mean().reset_index()
        report.append("\n### P1. Final parameters from five initialisations (5 seeds each, 100k steps)\n")
        report.append("If the parameters were being calibrated to the environment, final values from different "
                      "initial points would converge; if they merely drift, they stay near their initial value.\n")
        report.append(md_table(Ig, ["cond", "z0", "b0", lambda r: f"{r['z']:.3f}", lambda r: f"{r['dz']:+.3f}",
                                    lambda r: f"{r['beta']:.3f}", lambda r: f"{r['dbeta']:+.3f}"],
                               ["Cond", "z₀", "β₀", "z final", "Δz", "β final", "Δβ"]) + "\n")
        # variance decomposition: share of variance in final z explained by init
        dec = []
        for c, g in I.groupby("cond"):
            for p in ("z", "beta"):
                tot = g[p].var(ddof=0)
                between = g.groupby(["z0", "b0"])[p].mean().var(ddof=0)
                dec.append(dict(cond=c, param=p, share_explained_by_init=between / tot if tot > 0 else np.nan))
        D = pd.DataFrame(dec).pivot(index="cond", columns="param", values="share_explained_by_init").reset_index()
        D.to_csv(os.path.join(OUT, "P1_variance_explained_by_init.csv"), index=False)
        report.append("\nShare of the across-run variance of the final value explained by the initial value "
                      "(η², between-init / total):\n")
        report.append(md_table(D, ["cond", lambda r: f"{r['z']:.2f}", lambda r: f"{r['beta']:.2f}"],
                               ["Cond", "η²(z)", "η²(β)"]) + "\n")
        # also performance by init
        perf = SP[SP.tag.str.startswith("init")].groupby(["cond", "tag"])["rmst"].mean().unstack()
        base = S_main[(S_main.variant == "OR_adapt") & (S_main.seed < 47)].groupby("cond")["rmst"].mean()
        perf["init_z0.5_b1.0 (main)"] = base
        perf = perf.reset_index()
        perf.to_csv(os.path.join(OUT, "P1_rmst_by_init.csv"), index=False)
        report.append("\nRMST by initialisation (mean over 5 seeds):\n")
        pc = [c for c in perf.columns if c != "cond"]
        report.append(md_table(perf, ["cond"] + [lambda r, c=c: f"{r[c]:.1f}" for c in pc], ["Cond"] + pc) + "\n")

        fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
        for ax, c in zip(axes.flat, CONDITIONS):
            g = I[I.cond == c]
            for (z0, b0), gg in g.groupby(["z0", "b0"]):
                ax.scatter([z0], [b0], marker="x", color="k")
                ax.scatter(gg.z, gg.beta, s=18, alpha=.7)
                ax.annotate("", xy=(gg.z.mean(), gg.beta.mean()), xytext=(z0, b0),
                            arrowprops=dict(arrowstyle="->", color="#555"))
            ax.set_title(c)
            ax.set_xlabel("z_safe")
            ax.set_ylabel("β")
            ax.grid(alpha=.3)
        fig.suptitle("Initial (×) → final (•) learned parameters, 5 initialisations × 5 seeds", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "P1_init_to_final.png"), dpi=160)
        plt.close(fig)

    # P2/P3: fixed at learned values
    comb = pd.concat([S_main[S_main.variant.isin(["OR_adapt", "OR_fixed"])],
                      SP[SP.tag.isin(["fixed_at_own", "fixed_at_swap"])]])
    comb["arm"] = np.where(comb.tag == "base", comb.variant, "OR_fixed@" + comb.tag.str.replace("fixed_at_", ""))
    piv = comb.pivot_table(index=["cond", "seed"], columns="arm", values="rmst").reset_index()
    if "OR_fixed@own" in piv:
        rows = []
        for c, g in piv.groupby("cond"):
            for a1, a2 in [("OR_adapt", "OR_fixed@own"), ("OR_fixed@own", "OR_fixed"),
                           ("OR_fixed@own", "OR_fixed@swap"), ("OR_adapt", "OR_fixed")]:
                if a1 in g and a2 in g:
                    gg = g[[a1, a2]].dropna()
                    if len(gg) >= 3:
                        t = paired_test(gg[a1].values, gg[a2].values)
                        rows.append(dict(cond=c, contrast=f"{a1} vs {a2}", mean1=gg[a1].mean(),
                                         mean2=gg[a2].mean(), **t))
        P2 = pd.DataFrame(rows)
        for c, g in P2.groupby("cond"):
            P2.loc[g.index, "p_holm"] = holm(g.p_raw.values)
        P2["p_holm_global"] = holm(P2.p_raw.values)
        P2.to_csv(os.path.join(OUT, "P2_fixed_at_learned.csv"), index=False)
        report.append("\n### P2/P3. OR fixed at the learned values (own condition) and at the values learned "
                      "in the budget-swapped condition (10 seeds)\n")
        report.append("If Adaptive ≈ Fixed@own, the benefit lies in the parameter values, not in the "
                      "adaptation process; if Fixed@own > Fixed@swap, the learned values are "
                      "environment-specific (calibration).\n")
        report.append(md_table(P2, ["cond", "contrast", lambda r: f"{r['mean1']:.1f}", lambda r: f"{r['mean2']:.1f}",
                                    lambda r: f"{r['diff']:+.1f} [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]",
                                    lambda r: fmt_p(r["p_raw"]), lambda r: fmt_p(r["p_holm"]),
                                    lambda r: fmt_p(r["p_holm_global"])],
                               ["Cond", "Contrast", "mean 1", "mean 2", "Δ [95% CI]", "p_raw", "p_holm",
                                "p_holm_glob"]) + "\n")

    # P4: long training
    LT = SP[SP.tag == "long400k"]
    if not LT.empty:
        rows = []
        for c, g in LT.groupby("cond"):
            row = dict(cond=c)
            for v in MAIN:
                row[f"{v}_400k"] = g[g.variant == v].rmst.mean()
                row[f"{v}_100k"] = S_main[(S_main.cond == c) & (S_main.variant == v) & (S_main.seed < 47)].rmst.mean()
            rows.append(row)
        P4 = pd.DataFrame(rows)
        P4.to_csv(os.path.join(OUT, "P4_long_training.csv"), index=False)
        report.append("\n### P4. Convergence check: 100k vs 400k training steps (same 5 seeds)\n")
        cols = ["cond"]
        hdr = ["Cond"]
        for v in MAIN:
            cols.append(lambda r, v=v: f"{r[f'{v}_100k']:.1f} → {r[f'{v}_400k']:.1f}")
            hdr.append(LABEL[v])
        report.append(md_table(P4, cols, hdr) + "\n")
        ctl = contrast_table(LT, sorted(LT.cond.unique()), [("OR_adapt", "OR_fixed"), ("OR_adapt", "DQ"),
                                                          ("OR_adapt", "DR"), ("OR_fixed", "DQ"),
                                                          ("OR_fixed", "DR"), ("DQ", "DR")], "long")
        ctl.to_csv(os.path.join(OUT, "P4_long_contrasts.csv"), index=False)
        report.append("\nPairwise contrasts at 400k (n = 5 seeds; minimum attainable exact p = 0.0625, "
                      "so these are descriptive):\n")
        report.append(md_table(ctl, ["cond", "contrast", lambda r: f"{r['diff']:+.1f} [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]",
                                     lambda r: fmt_p(r["p_raw"])], ["Cond", "Contrast", "Δ [95% CI]", "p_raw"]) + "\n")
        # learned params at 400k
        lp = [dict(cond=m["cond"], seed=m["seed"], z=m["or_final"]["z_of_mean"], beta=m["or_final"]["beta_of_mean"])
              for m in meta if m["tag"] == "long400k" and m["variant"] == "OR_adapt"]
        if lp:
            lp = pd.DataFrame(lp).groupby("cond")[["z", "beta"]].agg(["mean", "std"]).reset_index()
            lp.columns = ["_".join(c).strip("_") for c in lp.columns]
            report.append("\nLearned parameters after 400k steps (init z=0.5, β=1.0):\n")
            report.append(md_table(lp, ["cond", lambda r: f"{r['z_mean']:.3f} ({r['z_std']:.3f})",
                                        lambda r: f"{r['beta_mean']:.3f} ({r['beta_std']:.3f})"],
                                   ["Cond", "z̄_safe", "β̄"]) + "\n")
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        for ax, c in zip(axes, ["C1", "C3", "C5", "C7"]):
            for v in MAIN:
                curves = []
                grid = np.linspace(0, 400_000, 81)
                for m in meta:
                    if m["tag"] == "long400k" and m["cond"] == c and m["variant"] == v:
                        st, ln = np.array(m["train_step"]), np.array(m["train_len"])
                        y = np.array([ln[(st > grid[i]) & (st <= grid[i + 1])].mean()
                                      if ((st > grid[i]) & (st <= grid[i + 1])).any() else np.nan
                                      for i in range(len(grid) - 1)])
                        curves.append(pd.Series(y).ffill().bfill().values)
                if curves:
                    ax.plot((grid[1:] + grid[:-1]) / 2, np.mean(curves, 0), color=COLOR[v], label=LABEL[v])
            ax.set_title(c)
            ax.set_xlabel("training steps")
            ax.grid(alpha=.3)
        axes[0].legend(fontsize=8)
        axes[0].set_ylabel("training episode length")
        fig.suptitle("Long training (400k steps, 5 seeds)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "P4_long_training_curves.png"), dpi=160)
        plt.close(fig)


def analyze_noise(report, S_main):
    ep, _ = load("N_noise")
    if ep.empty:
        return
    SN = seed_level(ep)
    SN.to_csv(os.path.join(OUT, "N_seed_level.csv"), index=False)
    report.append("\n## N. Observation-noise mechanism\n")
    report.append("Reward and termination are computed from the latent state; with the default setting, noise "
                  "affects only the policy's observation of v (Eq. 28).\n")
    # sweep
    base = S_main[(S_main.budget == 1.0) & (S_main.variant.isin(MAIN + ["Alive"]))].copy()
    sw = pd.concat([base, SN[SN.tag == "sweep"]])
    rows = []
    for (n, sig, v), g in sw.groupby(["n", "noise", "variant"]):
        lo, hi = boot_ci(g.rmst.values)
        rows.append(dict(n=n, noise=sig, variant=v, rmst=g.rmst.mean(), lo=lo, hi=hi, k=len(g)))
    SW = pd.DataFrame(rows)
    SW.to_csv(os.path.join(OUT, "N1_noise_sweep.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, n in zip(axes, (50, 100)):
        for v in MAIN + ["Alive"]:
            g = SW[(SW.n == n) & (SW.variant == v)].sort_values("noise")
            ax.errorbar(g.noise, g.rmst, yerr=[g.rmst - g.lo, g.hi - g.rmst], color=COLOR[v], label=LABEL[v],
                        marker="o", capsize=3)
        ax.set_title(f"n = {n}, b = 1.0")
        ax.set_xlabel("observation noise σ_obs")
        ax.set_ylabel("RMST (τ=1000)")
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "N1_noise_sweep.png"), dpi=160)
    plt.close(fig)
    piv = SW.pivot_table(index=["n", "variant"], columns="noise", values="rmst").reset_index()
    report.append("\n### N1. Noise sweep (b = 1.0; 10 seeds per cell; σ = 0 and 0.2 from the main study)\n")
    nc = [c for c in piv.columns if c not in ("n", "variant")]
    report.append(md_table(piv, ["n", lambda r: LABEL[r["variant"]]] + [lambda r, c=c: f"{r[c]:.1f}" for c in nc],
                           ["n", "Variant"] + [f"σ={c}" for c in nc]) + "\n")
    # degradation slope test: per seed, RMST(σ=0.4) - RMST(σ=0)
    rows = []
    for n in (50, 100):
        for v in MAIN:
            a = sw[(sw.n == n) & (sw.variant == v) & (sw.noise == 0.0)].set_index("seed").rmst
            b = sw[(sw.n == n) & (sw.variant == v) & (sw.noise == 0.4)].set_index("seed").rmst
            com = a.index.intersection(b.index)
            if len(com) >= 3:
                d = b.loc[com].values - a.loc[com].values
                lo, hi = boot_ci(d)
                rows.append(dict(n=n, variant=v, delta_0_to_04=d.mean(), lo=lo, hi=hi,
                                 rel=d.mean() / a.loc[com].mean()))
    if rows:
        DG = pd.DataFrame(rows)
        DG.to_csv(os.path.join(OUT, "N1_degradation.csv"), index=False)
        report.append("\nChange in RMST from σ=0 to σ=0.4 (same seeds; 95% bootstrap CI):\n")
        report.append(md_table(DG, ["n", lambda r: LABEL[r["variant"]],
                                    lambda r: f"{r['delta_0_to_04']:+.1f} [{r['lo']:+.1f}, {r['hi']:+.1f}]",
                                    lambda r: f"{100*r['rel']:+.0f}%"], ["n", "Variant", "Δ RMST", "relative"]) + "\n")

    # N2/N3
    rows = []
    for tag, title in (("rew_observed", "reward from noisy observation"),
                       ("noise_all", "noise on all 4n+1 inputs (original code)")):
        g = SN[SN.tag == tag]
        for c in sorted(g.cond.unique()):
            for v in MAIN:
                a = g[(g.cond == c) & (g.variant == v)].set_index("seed").rmst
                b = S_main[(S_main.cond == c) & (S_main.variant == v)].set_index("seed").rmst
                com = a.index.intersection(b.index)
                if len(com) >= 3:
                    t = paired_test(a.loc[com].values, b.loc[com].values)
                    rows.append(dict(manip=title, cond=c, variant=v, manip_mean=a.loc[com].mean(),
                                     base_mean=b.loc[com].mean(), **t))
    if rows:
        NM = pd.DataFrame(rows)
        NM["p_holm"] = np.nan
        for (mp, c), g in NM.groupby(["manip", "cond"]):
            NM.loc[g.index, "p_holm"] = holm(g.p_raw.values)
        NM.to_csv(os.path.join(OUT, "N2_N3_manipulations.csv"), index=False)
        report.append("\n### N2/N3. Direct manipulations in C2 and C6 (σ = 0.2): paired against the main study "
                      "(same seeds)\n")
        report.append(md_table(NM, ["manip", "cond", lambda r: LABEL[r["variant"]],
                                    lambda r: f"{r['base_mean']:.1f}", lambda r: f"{r['manip_mean']:.1f}",
                                    lambda r: f"{r['diff']:+.1f} [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]",
                                    lambda r: fmt_p(r["p_raw"]), lambda r: fmt_p(r["p_holm"])],
                               ["Manipulation", "Cond", "Variant", "baseline", "manipulated", "Δ [95% CI]",
                                "p_raw", "p_holm"]) + "\n")
        # rank of variants under manipulation
        for tag in ("reward from noisy observation", "noise on all 4n+1 inputs (original code)"):
            g = NM[NM.manip == tag]
            if g.empty:
                continue


def analyze_tuning(report):
    ep, _ = load("T_tune")
    if ep.empty:
        return
    ST = seed_level(ep)
    tab = ST.groupby(["variant", "tag", "cond"]).rmst.mean().unstack("cond").reset_index()
    tab.to_csv(os.path.join(OUT, "T_tuning_grid.csv"), index=False)
    best = json.load(open(os.path.join(RES, "tuned_configs.json"))) if os.path.exists(
        os.path.join(RES, "tuned_configs.json")) else {}
    report.append("\n## T/F. Modest per-method tuning\n")
    report.append("Grid: learning rate ∈ {1e-4, 3e-4, 1e-3} × reward normalisation (running return std, "
                  "as SB3 VecNormalize) ∈ {off, on}; tuned on C1 and C5 with three separate tuning seeds "
                  "(900–902, disjoint from the evaluation seeds); selection by mean rank across the two "
                  "conditions.\n")
    report.append(md_table(pd.DataFrame([dict(variant=LABEL[k], cfg=v) for k, v in best.items()]),
                           ["variant", "cfg"], ["Variant", "selected config"]) + "\n")
    report.append("\nTuning grid (RMST, mean over 3 tuning seeds):\n")
    report.append(md_table(tab, [lambda r: LABEL[r["variant"]], "tag", lambda r: f"{r['C1']:.1f}",
                                 lambda r: f"{r['C5']:.1f}"], ["Variant", "config", "C1", "C5"]) + "\n")
    # final tuned
    epF, _ = load("F_tuned")
    epA, _ = load("A_main")
    SA = seed_level(epA)
    parts = []
    for v in ALL_VARIANTS:
        tag = best.get(v, "lr0.0003_nr0")
        if tag == "lr0.0003_nr0":
            parts.append(SA[SA.variant == v])
        elif not epF.empty:
            SF = seed_level(epF)
            parts.append(SF[(SF.variant == v) & (SF.tag == tag)])
    STF = pd.concat(parts)
    STF.to_csv(os.path.join(OUT, "F_seed_level_tuned.csv"), index=False)
    TF = summary_table(STF, list(CONDITIONS), ALL_VARIANTS, friedman_variants=MAIN)
    TF.to_csv(os.path.join(OUT, "F_table_rmst_tuned.csv"), index=False)
    report.append("\n### F. All variants at their tuned configuration (RMST mean ± SD, 10 seeds)\n")
    cols = ["cond"] + [lambda r, v=v: f"{r[f'{v}_mean']:.1f} ± {r[f'{v}_sd']:.1f}" for v in ALL_VARIANTS]
    report.append(md_table(TF, cols, ["Cond"] + [LABEL[v] for v in ALL_VARIANTS]) + "\n")
    cols = ["cond", lambda r: f"χ²={r['friedman_chi2']:.2f}, p={fmt_p(r['friedman_p'])}"]
    report.append("\nFriedman (4 main treatments, tuned):\n" + md_table(TF, cols, ["Cond", "Friedman"]) + "\n")
    pairs = [("OR_adapt", "OR_fixed"), ("OR_adapt", "DQ"), ("OR_adapt", "DR"), ("OR_fixed", "DQ"),
             ("OR_fixed", "DR"), ("DQ", "DR"), ("OR_adapt", "Alive"), ("OR_adapt", "DR_alive"),
             ("OR_adapt", "DQ_alive")]
    CF = contrast_table(STF, list(CONDITIONS), pairs, "tuned")
    CF.to_csv(os.path.join(OUT, "F_table_contrasts_tuned.csv"), index=False)
    report.append("\nTuned contrasts (family = 9 contrasts per condition):\n")
    report.append(md_table(CF, ["cond", "contrast", lambda r: f"{r['diff']:+.1f} [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]",
                                lambda r: fmt_p(r["p_raw"]), lambda r: fmt_p(r["p_holm"]),
                                lambda r: fmt_p(r["p_bonf"]), lambda r: fmt_p(r["p_holm_global"])],
                           ["Cond", "Contrast", "Δ [95% CI]", "p_raw", "p_holm", "p_bonf", "p_holm_glob"]) + "\n")
    fig_forest(CF, "F_forest_tuned.png", "Tuned configurations: paired RMST differences")


def main():
    os.makedirs(FIG, exist_ok=True)
    report = ["# Revised MRP experiments — results\n",
              "Generated by `mrp_v2/analyze.py`. Unit of analysis: training seed (n = 10 unless stated). "
              "Outcome: RMST at τ = 1000 over 20 paired evaluation episodes (seeds 1000–1019).\n"]
    probe = os.path.join(OUT, "reward_scale_probe.csv")
    if os.path.exists(probe):
        P = pd.read_csv(probe)
        g = P[P.policy == "random"].groupby("variant")[["mean", "std", "frac_pos", "frac_neg"]].mean().reset_index()
        report.append("## R. Per-step reward sign and scale on identical trajectories (random-logit policy, "
                      "averaged over C1–C8)\n")
        report.append(md_table(g, [lambda r: LABEL.get(r["variant"], r["variant"]), lambda r: f"{r['mean']:+.4f}",
                                   lambda r: f"{r['std']:.4f}", lambda r: f"{r['frac_pos']:.2f}",
                                   lambda r: f"{r['frac_neg']:.2f}"],
                               ["Variant", "mean r_t", "SD r_t", "P(r_t>0)", "P(r_t<0)"]) + "\n")
    S, meta = analyze_main(report)
    if S is not None:
        analyze_learn(report, S)
        analyze_params(report, S, meta)
        analyze_noise(report, S)
    analyze_tuning(report)
    with open(os.path.join(OUT, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print("wrote", os.path.join(OUT, "RESULTS.md"))


if __name__ == "__main__":
    main()
