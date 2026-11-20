"""
review_v3_mrp_tables.py — LaTeX rows of the MRP tables of the paper, regenerated from the re-run with
the corrected dynamics (docs/results_v3_mrp/analysis).  Nothing is typed by hand; the rows are pasted
into output/main7_humanized.tex.

Tables: tab:sign (reward sign under the random policy), tab:mrp_main (RMST, 10^5 steps),
tab:mrp_geometry, tab:app_long (3x10^5 steps), tab:app_norm (reward normalisation) and the MRP block of
tab:noterm (viable fraction without termination).  Rounding conventions are those of the current text:
RMST mean with 0 decimals if >= 100 else 1, SD with 0 decimals if >= 10 else 1; reference policies as
integers; long-training table in integers; viable fraction with 3 decimals.

Output (new file): docs/results_v3_mrp/analysis/REVIEW_v3_mrp_tables.tex
Usage: python review_v3_mrp_tables.py
"""
import os


import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
A3 = os.path.join(HERE, "..", "docs", "results_v3_mrp", "analysis")
rd = lambda f: pd.read_csv(os.path.join(A3, f))  # noqa: E731


def r(x, dec):
    # Python float formatting, as in review_v3_contrast_table.py, so that the same mean is rounded
    # identically in every table of the paper.
    return f"{float(x):.{dec}f}"


def ms(m, s):
    sd = "0" if s == 0 else r(s, 0 if s >= 10 else 1)
    return f"${r(m, 0 if m >= 100 else 1)} \\pm {sd}$"


def ms_int(m, s):
    return f"${r(m, 0)} \\pm {r(s, 0)}$"


def signed(x):
    d = 3 if abs(float(r(x, 3))) > 0 else 4
    s = r(x, d)
    if s.startswith("-"):
        return f"$-{s[1:]}$"
    return f"$+{s}$"


def main():
    out = []
    # ---- tab:sign --------------------------------------------------------------------------
    P = rd("reward_scale_probe.csv")
    P = P[P.policy == "random"]
    avg = P.groupby("variant")[["mean", "std", "frac_pos", "frac_neg"]].mean()
    per = P.pivot(index="variant", columns="cond", values="mean")
    rows = [("OR ($z_{\\mathrm{safe}}=0.5$, $\\beta=1$)", "OR_fixed"), ("OR without penalty ($\\beta=0$)", "OR_noPen"),
            ("OR without zone ($z_{\\mathrm{safe}}=0$)", "OR_noZone"), ("OR, min aggregation", "OR_min"),
            ("OR, soft-min aggregation", "OR_softmin"), ("DQ (differential)", "DQ"), ("DR (level)", "DR"),
            ("Dead-zone drive", "DeadZone"), ("Alive (constant)", "Alive"), ("DR + alive", "DR_alive")]
    out.append("% tab:sign")
    for lab, v in rows:
        a = avg.loc[v]
        pairs = [per.loc[v, c] for c in ("C1", "C3", "C5", "C7")]
        for c1, c2 in (("C1", "C2"), ("C3", "C4"), ("C5", "C6"), ("C7", "C8")):
            assert abs(per.loc[v, c1] - per.loc[v, c2]) < 1e-12
        out.append(f"{lab} & {signed(a['mean'])} & {r(a['std'], 3)} & {r(a['frac_pos'], 2)} & {r(a['frac_neg'], 2)} & "
                   + " & ".join(signed(x) for x in pairs) + " \\\\")
    out.append("% fraction of steps with r_t == 0, averaged over C1-C8: " + ", ".join(
        f"{v}={1 - avg.loc[v, 'frac_pos'] - avg.loc[v, 'frac_neg']:.3f}" for v in avg.index))

    # ---- tab:mrp_main ----------------------------------------------------------------------
    T = rd("L_table_rmst.csv").set_index("cond")
    B = rd("policy_bounds.csv").set_index("cond")
    out.append("% tab:mrp_main")
    for c in T.index:
        cells = [ms(T.loc[c, f"{v}_mean"], T.loc[c, f"{v}_sd"]) for v in ("OR_fixed", "DQ", "DR", "Alive", "DR_alive", "DQ_alive")]
        refs = [r(B.loc[c, p], 0) for p in ("zero", "prop", "greedy", "oracle")]
        out.append(f"{c} & " + " & ".join(cells + refs) + " \\\\")
    out.append("% Friedman (OR, DQ, DR): " + "; ".join(
        f"{c}: p={T.loc[c, 'friedman_p']:.2g}, W={T.loc[c, 'kendall_W']:.2f}" for c in T.index))

    # ---- tab:mrp_geometry ------------------------------------------------------------------
    out.append("% tab:mrp_geometry")
    for c in ("C3", "C4", "C5", "C6"):
        cells = [ms(T.loc[c, f"{v}_mean"], T.loc[c, f"{v}_sd"])
                 for v in ("OR_fixed", "OR_noZone", "OR_noPen", "OR_min", "OR_softmin", "DeadZone")]
        out.append(f"{c} & " + " & ".join(cells) + " \\\\")

    # ---- tab:app_long and tab:app_norm (seed level) ------------------------------------------
    S = rd("L_seed_level.csv")
    for tag, name, fmt in (("shared_long300k", "tab:app_long", ms_int), ("shared_nr1", "tab:app_norm", ms)):
        X = S[S.tag == tag]
        g = X.groupby(["cond", "variant"]).rmst.agg(["mean", "std", "count"])
        out.append(f"% {name} (seed counts: {dict(g['count'].groupby(level=0).min())})")
        for c in sorted(X.cond.unique()):
            if tag == "shared_long300k" and c == "C1":
                continue  # C1: 5 seeds, descriptive only, not in the paper
            cells = [fmt(g.loc[(c, v), "mean"], g.loc[(c, v), "std"]) for v in ("OR_fixed", "DQ", "DR", "Alive", "DR_alive")]
            out.append(f"{c} & " + " & ".join(cells) + " \\\\")

    # ---- tab:noterm, MRP block ----------------------------------------------------------------
    N = rd("NT_seed_level.csv")
    R = rd("NT_reference_policies.csv")
    m = N.groupby(["cond", "variant"]).frac_viable.mean()
    out.append("% tab:noterm (MRP rows)")
    for c in ("C3", "C4", "C5", "C6", "C7", "C8"):
        cells = [r(m.loc[(c, v)], 3) for v in ("OR_fixed", "DQ", "DR", "DR_alive", "Alive", "DeadZone", "OR_noPen", "OR_noZone")]
        refs = [r(R[(R.cond == c) & (R.policy == p)].frac_viable.iloc[0], 3) for p in ("uniform", "proportional")]
        out.append(f"{c} & " + " & ".join(cells + refs) + " \\\\")

    path = os.path.join(A3, "REVIEW_v3_mrp_tables.tex")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; refusing to overwrite")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
