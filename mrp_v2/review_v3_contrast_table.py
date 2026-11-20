"""
review_v3_contrast_table.py — copy of review_r1_contrast_table.py in which every MRP block is read
from the re-run with the corrected dynamics (docs/results_v3_mrp/analysis); the foraging blocks are
read as before (docs/results_v2_noadapt/analysis and docs/results_v2/analysis).  Nothing is typed by
hand.

Changes with respect to review_r1_contrast_table.py:
  * MRP blocks (L_main, L_sign, L_geometry, L_long, L_norm, observation noise, NT) come from
    docs/results_v3_mrp/analysis.
  * C1 and C2 are now included in the MRP blocks with termination (with the corrected dynamics the
    rewards separate there); the main family also lists DQ vs DR; the normalisation block also lists
    OR vs DQ; the no-termination blocks also list DQ vs Alive and OR vs OR no penalty.
  * New block: change in RMST from sigma_obs = 0 to 0.2 (L_table_noise_effect.csv).
  * Caption: the sentence on C1/C2 is replaced; a missing space is fixed.

Output (new files; nothing existing is overwritten):
  docs/results_v3_mrp/analysis/REVIEW_v3_contrast_table.csv and .tex
Usage: python review_v3_contrast_table.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
A3 = os.path.join(HERE, "..", "docs", "results_v3_mrp", "analysis")   # MRP (v3 dynamics)
A0 = os.path.join(HERE, "..", "docs", "results_v2_noadapt", "analysis")  # foraging
A1 = os.path.join(HERE, "..", "docs", "results_v2", "analysis")          # foraging follow-ups

NAME = {"OR_fixed": "OR", "OR (Fixed)": "OR", "DQ": "DQ", "DR": "DR", "Alive": "Alive", "Alive (+1)": "Alive",
        "DR_alive": "DR + alive", "DR + alive": "DR + alive", "DQ_alive": "DQ + alive", "DQ + alive": "DQ + alive",
        "DeadZone": "Dead-zone", "Dead-zone": "Dead-zone", "OR_noZone": "OR no zone", "OR no zone": "OR no zone",
        "OR_noPen": "OR no pen.", "OR no penalty": "OR no pen.", "OR min-agg": "OR min", "OR soft-min": "OR soft-min", "OR_min": "OR min", "OR_softmin": "OR soft-min",
        "DQ_c001": "DQ + 0.01", "DQ_c01": "DQ + 0.1", "OR_m1": "OR $-$ 1"}

ROWS = []


def add(block, df, v1col="v1", v2col="v2", condcol="cond", posthoc=False, note=""):
    for _, r in df.iterrows():
        c = r[condcol]
        c = f"$k={int(c)}$" if condcol == "k" else str(c)
        if "kind" in df.columns and r["kind"] == "eval_det":
            c += " (det.)"
        ROWS.append(dict(block=block, contrast=f"{NAME[r[v1col]]} vs {NAME[r[v2col]]}", cond=c,
                         mean1=r["mean1"], mean2=r["mean2"], diff=r["diff"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                         rbc=r["rbc"], p_raw=r["p_raw"], p_holm=r["p_holm"], posthoc=posthoc))


def sel(df, conds=None, contrasts=None, **kw):
    m = np.ones(len(df), bool)
    if conds is not None:
        m &= df.cond.isin(conds)
    if contrasts is not None:
        m &= df.contrast.isin(contrasts)
    for k, v in kw.items():
        m &= (df[k] == v)
    return df[m]


def split_contrast(df):
    df = df.copy()
    df["v1"] = df.contrast.str.split(" vs ").str[0]
    df["v2"] = df.contrast.str.split(" vs ").str[1]
    return df


def build():
    C18 = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
    C38 = ["C3", "C4", "C5", "C6", "C7", "C8"]
    C36 = ["C3", "C4", "C5", "C6"]
    F14 = ["F1", "F2", "F3", "F4"]
    rd = lambda f, base=A0: pd.read_csv(os.path.join(base, f))  # noqa: E731

    # --- MRP with termination (v3 dynamics) ------------------------------------------------
    M = rd("L_table_main_contrasts.csv", A3)
    add("MRP, termination, RMST; main family (OR, DQ, DR: 3 contrasts per condition)",
        sel(M, C18, ["OR (Fixed) vs DR", "OR (Fixed) vs DQ", "DQ vs DR"]).sort_values(["contrast", "cond"]))
    S = rd("L_table_sign_controls.csv", A3)
    add("MRP, termination, RMST; sign family (5 contrasts per condition)", sel(S, C18).sort_values(["contrast", "cond"]))
    G = rd("L_table_geometry_controls.csv", A3)
    add("MRP, termination, RMST; geometry family (7 contrasts per condition, C3--C6)",
        sel(G, C36, ["OR (Fixed) vs Dead-zone", "OR (Fixed) vs OR no zone", "OR (Fixed) vs OR no penalty",
                     "OR (Fixed) vs OR min-agg", "OR (Fixed) vs OR soft-min"]).sort_values(["contrast", "cond"]))
    L = rd("L_table_long300k_contrasts.csv", A3)
    add("MRP, termination, RMST, $3\\times10^5$ steps (6 contrasts per condition)", L.sort_values(["contrast", "cond"]))
    N = rd("L_table_rewnorm_contrasts.csv", A3)
    add("MRP, termination, RMST, reward normalisation (6 contrasts per condition)",
        sel(N, None, ["OR (Fixed) vs DR", "OR (Fixed) vs DQ", "DQ vs DR", "DR + alive vs DR", "OR (Fixed) vs Alive (+1)"])
        .sort_values(["contrast", "cond"]))
    NZ = rd("L_table_noise_effect.csv", A3).rename(columns={"sigma02": "mean1", "sigma0": "mean2"})
    blk = ("MRP, termination, RMST, observation noise $\\sigma_{\\mathrm{obs}} = 0.2$ against 0 "
           "(same training seed; Holm over the four formulations of each pair of conditions)")
    for _, r in NZ.iterrows():
        ROWS.append(dict(block=blk, contrast=f"{NAME[r['variant']]}, noise vs none",
                         cond=str(r["pair"]).replace("→", "$\\to$"),
                         mean1=r["mean1"], mean2=r["mean2"], diff=r["diff"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                         rbc=r["rbc"], p_raw=r["p_raw"], p_holm=r["p_holm"], posthoc=False))

    # --- MRP without termination (v3 dynamics) ---------------------------------------------
    T = rd("NT_contrasts.csv", A3)
    T = T[T.metric == "frac_viable"]
    add("MRP, no termination, viable fraction; family vs Alive (7 contrasts per condition)",
        sel(T, C38, ["OR (Fixed) vs Alive (+1)", "DQ vs Alive (+1)", "DR vs Alive (+1)", "Dead-zone vs Alive (+1)"])
        .sort_values(["contrast", "cond"]))
    add("MRP, no termination, viable fraction; geometry family (6 contrasts per condition)",
        sel(T, C38, ["OR (Fixed) vs DQ", "OR (Fixed) vs DR", "OR (Fixed) vs Dead-zone", "OR (Fixed) vs OR no zone",
                     "OR (Fixed) vs OR no penalty", "DR + alive vs DR"]).sort_values(["contrast", "cond"]))

    # --- Foraging with termination --------------------------------------------------------
    FT = rd("FORAGE_F_term_contrasts.csv")
    add("Foraging, termination, RMST; family vs Alive (7 contrasts per condition)",
        sel(FT, F14, ["OR (Fixed) vs Alive (+1)", "DQ vs Alive (+1)"], metric="rmst").sort_values(["contrast", "cond"]))
    add("Foraging, termination, RMST; geometry family (6 contrasts per condition)",
        sel(FT, F14, ["OR (Fixed) vs DQ", "OR (Fixed) vs OR no penalty", "DR + alive vs DR"], metric="rmst")
        .sort_values(["contrast", "cond"]))
    PH = rd("FORAGE_F_term_posthoc_sign.csv")
    add("Foraging, termination, RMST; added after the pre-specified families (3 contrasts per condition)",
        split_contrast(sel(PH, F14, ["OR (Fixed) vs DR + alive", "DQ vs DR + alive"])).sort_values(["contrast", "cond"]),
        posthoc=True)
    add("Foraging, termination, in-zone fraction; geometry family (6 contrasts per condition)",
        sel(FT, F14, ["OR (Fixed) vs DQ"], metric="frac_in_zone"))

    # --- Foraging without termination -----------------------------------------------------
    FN = rd("FORAGE_F_noterm_contrasts.csv")
    add("Foraging, no termination, viable fraction; geometry family (6 contrasts per condition)",
        sel(FN, F14, ["OR (Fixed) vs DQ", "OR (Fixed) vs DR", "OR (Fixed) vs Dead-zone", "OR (Fixed) vs OR no penalty",
                      "OR (Fixed) vs OR no zone", "DR + alive vs DR"], metric="frac_viable", family="F2 geometry")
        .sort_values(["contrast", "cond"]))
    add("Foraging, no termination, time to first violation; geometry family (6 contrasts per condition)",
        sel(FN, F14, ["OR (Fixed) vs OR no zone"], metric="t_first_viol", family="F2 geometry"))

    # --- Number of needs -------------------------------------------------------------------
    for f, lab in (("FORAGE_ksweep_term_contrasts.csv", "termination, RMST"),
                   ("FORAGE_ksweep_noterm_contrasts.csv", "no termination, viable fraction")):
        K = split_contrast(rd(f))
        add(f"Foraging, number of needs, {lab} (3 contrasts per $k$)", K.sort_values(["contrast", "k"]), condcol="k")

    # --- Sign-shifted controls (pre-specified families a, b, c; Holm over F1--F6) ---------------
    S1 = rd("CLARIFY_S1_contrasts.csv", A1)
    for fam, lab, ph in (("a", "DQ + 0.01 vs DQ", False), ("b", "OR vs DQ + 0.01", False),
                         ("c", "OR $-$ 1 vs OR", False), ("d (descriptive)", "DQ + 0.1 vs Alive", True)):
        add("Foraging, termination, RMST, sign-shifted controls (Holm over F1--F6)"
            + ("; added after the pre-specified families" if ph else ""), S1[S1.family == fam], posthoc=ph)

    # --- Long training -------------------------------------------------------------------------
    S2 = rd("CLARIFY_S2_contrasts.csv", A1)
    add("Foraging, termination, RMST, $10^7$ steps, OR vs DQ (Holm over F1, F2, F5, F6)",
        S2[S2.family == "S2 — 10M"])
    add("Foraging, termination, RMST, $10^7$ steps, OR vs DQ, deterministic policy (Holm over F1--F6)",
        S2[S2.family == "S2 — 10M det"])
    R34 = rd("REVIEW_F34_10M_contrasts.csv", A1)
    add("Foraging, termination, RMST, $10^7$ steps, F3 and F4 (Holm over F3, F4); earlier study, not pre-specified",
        R34[(R34.v2 == "DQ") & (R34.kind == "eval")].sort_values("cond"), posthoc=True)
    add("Foraging, termination, RMST, $10^7$ steps (Holm over F5, F6); added after the pre-specified families",
        S2[S2.family == "S2 — 10M extra"].sort_values(["v1", "v2", "cond"]), posthoc=True)
    S3 = rd("CLARIFY_S3_contrasts.csv", A1)
    add("Foraging, termination, RMST, $3\\times10^7$ steps, F4 (single test; stochastic, then deterministic policy)", S3)
    S6 = rd("CLARIFY_S6_noterm_contrasts.csv", A1).rename(columns={"OR": "mean1", "DQ": "mean2"})
    S6["v1"], S6["v2"] = "OR_fixed", "DQ"
    for st, lab in ((3000000, "$3\\times10^6$"), (10000000, "$10^7$")):
        for kind, kl in (("eval", ""), ("eval_det", ", deterministic policy")):
            add(f"Foraging, no termination, viable fraction, {lab} steps{kl} (Holm over F3--F6)",
                S6[(S6.steps == st) & (S6.kind == kind)])
    IZ = rd("REVIEW_noterm_inzone_contrasts.csv", A1)
    add("Foraging, no termination, in-zone fraction, $10^7$ steps (Holm over F3--F6); added after the runs",
        IZ[(IZ.study.str.startswith("1e7")) & (IZ.kind == "eval")], posthoc=True)

    # --- Recovery ------------------------------------------------------------------------------
    RC = split_contrast(rd("FORAGE_recovery_contrasts.csv")).rename(columns={"m1": "mean1", "m2": "mean2"})
    add("Foraging, recovery success (6 contrasts per condition)",
        sel(RC, F14, ["OR_fixed vs Alive", "OR_fixed vs DQ", "OR_fixed vs OR_noPen"]).sort_values(["contrast", "cond"]))
    return pd.DataFrame(ROWS)


def num(x, dec):
    s = f"{x:.{dec}f}"
    if s.startswith("-"):
        s = s[1:] if float(s) == 0 else "$-$" + s[1:]
    return s


def pfmt(p):
    return f"{p:.3f}"


def to_latex(D):
    out = []
    out.append("{\\scriptsize\\setlength{\\tabcolsep}{4pt}")
    out.append("\\begin{longtable}{llrrrrrr}")
    cap = ("\\caption{Paired contrasts cited in the text: means of the two formulations, mean paired difference "
           "$\\Delta$ (first minus second) with bootstrap 95\\% confidence interval, rank-biserial correlation "
           "$r_{\\mathrm{rb}}$, exact two-sided Wilcoxon $p$ and Holm-adjusted $p$ ($n = 10$ training seeds; the "
           "smallest attainable raw $p$ is 0.002). The family over which $p$ is adjusted is given in each block "
           "heading. Contrasts marked $^\\dagger$ were not among the contrasts specified in advance. Endpoints are in "
           "their own units (RMST in steps; viable, in-zone and recovery fractions between 0 and 1). Unless stated "
           "otherwise the evaluation policy is the mean action (MRP) or the stochastic policy (foraging), and "
           "training lasted $10^5$ (MRP) or $3\\times10^6$ (foraging) steps. The MRP without termination was run "
           "in C3--C8 only.}\\label{tab:app_contrasts}\\\\")
    out.append(cap)
    head = ("\\toprule\nContrast & Cond & Mean 1 & Mean 2 & $\\Delta$ & 95\\% CI & $r_{\\mathrm{rb}}$ & "
            "$p$ / $p_{\\mathrm{Holm}}$ \\\\\n\\midrule")
    out.append(head + "\n\\endfirsthead")
    out.append("\\toprule\nContrast & Cond & Mean 1 & Mean 2 & $\\Delta$ & 95\\% CI & $r_{\\mathrm{rb}}$ & "
               "$p$ / $p_{\\mathrm{Holm}}$ \\\\\n\\midrule\n\\endhead")
    out.append("\\midrule\n\\multicolumn{8}{r}{\\textit{continued on next page}} \\\\\n\\endfoot")
    out.append("\\bottomrule\n\\endlastfoot")
    for block, g in D.groupby("block", sort=False):
        dec = 1 if "RMST" in block else (4 if block.startswith("MRP, no termination") else 3)
        out.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{block}}}}} \\\\")
        for _, r in g.iterrows():
            dag = "$^\\dagger$" if r.posthoc else ""
            out.append(f"{r.contrast}{dag} & {r.cond} & {num(r.mean1, dec)} & {num(r.mean2, dec)} & {num(r['diff'], dec)} & "
                       f"[{num(r.ci_lo, dec)}, {num(r.ci_hi, dec)}] & {num(r.rbc, 2)} & {pfmt(r.p_raw)} / {pfmt(r.p_holm)} \\\\")
    out.append("\\end{longtable}")
    out.append("}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    D = build()
    csv_out = os.path.join(A3, "REVIEW_v3_contrast_table.csv")
    tex_out = os.path.join(A3, "REVIEW_v3_contrast_table.tex")
    for f in (csv_out, tex_out):
        if os.path.exists(f):
            raise SystemExit(f"{f} exists; refusing to overwrite")
    D.to_csv(csv_out, index=False)
    tex = to_latex(D)
    with open(tex_out, "w", encoding="utf-8") as f:
        f.write(tex)
    print(len(D), "rows;", D.block.nunique(), "blocks")
