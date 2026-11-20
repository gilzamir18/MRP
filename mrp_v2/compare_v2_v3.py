"""
compare_v2_v3.py — side-by-side comparison of the MRP results with the legacy dynamics
(docs/results_v2_noadapt/analysis) and the corrected v3 dynamics (docs/results_v3_mrp/analysis),
checked against the predictions P1–P8 of docs/prereg_mrp_v3_2026-11-17.md.

Usage: python compare_v2_v3.py -> docs/results_v3_mrp/analysis/COMPARE_legacy_v3.md
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(HERE, "..", "docs", "results_v2_noadapt", "analysis")
NEW = os.path.join(HERE, "..", "docs", "results_v3_mrp", "analysis")
CONDS = [f"C{i}" for i in range(1, 9)]
out = ["# MRP: dinâmica legacy × v3\n",
       "Gerado por `mrp_v2/compare_v2_v3.py`. Fontes: `results_v2_noadapt/analysis` (legacy) e "
       "`results_v3_mrp/analysis` (v3). Células: legacy → v3.\n"]


def rd(d, f):
    return pd.read_csv(os.path.join(d, f))


def arrow(a, b, fmt="{:.1f}"):
    fa = "—" if pd.isna(a) else fmt.format(a)
    fb = "—" if pd.isna(b) else fmt.format(b)
    return f"{fa} → {fb}"


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def p_mark(p):
    return "" if pd.isna(p) else ("*" if p < 0.05 else "")


# 1. RMST means (equivariant actor, termination)
o, n = rd(OLD, "L_table_rmst.csv").set_index("cond"), rd(NEW, "L_table_rmst.csv").set_index("cond")
vs = ["OR_fixed", "DQ", "DR", "Alive", "DR_alive", "DQ_alive", "DeadZone", "OR_noZone", "OR_noPen",
      "OR_min", "OR_softmin"]
rows = []
for c in CONDS:
    r = {"cond": c}
    for v in vs:
        r[v] = arrow(o.loc[c].get(f"{v}_mean"), n.loc[c].get(f"{v}_mean"))
    rows.append(r)
out += ["## 1. RMST (ator equivariante, com término, 10^5 passos)\n", md(pd.DataFrame(rows))]

# 2. Reference policies
o, n = rd(OLD, "L_bounds.csv").set_index("cond"), rd(NEW, "L_bounds.csv").set_index("cond")
rows = [{"cond": c, **{p: arrow(o.loc[c, p], n.loc[c, p]) for p in ["zero", "random", "prop", "greedy", "oracle"]}}
        for c in CONDS]
out += ["## 2. Políticas de referência (RMST)\n", md(pd.DataFrame(rows))]


# 3. Contrasts (Holm within family x condition)
def contrasts(fname, label):
    o, n = rd(OLD, fname), rd(NEW, fname)
    key = ["family", "cond", "v1", "v2"]
    m = o.merge(n, on=key, how="outer", suffixes=("_o", "_n"))
    m = m[m.cond.isin(CONDS)]
    rows = []
    for _, r in m.iterrows():
        rows.append({"family": r["family"], "cond": r["cond"], "contrast": f"{r['v1']} − {r['v2']}",
                     "diff": arrow(r.get("diff_o"), r.get("diff_n")),
                     "p_holm": arrow(r.get("p_holm_o"), r.get("p_holm_n"), "{:.3f}"),
                     "sig": f"{p_mark(r.get('p_holm_o')) or '·'} → {p_mark(r.get('p_holm_n')) or '·'}"})
    return [f"## {label}\n", md(pd.DataFrame(rows))]


out += contrasts("L_table_main_contrasts.csv", "3. Contrastes principais (L_main)")
out += contrasts("L_table_sign_controls.csv", "4. Controles de sinal (L_sign)")
out += contrasts("L_table_geometry_controls.csv", "5. Controles de geometria (L_geometry)")

# 6. MLP vs uniform
o, n = rd(OLD, "A_bounds.csv").set_index("cond"), rd(NEW, "A_bounds.csv").set_index("cond")
rows = []
for c in CONDS:
    r = {"cond": c, "zero": arrow(o.loc[c, "zero"], n.loc[c, "zero"])}
    for v in ["OR_fixed", "DQ", "DR", "Alive"]:
        r[v] = arrow(o.loc[c, f"{v}_mean"], n.loc[c, f"{v}_mean"]) + \
            f" ({'+' if n.loc[c, f'{v}_vs_zero'] > 0 else '−'}{'*' if n.loc[c, f'{v}_p_holm'] < .05 else ''})"
    rows.append(r)
out += ["## 6. MLP original contra a alocação uniforme (v3: sinal da diferença, * p_Holm < 0,05)\n",
        md(pd.DataFrame(rows))]

# 7. No termination
o, n = rd(OLD, "NT_contrasts.csv"), rd(NEW, "NT_contrasts.csv")
key = ["family", "metric", "cond", "v1", "v2"]
m = o.merge(n, on=key, how="outer", suffixes=("_o", "_n"))
m = m[m.metric == "frac_viable"]
rows = [{"family": r["family"][:12], "cond": r["cond"], "contrast": f"{r['v1']} − {r['v2']}",
         "diff": arrow(r.get("diff_o"), r.get("diff_n"), "{:+.3f}"),
         "p_holm": arrow(r.get("p_holm_o"), r.get("p_holm_n"), "{:.3f}"),
         "sig": f"{p_mark(r.get('p_holm_o')) or '·'} → {p_mark(r.get('p_holm_n')) or '·'}"}
        for _, r in m.iterrows()]
out += ["## 7. Sem término (fração viável)\n", md(pd.DataFrame(rows))]

# 8. Reward sign under the random policy
o, n = rd(OLD, "reward_scale_probe.csv"), rd(NEW, "reward_scale_probe.csv")
o, n = o[o.policy == "random"], n[n.policy == "random"]
rows = []
for v in sorted(n.variant.unique()):
    r = {"variant": v}
    for pair in (("C1", "C2"), ("C3", "C4"), ("C5", "C6"), ("C7", "C8")):
        r["/".join(pair)] = arrow(o[(o.variant == v) & o.cond.isin(pair)]["mean"].mean(),
                                  n[(n.variant == v) & n.cond.isin(pair)]["mean"].mean(), "{:+.3f}")
    rows.append(r)
out += ["## 8. Recompensa média por passo sob a política aleatória\n", md(pd.DataFrame(rows))]

path = os.path.join(NEW, "COMPARE_legacy_v3.md")
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("wrote", path)
