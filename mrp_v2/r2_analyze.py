"""
r2_analyze.py — analyses of the second-review studies, as pre-registered in prereg/prereg_2026-11-20.md.

    python r2_analyze.py [E] [D] [P] [M]      (default: all)

E  R2_nt_norm   : replication check, entropy (A1), OR vs DQ with return normalisation (A2 rule)
D  R2_dz_sign   : dead-zone sign shifts (B1-B3)
P  R2_long_pilot: MRP checkpoints, stabilisation criterion (D1, D2)
M  NT_noterm    : min / soft-min without termination ("aggregation" family)
Outputs: docs/results_v2/analysis/R2*.csv and R2_RESULTS.md (appended per section).
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(os.path.dirname(HERE), "docs")
V2, V3 = os.path.join(DOCS, "results_v2"), os.path.join(DOCS, "results_v3_mrp")
OUT = os.path.join(V2, "analysis")
MD = os.path.join(OUT, "R2_RESULTS.md")
SEEDS = list(range(42, 52))


def holm(p):
    p = np.asarray([q if np.isfinite(q) else 1.0 for q in p], float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def wil(x, y):
    d = np.asarray(x) - np.asarray(y)
    if np.all(d == 0):
        return 1.0
    return float(wilcoxon(d, method="exact").pvalue) if np.sum(d != 0) >= 1 else 1.0


def boot_ci(d, rng, n=10_000):
    d = np.asarray(d)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return np.percentile(m, [2.5, 97.5])


def load(study, cond, var, tag):
    out = {}
    for f in glob.glob(os.path.join(V2, study, f"{cond}__{var}__{tag}__s*.json")):
        r = json.load(open(f))
        out[int(r["job"]["seed"])] = r
    return out


def seed_metric(r, pol, metric):
    ev = r["eval"] if pol == "stoch" else r["eval_det"]
    if metric == "rmst":
        return float(np.mean([min(e["length"], 500) for e in ev]))
    if metric == "entropy":
        return float(np.mean([e["entropy"] for e in r["diag"][pol]]))
    return float(np.mean([e[metric] for e in ev]))


def contrast_rows(fam, pairs, rng):
    """pairs: list of (label, cond, xdict{seed:val}, ydict{seed:val}); Holm over the list."""
    rows = []
    for lab, c, x, y in pairs:
        s = sorted(set(x) & set(y))
        xa, ya = np.array([x[k] for k in s]), np.array([y[k] for k in s])
        d = xa - ya
        lo, hi = boot_ci(d, rng) if len(d) > 1 else (np.nan, np.nan)
        rows.append(dict(family=fam, contrast=lab, cond=c, n=len(s), mean_a=xa.mean(), mean_b=ya.mean(),
                         diff=d.mean(), ci_lo=lo, ci_hi=hi, p=wil(xa, ya)))
    for r, a in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = a
    return rows


def md(text):
    with open(MD, "a", encoding="utf-8") as f:
        f.write(text + "\n")


# ---------------------------------------------------------------------------------------------
def study_E(rng):
    F36 = ["F3", "F4", "F5", "F6"]
    base_study = {"F3": "F_noterm", "F4": "F_noterm", "F5": "F_ksweep", "F6": "F_ksweep"}
    md("\n## R2-A: OR vs DQ sem término, normalização e entropia (R2_nt_norm)\n")
    # replication: R2 runs without normalisation vs the published runs, same seeds
    rep = []
    for c in F36:
        for v in ("OR_fixed", "DQ"):
            new, old = load("R2_nt_norm", c, v, "noterm"), load(base_study[c], c, v, "noterm")
            for s in sorted(set(new) & set(old)):
                same = new[s]["eval"] == old[s]["eval"] and new[s]["eval_det"] == old[s]["eval_det"]
                rep.append(dict(cond=c, variant=v, seed=s, identical=same,
                                fv_new=seed_metric(new[s], "stoch", "frac_viable"),
                                fv_old=seed_metric(old[s], "stoch", "frac_viable")))
    rep = pd.DataFrame(rep)
    rep.to_csv(os.path.join(OUT, "R2A_replication.csv"), index=False)
    md(f"Réplica (sem normalização vs F_noterm/F_ksweep, mesmas seeds): {int(rep.identical.sum())}/{len(rep)} idênticas bit a bit.\n")
    # seed-level table
    rows = []
    for tag in ("noterm", "noterm_nr"):
        for c in F36:
            for v in ("OR_fixed", "DQ"):
                for s, r in load("R2_nt_norm", c, v, tag).items():
                    rows.append(dict(tag=tag, cond=c, variant=v, seed=s,
                                     fv_stoch=seed_metric(r, "stoch", "frac_viable"),
                                     fv_det=seed_metric(r, "det", "frac_viable"),
                                     zone_stoch=seed_metric(r, "stoch", "frac_in_zone"),
                                     ent_stoch=seed_metric(r, "stoch", "entropy"),
                                     ent_det=seed_metric(r, "det", "entropy"),
                                     ent_train_last=r["upd_entropy"][-1][1]))
    sl = pd.DataFrame(rows)
    sl.to_csv(os.path.join(OUT, "R2A_seed_level.csv"), index=False)
    g = sl.groupby(["tag", "cond", "variant"]).mean(numeric_only=True).drop(columns="seed").round(4)
    md("Médias por condição (10 seeds):\n\n" + g.to_markdown() + "\n")
    get = lambda tag, c, v, m: dict(zip(sl[(sl.tag == tag) & (sl.cond == c) & (sl.variant == v)].seed,
                                       sl[(sl.tag == tag) & (sl.cond == c) & (sl.variant == v)][m]))
    allrows = []
    for pol in ("fv_stoch", "fv_det"):
        allrows += contrast_rows(f"(a) OR vs DQ, normalised, {pol}",
                                 [("OR - DQ", c, get("noterm_nr", c, "OR_fixed", pol), get("noterm_nr", c, "DQ", pol)) for c in F36], rng)
        allrows += contrast_rows(f"(ref) OR vs DQ, not normalised, {pol}",
                                 [("OR - DQ", c, get("noterm", c, "OR_fixed", pol), get("noterm", c, "DQ", pol)) for c in F36], rng)
    allrows += contrast_rows("(b) entropy OR vs DQ, not normalised, stoch",
                             [("OR - DQ", c, get("noterm", c, "OR_fixed", "ent_stoch"), get("noterm", c, "DQ", "ent_stoch")) for c in F36], rng)
    allrows += contrast_rows("(desc) entropy OR vs DQ, normalised, stoch",
                             [("OR - DQ", c, get("noterm_nr", c, "OR_fixed", "ent_stoch"), get("noterm_nr", c, "DQ", "ent_stoch")) for c in F36], rng)
    for v in ("OR_fixed", "DQ"):
        allrows += contrast_rows(f"(desc) normalised - not normalised, {v}, fv_stoch",
                                 [("nr - base", c, get("noterm_nr", c, v, "fv_stoch"), get("noterm", c, v, "fv_stoch")) for c in F36], rng)
    df = pd.DataFrame(allrows)
    df.to_csv(os.path.join(OUT, "R2A_contrasts.csv"), index=False)
    md("Contrastes:\n\n" + df.round(4).to_markdown(index=False) + "\n")
    a = df[df.family == "(a) OR vs DQ, normalised, fv_stoch"]
    nsig = int((a.p_holm < 0.05).sum())
    md(f"Regra A2: com normalização, OR−DQ (estocástica) significativo em {nsig}/4 condições "
       f"→ {'afirmação mantida' if nsig >= 2 else 'afirmação retirada/condicionada à escala'}"
       " (a regra retira se não significativo em ≥ 3 de 4).\n")
    b = df[df.family == "(b) entropy OR vs DQ, not normalised, stoch"]
    md(f"A1 (entropia DQ > OR em F3–F6): diferenças OR−DQ = {list(b['diff'].round(4))}, p_holm = {list(b.p_holm.round(3))}.\n")


def study_D(rng):
    md("\n## R2-B: dead-zone com deslocamento de sinal, com término (R2_dz_sign)\n")
    F16 = ["F1", "F2", "F3", "F4", "F5", "F6"]
    src = lambda c, v: (load("R2_dz_sign", c, v, "term") if v.startswith("DZ_") or (v == "DeadZone" and c in ("F5", "F6"))
                        else load("F_term", c, v, "term") if c in ("F1", "F2", "F3", "F4")
                        else load("F_ksweep", c, v, "term"))
    M = {}
    for c in F16:
        for v in ("DZ_c005", "DZ_alive", "DeadZone", "OR_fixed", "Alive"):
            M[(c, v)] = {s: seed_metric(r, "stoch", "rmst") for s, r in src(c, v).items()}
    tab = pd.DataFrame({v: [np.mean(list(M[(c, v)].values())) if M[(c, v)] else np.nan for c in F16]
                        for v in ("OR_fixed", "Alive", "DeadZone", "DZ_c005", "DZ_alive")}, index=F16).round(1)
    md("RMST média (estocástica):\n\n" + tab.to_markdown() + "\n")
    rows = []
    for fam, a, b in (("(a) DZ+0.05 vs DZ", "DZ_c005", "DeadZone"), ("(b) DZ+1 vs DZ", "DZ_alive", "DeadZone"),
                      ("(c) OR vs DZ+0.05", "OR_fixed", "DZ_c005"), ("(d) DZ+0.05 vs Alive", "DZ_c005", "Alive")):
        rows += contrast_rows(fam, [(f"{a} - {b}", c, M[(c, a)], M[(c, b)]) for c in F16], rng)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "R2B_contrasts.csv"), index=False)
    md("Contrastes (Holm sobre F1–F6 em cada família):\n\n" + df.round(4).to_markdown(index=False) + "\n")


def study_P():
    md("\n## R2-D: piloto de treino longo no MRP (R2_long_pilot)\n")
    rows = []
    for f in glob.glob(os.path.join(V3, "R2_long_pilot", "*.json")):
        r = json.load(open(f))
        j = r["job"]
        met = "frac_viable" if not j.get("terminate", True) else "length"
        pts = [(c["step"], c["eval"]) for c in (r.get("checkpoints") or [])] + [(j["timesteps"], r["eval"])]
        for step, ev in pts:
            rows.append(dict(cond=j["cond"], variant=j["variant"], term=j.get("terminate", True), seed=j["seed"],
                             step=step, value=float(np.mean([min(e[met], 1000) for e in ev]))))
    if not rows:
        md("(sem resultados)\n")
        return
    sl = pd.DataFrame(rows)
    sl.to_csv(os.path.join(OUT, "R2D_long_pilot_seed_level.csv"), index=False)
    g = sl.groupby(["term", "cond", "variant", "step"]).value.mean().unstack("step").round(3)
    md(g.to_markdown() + "\n")
    st = []
    for (term, c, v), sub in sl.groupby(["term", "cond", "variant"]):
        m = sub.groupby("step").value.mean().sort_index()
        if len(m) < 2:
            continue
        a, b = m.iloc[-2], m.iloc[-1]
        ok = (abs(b - a) / max(abs(a), 1e-9) < 0.05) if term else (abs(b - a) < 0.005)
        st.append(dict(term=term, cond=c, variant=v, last_two=f"{m.index[-2]}->{m.index[-1]}", a=a, b=b, stable=ok))
    st = pd.DataFrame(st)
    st.to_csv(os.path.join(OUT, "R2D_stabilisation.csv"), index=False)
    md("Critério de estabilização (dois últimos pontos):\n\n" + st.round(3).to_markdown(index=False) + "\n")


def study_M(rng):
    md("\n## R2-E: min e soft-min sem término no MRP (NT_noterm)\n")
    C38 = ["C3", "C4", "C5", "C6", "C7", "C8"]
    M = {}
    for c in C38:
        for v in ("OR_fixed", "OR_min", "OR_softmin", "Alive", "DQ"):
            M[(c, v)] = {}
            for f in glob.glob(os.path.join(V3, "NT_noterm", f"{c}__{v}__shared_noterm__s*.json")):
                r = json.load(open(f))
                M[(c, v)][int(r["job"]["seed"])] = float(np.mean([e["frac_viable"] for e in r["eval"]]))
    tab = pd.DataFrame({v: [np.mean(list(M[(c, v)].values())) if M[(c, v)] else np.nan for c in C38]
                        for v in ("OR_fixed", "OR_min", "OR_softmin", "DQ", "Alive")}, index=C38).round(4)
    md("Fração viável média:\n\n" + tab.to_markdown() + "\n")
    rows = []
    for c in C38:
        rows += contrast_rows(f"aggregation {c}", [("OR - min", c, M[(c, "OR_fixed")], M[(c, "OR_min")]),
                                                   ("OR - softmin", c, M[(c, "OR_fixed")], M[(c, "OR_softmin")]),
                                                   ("min - Alive", c, M[(c, "OR_min")], M[(c, "Alive")]),
                                                   ("softmin - Alive", c, M[(c, "OR_softmin")], M[(c, "Alive")])], rng)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "R2E_aggregation_contrasts.csv"), index=False)
    md("Contrastes (família 'agregação', Holm dentro de cada condição):\n\n" + df.round(4).to_markdown(index=False) + "\n")


if __name__ == "__main__":
    which = sys.argv[1:] or ["E", "D", "P", "M"]
    rng = np.random.default_rng(20261120)
    open(MD, "w", encoding="utf-8").write("# Resultados dos estudos do segundo parecer (prereg 2026-11-20)\n")
    for w in which:
        {"E": lambda: study_E(rng), "D": lambda: study_D(rng), "P": study_P, "M": lambda: study_M(rng)}[w]()
    print(open(MD, encoding="utf-8").read())
