"""
experiments.py — revised MRP experimental programme (response to review).

Stages (run in this order; each is resumable):

  A  main      12 reward variants x 8 conditions x 10 training seeds, shared
               default config (paper Table 2).  Variants:
                 main ......... OR_adapt, OR_fixed, DQ, DR
                 sign/offset .. Alive (r=+1), DR_alive (1-D), DQ_alive (1+dD)
                 geometry ..... DeadZone -max(0,|v|-z)^2, OR_noZone (z=0),
                                OR_noPen (beta=0), OR_min, OR_softmin
  T  tune      modest per-method tuning: lr {1e-4,3e-4,1e-3} x reward-norm
               {off,on} on C1 and C5 with 3 separate tuning seeds (900-902)
  F  tuned     final runs of every variant with its tuned config (only for
               variants whose best config differs from the default)
  P  params    calibration-vs-drift of Adaptive OR:
                 P1 four alternative initialisations (z0,beta0) in all 8 conds
                 P2 OR fixed at the values Adaptive OR learned in the SAME cond
                 P3 OR fixed at values learned in the budget-swapped cond
                 P4 long training (400k) for OR_adapt/OR_fixed/DQ/DR on C1,C3,C5,C7
  N  noise     mechanism tests:
                 N1 noise sweep sigma in {0.1, 0.4} (0 and 0.2 come from A)
                 N2 reward computed from the noisy observation (C2, C6)
                 N3 original noise placement (noise on all 4n+1 inputs; C2, C6)
  L  learn     main + sign variants with a permutation-equivariant (shared) actor,
               all 8 conds x 10 seeds, + 300k-step subset (C1, C3, C5; 5 seeds)
  L2 long      300k steps, C3 and C5, main + Alive + DR_alive, 10 seeds (adds to L)
  LG geometry  regime-geometry controls with the shared actor, C3-C6, 10 seeds
  LN rewnorm   main + Alive + DR_alive with reward normalisation, shared actor, C3-C6
  NT  noterm   no-termination control (shared actor): a variable crossing the viability
               bound is clipped to it and episodes always last 1000 steps, so a constant
               per-step reward carries no information.  Main + Alive + DR_alive + geometry
               (DeadZone, OR_noPen, OR_noZone), noise-free scarce/large conds C3, C5, C7.
               Primary endpoint (fixed before running): fraction of variable-steps within
               the viability bound (frac_viable).  C1 is at ceiling for any sensible policy.
  NT2 noterm   same as NT for the noisy conditions C4, C6, C8

Usage:  python experiments.py A [B ...] [--workers 15] [--out ../docs/results_v2]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from runner import CONDITIONS, run_many  # noqa: E402

MAIN = ["OR_adapt", "OR_fixed", "DQ", "DR"]
SIGN = ["Alive", "DR_alive", "DQ_alive"]
GEOM = ["DeadZone", "OR_noZone", "OR_noPen", "OR_min", "OR_softmin"]
ALL_VARIANTS = MAIN + SIGN + GEOM
SEEDS = list(range(42, 52))          # 10 training seeds (42-44 were the original three)
TUNE_SEEDS = [900, 901, 902]
TUNE_CONDS = ["C1", "C5"]
TUNE_GRID = [(lr, nr) for lr in (1e-4, 3e-4, 1e-3) for nr in (False, True)]
INITS = [(0.3, 0.5), (0.3, 2.0), (0.7, 0.5), (0.7, 2.0)]
SWAP = {"C1": "C3", "C3": "C1", "C2": "C4", "C4": "C2",
        "C5": "C7", "C7": "C5", "C6": "C8", "C8": "C6"}


def cfg_tag(lr, nr):
    return f"lr{lr:g}_nr{int(nr)}"


def stage_A(out):
    return [dict(study="A_main", out=out, cond=c, variant=v, seed=s)
            for c in CONDITIONS for v in ALL_VARIANTS for s in SEEDS]


def stage_T(out):
    return [dict(study="T_tune", out=out, cond=c, variant=v, seed=s, lr=lr,
                 norm_reward=nr, tag=cfg_tag(lr, nr))
            for c in TUNE_CONDS for v in ALL_VARIANTS for (lr, nr) in TUNE_GRID
            for s in TUNE_SEEDS]


def best_configs(out):
    """Pick, per variant, the config with the best mean rank across tuning conds."""
    import numpy as np
    rows = {}
    for p in glob.glob(os.path.join(out, "T_tune", "*.json")):
        r = json.load(open(p))
        j = r["job"]
        rm = float(np.mean([e["length"] for e in r["eval"]]))
        rows.setdefault((j["variant"], j["tag"], j["cond"]), []).append(rm)
    best = {}
    for v in ALL_VARIANTS:
        score = {}
        for c in TUNE_CONDS:
            vals = {cfg_tag(lr, nr): float(np.mean(rows.get((v, cfg_tag(lr, nr), c), [np.nan])))
                    for lr, nr in TUNE_GRID}
            order = sorted(vals, key=lambda k: -vals[k])
            for rank, k in enumerate(order):
                score[k] = score.get(k, 0) + rank
        best[v] = min(score, key=lambda k: score[k]) if score else cfg_tag(3e-4, False)
    return best


def stage_F(out):
    best = best_configs(out)
    with open(os.path.join(out, "tuned_configs.json"), "w") as f:
        json.dump(best, f, indent=2)
    jobs = []
    for v, tag in best.items():
        if tag == cfg_tag(3e-4, False):
            continue  # identical to stage A
        lr = float(tag.split("_")[0][2:])
        nr = tag.endswith("nr1")
        jobs += [dict(study="F_tuned", out=out, cond=c, variant=v, seed=s, lr=lr,
                      norm_reward=nr, tag=tag) for c in CONDITIONS for s in SEEDS]
    return jobs


def learned_values(out):
    import numpy as np
    vals = {}
    for p in glob.glob(os.path.join(out, "A_main", "*__OR_adapt__base__*.json")):
        r = json.load(open(p))
        f = r["or_final"]
        vals.setdefault(r["job"]["cond"], []).append((f["z_of_mean"], f["beta_of_mean"]))
    return {c: tuple(np.mean(np.array(v), axis=0)) for c, v in vals.items()}


def stage_P(out):
    jobs = []
    p_seeds = SEEDS[:5]
    for c in CONDITIONS:
        for (z0, b0) in INITS:
            jobs += [dict(study="P_params", out=out, cond=c, variant="OR_adapt", seed=s,
                          z_init=z0, beta_init=b0, tag=f"init_z{z0}_b{b0}") for s in p_seeds]
    lv = learned_values(out)
    for c in CONDITIONS:
        if c in lv:
            z, b = lv[c]
            jobs += [dict(study="P_params", out=out, cond=c, variant="OR_fixed", seed=s,
                          fixed_z=z, fixed_beta=b, tag="fixed_at_own") for s in SEEDS]
        if SWAP[c] in lv:
            z, b = lv[SWAP[c]]
            jobs += [dict(study="P_params", out=out, cond=c, variant="OR_fixed", seed=s,
                          fixed_z=z, fixed_beta=b, tag="fixed_at_swap") for s in SEEDS]
    for c in ["C1", "C3", "C5", "C7"]:
        jobs += [dict(study="P_params", out=out, cond=c, variant=v, seed=s,
                      timesteps=400_000, tag="long400k") for v in MAIN for s in p_seeds]
    return jobs


def stage_N(out):
    jobs = []
    for n in (50, 100):
        for sig in (0.1, 0.4):
            cname = f"n{n}_s{sig}_b1"
            jobs += [dict(study="N_noise", out=out, cond=cname, n=n, noise=sig, budget=1.0,
                          variant=v, seed=s, tag="sweep")
                     for v in MAIN + ["Alive"] for s in SEEDS]
    for c in ("C2", "C6"):
        jobs += [dict(study="N_noise", out=out, cond=c, variant=v, seed=s,
                      reward_source="observed", tag="rew_observed") for v in MAIN for s in SEEDS]
        jobs += [dict(study="N_noise", out=out, cond=c, variant=v, seed=s,
                      noise_on="all", tag="noise_all") for v in MAIN for s in SEEDS]
    return jobs


def stage_L(out):
    """Learnability: the joint 2x64 MLP of the paper does not learn the MRP (stage A: every
    trained agent ends at or below the untrained/uniform policy). Repeat the core comparison
    with a permutation-equivariant actor (same small MLP applied to each variable's
    features), which does learn in a pilot; plus a 300k-step subset."""
    jobs = [dict(study="L_learn", out=out, cond=c, variant=v, seed=s, arch="shared", tag="shared")
            for c in CONDITIONS for v in MAIN + SIGN for s in SEEDS]
    jobs += [dict(study="L_learn", out=out, cond=c, variant=v, seed=s, arch="shared",
                  timesteps=300_000, tag="shared_long300k")
             for c in ("C1", "C3", "C5") for v in MAIN for s in SEEDS[:5]]
    return jobs


def stage_L2(out):
    """Long-training check with enough seeds for tests and with the survival-bonus controls:
    C3 and C5 (where 300k steps separate the rewards), 10 seeds, main + Alive + DR_alive."""
    return [dict(study="L_learn", out=out, cond=c, variant=v, seed=s, arch="shared",
                 timesteps=300_000, tag="shared_long300k")
            for c in ("C3", "C5") for v in MAIN + ["Alive", "DR_alive"] for s in SEEDS]


L_SEP_CONDS = ["C3", "C4", "C5", "C6"]   # where the learning agent separates the rewards
# (C1/C2: every reward near the 1000-step ceiling; C7/C8: every reward within ~1 step of the floor)


def stage_LG(out):
    """Regime-geometry controls with the learning (shared) agent."""
    return [dict(study="L_learn", out=out, cond=c, variant=v, seed=s, arch="shared", tag="shared")
            for c in L_SEP_CONDS for v in GEOM for s in SEEDS]


def stage_LN(out):
    """Reward-scale control with the learning agent: running-return reward normalisation."""
    return [dict(study="L_learn", out=out, cond=c, variant=v, seed=s, arch="shared",
                 norm_reward=True, tag="shared_nr1")
            for c in L_SEP_CONDS for v in MAIN + ["Alive", "DR_alive"] for s in SEEDS]


NT_VARIANTS = MAIN + ["Alive", "DR_alive", "DeadZone", "OR_noPen", "OR_noZone"]


def _nt(out, conds, study="NT_noterm", seeds=SEEDS, **kw):
    return [dict(study=study, out=out, cond=c, variant=v, seed=s, arch="shared",
                 terminate=False, tag="shared_noterm", **kw)
            for c in conds for v in NT_VARIANTS for s in seeds]


def stage_NTP(out):
    """Short pilot (30k steps, 1 seed, C3) written to a separate study folder."""
    return _nt(out, ["C3"], study="NT_pilot", seeds=[42], timesteps=30_000)


def stage_NT(out):
    return _nt(out, ["C3", "C5", "C7"])


def stage_NT2(out):
    return _nt(out, ["C4", "C6", "C8"])


# ---- second review (2026-11-20; pre-registered in prereg/prereg_2026-11-20.md) -------------
R2_LONG_VARIANTS = ["OR_fixed", "DQ", "DR", "Alive", "DR_alive"]
R2_CP_TERM = [100_000, 200_000, 300_000, 500_000, 1_000_000, 1_500_000]       # final 2e6
R2_CP_NOTERM = [100_000, 200_000, 300_000, 500_000, 750_000]                  # final 1e6


def stage_R2P(out):
    """Long-training pilot: does learning in C5 (where 3e5 steps already helped) and C7 (where
    every agent stays near uniform allocation) stabilise?  3 seeds, exact checkpoints; the
    1e5 checkpoint reproduces the L_learn / NT_noterm runs with the same seeds."""
    nt = [dict(study="R2_long_pilot", out=out, cond=c, variant=v, seed=s, arch="shared",
               terminate=False, tag="shared_noterm_1M", timesteps=1_000_000, checkpoints=R2_CP_NOTERM)
          for c in ("C5", "C7") for v in R2_LONG_VARIANTS for s in SEEDS[:3]]
    tm = [dict(study="R2_long_pilot", out=out, cond=c, variant=v, seed=s, arch="shared",
               tag="shared_2M", timesteps=2_000_000, checkpoints=R2_CP_TERM)
          for c in ("C5", "C7") for v in R2_LONG_VARIANTS for s in SEEDS[:3]]
    return nt + tm


def stage_R2M(out):
    """Min and soft-min aggregation without termination (C3-C8, 10 seeds, 1e5 steps), with
    OR as reference from NT_noterm (same seeds and settings)."""
    return [dict(study="NT_noterm", out=out, cond=c, variant=v, seed=s, arch="shared",
                 terminate=False, tag="shared_noterm")
            for c in ("C3", "C4", "C5", "C6", "C7", "C8") for v in ("OR_min", "OR_softmin") for s in SEEDS]


STAGES = {"A": stage_A, "T": stage_T, "F": stage_F, "P": stage_P, "N": stage_N, "L": stage_L,
          "L2": stage_L2, "LG": stage_LG, "LN": stage_LN,
          "NTP": stage_NTP, "NT": stage_NT, "NT2": stage_NT2,
          "R2P": stage_R2P, "R2M": stage_R2M}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=list(STAGES))
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "docs", "results_v2"))
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    for st in a.stages:
        run_many(STAGES[st](out), workers=a.workers, label=st)
