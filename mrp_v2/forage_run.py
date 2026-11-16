"""
forage_run.py — jobs and stages for the foraging gridworld (Crafter replacement).

    python forage_run.py STAGE [--out DIR] [--workers N]

Stages
  FP   pilot: 9 variants x {F2, K4d016} x {termination, no-termination}, 1 seed
  FT   main, with termination: 9 variants x 4 conditions x 10 seeds
  FN   no-termination control: same grid, terminate=False
Results: <out>/<study>/<cond>__<variant>__<tag>__s<seed>.json (resumable).

Pre-specified analysis (fixed 2026-11-10, after the pilot/calibration and before FT/FN):
  * unit = training seed (10 seeds); 20 paired evaluation episodes (seeds 1000-1019),
    stochastic policy with a fixed action RNG (primary) and argmax (secondary);
  * primary endpoint FT: RMST of episode length at tau = 500;
    primary endpoint FN: fraction of variable-steps within viability (|v| <= 1);
  * secondary: fraction of variable-steps in the zone (|v| <= 0.5), mean |v|,
    over-consumption share (consume actions taken with v >= 0), deterministic eval;
  * families/tests as in the MRP NT analysis (F1 = each variant vs Alive;
    F2 = OR_fixed vs DR/DQ/DeadZone/OR_noPen/OR_noZone, OR_adapt vs OR_fixed,
    DR_alive vs DR); exact Wilcoxon, Holm within condition x family x metric;
  * references: uniform-random and greedy heuristic on the same episodes.
Conditions were calibrated so that the Alive control does not reach the ceiling.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import sys
import time
import traceback
from multiprocessing import get_context

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FCONDS = {
    "F1": dict(k=2, decay_mu=0.008),
    "F2": dict(k=2, decay_mu=0.016),
    "F3": dict(k=4, decay_mu=0.008),
    "F4": dict(k=4, decay_mu=0.010),
    # number-of-needs sweep (same depletion as F1/F3): k = 6 and k = 8
    "F5": dict(k=6, decay_mu=0.008),
    "F6": dict(k=8, decay_mu=0.008),
    # calibration / pilot only (the pilot study F_pilot used k=4, decay 0.016 under the name F4)
    "K4d016": dict(k=4, decay_mu=0.016),
    "K4d012": dict(k=4, decay_mu=0.012),
    "K4d010": dict(k=4, decay_mu=0.010),
}
VARIANTS = ["OR_fixed", "OR_adapt", "DQ", "DR", "DR_alive", "Alive", "DeadZone", "OR_noZone", "OR_noPen"]
SEEDS = list(range(42, 52))
MAIN_CONDS = ["F1", "F2", "F3", "F4"]
EVAL_SEEDS = list(range(1000, 1020))
DEFAULTS = dict(timesteps=3_000_000, n_envs=16, terminate=True, W=9, lr=3e-4, ent_coef=0.01)


def job_id(j):
    return f"{j['cond']}__{j['variant']}__{j['tag']}__s{j['seed']}"


def run_job(job):
    import numpy as np
    import torch
    from forage_env import ForageEnv, make_reward_fn
    from forage_ppo import Agent, train, evaluate
    torch.set_num_threads(1)
    j = {**DEFAULTS, **job}
    path = os.path.join(j["out"], j["study"], job_id(j) + ".json")
    if os.path.exists(path):
        return path, "skip", 0.0
    t0 = time.time()
    c = FCONDS[j["cond"]]
    k = c["k"]
    kw = dict(W=j["W"], decay_mu=c["decay_mu"])
    env = ForageEnv(j["n_envs"], k, make_reward_fn(j["variant"]), terminate=j["terminate"], **kw)
    zi, bi = j.get("z_init", 0.5), j.get("beta_init", 1.0)
    ag = Agent(env.obs_dim, env.n_actions, learn_or=(j["variant"] == "OR_adapt"), z_init=zi, beta_init=bi)
    log = train(env, ag, j["timesteps"], seed=int(j["seed"]), lr=j["lr"], ent_coef=j["ent_coef"],
                outer=(j["variant"] == "OR_outer"), outer_kw=dict(z_init=zi, beta_init=bi))
    ev = evaluate(ag, k, kw, EVAL_SEEDS, terminate=j["terminate"], deterministic=False)
    ev_det = evaluate(ag, k, kw, EVAL_SEEDS, terminate=j["terminate"], deterministic=True)
    rec = None
    if j.get("recovery"):
        from forage_ppo import recovery_test
        # deficit fixed a priori: 25 steps to violation at the mean depletion rate
        # (calibrated: greedy heuristic recovers in 99-100% of episodes, random in 0-2.5%)
        rec = recovery_test(ag, k, kw, EVAL_SEEDS, deficit=-(1.0 - 25 * c["decay_mu"]))
        os.makedirs(os.path.join(j["out"], j["study"]), exist_ok=True)
        torch.save(ag.state_dict(), path.replace(".json", ".pt"))
    res = {"job": j, "cond_params": {**c, **kw}, "eval": ev, "eval_det": ev_det, "recovery": rec,
           "train_curve": {"step": log["ep_end_step"], "len": log["ep_len"], "ret": log["ep_ret"]},
           "or_traj": log["or_traj"], "or_final": ag.or_summary() or log.get("outer_final"), "wall_s": time.time() - t0}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(res, f)
    os.replace(path + ".tmp", path)
    return path, "ok", res["wall_s"]


def _safe(job):
    try:
        return run_job(job)
    except Exception:
        return job_id({**DEFAULTS, **job}), "ERROR\n" + traceback.format_exc(), 0.0


def run_many(jobs, workers, label):
    todo = [j for j in jobs if not os.path.exists(os.path.join(j["out"], j["study"], job_id({**DEFAULTS, **j}) + ".json"))]
    print(f"[{label}] {len(jobs)} jobs, {len(todo)} to run, {workers} workers", flush=True)
    if not todo:
        return
    t0, done = time.time(), 0
    with get_context("spawn").Pool(workers, maxtasksperchild=10) as pool:
        for path, status, wall in pool.imap_unordered(_safe, todo):
            done += 1
            el = time.time() - t0
            if status.startswith("ERROR"):
                print(f"  !! {path}: {status}", flush=True)
            if done % 10 == 0 or done == len(todo) or status.startswith("ERROR") or len(todo) <= 40:
                print(f"  [{label}] {done}/{len(todo)} last={wall:.0f}s elapsed={el/60:.1f}m "
                      f"eta={el/done*(len(todo)-done)/60:.1f}m", flush=True)


def _grid(out, study, conds, seeds, terminate, tag, **kw):
    return [dict(out=out, study=study, cond=c, variant=v, seed=s, terminate=terminate, tag=tag, **kw)
            for c in conds for v in VARIANTS for s in seeds]


def stage_FP(out, **kw):
    return (_grid(out, "F_pilot", ["F2", "K4d016"], [42], True, "term", **kw)
            + _grid(out, "F_pilot", ["F2", "K4d016"], [42], False, "noterm", **kw))


def stage_FT(out, **kw):
    return _grid(out, "F_term", MAIN_CONDS, SEEDS, True, "term", **kw)


def stage_FN(out, **kw):
    return _grid(out, "F_noterm", MAIN_CONDS, SEEDS, False, "noterm", **kw)


def stage_FC(out, **kw):
    return [dict(out=out, study="F_calib", cond=c, variant=v, seed=42, terminate=True, tag="term", **kw)
            for c in ["F1", "F3", "K4d010", "K4d012"] for v in ["OR_fixed", "DQ", "Alive"]]


def stage_FR(out, **kw):
    """Retrain the FT models with saved weights + recovery test (same seeds as FT)."""
    return _grid(out, "F_term_rec", MAIN_CONDS, SEEDS, True, "term", recovery=True, **kw)


Z_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
B_GRID = [0.0, 0.5, 1.0, 2.0, 4.0]


def stage_FZ(out, **kw):
    """Optimal-reward landscape: fixed OR on a (z, beta) grid, judged by external survival."""
    return [dict(out=out, study="F_zbgrid", cond=c, variant=f"ORg_z{z:.2f}_b{b:.2f}", seed=s,
                 terminate=True, tag="term", **kw)
            for c in MAIN_CONDS for z in Z_GRID for b in B_GRID for s in SEEDS[:5]]


def stage_FO(out, **kw):
    """Non-circular adaptation: (z, beta) updated by an outer loop on external fitness."""
    return ([dict(out=out, study="F_term", cond=c, variant="OR_outer", seed=s, terminate=True,
                  tag="term", **kw) for c in MAIN_CONDS for s in SEEDS]
            + [dict(out=out, study="F_noterm", cond=c, variant="OR_outer", seed=s, terminate=False,
                    tag="noterm", **kw) for c in MAIN_CONDS for s in SEEDS])


# ---- follow-up studies (2026-11-15) -------------------------------------------------------
INITS = [(0.30, 2.0), (0.30, 4.0), (0.88, 0.10)]   # punitive (moderate), punitive (strong), saturated


def stage_FI(out, **kw):
    """Does adaptation help when it STARTS from a bad calibration?  Joint (circular) and
    outer-loop adaptation started at three points outside the good band of the (z, beta)
    landscape; Fixed OR at the same points (for (0.30, 2.0) and (0.30, 4.0) it is already in
    F_zbgrid with the same seeds)."""
    jobs = []
    for (z, b) in INITS:
        tag = f"init_z{z:.2f}_b{b:.2f}"
        for c in MAIN_CONDS:
            for sd in SEEDS[:5]:
                for v in ("OR_adapt", "OR_outer"):
                    jobs.append(dict(out=out, study="F_init", cond=c, variant=v, seed=sd, terminate=True,
                                     tag=tag, z_init=z, beta_init=b, **kw))
                if (z, b) == (0.88, 0.10):
                    jobs.append(dict(out=out, study="F_init", cond=c, variant=f"ORg_z{z:.2f}_b{b:.2f}",
                                     seed=sd, terminate=True, tag=tag, **kw))
    return jobs


def stage_FK(out, **kw):
    """Number-of-needs sweep: k = 6 and 8 (k = 2 and 4 are F1 and F3 of the main study)."""
    return [dict(out=out, study="F_ksweep", cond=c, variant=v, seed=sd, terminate=t,
                 tag="term" if t else "noterm", **kw)
            for c in ("F5", "F6") for v in ("OR_fixed", "DQ", "Alive") for t in (True, False) for sd in SEEDS]


def stage_FL(out, **kw):
    """Long training (1e7 steps): does Adaptive OR lose performance as it drifts towards the
    saturated region?  Fixed OR with the same seeds as reference."""
    kw = {**kw, "timesteps": 10_000_000}
    return [dict(out=out, study="F_long", cond=c, variant=v, seed=sd, terminate=True, tag="term10M", **kw)
            for c in ("F1", "F3") for v in ("OR_adapt", "OR_fixed") for sd in SEEDS[:5]]


def stage_FL2(out, **kw):
    """Convergence check for k = 4 (1e7 steps): does OR ~ DQ in F3/F4 survive longer training?
    Same study/tag as FL, so F3 OR_fixed is reused (skipped)."""
    kw = {**kw, "timesteps": 10_000_000}
    return [dict(out=out, study="F_long", cond=c, variant=v, seed=sd, terminate=True, tag="term10M", **kw)
            for c in ("F3", "F4") for v in ("OR_fixed", "DQ", "Alive") for sd in SEEDS[:5]]


def stage_FX(out, **kw):
    """All three follow-ups in one pool (long runs first to shorten the tail)."""
    return stage_FL(out, **kw) + stage_FI(out, **kw) + stage_FK(out, **kw)


STAGES = {"FI": stage_FI, "FK": stage_FK, "FL": stage_FL, "FL2": stage_FL2, "FX": stage_FX, "FZ": stage_FZ, "FO": stage_FO, "FR": stage_FR, "FC": stage_FC, "FP": stage_FP, "FT": stage_FT, "FN": stage_FN}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "docs", "results_v2"))
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--timesteps", type=int, default=None)
    a = ap.parse_args()
    extra = {} if a.timesteps is None else {"timesteps": a.timesteps}
    for st in a.stages:
        run_many(STAGES[st](a.out, **extra), a.workers, st)
