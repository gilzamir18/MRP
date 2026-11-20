"""
runner.py — runs one MRP training+evaluation job, or a list of them in parallel.

Every job is a dict; results go to ``<out>/<study>/<job_id>.json`` and existing
files are skipped, so a crashed/interrupted batch can simply be relaunched.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import sys
import time
import traceback
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

CONDITIONS = {
    "C1": dict(n=50, noise=0.0, budget=1.0),
    "C2": dict(n=50, noise=0.2, budget=1.0),
    "C3": dict(n=50, noise=0.0, budget=0.2),
    "C4": dict(n=50, noise=0.2, budget=0.2),
    "C5": dict(n=100, noise=0.0, budget=1.0),
    "C6": dict(n=100, noise=0.2, budget=1.0),
    "C7": dict(n=100, noise=0.0, budget=0.2),
    "C8": dict(n=100, noise=0.2, budget=0.2),
}

DEFAULTS = dict(timesteps=100_000, lr=3e-4, norm_reward=False, n_steps=1024,
                batch_size=64, n_epochs=10, z_init=0.5, beta_init=1.0,
                fixed_z=None, fixed_beta=None, reward_source="latent",
                noise_on="state", log_std_init=0.0, norm_obs=False, arch="mlp", terminate=True, eval_seeds=list(range(1000, 1020)), tag="base")


def job_id(job):
    return f"{job['cond']}__{job['variant']}__{job.get('tag', 'base')}__s{job['seed']}"


def run_job(job):
    import torch
    from mrp_env import MRPEnv, make_reward_fn
    from ppo_agent import ActorCritic, train, evaluate

    torch.set_num_threads(1)
    j = {**DEFAULTS, **job}
    out_path = os.path.join(j["out"], j["study"], job_id(j) + ".json")
    if os.path.exists(out_path):
        return out_path, "skip", 0.0
    t0 = time.time()
    seed = int(j["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    c = {**CONDITIONS.get(j["cond"], {}), **{k: j[k] for k in ("n", "noise", "budget") if k in j}}
    learn = j["variant"] == "OR_adapt"
    rfn = make_reward_fn(j["variant"], fixed_z=j["fixed_z"], fixed_beta=j["fixed_beta"])
    mk = lambda: MRPEnv(c["n"], c["budget"], c["noise"], rfn,
                        reward_source=j["reward_source"], noise_on=j["noise_on"],
                        terminate=j["terminate"])
    env = mk()
    ac = ActorCritic(env.obs_dim, env.act_dim, learn_or=learn,
                     z_init=j["z_init"], beta_init=j["beta_init"],
                     log_std_init=j["log_std_init"], arch=j["arch"])
    # intermediate evaluations (2026-11-20, second review): same deterministic evaluation as the
    # final one, at exact checkpoints; partial results are written as the run progresses
    cp_res = []

    def on_cp(c):
        ev_env = MRPEnv(c_["n"], c_["budget"], c_["noise"], rfn, noise_on=j["noise_on"],
                        terminate=j["terminate"])
        cp_res.append({"step": int(c), "eval": evaluate(ev_env, ac, j["eval_seeds"]),
                       "wall_s": time.time() - t0})
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path + ".partial.tmp", "w", encoding="utf-8") as f:
            json.dump({"job": {k: v for k, v in j.items() if k != "eval_seeds"}, "checkpoints": cp_res}, f)
        os.replace(out_path + ".partial.tmp", out_path + ".partial")

    c_ = c
    log = train(env, ac, total_timesteps=j["timesteps"], n_steps=j["n_steps"],
                batch_size=j["batch_size"], n_epochs=j["n_epochs"], lr=j["lr"],
                norm_reward=j["norm_reward"], norm_obs=j["norm_obs"],
                episode_seed_rng=np.random.default_rng(10_000 + seed),
                action_rng=np.random.default_rng(20_000 + seed),
                checkpoints=j.get("checkpoints"), on_checkpoint=on_cp)
    eval_env = MRPEnv(c["n"], c["budget"], c["noise"], rfn, noise_on=j["noise_on"],
                      terminate=j["terminate"])
    ev = evaluate(eval_env, ac, j["eval_seeds"])

    # thin the OR-parameter trajectory (one entry per update is ~100 entries: keep all)
    res = {
        "job": {k: v for k, v in j.items() if k != "eval_seeds"},
        "cond_params": c,
        "dynamics": env.dynamics,
        "eval": ev,
        "train_curve": {"step": log["ep_end_step"], "len": log["ep_len"]},
        "or_traj": ({"step": log["upd_step"], "params": log["or_params"]} if learn else None),
        "or_final": ac.or_params_summary(),
        "checkpoints": cp_res or None,
        "wall_s": time.time() - t0,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(res, f)
    os.replace(tmp, out_path)
    return out_path, "ok", res["wall_s"]


def _safe(job):
    try:
        return run_job(job)
    except Exception:
        return job_id({**DEFAULTS, **job}), "ERROR\n" + traceback.format_exc(), 0.0


def run_many(jobs, workers=15, label=""):
    todo = [j for j in jobs if not os.path.exists(
        os.path.join(j["out"], j["study"], job_id({**DEFAULTS, **j}) + ".json"))]
    print(f"[{label}] {len(jobs)} jobs, {len(todo)} to run, {workers} workers", flush=True)
    if not todo:
        return
    t0 = time.time()
    done = 0
    ctx = get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=20) as pool:
        for path, status, wall in pool.imap_unordered(_safe, todo):
            done += 1
            if status.startswith("ERROR"):
                print(f"  !! {path}: {status}", flush=True)
            elapsed = time.time() - t0
            eta = elapsed / done * (len(todo) - done)
            if done % 10 == 0 or done == len(todo) or status.startswith("ERROR"):
                print(f"  [{label}] {done}/{len(todo)}  last={wall:.0f}s  "
                      f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m", flush=True)
