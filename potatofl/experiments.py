"""The federated experiment grid. One JSON line per finished run in RESULTS/fl_full/<name>.jsonl;
finished runs are skipped, so an interrupted grid can simply be restarted.

    python -m potatofl.pipeline federate pilot       # local learning rate, chosen on validation
    python -m potatofl.pipeline federate core        # the grid used in the report
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from . import config as C
from . import data as D
from . import federated as F

RES = C.RESULTS / "fl_full"
_DATA: dict = {}


def data():
    if not _DATA:
        _DATA["X"], _DATA["y"], _DATA["classes"], _ = D.load_data()
        if C.SMOKE:
            y = _DATA["y"]
            keep = np.concatenate([np.where(y == c)[0][:12] for c in range(len(_DATA["classes"]))])
            _DATA["X"], _DATA["y"] = _DATA["X"][keep], y[keep]
    return _DATA["X"], _DATA["y"], len(_DATA["classes"])


def append(name, row):
    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / f"{name}.jsonl", "a") as f:
        f.write(json.dumps(row, default=float) + "\n")


def done(name):
    p = RES / f"{name}.jsonl"
    return {json.dumps(json.loads(l)["key"], sort_keys=True) for l in p.read_text().splitlines()} if p.exists() else set()


def run_one(name, key, seed, init, K, alpha, scope="full", save=False, **kw):
    if json.dumps(key, sort_keys=True) in done(name):
        return
    X, y, n_classes = data()
    sp = np.load(C.MODELS / "v0" / f"seed{seed}" / "split.npz")
    rng = np.random.default_rng(1000 * seed + int(100 * alpha) + K)
    try:
        parts = F.dirichlet_partition(y, sp["clients"], K, alpha, rng, min_size=2 if C.SMOKE else 10)
    except RuntimeError as err:
        append(name, {"key": key, "skipped": str(err)})
        return
    print(f"[{time.strftime('%H:%M:%S')}] {name} {key}", flush=True)
    t = time.time()
    r = F.run_full(X, y, parts, sp["val"], sp["test"], n_classes, init, seed, scope=scope,
                   save_to=(RES / "models" / ("_".join(map(str, key)) + ".weights.h5")) if save else None,
                   ckpt=RES / "ckpt" / (name + "_" + "_".join(map(str, key)) + ".pkl"), **kw)
    h = r["history"]
    append(name, {"key": key, "seed": seed, "init": init, "K": K, "alpha": alpha, "scope": scope, "skew_js": F.label_skew(parts, y, n_classes),
                  "test_f1": r["test_macro_f1"], "test_acc": r["test_acc"], "n_params": r["n_params"],
                  "val_f1_curve": [x["val_macro_f1"] for x in h], "val_rounds": [x["round"] for x in h],
                  "bytes_up_total": h[-1]["bytes_up_total"], "bytes_down_total": h[-1]["bytes_down_total"],
                  "minutes": (time.time() - t) / 60, **{k: v for k, v in kw.items() if k != "dp"}, "dp": kw.get("dp")})
    print(f"  TEST macro-F1 {r['test_macro_f1']:.3f} acc {r['test_acc']:.3f} ({(time.time() - t) / 60:.1f} min)", flush=True)


def pilot():
    """Pick the local learning rate on validation macro-F1: 6 rounds, alpha 0.5, seed 42, init B."""
    rounds = 2 if C.SMOKE else 6
    for lr in (1e-5, 3e-5, 1e-4):
        run_one("pilot", ["pilot", lr], C.SEED, "B", 3 if C.SMOKE else 10, 0.5, rounds=rounds, eval_every=2, lr=lr)


def best_lr():
    rows = [json.loads(l) for l in (RES / "pilot.jsonl").read_text().splitlines()]
    return max((r for r in rows if "lr" in r), key=lambda r: r["val_f1_curve"][-1])["lr"]


def core(lr=None):
    lr = lr or best_lr()
    K = 3 if C.SMOKE else 10
    base = dict(rounds=2 if C.SMOKE else 30, local_epochs=1, lr=lr)
    for s in (C.SEEDS[:1] if C.SMOKE else C.SEEDS):
        # most informative cells first, so an interrupted grid still answers the main questions
        run_one("core", ["alpha", s, 0.5, "fedavg"], s, "B", K, 0.5, save=True, **base)  # the shipped model
        run_one("core", ["alpha", s, 100.0, "fedavg"], s, "B", K, 100.0, **base)
        for scope in ("partial", "head"):
            run_one("core", ["scope", s, scope, 0.5], s, "B", K, 0.5, scope=scope, **base)
        run_one("core", ["alpha", s, 0.1, "fedavg"], s, "B", K, 0.1, **base)
        run_one("core", ["init", s, "A", 0.5], s, "A", K, 0.5, **base)
        run_one("core", ["alpha", s, 1.0, "fedavg"], s, "B", K, 1.0, **base)
        for alpha in (0.5, 0.1):
            run_one("core", ["alpha", s, alpha, "fedprox"], s, "B", K, alpha, mu=0.1, **base)
        print("core seed", s, "done", flush=True)


def review(rounds=int(os.environ.get("POTATOFL_REVIEW_ROUNDS", 10))):
    """Short real run for a reviewer: start from the shipped v0, 10 federated rounds, FedAvg, alpha 0.5, seed 42, the same
    local learning rate as the grid. Saves the model for the `bundle` stage."""
    run_one("review", ["alpha", C.SEED, 0.5, "fedavg"], C.SEED, "B", 10, 0.5, save=True, rounds=rounds, local_epochs=1, lr=1e-5, eval_every=5)
