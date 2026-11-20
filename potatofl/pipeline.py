"""Command line for the whole pipeline. Each stage reads what the previous one wrote under ./work.

    python -m potatofl.pipeline ingest                       # raw folders -> cleaned index
    python -m potatofl.pipeline central --pool 0.4           # model v0 (central pool)
    python -m potatofl.pipeline central --pool 0             # centralised reference (all training data)
    python -m potatofl.pipeline federate pilot|core          # federated experiments
    python -m potatofl.pipeline bundle                       # calibrate the shipped model, write artifacts/
    python -m potatofl.pipeline report                       # tables and figures
    python -m potatofl.pipeline review                       # reviewer path: shipped v0 + 10 federated rounds + calibration (~25 min)
    python -m potatofl.pipeline all                          # everything above for seed 42 (hours on a CPU)

POTATOFL_SMOKE=1 shrinks every stage to a few images and rounds, to check the pipeline end to end in minutes.
"""
from __future__ import annotations

import argparse

from . import config as C


def main(argv=None):
    ap = argparse.ArgumentParser(prog="potatofl.pipeline", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("ingest")
    c = sub.add_parser("central")
    c.add_argument("--pool", type=float, default=C.POOL)
    c.add_argument("--seed", type=int, default=C.SEED)
    f = sub.add_parser("federate")
    f.add_argument("which", choices=["pilot", "core"])
    f.add_argument("--lr", type=float, default=None, help="local learning rate; default: the best of the pilot")
    b = sub.add_parser("bundle")
    b.add_argument("--weights", default=None, help="default: the alpha=0.5 FedAvg model saved by the core grid")
    b.add_argument("--seed", type=int, default=C.SEED)
    r = sub.add_parser("report")
    r.add_argument("--models", default=None)
    r.add_argument("--results", default=None)
    r.add_argument("--out", default=None)
    sub.add_parser("review", help="reviewer path: shipped v0 -> 10 federated rounds -> calibrate (about 25 min on 8 CPU cores, needs the real data)")
    sub.add_parser("all")
    a = ap.parse_args(argv)

    if a.stage in ("ingest", "all"):
        from . import ingest
        ingest.run()
    if a.stage in ("central", "all"):
        from . import central
        if a.stage == "all":
            for pool in (C.POOL, 0):
                central.run(C.SEED, pool)
        else:
            central.run(a.seed, a.pool)
    if a.stage in ("federate", "all"):
        from . import experiments
        if a.stage == "all" or a.which == "pilot":
            experiments.pilot()
        if a.stage == "all" or a.which == "core":
            experiments.core(getattr(a, "lr", None))
    if a.stage == "review":
        import shutil
        from . import bundle, experiments
        v0 = C.MODELS / "v0" / f"seed{C.SEED}"
        if not (v0 / "best.weights.h5").exists():
            v0.mkdir(parents=True, exist_ok=True)
            for f in (C.ARTIFACTS / "v0").glob("*"):
                shutil.copy(f, v0 / f.name)
        experiments.review()
        out = C.WORK / "review" / "artifacts"
        bundle.run(str(C.RESULTS / "fl_full" / "models" / f"alpha_{C.SEED}_0.5_fedavg.weights.h5"), C.SEED, out)
        print(f"\nReviewer run written to {out}. Shipped model: test macro-F1 0.805 after 30 rounds; this one after 10 rounds, so expect about 0.80.")
    if a.stage in ("bundle", "all"):
        from . import bundle
        w = getattr(a, "weights", None) or str(C.RESULTS / "fl_full" / "models" / "alpha_42_0.5_fedavg.weights.h5")
        bundle.run(w, getattr(a, "seed", C.SEED))
    if a.stage in ("report", "all"):
        from . import report
        report.run(getattr(a, "models", None), getattr(a, "results", None), getattr(a, "out", None))


if __name__ == "__main__":
    main()
