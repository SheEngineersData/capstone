"""Turn the raw run records into the tables and figures of the report.

Reads  <models>/{v0,central}/seed*/summary.json and <results>/fl_full/core.jsonl
Writes <out>/tables.json, tables.tex (\\input by the report), fig_alpha.pdf, fig_curves.pdf, fig_scope.pdf

    python -m potatofl.pipeline report
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as C


def load(models: Path, results: Path):
    rows = [json.loads(l) for l in (results / "fl_full" / "core.jsonl").read_text().splitlines()]
    rows = [r for r in rows if "test_f1" in r]
    summ = {k: {} for k in ("v0", "central")}
    for k in summ:
        for f in sorted((models / k).glob("seed*/summary.json")):
            summ[k][int(f.parent.name[4:])] = json.loads(f.read_text())
    return rows, summ


def ms(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=0)), len(v)


def fmt(v, d=3, show_n=True):
    m, s, n = ms(v)
    return f"{m:.{d}f}" + (f" $\\pm$ {s:.{d}f}" if n > 1 else "") + (f" (n={n})" if show_n else "")


def run(models: Path | None = None, results: Path | None = None, out: Path | None = None, bundle_dir: Path | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models, results = Path(models or C.MODELS), Path(results or C.RESULTS)
    out = Path(out or results)
    out.mkdir(parents=True, exist_ok=True)
    rows, summ = load(models, results)
    key = lambda r: r["key"]
    avg = {(r["alpha"], r["seed"]): r for r in rows if key(r)[0] == "alpha" and key(r)[3] == "fedavg"}
    prox = {(r["alpha"], r["seed"]): r for r in rows if key(r)[0] == "alpha" and key(r)[3] == "fedprox"}
    initA = {r["seed"]: r for r in rows if key(r)[0] == "init"}
    scope = {(r["scope"], r["seed"]): r for r in rows if key(r)[0] == "scope"}
    f1 = lambda d, a: [r["test_f1"] for (al, s), r in d.items() if al == a]
    v0 = [s["test"]["macro_f1"] for s in summ["v0"].values()]
    cen = [s["test"]["macro_f1"] for s in summ["central"].values()]
    T = {"v0": fmt(v0) if v0 else "n/a", "central": fmt(cen) if cen else "n/a"}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    BLUE, ORANGE, GREY = "#2a6f97", "#d9822b", "#777777"

    # --- table 1: label skew sweep
    alphas = sorted({a for a, _ in avg}, reverse=True)
    t1 = []
    for a in alphas:
        r0 = next(r for (al, s), r in avg.items() if al == a)
        t1.append({"alpha": a, "skew_js": r0["skew_js"], "fedavg": f1(avg, a), "fedprox": f1(prox, a)})
    lines = ["\\begin{tabular}{rrcc}", "\\toprule", "$\\alpha$ & label skew (JS) & FedAvg macro-F1 & FedProx macro-F1 \\\\", "\\midrule"]
    for r in t1:
        lines.append(f"{r['alpha']:g} & {r['skew_js']:.2f} & {fmt(r['fedavg'])} & {fmt(r['fedprox']) if r['fedprox'] else '--'} \\\\")
    lines += ["\\midrule", f"\\multicolumn{{2}}{{l}}{{v0 (central pool only)}} & \\multicolumn{{2}}{{c}}{{{T['v0']}}} \\\\",
              f"\\multicolumn{{2}}{{l}}{{Centralised, all data}} & \\multicolumn{{2}}{{c}}{{{T['central']}}} \\\\", "\\bottomrule", "\\end{tabular}"]
    tex = {"skew": "\n".join(lines)}

    # --- table 2: initialisation and update scope
    full = f1(avg, 0.5)
    t2 = [("Full network from v0 (default)", full, avg[(0.5, next(iter(s for a, s in avg if a == 0.5)))]["n_params"])]
    t2 += [(f"{'Last blocks' if k == 'partial' else 'Classifier head only'} from v0", [r["test_f1"] for (sc, s), r in scope.items() if sc == k],
            next((r["n_params"] for (sc, s), r in scope.items() if sc == k), 0)) for k in ("partial", "head")]
    if initA:
        t2.append(("Full network, no central stage (empty head)", [r["test_f1"] for r in initA.values()], next(iter(initA.values()))["n_params"]))
    lines = ["\\begin{tabular}{lrrc}", "\\toprule", "Setting ($\\alpha=0.5$) & parameters sent & MB per client per round & macro-F1 \\\\", "\\midrule"]
    for name, v, n in t2:
        lines.append(f"{name} & {n:,} & {n * 4 / 1e6:.2f} & {fmt(v) if v else '--'} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex["scope"] = "\n".join(lines)
    (out / "tab_skew.tex").write_text(tex["skew"])
    (out / "tab_scope.tex").write_text(tex["scope"])
    bundle = Path(bundle_dir or C.BUNDLE)
    if (bundle / "calibration.json").exists():
        import pandas as pd
        cal = json.loads((bundle / "calibration.json").read_text())
        t = cal["test"]
        (out / "tab_tiers.tex").write_text(" \\\\\n".join(
            f"{k} & {t['tiers'][k]['share'] * 100:.1f}\\% & " + ("--" if t['tiers'][k]['accuracy'] is None else f"{t['tiers'][k]['accuracy'] * 100:.1f}\\%") for k in ("very likely", "possible", "unsure")))
        pc = pd.read_csv(bundle / "per_class.csv", index_col=0)
        pcl = ["\\begin{tabular}{lrrrr}", "\\toprule", "Class & precision & recall & F1 & test photos \\\\", "\\midrule"]
        pcl += [f"{n.replace('/', ': ')} & {r['precision']:.2f} & {r['recall']:.2f} & {r['f1-score']:.2f} & {int(r['support'])} \\\\" for n, r in pc.iterrows()]
        (out / "tab_perclass.tex").write_text("\n".join(pcl + ["\\bottomrule", "\\end{tabular}"]))
        f3 = lambda v: f"{v:.3f}"
        macros = {"MODELF": f3(t["macro_f1"]), "VZERO": fmt(v0, show_n=False), "CENTRAL": fmt(cen, show_n=False), "TEMP": f"{cal['temperature']:.2f}",
                  "ECEB": f3(t["ece_before"]), "ECEA": f3(t["ece_after"]), "NLLB": f3(t["nll_before"]), "NLLA": f3(t["nll_after"])}
        (out / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()))

    # --- figures
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    xs = np.array(alphas)
    for d, lab, c in ((avg, "FedAvg", BLUE), (prox, "FedProx ($\\mu$=0.1)", ORANGE)):
        pts = [(a, f1(d, a)) for a in alphas if f1(d, a)]
        if pts:
            ax.errorbar([a for a, _ in pts], [np.mean(v) for _, v in pts], yerr=[np.std(v) for _, v in pts], marker="o", ms=4, capsize=2, label=lab, color=c)
    if v0:
        ax.axhline(np.mean(v0), ls=":", color=GREY); ax.text(xs.min(), np.mean(v0) - 0.006, "v0 (central pool)", color=GREY, fontsize=7, va="top")
    if cen:
        ax.axhline(np.mean(cen), ls="--", color=GREY); ax.text(xs.min(), np.mean(cen) + 0.004, "centralised, all data", color=GREY, fontsize=7)
    ax.set_xscale("log"); ax.set_xlabel("Dirichlet $\\alpha$ (small = more skewed clients)"); ax.set_ylabel("test macro-F1"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_alpha.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for r in rows:
        k = key(r)
        if k[0] == "alpha" and k[2] == 0.5 and k[3] == "fedavg":
            ax.plot(r["val_rounds"], r["val_f1_curve"], color=BLUE, alpha=0.8, label="from v0" if k[1] == C.SEED else None)
        if k[0] == "init":
            ax.plot(r["val_rounds"], r["val_f1_curve"], color=ORANGE, alpha=0.8, label="no central stage" if k[1] == C.SEED else None)
    ax.set_xlabel("federated round"); ax.set_ylabel("validation macro-F1"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_curves.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for name, v, n in t2[:3]:
        if v:
            ax.scatter(n * 4 / 1e6, np.mean(v), s=40, color=BLUE); ax.annotate(name.split(" from")[0], (n * 4 / 1e6, np.mean(v)), textcoords="offset points", xytext=(5, -10), fontsize=7)
    ax.set_ylim(0.74, 0.87); ax.set_xlim(0.04, 20)
    if cen: ax.axhline(np.mean(cen), ls="--", color=GREY)
    ax.set_xscale("log"); ax.set_xlabel("MB uploaded per client per round (float32)"); ax.set_ylabel("test macro-F1")
    fig.tight_layout(); fig.savefig(out / "fig_scope.pdf"); plt.close(fig)

    (out / "tables.json").write_text(json.dumps({
        "v0_f1": v0, "central_f1": cen, "skew": [{k: v for k, v in r.items()} for r in t1],
        "scope": [{"name": n, "f1": v, "params": p} for n, v, p in t2],
        "init_A_f1": [r["test_f1"] for r in initA.values()],
        "bytes_per_run_full_mb": avg[(0.5, C.SEED)]["bytes_up_total"] / 1e6 if (0.5, C.SEED) in avg else None,
    }, indent=1, default=float))
    print("wrote", out)
