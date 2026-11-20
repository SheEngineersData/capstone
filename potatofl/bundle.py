"""Calibrate the final federated model and write the artifact bundle the service loads.

  1. temperature scaling fitted on the validation split (one number that stretches or shrinks the logits)
  2. test-set check: accuracy, macro-F1, expected calibration error before and after, accuracy per confidence tier
  3. reference input statistics of the training images, the baseline for drift monitoring
  4. a few held-out sample photos for the README demo

    python -m potatofl.pipeline bundle --weights <final model weights> --seed 42
"""
from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

from . import config as C
from . import data as D
from . import infer
from . import model as M

UNSURE, VERY_LIKELY = 0.5, 0.9  # confidence tiers shown to the farmer
MIN_F1 = 0.75   # quality gate: the test macro-F1 a model must reach to be written to artifacts/
MAX_ECE = 0.10  # quality gate: after temperature scaling, the calibration error must stay below this


def quality_gate(test: dict) -> list[str]:
    """Reasons the model must not be deployed (empty list = passes). Smoke runs use a handful of photos, so the gate does not apply to them."""
    if C.SMOKE:
        return []
    why = []
    if test["macro_f1"] < MIN_F1:
        why.append(f"test macro-F1 {test['macro_f1']:.3f} is below {MIN_F1}")
    if test["ece_after"] > MAX_ECE:
        why.append(f"calibration error {test['ece_after']:.3f} is above {MAX_ECE}")
    return why


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits, y) -> float:
    nll = lambda t: -np.log(softmax(logits / t)[np.arange(len(y)), y] + 1e-12).mean()
    return float(minimize_scalar(nll, bounds=(0.2, 10.0), method="bounded").x)


def ece(p, y, bins=10) -> float:
    conf, pred = p.max(1), p.argmax(1)
    edges, tot = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(tot)


def reference_stats(df: pd.DataFrame) -> pd.DataFrame:
    def one(path):
        rgb, (w, h), _ = infer.load_image(Path(path).read_bytes())
        return infer.image_stats(rgb, w, h)
    with ThreadPoolExecutor() as ex:
        return pd.DataFrame(list(ex.map(one, df.path)))


def run(weights: str, seed: int = C.SEED, out: Path = C.BUNDLE, version: str = "v1.0"):
    out = Path(out)
    X, y, classes, df = D.load_data()
    sp = np.load(C.MODELS / "v0" / f"seed{seed}" / "split.npz")
    model = M.build_model(len(classes), pretrained=False)
    model.load_weights(weights)
    head = model.get_layer("head")
    pre = M.keras.Model(model.input, model.get_layer("drop").output)  # features before the head; head is linear
    W, b = head.get_weights()

    def logits(idx):
        return pre.predict(X[idx].astype("float32"), batch_size=128, verbose=0) @ W + b

    lv, lt = logits(sp["val"]), logits(sp["test"])
    yv, yt = y[sp["val"]], y[sp["test"]]
    T = fit_temperature(lv, yv)
    pt_raw, pt_cal = softmax(lt), softmax(lt / T)
    conf, pred = pt_cal.max(1), pt_cal.argmax(1)
    tiers = {"very likely": conf >= VERY_LIKELY, "possible": (conf >= UNSURE) & (conf < VERY_LIKELY), "unsure": conf < UNSURE}
    cal = {
        "version": version, "temperature": T, "unsure": UNSURE, "very_likely": VERY_LIKELY,
        "test": {
            "accuracy": float((pred == yt).mean()), "macro_f1": float(f1_score(yt, pred, average="macro", labels=np.arange(len(classes)), zero_division=0)),
            "ece_before": ece(pt_raw, yt), "ece_after": ece(pt_cal, yt), "nll_before": float(-np.log(pt_raw[np.arange(len(yt)), yt] + 1e-12).mean()),
            "nll_after": float(-np.log(pt_cal[np.arange(len(yt)), yt] + 1e-12).mean()),
            "tiers": {k: {"share": float(m.mean()), "accuracy": float((pred[m] == yt[m]).mean()) if m.any() else None} for k, m in tiers.items()},
        },
    }
    problems = quality_gate(cal["test"])
    if problems:  # nothing has been written yet, so the previous artifacts stay as they were
        print("QUALITY GATE FAILED, not writing artifacts: " + "; ".join(problems))
        raise SystemExit(1)
    print(f"Quality gate passed (macro-F1 >= {MIN_F1}, calibration error <= {MAX_ECE})" if not C.SMOKE else "Quality gate skipped in smoke mode")
    out.mkdir(parents=True, exist_ok=True)
    ev = {"targets": yt, "preds": pred}
    M.per_class_report(ev, classes).round(3).to_csv(out / "per_class.csv")
    M.confusion(ev, classes).to_csv(out / "confusion.csv")
    (out / "calibration.json").write_text(json.dumps(cal, indent=1))
    (out / "classes.json").write_text(json.dumps(classes))
    shutil.copy(weights, out / "model.weights.h5")

    ref_idx = np.concatenate([sp["train"], sp["clients"]]) if len(sp["clients"]) else sp["train"]
    reference_stats(df.iloc[ref_idx]).to_csv(out / "reference_stats.csv", index=False)

    rng = np.random.default_rng(seed)
    samp = out.parent / "samples"
    samp.mkdir(exist_ok=True)
    for f in samp.glob("*.jpg"):
        f.unlink()
    for c in range(len(classes)):  # one held-out photo per class
        cand = sp["test"][yt == c]
        if len(cand):
            i = int(rng.choice(cand))
            with Image.open(df.path.iloc[i]) as im:
                im = im.convert("RGB")
                im.thumbnail((512, 512))
                im.save(samp / (classes[c].replace("/", "_").replace(" ", "_") + ".jpg"), quality=88)
    print(json.dumps(cal["test"], indent=1))
