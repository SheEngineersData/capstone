"""Centralised fine-tuning of the ImageNet MobileNetV2 on the public leaf + tuber data.

pool > 0  : model v0, trained on a central pool (default 40% of training images). It is the starting
            point of federated learning; the other 60% stay with the simulated farms and v0 never sees them.
pool == 0 : the centralised reference, trained on every training image. The upper bound federation is judged against.

    python -m potatofl.pipeline central --pool 0.4 --seed 42
"""
from __future__ import annotations

import time

import numpy as np

from . import config as C
from . import data as D
from . import model as M

keras, tf = M.keras, M.tf


def out_dir(seed: int, pool: float):
    return C.MODELS / ("v0" if pool else "central") / f"seed{seed}"


def run(seed: int = C.SEED, pool: float = C.POOL):
    head_epochs, ft_epochs = (1, 1) if C.SMOKE else (5, 25)
    batch = 32
    out = out_dir(seed, pool)
    out.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(seed)

    X, y, classes, df = D.load_data()
    if C.SMOKE:  # a tiny stratified subset
        keep = np.concatenate([np.where(y == c)[0][:12] for c in range(len(classes))])
        X, y = X[keep], y[keep]
    tr, va, te = D.make_split(y, seed=seed)
    clients = np.array([], dtype=int)
    if pool:
        tr, clients = D.train_test_split(tr, train_size=pool, stratify=y[tr], random_state=seed)
    print(f"classes={len(classes)} images={len(y)} train={len(tr)} val={len(va)} test={len(te)} seed={seed} pool={pool}")

    model = M.build_model(len(classes), pretrained=True)
    cw = M.class_weights(y, tr)
    train_ds = M.make_ds(X, y, tr, batch, train=True, seed=seed)
    history, best = [], {"f1": -1.0}

    class ValF1(keras.callbacks.Callback):
        """Scores macro-F1 on validation each epoch and keeps the best weights."""
        def __init__(self, phase):
            super().__init__()
            self.phase = phase

        def on_epoch_begin(self, epoch, logs=None):
            self.t0 = time.time()

        def on_epoch_end(self, epoch, logs=None):
            ev = M.evaluate(self.model, X, y, va)
            row = {"phase": self.phase, "epoch": epoch + 1, "train_loss": logs["loss"], "train_acc": logs["accuracy"],
                   "val_acc": ev["acc"], "val_macro_f1": ev["macro_f1"], "secs": time.time() - self.t0}
            history.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            if ev["macro_f1"] > best["f1"]:
                best["f1"] = ev["macro_f1"]
                self.model.save_weights(str(out / "best.weights.h5"))

    t_all = time.time()
    M.set_trainable(model, "head")
    M.compile_model(model, 1e-3)
    model.fit(train_ds, epochs=head_epochs, class_weight=cw, callbacks=[ValF1("head")], verbose=0)
    M.set_trainable(model, "all")
    steps = int(np.ceil(len(tr) / batch)) * ft_epochs
    M.compile_model(model, keras.optimizers.schedules.CosineDecay(1e-4, steps, alpha=0.05))
    model.fit(train_ds, epochs=ft_epochs, class_weight=cw, callbacks=[ValF1("finetune")], verbose=0)
    train_secs = time.time() - t_all

    model.load_weights(str(out / "best.weights.h5"))
    ev = M.evaluate(model, X, y, te)
    rep = M.per_class_report(ev, classes)
    rep["n_train"] = np.bincount(y[tr], minlength=len(classes))
    rep.round(3).to_csv(out / "per_class.csv")
    M.confusion(ev, classes).to_csv(out / "confusion.csv")
    np.savez(out / "split.npz", train=tr, val=va, test=te, clients=clients)
    M.save_json({
        "seed": seed, "smoke": C.SMOKE, "classes": classes, "n_images": int(len(y)),
        "split": {"train": len(tr), "val": len(va), "test": len(te)},
        "test": {"accuracy": ev["acc"], "macro_f1": ev["macro_f1"]},
        "best_val_macro_f1": best["f1"], "train_seconds": train_secs, "model": M.head_and_size(model),
        "pool_fraction": pool,
        "config": {"img": C.IMG_SIZE, "batch": batch, "head_epochs": head_epochs, "ft_epochs": ft_epochs, "class_weighted": True, "pretrained": "imagenet"},
        "history": history,
    }, out / "summary.json")
    print(f"TEST acc {ev['acc']:.3f} macro-F1 {ev['macro_f1']:.3f}  ({train_secs / 60:.1f} min)")
