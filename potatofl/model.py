"""Model definition, local training helpers and evaluation (TensorFlow, Keras 2 via tf_keras)."""
from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .config import IMG_SIZE, SEED


# name -> (constructor, needs our 0-255 -> [-1, 1] rescale). The others scale their own input.
ARCHS = {
    "mobilenetv2": (keras.applications.MobileNetV2, True),
    "mobilenetv3small": (keras.applications.MobileNetV3Small, False),
    "mobilenetv3large": (keras.applications.MobileNetV3Large, False),
    "efficientnetb0": (keras.applications.EfficientNetB0, False),
    "resnet50": (keras.applications.ResNet50, False),
}


def build_model(num_classes: int, pretrained: bool = True, size: int = IMG_SIZE, alpha: float = 1.0, arch: str = "mobilenetv2"):
    """Backbone + dropout + softmax head. Raw 0-255 pixels go in, so exported models need no separate preprocessing."""
    ctor, rescale = ARCHS[arch]
    kw = {"alpha": alpha} if arch.startswith("mobilenet") else {}
    base = ctor(input_shape=(size, size, 3), include_top=False, weights="imagenet" if pretrained else None, pooling="avg", **kw)
    inp = keras.Input((size, size, 3), name="image")
    x = keras.layers.Rescaling(1 / 127.5, offset=-1.0, name="rescale")(inp) if rescale else inp
    x = base(x)
    x = keras.layers.Dropout(0.2, name="drop")(x)
    out = keras.layers.Dense(num_classes, activation="softmax", name="head")(x)
    return keras.Model(inp, out, name=f"potato_{arch}")


def get_base(model):
    return next(l for l in model.layers if isinstance(l, keras.Model))


def set_trainable(model, mode: str):
    """'head': only the classifier. 'all': everything except BatchNorm (kept in inference mode)."""
    base = get_base(model)
    for layer in model.layers:
        layer.trainable = False
    model.get_layer("head").trainable = True
    if mode == "all":
        base.trainable = True
        for layer in base.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False


def augment(x, y):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)  # leaves and tubers have no canonical orientation
    x = tf.image.random_brightness(x, 25.0)
    x = tf.image.random_contrast(x, 0.8, 1.2)
    pad = 10
    x = tf.pad(x, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
    x = tf.image.random_crop(x, [IMG_SIZE, IMG_SIZE, 3])
    return tf.clip_by_value(x, 0.0, 255.0), y


def make_ds(X, y, idx, batch=32, train=False, seed=SEED):
    ds = tf.data.Dataset.from_tensor_slices((X[idx].astype("float32"), y[idx].astype("int32")))
    if train:
        ds = ds.shuffle(len(idx), seed=seed, reshuffle_each_iteration=True).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def class_weights(y, idx, cap: float = 10.0) -> dict:
    """Inverse-frequency weights from the labels this learner holds. Capped, mean sample weight 1."""
    n = int(y.max()) + 1
    counts = np.bincount(y[idx], minlength=n).astype(float)
    w = np.where(counts > 0, counts.sum() / (np.count_nonzero(counts) * np.maximum(counts, 1)), 0.0)
    w = np.minimum(w, cap)
    w = w / w[y[idx]].mean()
    return {i: float(v) for i, v in enumerate(w)}


def compile_model(model, lr: float, loss="sparse_categorical_crossentropy"):
    """`lr` may be a number or a Keras learning-rate schedule."""
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss, metrics=["accuracy"])


# --------------------------------------------------------------- evaluation
def evaluate(model, X, y, idx, batch=128):
    probs = model.predict(X[idx].astype("float32"), batch_size=batch, verbose=0)
    pred = probs.argmax(1)
    t = y[idx]
    return {
        "acc": float((pred == t).mean()),
        "macro_f1": float(f1_score(t, pred, average="macro", labels=np.unique(y))),
        "probs": probs, "preds": pred, "targets": t,
    }


def per_class_report(ev, classes):
    rep = classification_report(ev["targets"], ev["preds"], labels=range(len(classes)), target_names=classes, output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T.loc[classes, ["precision", "recall", "f1-score", "support"]]


def confusion(ev, classes):
    return pd.DataFrame(confusion_matrix(ev["targets"], ev["preds"], labels=range(len(classes))), index=classes, columns=classes)


def head_and_size(model):
    """Parameter counts and uncompressed float32 size, for the communication-cost analysis."""
    head = sum(int(np.prod(w.shape)) for w in model.get_layer("head").weights)
    total = model.count_params()
    return {"head_params": head, "total_params": total, "head_kb_fp32": head * 4 / 1024, "total_mb_fp32": total * 4 / 1e6}


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=float))
