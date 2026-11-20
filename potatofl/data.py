"""Data loading, caching and splitting. Reads the cleaned index written by the ingest stage."""
from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

from . import config as C

IMG_SIZE, SEED = C.IMG_SIZE, C.SEED


# --------------------------------------------------------------------- data
def load_joint_index() -> pd.DataFrame:
    """Leaf and tuber clean indexes as one table with a joint label such as 'Leaf/Fungi'."""
    parts = [pd.read_csv(C.INDEX / f"{n}_clean.csv") for n in ("leaf", "tuber")]
    df = pd.concat(parts, ignore_index=True)
    df["joint"] = df.organ + "/" + df.label
    df["path"] = df.path.map(lambda p: str(C.RAW / p))
    # Sort by dataset folder name, then label, then file name, so the order does not depend on where the data was unpacked.
    df["_k"] = df.organ.map({"Tuber": "0", "Leaf": "1"}) + "/" + df.label + "/" + df.path.map(lambda p: Path(p).name)
    return df.sort_values("_k").drop(columns="_k").reset_index(drop=True)


def _load_one(path: str, size: int) -> np.ndarray:
    with Image.open(path) as im:
        im.draft("RGB", (size * 2, size * 2))  # fast JPEG downscale on decode
        im = ImageOps.exif_transpose(im).convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def load_data(size: int = IMG_SIZE):
    """Returns (X uint8 [N,size,size,3], y int labels, class names, dataframe). Cached on disk."""
    df = load_joint_index()
    classes = sorted(df.joint.unique())
    y = df.joint.map({c: i for i, c in enumerate(classes)}).to_numpy()
    key = hashlib.md5(",".join(df.md5).encode()).hexdigest()[:8]
    cache = C.WORK / "cache" / f"joint_{size}_{key}"
    cache.mkdir(parents=True, exist_ok=True)
    if not (cache / "X.npy").exists():
        t = time.time()
        with ThreadPoolExecutor(os.cpu_count()) as ex:
            X = np.stack(list(ex.map(lambda p: _load_one(p, size), df.path)))
        np.save(cache / "X.npy", X)
        print(f"cached {len(X):,} images in {time.time() - t:.0f}s")
    return np.load(cache / "X.npy"), y, classes, df


def make_split(y, seed: int = SEED, val=0.15, test=0.15):
    """Stratified train/val/test indices. Same seed gives the same split everywhere."""
    idx = np.arange(len(y))
    tv, te = train_test_split(idx, test_size=test, stratify=y, random_state=seed)
    tr, va = train_test_split(tv, test_size=val / (1 - test), stratify=y[tv], random_state=seed)
    return tr, va, te


