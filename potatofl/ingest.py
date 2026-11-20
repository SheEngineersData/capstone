"""Stage 1 of the pipeline: ingest the raw image folders and clean them.

Reads every image under RAW/leaf and RAW/tuber (the folder name is the label), then removes
unreadable files, the catch-all tuber class, byte-identical and near-identical photos
(64-bit difference hash), and any group of look-alikes whose labels disagree.
Writes work/index/{leaf,tuber}_clean.csv, removed_files.csv and cleaning_report.json.

    python -m potatofl.pipeline ingest
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from . import config as C

POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def build_index(root: Path, organ: str) -> pd.DataFrame:
    rows = [{"organ": organ, "label": p.parent.name, "path": str(p.relative_to(C.RAW))}
            for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in C.IMG_EXT]
    return pd.DataFrame(rows)


def probe(path: str) -> dict:
    full = C.RAW / path
    try:
        with Image.open(full) as im:
            im.load()
            w, h, mode = im.width, im.height, im.mode
        return {"width": w, "height": h, "mode": mode, "error": None, "md5": hashlib.md5(full.read_bytes()).hexdigest()}
    except Exception as e:  # unreadable or truncated file
        return {"width": None, "height": None, "mode": None, "error": f"{type(e).__name__}", "md5": None}


def dhash(path: str) -> np.ndarray:
    """64-bit difference hash: does each pixel of a 9x8 grey thumbnail exceed its left neighbour?"""
    with Image.open(C.RAW / path) as im:
        im.draft("L", (64, 64))
        g = np.asarray(ImageOps.exif_transpose(im).convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


def near_pairs(H: np.ndarray, max_bits: int):
    out = []
    for i in range(len(H) - 1):
        d = POP[H[i] ^ H[i + 1:]].sum(1)
        out += [(i, i + 1 + j, int(d[j])) for j in np.where(d <= max_bits)[0]]
    return out


def components(n: int, pairs):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, j, _ in pairs:
        parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def clean(df: pd.DataFrame, organ: str):
    df = df.sort_values("path").reset_index(drop=True).copy()
    df["reason"] = ""
    alive = lambda: df.reason == ""
    rep = {"raw": len(df)}

    def step(name, mask, reason):
        df.loc[mask, "reason"] = reason
        rep[name] = int(alive().sum())

    step("after_unreadable", df.error.notna(), "unreadable")
    step("after_excluded", alive() & df.label.isin(C.DROP_CLASSES[organ]), "excluded class")
    rep["excluded_class_files"] = rep["after_unreadable"] - rep["after_excluded"]

    a = df[alive()]
    conflict = a.groupby("md5").label.transform("nunique") > 1
    step("after_exact_cross", df.index.isin(a.index[conflict]), "exact duplicate, conflicting labels")
    a = df[alive()]
    step("after_exact_same", df.index.isin(a.index[a.duplicated("md5", keep="first")]), "exact duplicate, same class")

    a = df[alive()]
    with ThreadPoolExecutor() as ex:
        H = np.stack(list(ex.map(dhash, a.path)))
    groups = components(len(a), near_pairs(H, C.NEAR_DUP_MAX_BITS))
    labels = a.label.to_numpy(dtype=object)
    drop_cross, drop_same = [], []
    for g in groups:
        if len(set(labels[g])) > 1:
            drop_cross += g
        else:
            drop_same += g[1:]
    step("after_near_cross", df.index.isin(a.index[drop_cross]), "near duplicate, conflicting labels")
    step("final", df.index.isin(a.index[drop_same]), "near duplicate, same class")
    c = df[alive()].label.value_counts()
    r = df.label.value_counts()
    rep.update({"n_classes": int(c.size), "imbalance_raw": round(r.max() / r.min(), 1), "imbalance_clean": round(c.max() / c.min(), 1)})
    return df[alive()].drop(columns="reason"), df[~alive()], rep


def run():
    out = C.INDEX
    out.mkdir(parents=True, exist_ok=True)
    report, removed = {}, []
    for organ, folder in C.DATASETS.items():
        root = C.RAW / folder
        if not root.exists():
            raise SystemExit(f"missing {root}. See README, section 'Data', for the two datasets to download.")
        df = build_index(root, organ)
        with ThreadPoolExecutor() as ex:
            df = pd.concat([df, pd.DataFrame(list(ex.map(probe, df.path)))], axis=1)
        kept, gone, report[organ] = clean(df, organ)
        kept.to_csv(out / f"{folder}_clean.csv", index=False)
        removed.append(gone[["organ", "label", "path", "md5", "reason"]])
        print(f"{organ}: {report[organ]['raw']} raw -> {report[organ]['final']} kept, {report[organ]['n_classes']} classes")
    pd.concat(removed).to_csv(out / "removed_files.csv", index=False)
    (out / "cleaning_report.json").write_text(json.dumps(report, indent=1))
