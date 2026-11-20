"""Inference: the exported model, calibrated confidence and the Grad-CAM highlight.

The model is a MobileNetV2 feature extractor, global average pooling, dropout and a linear softmax
head. For that structure the Grad-CAM heat-map of class c is exactly the feature map weighted by the
head weights of class c (the gradient of the logit with respect to each feature-map cell is W[:, c] / HW),
so one forward pass gives the prediction and the heat-map and no gradient tape is needed.

An artifact bundle is a folder with: model.weights.h5, classes.json, calibration.json, reference_stats.csv.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from . import config as C
from . import model as M

keras, tf = M.keras, M.tf


def image_stats(rgb128: np.ndarray, width: int, height: int) -> dict:
    """Cheap input descriptors used for drift monitoring. rgb128 is the resized uint8 image the model sees."""
    g = rgb128.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    mx, mn = rgb128.max(2).astype(np.float32), rgb128.min(2).astype(np.float32)
    return {
        "brightness": float(g.mean()), "contrast": float(g.std()), "sharpness": float(lap.var()),
        "saturation": float(((mx - mn) / np.maximum(mx, 1)).mean()), "green_share": float(rgb128[..., 1].mean() / max(rgb128.mean(), 1e-6)),
        "width": int(width), "height": int(height),
    }


def load_image(raw: bytes, size: int = C.IMG_SIZE):
    """Decode, fix the EXIF rotation, convert to RGB. Returns (uint8 [size,size,3], original (w, h), PIL image for the overlay)."""
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        im = ImageOps.exif_transpose(im).convert("RGB")
    return np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.uint8), im.size, im


class Classifier:
    def __init__(self, bundle: str | Path):
        bundle = Path(bundle)
        self.classes = json.loads((bundle / "classes.json").read_text())
        cal = json.loads((bundle / "calibration.json").read_text())
        self.temperature, self.unsure, self.very_likely = cal["temperature"], cal["unsure"], cal["very_likely"]
        self.version = cal.get("version", "v1")
        full = M.build_model(len(self.classes), pretrained=False)
        full.load_weights(str(bundle / "model.weights.h5"))
        base = M.get_base(full)
        fmap = keras.Model(base.input, base.get_layer("out_relu").output)
        inp = keras.Input((C.IMG_SIZE, C.IMG_SIZE, 3))
        self.backbone = keras.Model(inp, fmap(full.get_layer("rescale")(inp)))
        self.W, self.b = (w.astype(np.float32) for w in full.get_layer("head").get_weights())

    def logits_and_maps(self, x: np.ndarray):
        fm = self.backbone.predict(x.astype("float32"), verbose=0)  # [N, h, w, 1280]
        return fm.mean((1, 2)) @ self.W + self.b, fm

    def calibrated(self, logits: np.ndarray) -> np.ndarray:
        z = logits / self.temperature
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def tier(self, conf: float) -> str:
        return "very likely" if conf >= self.very_likely else ("possible" if conf >= self.unsure else "unsure")

    def predict(self, rgb128: np.ndarray, top: int = 3, with_cam: bool = False):
        logits, fm = self.logits_and_maps(rgb128[None])
        p = self.calibrated(logits)[0]
        order = np.argsort(-p)[:top]
        out = {"label": self.classes[order[0]], "confidence": float(p[order[0]]), "tier": self.tier(float(p[order[0]])),
               "top": [{"label": self.classes[i], "probability": float(p[i])} for i in order], "model_version": self.version}
        cam = None
        if with_cam:
            cam = np.maximum(fm[0] @ self.W[:, order[0]], 0)
            cam = cam / cam.max() if cam.max() > 0 else cam
        return out, cam


def overlay_png(im, cam: np.ndarray, alpha: float = 0.45) -> bytes:
    """Red-yellow heat-map of `cam` (h x w in [0,1]) blended over the original photo, as PNG bytes."""
    w, h = im.size
    heat = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0
    colour = np.stack([np.ones_like(heat), np.clip(2 * (1 - heat), 0, 1), np.zeros_like(heat)], -1) * 255
    base = np.asarray(im, dtype=np.float32)
    a = (alpha * heat)[..., None]
    out = Image.fromarray((base * (1 - a) + colour * a).astype(np.uint8))
    if max(out.size) > 640:
        out.thumbnail((640, 640))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()
