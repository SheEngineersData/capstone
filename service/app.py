"""FastAPI service: send a potato leaf or tuber photo, get back the disease, a calibrated confidence and the highlight.

    POST /predict?heatmap=true   multipart form field "file" (JPEG/PNG)   -> JSON
    GET  /healthz, /health       liveness
    GET  /model, /model-info     version, classes, test metrics of the shipped model
    GET  /drift                  Evidently drift summary of recent requests vs the training photos
    GET  /drift/report           the full Evidently HTML report
    GET  /                       the web page: upload or take a photo, see the answer, confidence and heat-map
    GET  /samples.json           held-out test photos (with their true class) offered on the page
"""
from __future__ import annotations

import base64
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from potatofl import infer

from .monitor import Monitor

BUNDLE = Path(os.environ.get("ARTIFACT_DIR", Path(__file__).resolve().parent.parent / "artifacts"))
DATA = Path(os.environ.get("DATA_DIR", "/tmp/potatofl"))
MAX_BYTES = 10 * 1024 * 1024
state: dict = {}

ADVICE = {
    "very likely": "High confidence. Still confirm with an extension officer before treating or discarding seed.",
    "possible": "Moderate confidence. Take a second photo in daylight, closer to the affected area.",
    "unsure": "Low confidence. The model cannot tell. Retake the photo or ask an extension officer.",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["clf"] = infer.Classifier(BUNDLE)
    state["mon"] = Monitor(DATA / "requests.jsonl", BUNDLE / "reference_stats.csv")
    state["cal"] = json.loads((BUNDLE / "calibration.json").read_text())
    yield


app = FastAPI(title="Potato disease classifier", version="1.0", lifespan=lifespan)


@app.get("/healthz")
@app.get("/health")
def healthz():
    return {"status": "ok", "model_version": state["clf"].version}


@app.get("/model")
@app.get("/model-info")
def model_info():
    c = state["clf"]
    return {"version": c.version, "classes": c.classes, "temperature": c.temperature, "tiers": {"unsure_below": c.unsure, "very_likely_from": c.very_likely},
            "test": state["cal"]["test"]}


@app.post("/predict")
async def predict(file: UploadFile = File(...), heatmap: bool = False, top: int = 3):
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "image larger than 10 MB")
    try:
        rgb, (w, h), im = infer.load_image(raw)
    except Exception:
        raise HTTPException(422, "could not read the file as an image (send a JPEG or PNG photo)")
    if min(w, h) < 64:
        raise HTTPException(422, f"image too small ({w}x{h}); at least 64 pixels on the short side")
    clf: infer.Classifier = state["clf"]
    result, cam = clf.predict(rgb, top=max(1, min(top, 16)), with_cam=heatmap)
    result["advice"] = ADVICE[result["tier"]]
    if heatmap:
        result["heatmap_png_base64"] = base64.b64encode(infer.overlay_png(im, cam)).decode()
    state["mon"].record(infer.image_stats(rgb, w, h), result)
    return result


@app.get("/drift")
def drift():
    return state["mon"].drift()


@app.get("/drift/report", response_class=HTMLResponse)
def drift_report():
    return state["mon"].html()


STATIC = Path(__file__).resolve().parent / "static"
SAMPLES = Path(os.environ.get("SAMPLES_DIR", Path(__file__).resolve().parent.parent / "samples"))
app.mount("/static", StaticFiles(directory=STATIC), name="static")
if SAMPLES.exists():
    app.mount("/samples", StaticFiles(directory=SAMPLES), name="samples")


@app.get("/samples.json")
def samples():
    """Held-out test photos, one per class, with their true class (from the file name, e.g. Tuber_Black_Scurf.jpg)."""
    out = []
    for f in sorted(SAMPLES.glob("*.jpg")) if SAMPLES.exists() else []:
        organ, _, rest = f.stem.partition("_")
        out.append({"url": f"/samples/{f.name}", "truth": f"{organ}/{rest.replace('_', ' ')}"})
    return out


@app.get("/", response_class=FileResponse)
def index():
    return FileResponse(STATIC / "index.html")
