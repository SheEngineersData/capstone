import base64
import importlib
import io
import json
import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from potatofl import config as C
from potatofl import infer
from potatofl import model as M

CLASSES = [f"Leaf/c{i}" for i in range(5)] + [f"Tuber/c{i}" for i in range(5)]


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    """A throw-away bundle with a random-weight model, so the tests need no trained artifacts."""
    d = tmp_path_factory.mktemp("bundle")
    M.tf.keras.utils.set_random_seed(0)
    M.build_model(len(CLASSES), pretrained=False).save_weights(str(d / "model.weights.h5"))
    (d / "classes.json").write_text(json.dumps(CLASSES))
    (d / "calibration.json").write_text(json.dumps({"version": "test", "temperature": 1.5, "unsure": 0.5, "very_likely": 0.9, "test": {}}))
    rng = np.random.default_rng(0)
    n = 4000  # as large as the real reference (over 1,000 rows), where Evidently's default method is not a p-value test
    pd.DataFrame({"brightness": rng.normal(120, 20, n), "contrast": rng.normal(50, 8, n), "sharpness": rng.gamma(2, 300, n),
                  "saturation": rng.normal(0.4, 0.05, n), "green_share": rng.normal(1.0, 0.05, n),
                  "width": 1500, "height": 1500}).to_csv(d / "reference_stats.csv", index=False)
    return d


@pytest.fixture(scope="module")
def client(bundle_dir, tmp_path_factory):
    os.environ["ARTIFACT_DIR"] = str(bundle_dir)
    os.environ["DATA_DIR"] = str(tmp_path_factory.mktemp("data"))
    from service import app as appmod
    importlib.reload(appmod)
    with TestClient(appmod.app) as c:
        yield c


def jpeg(w=300, h=200, seed=0, brightness=128):
    a = np.random.default_rng(seed).integers(0, 255, (h, w, 3)).astype(np.float32) * (brightness / 128)
    buf = io.BytesIO()
    Image.fromarray(a.clip(0, 255).astype(np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["model_version"] == "test"


def test_predict_returns_calibrated_distribution(client):
    r = client.post("/predict", files={"file": ("a.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 200
    j = r.json()
    assert j["label"] in CLASSES and 0 < j["confidence"] <= 1 and j["tier"] in {"very likely", "possible", "unsure"}
    assert len(j["top"]) == 3 and j["top"][0]["label"] == j["label"]
    assert j["top"][0]["probability"] >= j["top"][1]["probability"]
    assert "heatmap_png_base64" not in j


def test_predict_with_heatmap_returns_png(client):
    j = client.post("/predict?heatmap=true", files={"file": ("a.jpg", jpeg(400, 300), "image/jpeg")}).json()
    assert Image.open(io.BytesIO(base64.b64decode(j["heatmap_png_base64"]))).format == "PNG"


def test_rejects_non_images_and_tiny_images(client):
    assert client.post("/predict", files={"file": ("a.txt", b"not an image", "text/plain")}).status_code == 422
    assert client.post("/predict", files={"file": ("a.jpg", jpeg(32, 32), "image/jpeg")}).status_code == 422


def test_model_endpoint(client):
    j = client.get("/model").json()
    assert j["classes"] == CLASSES and j["temperature"] == 1.5


def test_drift_waits_for_enough_requests_then_flags_dark_photos(client):
    pytest.importorskip("evidently")
    assert client.get("/drift").json()["status"] == "insufficient data"
    for i in range(40):  # photos far darker than the reference
        client.post("/predict", files={"file": ("a.jpg", jpeg(seed=i, brightness=20), "image/jpeg")})
    j = client.get("/drift").json()
    assert j["status"] == "drift" and "brightness" in j["drifted_columns"]
    assert len(client.get("/drift/report").text) > 10_000


def test_cam_matches_gradcam_from_autograd(bundle_dir):
    """The closed-form heat-map equals Grad-CAM computed with gradients."""
    clf = infer.Classifier(bundle_dir)
    tf = M.tf
    rgb = np.random.default_rng(1).integers(0, 255, (C.IMG_SIZE, C.IMG_SIZE, 3)).astype(np.uint8)
    res, cam = clf.predict(rgb, with_cam=True)
    c = CLASSES.index(res["label"])
    fm = tf.constant(clf.backbone.predict(rgb[None].astype("float32"), verbose=0))
    with tf.GradientTape() as tape:
        tape.watch(fm)
        logits = tf.reduce_mean(fm, (1, 2)) @ clf.W + clf.b
        score = logits[0, c]
    alpha = tf.reduce_mean(tape.gradient(score, fm), (1, 2))
    ref = tf.nn.relu(tf.reduce_sum(fm[0] * alpha[0], -1)).numpy()
    ref = ref / ref.max() if ref.max() > 0 else ref
    assert np.allclose(cam, ref, atol=1e-4)


def test_web_page_and_samples(client):
    page = client.get("/")
    assert page.status_code == 200 and "Potato check" in page.text
    assert client.get("/static/fonts/bricolage.woff2").status_code == 200
    lst = client.get("/samples.json").json()
    assert all(s["truth"].count("/") == 1 for s in lst)
    if lst:
        assert client.get(lst[0]["url"]).status_code == 200


def test_class_style_aliases(client):
    assert client.get("/health").json() == client.get("/healthz").json()
    assert client.get("/model-info").json() == client.get("/model").json()


def test_drift_flags_a_shift_in_a_single_column(bundle_dir, tmp_path):
    """A blurry window moves sharpness only. A 'half of the columns' rule would miss it; the monitor must not."""
    pytest.importorskip("evidently")
    from service.monitor import Monitor
    mon = Monitor(tmp_path / "log.jsonl", bundle_dir / "reference_stats.csv")
    ref = mon.reference
    cur = ref.sample(80, random_state=0).copy()
    cur["sharpness"] = cur["sharpness"] * 0.05  # blurrier photos, everything else as in the reference
    for _, r in cur.iterrows():
        mon.record(r.to_dict(), {"confidence": 0.8, "label": "Leaf/c0", "tier": "possible"})
    j = mon.drift()
    assert j["status"] == "drift" and j["drifted_columns"] == ["sharpness"]


def test_drift_stays_quiet_on_in_distribution_photos(bundle_dir, tmp_path):
    pytest.importorskip("evidently")
    from service.monitor import Monitor
    mon = Monitor(tmp_path / "log2.jsonl", bundle_dir / "reference_stats.csv")
    cur = mon.reference.sample(80, random_state=3)
    for _, r in cur.iterrows():
        mon.record(r.to_dict(), {"confidence": 0.8, "label": "Leaf/c0", "tier": "possible"})
    assert mon.drift()["status"] == "ok"
