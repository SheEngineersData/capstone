"""Input drift monitoring with Evidently AI.

Every prediction request appends one line of input descriptors (brightness, contrast, sharpness, colour,
image size) plus the model's confidence to a JSONL log. `drift()` compares the most recent requests with the
reference statistics of the training images using Evidently's data-drift report (per-column two-sample
Kolmogorov-Smirnov test, p < 0.05 / 7 per column). Drift is reported when any column drifts. A farmer's phone photos that are darker, blurrier or smaller than the training
photos show up here before accuracy can be measured, because the service never sees the true label.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pandas as pd

FEATURES = ["brightness", "contrast", "sharpness", "saturation", "green_share", "width", "height"]
MIN_REQUESTS = 30   # fewer than this and a drift test says little
ALPHA = 0.05 / len(FEATURES)  # one K-S test per column, Bonferroni-corrected so a clean window raises a false alarm about 5% of the time at most
WINDOW = 500        # compare the latest requests only


class Monitor:
    def __init__(self, log_path: str | Path, reference_csv: str | Path):
        self.log = Path(log_path)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.reference = pd.read_csv(reference_csv)[FEATURES]
        self._lock = threading.Lock()

    def record(self, stats: dict, result: dict):
        row = {"ts": time.time(), **{k: stats[k] for k in FEATURES}, "confidence": result["confidence"], "label": result["label"], "tier": result["tier"]}
        with self._lock, open(self.log, "a") as f:
            f.write(json.dumps(row) + "\n")

    def current(self) -> pd.DataFrame:
        if not self.log.exists():
            return pd.DataFrame(columns=["ts", *FEATURES, "confidence", "label", "tier"])
        with self._lock:
            lines = self.log.read_text().splitlines()[-WINDOW:]
        return pd.DataFrame([json.loads(l) for l in lines])

    def _report(self, cur: pd.DataFrame):
        from evidently import Dataset, DataDefinition, Report
        from evidently.presets import DataDriftPreset
        # All seven columns are declared numeric so every one gets the two-sample K-S test. Left to itself Evidently
        # types width and height as categorical (few distinct values) and uses a Jensen-Shannon distance that
        # false-alarms on small windows, and it switches numeric columns to a Wasserstein distance (not a p-value)
        # when the reference has more than 1,000 rows. Pinning K-S keeps one decision rule: drift when p < 0.05.
        definition = DataDefinition(numerical_columns=FEATURES)
        as_data = lambda df: Dataset.from_pandas(df[FEATURES], data_definition=definition)
        return Report([DataDriftPreset(num_method="ks", num_threshold=ALPHA, drift_share=0.1)]).run(as_data(cur), as_data(self.reference))

    def drift(self) -> dict:
        cur = self.current()
        out = {"requests_logged": int(len(cur)), "min_requests": MIN_REQUESTS}
        if len(cur) < MIN_REQUESTS:
            return {**out, "status": "insufficient data", "drifted_columns": []}
        res = self._report(cur).dict()
        cols = {}
        for m in res["metrics"]:
            name = m.get("metric_name", "")
            if name.startswith("ValueDrift"):
                col = m["config"]["column"]
                thr = m["config"].get("threshold", 0.05)
                is_distance = "wasserstein" in name.lower() or "distance" in name.lower()  # distances drift above the threshold, p-values below it
                cols[col] = {"score": float(m["value"]), "drifted": bool(m["value"] > thr if is_distance else m["value"] < thr)}
        drifted = sorted(c for c, v in cols.items() if v["drifted"])
        share = len(drifted) / max(len(cols), 1)
        # Any one column is enough: a darker, blurrier or smaller photo moves some columns and leaves others (colour ratios,
        # for instance) untouched, so "half of the columns" would miss a large real shift.
        return {**out, "status": "drift" if drifted else "ok", "share_drifted": share, "drifted_columns": drifted, "columns": cols,
                "mean_confidence": float(cur.confidence.mean()), "share_unsure": float((cur.tier == "unsure").mean())}

    def html(self) -> str:
        cur = self.current()
        if len(cur) < MIN_REQUESTS:
            return f"<p>Only {len(cur)} requests logged; at least {MIN_REQUESTS} are needed for a drift report.</p>"
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.html"
            self._report(cur).save_html(str(p))
            return p.read_text()
