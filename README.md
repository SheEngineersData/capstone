# Federated potato disease detection: pipeline and model service

DSA 8401 Applied Machine Learning, capstone. Gracious Chepkemoi Ngetich, MSc Data Science and Analytics, Strathmore University.

A system that classifies a photo of a potato **leaf** (7 classes) or **tuber** (9 classes) into one of 16 disease or healthy classes, trained with **federated learning** so that each simulated farm keeps its photos and shares only model updates. A FastAPI service returns the class, a calibrated confidence, a plain-language confidence tier and a Grad-CAM heat-map, and logs input statistics to an Evidently AI drift monitor. Everything runs from scripts and a Docker container.

| Result (test set, 1,016 photos, seed 42) | macro-F1 |
|---|---|
| v0: centralised model on a 40% central pool | 0.788 |
| **Federated full-network model (shipped), 10 clients, Dirichlet alpha 0.5** | **0.805** |
| Centralised on all training data (upper bound) | 0.858 |

FedAvg gains 1.7 points over v0 on seed 42 but loses 0.3 on seed 43, so averaged over seeds it matches v0 (about 0.80) and stays five points below centralised training. FedProx at alpha 0.5 is above v0 and above FedAvg on both seeds (0.811 on two seeds). The finding that matters for deployment is that updating only the classifier head (80 KB per round, against 8.8 MB for the whole network) costs no measurable accuracy.

The grid (label skew, FedProx, no central stage, update scope) was run in full on seed 42 and on every cell of seed 43 except FedProx at alpha 0.1; the report states the number of seeds behind every figure. The discussion is in [docs/report.pdf](docs/report.pdf).

## Architecture

```
 A. Training (offline, scripted: python -m potatofl.pipeline ...)

 raw photos --ingest--> cleaned index --central--> v0 (40% pool) --federate--> global model
 (2 datasets)  dedupe,    work/index       |           |   10 simulated clients,          |
               drop junk                  |           |   Dirichlet label skew,          |
                                 centralised reference|   FedAvg / FedProx, full network |
                                          |                                              v
                                          +------------------------------------> bundle: calibrate (temperature),
                                                                                  reference stats, artifacts/

 B. Serving (Docker)

 phone / curl --POST /predict--> FastAPI --> MobileNetV2 + head --> class, calibrated confidence, tier
                                    |                           \-> Grad-CAM heat-map (PNG)
                                    \--> request log --> Evidently drift report (/drift, /drift/report)
```

## Repository layout

| Path | Content |
|---|---|
| `potatofl/` | pipeline package: `ingest`, `data`, `model`, `central`, `federated`, `experiments`, `bundle`, `report`, `infer`, `pipeline` (CLI) |
| `service/` | FastAPI app (`app.py`) and Evidently drift monitor (`monitor.py`) |
| `artifacts/` | the shipped model bundle: weights, class list, calibration, reference input statistics, test metrics |
| `results/` | raw run records (`fl_full/*.jsonl`), per-seed summaries, tables and figures used in the report |
| `tests/` | pytest suite (partitioning, compression, privacy clipping, calibration, API, Grad-CAM exactness, drift) |
| `docs/` | final report (`report.tex`, `report.pdf`) |
| `samples/` | one held-out photo per class for trying the service |
| `Dockerfile`, `docker-compose.yml` | the service container |
| `Makefile`, `.github/workflows/ci.yml` | `make test / smoke / review / serve / full`; CI runs the tests and the smoke run on every push |
| `data/mini/` | 192 photos (12 per class, 256 px) for the no-download smoke run |

## Checking this submission: four levels, shortest first

You do not need to rerun the multi-day experiment grid to check the work. Each level proves more and costs more; the recorded runs of the grid are in `results/` and every table and figure in the report is rebuilt from them.

| Level | Command | Time | What it shows |
|---|---|---|---|
| 0 | `docker compose up --build -d`, then the `curl` calls below, and `make test` | about 10 min (mostly the image build) | The deployed service works with the shipped model; 21 tests pass |
| 1 | `make smoke` | about 10 min, no download | Every pipeline stage (ingest, central, federate, bundle, report) runs end to end on the 192-photo mini dataset in `data/mini` |
| 2 | `make review` | about 25 min, needs the real data | Ingest of the real data, then 10 real federated rounds from the shipped v0 (`artifacts/v0`), then calibration; test macro-F1 should come out near 0.80 (the shipped model uses 30 rounds). The result is written to `work/review/` and does not touch `artifacts/` |
| 3 | `make full` | about 7 h per seed on 8 CPU cores (about 20 h for 3 seeds) | Central models and the whole experiment grid; resumable, skips finished runs |

Rebuild the report's tables and figures from the shipped run records in seconds:
`python -m potatofl.pipeline report --models results/models --results results --out /tmp/report_check`

## Run the service (no training needed)

Needs Docker only.

```bash
docker compose up --build -d
curl http://localhost:8000/health        # also /healthz
curl -F file=@samples/Leaf_Healthy.jpg "http://localhost:8000/predict?heatmap=true" | head -c 600
```

Open <http://localhost:8000/> for the web page: take or choose a photo, or click one of the held-out test photos, and see the answer, the confidence gauge and the heat-map (it also works on a phone; `/?try=Leaf_Healthy` opens straight onto a sample). The page uses self-hosted Bricolage Grotesque and Atkinson Hyperlegible fonts (SIL Open Font License). Other pages: <http://localhost:8000/docs> for the API schema, `GET /model` (also `/model-info`) for the model card numbers and `GET /drift` for the drift summary (it needs at least 30 logged requests). `docker compose down` stops it; the request log is kept in a Docker volume.

Response of `POST /predict` for `samples/Leaf_Healthy.jpg` (`heatmap=true` adds `heatmap_png_base64`):

```json
{
 "label": "Leaf/Healthy",
 "confidence": 0.919,
 "tier": "very likely",
 "top": [
  {
   "label": "Leaf/Healthy",
   "probability": 0.919
  },
  {
   "label": "Leaf/Virus",
   "probability": 0.057
  },
  {
   "label": "Leaf/Phytopthora",
   "probability": 0.016
  }
 ],
 "model_version": "v1.0",
 "advice": "High confidence. Still confirm with an extension officer before treating or discarding seed."
}
```

The 16 sample photos are held-out test photos, one per class. The shipped model gets 12 of the 16 right and is "very likely" on 11 of them; the four misses are all leaf photos (Bacteria read as Fungi, Fungi as Phytophthora, Pest as Fungi, Virus as Bacteria) and none is above "possible", which is what the tiers are for. Test-set numbers for the whole set are in `GET /model`.

Errors: 422 for files that are not images or are under 64 px, 413 above 10 MB.

## Reproduce the pipeline from scratch

Python 3.12, CPU is enough (the numbers below are for an 8-core laptop).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

**Data.** Download the two public datasets and unpack them so that the class folders sit directly under `data/raw/leaf` and `data/raw/tuber` (override with `POTATOFL_RAW`):

* Leaf: Shabrina et al. (2023), *Potato leaf disease dataset in uncontrolled environment*, Mendeley Data. Folders `Bacteria, Fungi, Healthy, Nematode, Pest, Phytopthora, Virus` go under `data/raw/leaf/`.
* Tuber: Islam and Afrin (2025), *PotatoCare*, Mendeley Data, DOI 10.17632/7vm7xskfg4.2. Folders `Black Scurf, Blackleg, Blackspot Bruising, Brown Rot, Common Scab, Dry Rot, Healthy Potatoes, Miscellaneous, Pink Rot, Soft Rot` go under `data/raw/tuber/`.

**Stages** (each reads what the previous one wrote under `work/`):

```bash
python -m potatofl.pipeline ingest                  # 1 min: clean index and cleaning_report.json
python -m potatofl.pipeline central --pool 0.4      # 25 min: model v0
python -m potatofl.pipeline central --pool 0        # 25 min: centralised reference
python -m potatofl.pipeline federate pilot          # 30 min: picks the local learning rate on validation
python -m potatofl.pipeline federate core           # about 7 h per seed (9 runs of 40 to 60 min, three of them much shorter); resumable, skips finished runs
python -m potatofl.pipeline bundle                  # calibrate and write artifacts/ (refuses a model with test macro-F1 < 0.75 or calibration error > 0.10)
python -m potatofl.pipeline report                  # tables and figures
```

`--seed 43` and `--seed 44` on `central` give the other seeds (the grid needs v0 for all three). To check every stage works in a few minutes before committing hours, set `POTATOFL_SMOKE=1`; smoke runs write under `work/smoke/` and never touch `artifacts/`.

Clean-up decisions are in `work/index/cleaning_report.json` and `removed_files.csv`: unreadable files, the catch-all tuber folder `Miscellaneous`, byte-identical copies and near-duplicates (64-bit difference hash, at most 4 differing bits) are removed, and duplicates with conflicting labels are dropped in both copies. Leaf 3,076 to 2,979 photos, tuber 3,905 to 3,794.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Design choices in one paragraph each

* **Why federated, and why the full network.** Farms and cooperatives are unlikely to pool photos on one server. The system trains the whole MobileNetV2 (2.2 M parameters, 8.8 MB per update in float32) on the clients so that the farms' photos can adapt the features. The experiments then compare this with updating only the last blocks or only the classifier head; in these runs the cheaper scopes were as accurate, so the full-network update buys nothing measurable at this scale.
* **Why v0 and a central pool.** Starting from a model trained on a public pool (40% of the training photos) is what makes federation work with only 10 clients: without that stage the full network collapses (macro-F1 0.29 versus 0.81 in the seed-42 run).
* **Why calibration and tiers.** A softmax score is not a probability. Temperature scaling on the validation split lowers the expected calibration error from 0.050 to 0.029 on the test set; the tiers (very likely at 0.90 and above, possible from 0.50, otherwise unsure) are then honest: 98% of "very likely" answers are right on the test set, 34% of "unsure" ones.
* **Why Grad-CAM in closed form.** For global average pooling followed by a linear head the Grad-CAM map of class c is exactly the last feature map weighted by that class's head weights, so the service needs one forward pass; a test checks it against autograd Grad-CAM.
* **Why Evidently and these features.** The service never sees the true label, so accuracy cannot be monitored. It can see the photo: brightness, contrast, sharpness, saturation, green share and size are compared with the training photos using Evidently's two-sample K-S test per column; drift is reported when any column has p < 0.05/7.
* **Ruled out.** Vision Transformers and larger CNNs (too heavy for phones on slow links), LLM or RAG components (nothing to add to image classification), SHAP (Grad-CAM answers "where in the photo" directly), and sample-level DP-SGD (see the report).

## Limitations

* Both datasets are public and not Kenyan; clients are simulated by a Dirichlet label split, so the numbers measure the effect of label skew only, not camera, lighting or geography shift.
* Test photos come from the same two datasets as the training photos; field accuracy will be lower. The drift monitor exists for that reason.
* Leaf photos are 1500 px and tuber photos 224 px, and the two sets differ visually, so organ is partly recognisable from image statistics alone.
* Some classes are weak: leaf Fungi, Pest and Phytopthora (F1 0.62 to 0.67) and the rare tuber classes Blackleg and Common Scab (7 test photos each, F1 about 0.55 to 0.59). Per-class scores are in `artifacts/per_class.csv`.
* The model gives advice, not a diagnosis; the tier text tells the user to confirm with an extension officer.

## AI assistance

See [AI_USE.md](AI_USE.md).
