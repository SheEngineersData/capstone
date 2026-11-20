# Reviewer shortcuts. Python 3.12 with requirements.txt installed; TF_USE_LEGACY_KERAS is set by the code.
PY ?= python
.PHONY: test smoke review serve full

test:      ## 1 min: unit and API tests
	$(PY) -m pytest -q tests

smoke:     ## about 5 min, no download: every stage on the 192-photo mini dataset (data/mini), written under work/smoke
	POTATOFL_SMOKE=1 POTATOFL_RAW=data/mini $(PY) -m potatofl.pipeline all

review:    ## about 25 min, needs data/raw: ingest, shipped v0, 10 real federated rounds, calibration
	$(PY) -m potatofl.pipeline ingest
	$(PY) -m potatofl.pipeline review

serve:     ## the model service in Docker on http://localhost:8000
	docker compose up --build -d

full:      ## about 7 h per seed: central models and the whole experiment grid (resumable)
	$(PY) -m potatofl.pipeline ingest
	for s in 42 43 44; do $(PY) -m potatofl.pipeline central --pool 0.4 --seed $$s; $(PY) -m potatofl.pipeline central --pool 0 --seed $$s; done
	$(PY) -m potatofl.pipeline federate pilot
	$(PY) -m potatofl.pipeline federate core
	$(PY) -m potatofl.pipeline bundle
	$(PY) -m potatofl.pipeline report
