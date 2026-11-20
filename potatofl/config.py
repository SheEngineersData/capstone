"""Paths and constants. Everything can be moved with two environment variables.

POTATOFL_RAW   folder holding the two downloaded datasets (default: ./data/raw)
POTATOFL_WORK  folder for everything the pipeline writes (default: ./work)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = Path(os.environ.get("POTATOFL_RAW", ROOT / "data" / "raw"))
WORK = Path(os.environ.get("POTATOFL_WORK", ROOT / "work"))
ARTIFACTS = ROOT / "artifacts"

# folder name of each dataset inside RAW, and the organ it holds
DATASETS = {"Leaf": "leaf", "Tuber": "tuber"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DROP_CLASSES = {"Leaf": [], "Tuber": ["Miscellaneous"]}  # catch-all folder of unrelated conditions
NEAR_DUP_MAX_BITS = 4  # of 64 dHash bits: pairs inspected at this distance were the same photo

IMG_SIZE = 128
SEED = 42
SEEDS = (42, 43, 44)
POOL = 0.4  # share of training images held centrally to train v0; the rest belong to the simulated farms

SMOKE = os.environ.get("POTATOFL_SMOKE") == "1"  # tiny end-to-end run that checks every stage in minutes
_RUN = WORK / "smoke" if SMOKE else WORK
INDEX = _RUN / "index"          # cleaned image index
MODELS = _RUN / "models"        # v0 / central checkpoints
RESULTS = _RUN / "results"      # one JSON line per finished federated run, tables and figures
BUNDLE = _RUN / "artifacts" if SMOKE else ARTIFACTS  # smoke runs never overwrite the shipped artifacts
