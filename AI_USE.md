# AI use disclosure

**Tool:** Claude Code (Anthropic), used as a coding and writing assistant throughout the project.

## What the assistant did

* Wrote first drafts of most code in `potatofl/`, `service/` and `tests/`, working from my design (federated full-network training of a MobileNetV2, Dirichlet client split, FedAvg and FedProx, calibrated confidence tiers, Grad-CAM, Evidently drift monitoring).
* Ran experiments I specified, restarted them after laptop reboots, and assembled tables from the run records.
* Drafted the README and report text from the results; I edited them.
* Helped debug (for example a find-and-replace that broke an attribute name, and a Grad-CAM test that was decided by floating-point noise).

## What I did

* Chose the problem, the datasets, the techniques and what to leave out, and the experimental design (grid of label skew, algorithm, initialisation and update scope; three seeds).
* Decided the scope and what to claim, and read every result before it went into the report.
* Checked the numbers against the raw run records in `results/fl_full/*.jsonl`.

## What was not AI-generated

The datasets (Shabrina et al. 2023; Islam and Afrin 2025) and the pretrained ImageNet MobileNetV2 weights from Keras Applications are third-party. All reported numbers come from code in this repository, not from the assistant's own estimates.
