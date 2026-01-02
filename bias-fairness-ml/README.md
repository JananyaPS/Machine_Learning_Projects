# Bias & Fairness Auditing + Mitigation in ML (Netflix‑Style)

An end‑to‑end reference implementation that demonstrates how to:
1. Train a baseline classifier,
2. Audit model behavior across sensitive attributes (e.g., sex, race),
3. Compute fairness metrics (Demographic Parity, Equalized Odds),
4. Apply a practical mitigation (group‑specific decision thresholds),
5. Produce reproducible reports and track the accuracy ↔ fairness trade‑off.

This repository is intended as a reproducible lab and template for practitioners who want a clear, auditable workflow for fairness evaluations in personalization and recommender pipelines.

---

## Contents

- `download_data.py` — download or prepare raw dataset
- `make_dataset.py` — preprocessing, train/test split, and synthetic-data mode
- `train.py` — train baseline model (configurable)
- `evaluate.py` — compute accuracy and fairness metrics; generate reports/plots
- `mitigate.py` — apply group‑specific thresholds and evaluate trade‑offs
- `config.py` / `config.yml` (example) — configuration for experiments
- `requirements.txt` — pinned dependencies
- `examples/` — sample commands and small sample data (if included)
- `outputs/` — model artifacts, metrics, reports (generated at runtime)

---

## Quickstart (reproduce the full pipeline)

Prerequisites
- Python 3.9–3.11 recommended
- 4+ GB free RAM (less for synthetic/sample mode)

1) Create and activate a virtual environment
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) (Optional) Download or create dataset
- If a public dataset is used:
```bash
python download_data.py --out data/raw/
```
- To run quickly with a small synthetic dataset (fast smoke test):
```bash
python make_dataset.py --mode synthetic --out data/raw/sample.csv
```

4) Prepare / preprocess dataset
```bash
python make_dataset.py \
  --in data/raw/ \
  --out data/processed/ \
  --sensitive sex,race \
  --label clicked \
  --seed 42
```

5) Train baseline model
```bash
python train.py \
  --config config.yml \
  --data data/processed/train.csv \
  --out outputs/model.pkl \
  --seed 42
```

6) Evaluate and compute fairness metrics
```bash
python evaluate.py \
  --model outputs/model.pkl \
  --data data/processed/test.csv \
  --out outputs/eval_metrics.json \
  --report outputs/report.html
```

7) Apply group threshold mitigation and re-evaluate
```bash
python mitigate.py \
  --preds outputs/predictions.csv \
  --sensitive sex \
  --target_metric demographic_parity \
  --parity_tolerance 0.02 \
  --out outputs/mitigated_preds.csv

python evaluate.py \
  --preds outputs/mitigated_preds.csv \
  --data data/processed/test.csv \
  --out outputs/mitigated_metrics.json \
  --report outputs/mitigated_report.html
```

8) Verify expected output files:
- `outputs/eval_metrics.json` and `outputs/mitigated_metrics.json` (JSON with accuracy and fairness metrics)
- `outputs/report.html` (interactive/visual report)
- `outputs/model.pkl` (trained model artifact)

---

## Data

Describe the dataset used for experiments. If you are using a public dataset, include the exact name and link. If private, provide a synthetic example and the schema below.

Recommended schema (example)
- id: unique identifier (string/int)
- feature_1, feature_2, ...: numeric or categorical features
- label: binary outcome (0/1) — define what "positive" means (e.g., selected, clicked)
- sensitive attributes (one or more): e.g., `sex` (M/F), `race` (A/B/C)

Data notes
- Always document which column is used as the sensitive attribute and how missing/unknown values are handled.
- Provide counts per group in README or report to contextualize metric stability (small groups → high variance).

If dataset is proprietary or large
- Provide instructions to obtain it.
- Provide a small, synthetic sample in `examples/` for quick experiments.

---

## Configuration

A sample `config.yml` (or `config.py`) to control experiments:

```yaml
python_version: "3.10"
model:
  type: logistic_regression
  random_seed: 42
  params:
    C: 1.0
    penalty: l2

data:
  label_col: clicked
  sensitive_cols: [sex, race]
  test_size: 0.2
  random_state: 42

fairness:
  mitigation:
    method: group_thresholds
  parity_tolerance: 0.02
  metrics: [demographic_parity, equalized_odds]

outputs:
  dir: outputs/
```

All scripts accept `--config` or relevant CLI flags; prefer config files for reproducibility.

---

## Metrics & Definitions

- Accuracy: standard classification accuracy on the test set.
- Demographic Parity (selection rate parity): difference in positive outcome rate between groups.
- Equalized Odds: difference in true positive and false positive rates across groups.
- Report contains per-group metrics and an accuracy–fairness curve when applying different thresholds.

Recommendation: report point estimates plus 95% confidence intervals (bootstrap) for fairness metrics to convey uncertainty.

---

## Mitigation approach

This repo demonstrates a simple, practical mitigation:
- group‑specific decision thresholds: choose different score thresholds per group to bring selection rates closer together while tracking accuracy loss.

Notes
- This is a post‑processing technique (no re‑training).
- Works when group membership is available at decision time.
- Provide trade‑off curves so stakeholders can choose operating points.

Alternatives to consider (for future work):
- reweighing / preprocessing
- in‑processing methods (fairness‑aware loss)
- constrained optimization (e.g., equalized odds constraints)

---

## Reproducibility & best practices

- Pin Python and package versions in `requirements.txt`.
- Use deterministic train/test splits and set seeds (`--seed` or `random_state`).
- Record experiments (hyperparameters, metrics, Git commit hash). Use MLflow or a simple CSV log.
- Keep sensitive columns and raw data out of public outputs/reports unless permitted.

Example minimal experiment log (CSV)
```
run_id,git_sha,config,accuracy,demographic_parity,equalized_odds,timestamp
```

---

## Testing & CI

Add simple smoke tests to ensure the pipeline runs on a tiny dataset:

- `tests/test_smoke.py` should:
  - run `make_dataset.py --mode synthetic`
  - run `train.py` with small config and verify `outputs/model.pkl` exists
  - run `evaluate.py` and confirm `outputs/eval_metrics.json` contains expected keys

Example minimal GitHub Actions workflow (`.github/workflows/ci.yml`):
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - name: Install deps
        run: pip install -r requirements.txt
      - name: Run smoke tests
        run: pytest -q
```

---

## Privacy, Ethics & Responsible Use

- Sensitive attributes may be protected personal data. Use only with appropriate consent and legal basis.
- Document data provenance, consent, and retention policies.
- Avoid exposing PII in public reports. Aggregate results by group and report counts.
- Include an Ethics section in stakeholder-facing reports describing limitations (coverage, sample sizes, model scope).

---

## Troubleshooting

- If `pip install` fails: confirm Python version and upgrade pip: `python -m pip install --upgrade pip`.
- If training crashes on memory: reduce dataset size or use `--mode synthetic` for a small run.
- If fairness metrics are unstable: check group sample sizes; use bootstrapping to estimate variance.

---

## Example outputs

`outputs/eval_metrics.json` (example)
```json
{
  "accuracy": 0.82,
  "demographic_parity": {
    "overall": 0.12,
    "by_group": {"M": 0.43, "F": 0.31},
    "difference": 0.12
  },
  "equalized_odds": {
    "tpr_diff": 0.07,
    "fpr_diff": 0.03
  }
}
```

`outputs/report.html` — interactive HTML showing:
- per-group selection rates
- ROC and calibration plots per group
- accuracy–fairness tradeoff plot (threshold sweep)

---

## Contributing

Contributions welcome. Suggested workflow:
1. Fork the repo.
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Add tests and update `requirements.txt` if needed.
4. Open a PR and describe the change, include example outputs.

Please follow Conventional Commits style for commit messages (e.g., `feat:`, `fix:`, `docs:`).

Suggested commit for README changes:
```
docs(readme): add Quickstart, config example, data schema and reproducibility notes
```

---

## License & Attribution

Add a LICENSE file (e.g., MIT) and include license assignment here.

If this project references public datasets or prior work, cite them in this section.

---

## Contact & Acknowledgements

Author: JananyaPS  
Contact: (add email or GitHub handle)  

Acknowledgements: (list libraries, inspiration, or collaborators)

---

If you’d like, I can:
- generate a ready-to-commit `config.yml` and a pinned `requirements.txt` suitable for this codebase,
- create the smoke test file and a minimal GitHub Actions CI workflow,
- or produce a template `outputs/report.html` with example figures.

Which of these should I prepare next?
