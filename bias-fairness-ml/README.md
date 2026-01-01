# Bias & Fairness Auditing + Mitigation in ML (Netflix-Style)

This project demonstrates an end-to-end workflow to:
1) Train a baseline classifier,
2) Audit bias across sensitive attributes (e.g., sex, race),
3) Quantify fairness metrics (Demographic Parity, Equalized Odds),
4) Apply a mitigation strategy (group-specific decision thresholds),
5) Track the accuracy–fairness trade-off with reproducible reports.

## Why this matters (streaming context)
In recommender systems and personalization pipelines, decisions can unintentionally produce skewed outcomes
(e.g., systematically different selection rates across demographic groups). This repo focuses on:
- measurable fairness metrics,
- transparent reporting,
- a practical mitigation approach.

---

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
