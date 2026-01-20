# End-to-End ML Pipeline: User Engagement Prediction (MovieLens)

Production-style machine learning system that predicts the probability of user engagement from historical interaction data using offline training and evaluation.

---

## Dataset
MovieLens Latest Small from GroupLens.

Problem it solves

User engagement signals (clicks, views, interactions) power:
- Personalization systems
- Ranking and recommendation pipelines
- Retention and growth analytics

Rather than heuristic rules, this project models engagement **probabilistically** using a structured ML pipeline, enabling consistent scoring, evaluation, and interpretability suitable for downstream production systems.

## Approach
- Time-aware split: last interaction per user is held out for testing
- Feature engineering:
  - User historical behavior (count, mean rating, engagement rate)
  - Movie historical popularity/quality (count, mean rating, engagement rate)
  - Temporal features (hour, day-of-week)
  - Genre multi-hot features
- Model: Logistic Regression pipeline (scaler + classifier)
- Explainability:
  - Permutation feature importance
  - SHAP summary plot
---

Key results (offline)

Metric        Validation        Test
ROC-AUC      ~0.75–0.85*        ~0.75–0.85*
Accuracy     ~0.70–0.80*        ~0.70–0.80*
RMSE         ~0.35–0.45*        ~0.35–0.45*

*Synthetic / sample dataset; fully reproducible via configuration.

---

Tech stack

Python · Pandas · NumPy · Scikit-learn · SHAP · Pytest

---

Quickstart (2 minutes)

```bash
# 1. Setup environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download / load data
python src/download_data.py

# 3. Build dataset
python src/make_dataset.py

# 4. Train model
python src/train.py

# 5. Evaluate performance
python src/evaluate.py

# 6. Explain predictions
python src/explain.py



python src/explain.py
