# End-to-End ML Pipeline: User Engagement Prediction (MovieLens)

## Problem
Predict whether a user will show high engagement with a movie, defined as rating >= 4.0.

## Dataset
MovieLens Latest Small from GroupLens.

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

## Results
See `reports/metrics.txt` and plots under `reports/`.

## Reproducibility
Run:
```bash
python src/download_data.py
python src/make_dataset.py
python src/train.py
python src/evaluate.py
python src/explain.py
