# Explainable & Trustworthy Recommendation System

Hybrid Recommender with Feature Attribution, Counterfactual Explanations, and Trust Metrics

---

Table of contents
- Project overview
- Key capabilities
- Repository structure
- Quickstart (one-liners)
- Installation / environment
- Data (download & preprocessing)
- Usage (scripts & examples)
- Methods (brief)
  - Hybrid model
  - Feature attribution
  - Counterfactuals
  - Trust metrics
- Evaluation & reproducibility
- Example outputs
- Development notes
- Citation & license
- Contact

## Project overview
This project implements an end-to-end, explainable recommendation system inspired by large-scale streaming platforms. It focuses not only on ranking accuracy, but also on transparency, robustness and user trust by providing:

- Feature-level attribution ("Why was this title recommended?")
- Counterfactual explanations ("If you hadn't liked X, you would see Y")
- Trust and robustness metrics (coverage, novelty, diversity, stability)

The implementation is a production-minded demonstration (scripts, config, saved models, and reports), using MovieLens 1M as the example dataset.

## Key capabilities
- Hybrid recommender (LightFM) combining implicit user interactions with user/item side-features.
- Feature-attribution explanations showing feature contributions (e.g., genre affinity).
- Counterfactual explanations that identify influential past interactions and show how recommendations change when they are removed.
- Trust and robustness metrics: catalog coverage, novelty (popularity bias), genre diversity, and a stability proxy.

## Repository structure
Top-level layout (folders and primary files)

explainability-trust-recs/
├── README.md

├── requirements.txt

├── data/                  # auto-created; do not commit raw data

├── models/                # saved models (checkpoints, serialized artifacts)

├── reports/               # explanations & trust metrics outputs (CSV/JSON/plots)

└── src/
    ├── config.py          # default configuration / hyperparameters
    
    ├── download_data.py   # script to download MovieLens or other datasets
    
    ├── make_dataset.py    # process raw data → implicit interactions + feature matrices
    
    ├── train.py           # train LightFM hybrid model and save artifacts to models/
    
    ├── recommend.py       # generate recommendations from a saved model
    
    ├── explain.py         # compute feature attributions and counterfactuals
    
    ├── trust_metrics.py   # compute coverage / novelty / diversity / stability
    
    └── run_all.py         # convenience script to run the full pipeline end-to-end

## Quickstart (one-liners)
1. Create environment and install dependencies:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the full pipeline (download → preprocess → train → explain → metrics):
```bash
python src/run_all.py
```

3. Or run steps manually:
```bash
python src/download_data.py --dataset movielens-1m --output data/
python src/make_dataset.py --input data/movielens-1m --output data/processed/
python src/train.py --config src/config.py --output models/
python src/recommend.py --model models/best_model.npz --user-id 123 --topk 10 --output reports/recommendations_user123.json
python src/explain.py --model models/best_model.npz --user-id 123 --topk 10 --output reports/explanations_user123.json
python src/trust_metrics.py --model models/best_model.npz --data data/processed/ --output reports/trust_metrics.json
```

## Installation / environment
- Python: 3.8+ recommended
- Create a virtual environment (venv or conda)
- Install:
```bash
pip install -r requirements.txt
```
- requirements.txt includes LightFM, pandas, numpy, scipy, scikit-learn, matplotlib/plotly (for plots), and any utilities (argparse/typer).

Optional: a Dockerfile or binder configuration can be added to reproduce the environment in CI or demo.

## Data (download & preprocessing)
This project uses MovieLens 1M by default (GroupLens). The dataset must be downloaded from:
https://grouplens.org/datasets/movielens/1m/

Scripts:
- src/download_data.py
  - Downloads and unpacks movielens-1m (or verifies a local copy)
  - Example:
    ```bash
    python src/download_data.py --dataset movielens-1m --output data/
    ```
- src/make_dataset.py
  - Converts explicit ratings to implicit interactions (e.g., threshold rating >= 4)
  - Builds user and item side feature matrices (one-hot or multi-hot for genres; simple demographics for users)
  - Saves processed matrices/scipy sparse files into data/processed/

Notes:
- The pipeline assumes implicit feedback for training (binary interactions).
- All raw data should remain out of the repository (data/ is listed as auto-created). Commit only small derived artifacts if necessary.

## Usage (scripts & examples)
All scripts support --help for available options, e.g.:
```bash
python src/train.py --help
```

Common workflows:
- Train a model:
  ```bash
  python src/train.py --config src/config.py --save-dir models/
  ```
  Outputs: model weight file(s) in models/, a training log, and optionally validation metrics.

- Generate recommendations for a user:
  ```bash
  python src/recommend.py --model models/best_model.npz --user-id 123 --topk 10 --output reports/recs_user123.json
  ```

- Produce explanations (feature attribution + counterfactuals):
  ```bash
  python src/explain.py --model models/best_model.npz --user-id 123 --topk 10 --output reports/explanations_user123.json
  ```

- Compute trust/robustness metrics for your test set:
  ```bash
  python src/trust_metrics.py --model models/best_model.npz --data data/processed/ --output reports/trust_metrics.json
  ```

- End-to-end run:
  ```bash
  python src/run_all.py --config src/config.py
  ```

## Methods (brief)
This section gives concise descriptions of the methods implemented. For detailed implementation, see corresponding modules in src/.

Hybrid model
- LightFM (implicit) is used with WARP loss optimized for top-k ranking.
- Model inputs: interaction matrix (users × items), user features, item features.
- Embedding dimensionality, epochs, and other hyperparameters are in src/config.py.

Feature attribution
- We compute feature-level contribution scores to each recommendation using the model's linear combination of embeddings with side-feature effects.
- Attribution is reported per recommendation as a list of contributing features (e.g., genres) with signed scores (positive => increases score).
- The implementation is in src/explain.py; the approach is model-aware (uses LightFM weights) rather than an expensive model-agnostic explainer, for speed.

Counterfactual explanations
- For each top recommendation, we compute the most influential past interaction(s) by:
  - Removing a single past positive interaction for the user, re-scoring the top candidates, and measuring rank change (single-interaction ablation).
  - We report the interaction whose removal causes the largest negative impact on the recommended item's rank.
- This brute-force single-interaction ablation is simple and interpretable; depending on scale, an approximate influence or importance heuristic can be used.
- Implemented in src/explain.py.

Trust metrics
- Catalog coverage: fraction of catalog that appears in top-K lists across users.
- Novelty: average popularity (lower = more novel); often measured as average inverse log-popularity of recommended items.
- Genre diversity: intra-list diversity — the proportion of unique genres within recommendation lists and diversity aggregated across users.
- Stability (proxy): compare recommendation lists before and after small perturbations (e.g., random drop of 1–2 interactions) and compute average rank or Jaccard change.
- Implemented in src/trust_metrics.py; metric definitions and aggregation options are configurable.

## Evaluation & reproducibility
- Ranking metrics: Recall@K, Precision@K, NDCG@K are computed on holdout interactions.
- Trust metrics (see above) are computed on the same evaluation splits.
- Reproducibility:
  - Use src/config.py to set seeds, hyperparameters, and data splits.
  - Keep the model checkpoint saved in models/.
  - Save experiment logs and metrics in reports/ with timestamped filenames.

Example evaluation flow:
1. Split users/interactions (stratified or leave-one-out) — see make_dataset.py.
2. Train on train split, validate on validation split, evaluate on test split.
3. Record both ranking and trust metrics.

## Example outputs
- Recommendation (JSON):
```json
{
  "user_id": 123,
  "topk":

