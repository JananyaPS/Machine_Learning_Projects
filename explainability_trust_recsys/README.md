# Explainable & Trustworthy Recommendation System

Hybrid Recommender with Feature Attribution, Counterfactual Explanations, and Trust Metrics

---

Table of contents
- Project overview
- Key capabilities
- Repository structure (visual)
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

## Repository structure (visual)
Top-level layout (folders and primary files) — shown here in the same style as the project documentation:

```
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
```

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
- requirements.txt includes LightFM, pandas, numpy, scipy, scikit-learn, matplotlib/plotly for plots, and any CLI utilities used by scripts.

Optional: add a Dockerfile or Binder configuration if you want a reproducible container for demos.

## Data (download & preprocessing)
This project uses MovieLens 1M by default (GroupLens). Download from:
https://grouplens.org/datasets/movielens/1m/

Scripts:
- src/download_data.py
  - Downloads and unpacks movielens-1m (or validates a supplied local copy).
  - Example:
    ```bash
    python src/download_data.py --dataset movielens-1m --output data/
    ```
- src/make_dataset.py
  - Converts explicit ratings into implicit interactions (e.g., binarize rating >= 4).
  - Builds user and item side feature matrices (one-hot or multi-hot for genres; user demographics).
  - Saves processed matrices (sparse formats) into data/processed/.

Notes:
- data/ is auto-created and should not contain large raw files committed to Git.
- The pipeline expects implicit feedback by default; adjust thresholds in src/config.py.

## Usage (scripts & examples)
All scripts support --help; e.g.:
```bash
python src/train.py --help
```

Common workflows:
- Train:
  ```bash
  python src/train.py --config src/config.py --save-dir models/
  ```
  Outputs model files in models/, plus logs and optional validation metrics.

- Recommend for a single user:
  ```bash
  python src/recommend.py --model models/best_model.npz --user-id 123 --topk 10 --output reports/recs_user123.json
  ```

- Explain (feature attribution + counterfactuals):
  ```bash
  python src/explain.py --model models/best_model.npz --user-id 123 --topk 10 --output reports/explanations_user123.json
  ```

- Compute trust metrics:
  ```bash
  python src/trust_metrics.py --model models/best_model.npz --data data/processed/ --output reports/trust_metrics.json
  ```

- End-to-end:
  ```bash
  python src/run_all.py --config src/config.py
  ```

## Methods (brief)
See corresponding modules in src/ for implementation details.

Hybrid model
- LightFM (implicit) with WARP loss for top-k ranking.
- Inputs: user×item interaction matrix and side-feature matrices for users/items.
- Hyperparameters (embedding dim, epochs, learning rate) live in src/config.py.

Feature attribution
- Compute feature-level contributions by leveraging model weights and side-feature embeddings.
- Explanations are signed contributions (positive increases score).
- Implemented to be model-aware for efficiency (src/explain.py).

Counterfactuals
- Single-interaction ablation: remove one past positive interaction, re-score items and measure the rank change.
- The interaction whose removal most decreases rank for a recommended item is reported as the influential one.
- Brute-force but simple and interpretable; alternatives (approximate influence functions) are discussed in code comments.

Trust metrics
- Catalog coverage: fraction of items that appear across users' top-K lists.
- Novelty: average item popularity measure (e.g., inverse log-popularity).
- Genre diversity: intra-list diversity and aggregated diversity across users.
- Stability: proxy measured by change in recommendations after small perturbations (e.g., random drop of interactions); Jaccard or rank-based measures used.
- Implementations available in src/trust_metrics.py with configurable aggregation.

## Evaluation & reproducibility
- Ranking metrics: Recall@K, Precision@K, NDCG@K on holdout interactions.
- Trust metrics computed on the same evaluation splits.
- Use src/config.py to control seeds, data split strategy, and hyperparameters.
- Save checkpoints and logs to models/ and reports/ with timestamps.



