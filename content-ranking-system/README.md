# Content Recommendation Ranking System (Offline + Online)

An end-to-end Learning-to-Rank system that mimics Netflix’s core ranking problem —
ranking candidate content for a user using offline training and online inference.
## Why Ranking Matters

Modern recommender systems operate in two stages:
1. Retrieval — fast candidate generation
2. Ranking — precise ordering of items shown to users

This project focuses on the ranking stage, which directly impacts engagement,
watch-time, and user satisfaction.
## Tech Stack

- Python
- LightGBM (LambdaRank)
- pandas / numpy
- FastAPI
- PyYAML
- pytest + GitHub Actions
## System Architecture

Raw Interaction Logs
→ Negative Sampling
→ Feature Engineering (User / Item / Context)
→ Learning-to-Rank Training (LambdaMART)
→ Model Registry
→ Online Ranking API (FastAPI)
## Data

Synthetic interaction data is generated to simulate real streaming behavior:
- Users, content items, and sessions
- Implicit feedback (click, play, watch time)
- Graded relevance labels (0–3)

Labels:
0 = no engagement  
1 = click  
2 = short play  
3 = long play
## Modeling Approach

- Problem formulation: Learning-to-Rank
- Query group: (user_id, session_id)
- Model: LightGBM LambdaRank
- Objective: NDCG optimization
- Training includes:
  - Negative sampling
  - Time-based train/val/test splits
  - Early stopping
## Evaluation Metrics

- NDCG@K — evaluates ranking quality with graded relevance
- MAP@K — evaluates precision of relevant items

Metrics are computed per (user, session) group.
## How to Run

### Generate data
python -m src.ranking.data.generate_interactions --config configs/ranker.yaml

### Train ranking model
python -m src.ranking.models.train_ltr --config configs/ranker.yaml

### Start ranking API
uvicorn src.ranking.api.app:app --reload
## Online Ranking API

POST /rank

Input:
- user_id
- candidate item_ids
- request context (device, time)

Output:
- ranked list of items with scores
## Repository Structure

src/ranking/
- data/        # interaction processing & negative sampling
- features/    # user, item, context features
- models/      # training, evaluation, registry
- inference/   # online ranking logic
- api/         # FastAPI service
##  

Built an end-to-end content ranking system using Learning-to-Rank (LightGBM LambdaRank),
including offline feature engineering, negative sampling, NDCG/MAP evaluation,
and an online FastAPI ranking service.
