# Explainable & Trustworthy Recommendation System  
**Hybrid Recommender with Feature Attribution, Counterfactual Explanations, and Trust Metrics**

## Overview
This project implements an **end-to-end, explainable recommendation system** inspired by large-scale streaming platforms.  
Beyond ranking accuracy, the system emphasizes **transparency, user trust, and robustness** by answering:

- *Why was this title recommended?*
- *What user behavior most influenced the recommendation?*
- *How would recommendations change if a key interaction were removed?*
- *Are recommendations biased toward popularity or lacking diversity?*

The project is designed to demonstrate **production-ready ML thinking**, not just model training.

---

## Key Capabilities

### 1. Hybrid Recommendation Model
- Uses **LightFM** with implicit feedback
- Combines:
  - User interaction history
  - User side features (demographics)
  - Item side features (genres)
- Optimized using **WARP loss** for ranking quality

### 2. Feature-Attribution Explanations (“Why this title?”)
- Computes **feature-level contribution scores**
- Explains recommendations using interpretable signals such as:
  - Genre affinity
- Example:
  > *Recommended because of strong alignment with Action (+0.42) and Thriller (+0.31)*

### 3. Counterfactual Explanations (“What if?”)
- Identifies the **most influential past interaction**
- Generates post-hoc explanations:
  - *“Because you liked X…”*
  - *“If you hadn’t liked X, the top recommendation would be Y”*
- Helps assess **causal sensitivity** and model robustness

### 4. Trust & Robustness Metrics
In addition to accuracy, the system computes:
- **Catalog coverage** – Are we exposing the full catalog?
- **Novelty** – Are recommendations overly popular?
- **Genre diversity** – Are recommendation lists varied?
- **Stability (proxy)** – Do recommendations change drastically under small perturbations?

---

## Dataset
**MovieLens 1M** (GroupLens Research)  
- ~1M ratings
- 6,000 users
- 4,000 movies
- Rich metadata (genres, demographics)

Dataset source:  
https://grouplens.org/datasets/movielens/1m/

---

## Repository Structure

