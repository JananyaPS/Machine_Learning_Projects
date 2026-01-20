# 🚀 Machine Learning Projects

A curated portfolio of **production-oriented machine learning systems** demonstrating skills of:
- scalable data processing  
- ranking & personalization  
- engagement prediction  
- fairness & explainability  
- reproducible ML engineering  

Each project reflects strong engineering ownership with modular design, real-world ML practices, and clear evaluation discipline.

---

## 📂 Projects Overview

### **1️⃣ Content Ranking System (Search & Recommendations)**  
📁 `content-ranking-system/`

A full **Learning-to-Rank pipeline** similar to real-world search & recommendation systems.

**Highlights**
- Candidate generation + re-ranking workflow  
- User, item, and context feature pipelines  
- Negative sampling for implicit feedback  
- Time-aware train/validation/test splits to prevent leakage  
- Ranking metrics: **NDCG, MAP, Recall@K**  
- Lightweight inference pipeline for low-latency scoring  
- Clean architecture following production ML patterns  

**Tech Stack:** Python, LightGBM/XGBoost (LTR), Pandas, NumPy, FastAPI, GitHub Actions

---

### **2️⃣ User Engagement Prediction**  
📁 `User-Engagement-Prediction/`

Predicts user engagement likelihood to support personalization, ranking, and retention strategies.

**Highlights**
- End-to-end ML workflow (preprocessing → modeling → evaluation)  
- Behavioral and temporal feature engineering  
- Model comparison with strong validation discipline  
- Decision-aligned metrics: **ROC-AUC, RMSE**  
- Modular, reproducible experiment structure  

**Tech Stack:** Python, Scikit-learn, Pandas

---

### **3️⃣ Explainability & Trust in Recommender Systems**  
📁 `explainability_trust_recsys/`

Improves transparency and trust in model predictions using explainability techniques.

**Highlights**
- Post-hoc explanation methods  
- Feature attribution and interpretation  
- Separation of model logic from explainability layer  
- Stakeholder-friendly explanation outputs  

**Tech Stack:** Python, SHAP/LIME, Data Visualization

---

### **4️⃣ Bias & Fairness in Machine Learning**  
📁 `bias-fairness-ml/`

Analyzes and mitigates bias across sensitive attributes in machine learning models.

**Highlights**
- Fairness metrics and disparity analysis  
- Evaluation of subgroup performance gaps  
- Responsible ML practices aligned with industry standards  
- Reporting of fairness implications and trade-offs  

**Tech Stack:** Python, Statistical Analysis, Fairness Libraries

---

## 🧱 Core Engineering Principles

- **Modular architecture:** data → features → models → evaluation → inference  
- **Reproducibility:** config-driven pipelines and deterministic splits  
- **Evaluation discipline:** metrics aligned with ranking, engagement & fairness goals  
- **Production awareness:** low-latency inference design and clean API patterns  
- **Readable documentation:** recruiter-friendly, organized, and maintainable  

---




