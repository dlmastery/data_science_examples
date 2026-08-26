# CRISP-DM Master's Data Science Platform

A comprehensive 7-phase master's degree level curriculum and interactive analysis platform conducting an end-to-end CRISP-DM workflow on the **Kaggle Census & Income ($N=2,500$) dataset**.

![CRISP-DM EDA & Correlations](./screenshots/crispdm_eda_correlations.png)

---

## 🔬 Curriculum & Methodology Highlights

1. **Phase 1: Understanding & EDA**: Pearson multivariate correlation matrix, summary distributions, and rank-preserving median imputation.
2. **Phase 2: Clustering Topology**: K-Means ($k=4$) and Gaussian Mixture Models with silhouette optimization ($s=0.46$).
3. **Phase 3: Outlier Isolation**: Isolation Forest contamination tuning ($c=0.05$).
4. **Phase 4: Income Regression Tournament**: Gradient Boosting ($R^2 = 0.91$) vs Baseline OLS ($R^2 = 0.74$) with live salary calculator.
5. **Phase 5: Association Pattern Mining**: Apriori rule extraction with Lift $2.45\times$.
6. **Phase 6: Locality-Sensitive Hashing (LSH)**: Sub-linear Cosine Random Hyperplane nearest neighbor search ($14.8\times$ query speedup).
7. **Phase 7: Synthesis & Chapter Quizzes**: Progressive interactive master's level quizzes.

![CRISP-DM Regression Predictor](./screenshots/crispdm_regression_predictor.png)

---

## 🚀 Quick Start

```bash
# Backend (FastAPI on Port 8010)
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8010

# Frontend (Vite React on Port 5183)
cd frontend
npm install
npm run dev
```
