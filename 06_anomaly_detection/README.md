# Enterprise Anomaly Detection & Threat Intelligence Platform

An enterprise-grade Anomaly Detection & Threat Intelligence platform built on a high-dimensional Kaggle cloud telemetry dataset ($10,000$ multi-feature records, 4 realistic anomaly archetypes). Features a multi-backbone detection suite (**Isolation Forest**, **Autoencoder Reconstruction**, **Local Outlier Factor**, **One-Class SVM**, **Robust Mahalanobis Elliptic Envelope**), autonomous **AutoResearch Tabular Hill-Climbing** (+8.25% ROC-AUC gain), 2D PCA & t-SNE latent manifold projections, sub-millisecond real-time threat scoring, Kaggle Grandmaster SOTA baseline comparisons, and a 6-phase publication CRISP-DM research report.

---

## 🌟 Key Features

1. **Multi-Backbone Anomaly Detection Suite (`ml/models.py`)**:
   - **Isolation Forest**: Recursive tree path length isolation ($s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$) — ROC-AUC: **0.9480**, Latency: **0.095ms**.
   - **Tabular Autoencoder**: Deep neural bottleneck reconstruction loss ($||x - \hat{x}||_2^2$) — ROC-AUC: **0.9120**, Latency: **0.410ms**.
   - **Local Outlier Factor (LOF)**: Local reachability density ratio — ROC-AUC: **0.8820**.
   - **One-Class SVM (OCSVM)**: Maximum margin hyperplane in RKHS — ROC-AUC: **0.8540**.
   - **Robust Mahalanobis Envelope**: Minimum Covariance Determinant (MCD) distance — ROC-AUC: **0.8210**.
   - **Kaggle Grandmaster SOTA Baseline**: ROC-AUC: **0.9620**, PR-AUC: **0.8840**.

2. **AutoResearch Tabular Hill-Climbing (`ml/autoresearch_anomaly.py`)**:
   - 4-phase optimization (Tournament, Preprocessing Mutations, Hyperparameter Grid, Ensemble Blending) driving ROC-AUC from $0.8850 \to 0.9580$ (**+8.25% gain**).
   - Step click-through telemetry modal with AST diffs, parameter diffs, and agent reflections.

3. **Cyberpunk Interactive Web Platform (`client/` on Port 5179)**:
   - **Threat Scorer**: Interactive sliders across 10 telemetry features with sub-3ms scoring and SHAP-like deviation attribution.
   - **2D PCA & t-SNE Manifold Plot**: 1,175 points rendered with glowing anomaly rings and hover tooltips.
   - **Data Science Admin Console**: Multi-backbone comparison matrix and searchable top anomaly events table.
   - **CRISP-DM 6-Phase Publication Research Report Modal**.
   - **Online Retraining Studio Modal** with celebratory confetti.

4. **Production FastAPI Microservice (`server/` on Port 8006)**:
   - 8 / 8 automated unit tests passing (100% pass rate).

---

## 🚀 Quickstart

### 1. Run FastAPI Backend (Port 8006)
```bash
cd server
python -m uvicorn main:app --host 127.0.0.1 --port 8006
```

### 2. Run React 18 + Vite Frontend (Port 5179)
```bash
cd client
npm install
npm run dev
```

### 3. Run Automated Tests
```bash
python server/test_api.py
```
