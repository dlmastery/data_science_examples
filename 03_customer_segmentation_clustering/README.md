# Customer Intelligence & Unsupervised Segmentation Platform

An enterprise-grade Unsupervised Machine Learning & Customer Segmentation platform based on the popular Kaggle Customer Personality / Retail dataset.

---

## 🌟 Key Features
- 🏛️ **Multi-Backbone Clustering Tournament**:
  - **K-Means++** (Lloyd's algorithm)
  - **Gaussian Mixture Models (GMM)** (Soft EM clustering)
  - **Hierarchical Agglomerative Clustering** (Ward linkage)
  - **DBSCAN** (Density spatial clustering with noise detection)
  - **Spectral Clustering** (Graph Laplacian manifold clustering)
- 🗺️ **2D PCA & t-SNE Manifold Projections**:
  - Interactive SVG scatter canvas with 69.6% explained variance tracking.
- ⛰️ **AutoResearch Tabular Hill-Climbing Engine**:
  - 4-Phase autonomous optimization (Backbone Battle, Feature Evolution, Hyperparameter Tuning, Consensus Ensembling) with AST code diffs and step click-through inspector modal.
- 📄 **CRISP-DM Publication Research Report**:
  - 6-Phase interactive research report modal accessible directly in the UI.
- 🛍️ **Real-Time Customer Persona Classifier**:
  - Instant cluster assignment, confidence calculation, and tailored promotional blueprint.
- 🧪 **100% Verified**:
  - 7/7 automated API tests passed.
  - Chrome DevTools Protocol visual audit passing with 0 network errors.

---

## 🚀 Quickstart Guide

### 1. Requirements
- Python 3.10+
- `scikit-learn`, `pandas`, `numpy`, `fastapi`, `uvicorn`
- Node.js 18+

### 2. Training Models & Running AutoResearch
```bash
cd ml
python train.py
python autoresearch_clustering.py
```

### 3. Running the Backend Server (Port 8003)
```bash
cd server
python -m uvicorn main:app --host 127.0.0.1 --port 8003
```

### 4. Running the Frontend Web Application (Port 5176)
```bash
cd client
npm install
npm run dev
```

Open your browser at **`http://localhost:5176`**!
