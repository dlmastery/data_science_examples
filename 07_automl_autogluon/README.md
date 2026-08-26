# AutoGluon AutoML Multi-Task Platform & Stacking DAG

An enterprise-grade AutoML platform illustrating state-of-the-art **AutoGluon Multi-Layer Stacking Ensemble** architectures across canonical Kaggle data science tasks (**Tabular Classification** and **Tabular Regression**). Features multi-backbone learners (**LightGBM**, **CatBoost**, **XGBoost**, **Random Forest**, **Extra Trees**, **NeuralNet FastAI / Tabular MLP**, and **Level 2/3 Weighted Ensembles**), autonomous **AutoResearch Tabular Hill-Climbing** (+5.94% ROC-AUC gain), an interactive 3-Level Stacking DAG SVG visualizer, real-time multi-task inference, Kaggle Grandmaster SOTA baseline comparisons, a 6-phase publication CRISP-DM research report, and full Git integration.

---

## 🌟 Key Features

1. **AutoGluon Multi-Layer Stacking Architecture (`ml/autogluon_stacking.py`)**:
   - **Level 1 Base Learners**: LightGBM, CatBoost, XGBoost, Random Forest, Extra Trees, NeuralNet FastAI with 5-fold cross-validation.
   - **Level 2 Stacking**: Concatenates Level 1 Out-of-Fold (OOF) cross-validation predictions with raw features:
     $$X_{\text{Level 2}} = [X_{\text{raw}}, \hat{y}_{\text{OOF}}^{(1)}, \hat{y}_{\text{OOF}}^{(2)}, \dots, \hat{y}_{\text{OOF}}^{(K)}]$$
   - **Level 3 Meta-Learner (Caruana Greedy Forward Selection)**:
     - Iteratively selects models $M$ with non-negative weights $w_j$ to maximize the target validation metric:
       $$w^* = \operatorname{argmax}_w \text{Metric}\left(\sum_{j=1}^M w_j \hat{y}_j\right), \quad \sum w_j = 1$$
   - **Kaggle Grandmaster SOTA Baseline Comparison**: Hand-tuned 20-model ensemble (Classification ROC-AUC: **0.9460**, Regression R²: **0.9380**).

2. **AutoResearch Tabular Hill-Climbing (`ml/autoresearch_automl.py`)**:
   - 4-Phase autonomous search loop (Base Learner Tournament, Stacking Activation, Caruana Weight Tuning, Latency/Feature Pruning) driving classification ROC-AUC from $0.8920 \to 0.9450$ (**+5.94% gain**).
   - Step click-through telemetry tracking with AST code diffs, parameter diffs, and agent reflections.

3. **Interactive Multi-Task Web Platform (`client/` on Port 5180)**:
   - **Multi-Task Predictor**: Real-time slider simulator switching between Customer Churn (Classification) and Diamond Valuation (Regression).
   - **3-Level Stacking DAG SVG Visualizer**: Interactive architectural graph displaying Base Learners $\to$ Level 2 Stacker $\to$ Level 3 Caruana ensemble.
   - **Data Science Admin Console**: AutoGluon Leaderboard with preset filter buttons (`best_quality`, `high_quality`, `fast_training`), AutoResearch trajectory chart with **Step Click-Through Inspector Modal**, and Permutation/Gini feature importance charts.
   - **CRISP-DM 6-Phase Publication Research Report Modal**.
   - **Online Presets Studio Modal** with celebratory confetti.

4. **Production FastAPI Microservice (`server/` on Port 8007)**:
   - 8 / 8 automated unit tests passing (100% pass rate).

---

## 🚀 Quickstart

### 1. Run FastAPI Backend (Port 8007)
```bash
cd server
python -m uvicorn main:app --host 127.0.0.1 --port 8007
```

### 2. Run React 18 + Vite Frontend (Port 5180)
```bash
cd client
npm install
npm run dev
```

### 3. Run Automated Tests
```bash
python server/test_api.py
```
