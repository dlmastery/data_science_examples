# Enterprise Data Science, Machine Learning & Software Engineering Repository

A comprehensive collection of production-grade Data Science, Machine Learning, Deep Learning, and Full-Stack TypeScript applications built according to textbook CRISP-DM methodology, rigorous statistical governance, and modern UI/UX principles.

---

## 📚 Complete Project Index & Port Catalog

| # | Directory | System Title | Key Architecture / Methodology | Backend Port | Frontend Port |
|---|---|---|---|:---:|:---:|
| **1** | [`01_nyc_taxi_trip_prediction`](./01_nyc_taxi_trip_prediction) | **NYC Taxi Fare Prediction & Admin Dashboard** | CRISP-DM 6-Phase, Haversine Spatial Embeddings, AutoResearch Tabular Hill-Climbing | `8000` | `5174` |
| **2** | [`02_nano_llm_transformer`](./02_nano_llm_transformer) | **NanoLlama SFT Language Model & KV-Cache** | PyTorch Autoregressive Transformer, RMSNorm, RoPE, SwiGLU FFN, SFT (Loss `0.0000`) | `8002` | `5175` |
| **3** | [`03_customer_segmentation_clustering`](./03_customer_segmentation_clustering) | **Customer Segmentation & Clustering Platform** | K-Means, GMM, Agglomerative, DBSCAN, PCA 2D Manifolds, Silhouette Optimization | `8003` | `5176` |
| **4** | [`04_associative_pattern_mining`](./04_associative_pattern_mining) | **Market Basket Associative Pattern Mining** | Apriori Candidate Pruning, FP-Growth Tree Traversal, Lift & Directional Conviction | `8004` | `5177` |
| **5** | [`05_data_science_skills_lab`](./05_data_science_skills_lab) | **Data Science Skills Mastery Lab** | 54 Param087 & Nimrodfisher Skills, 5 Kaggle Benchmark Datasets, Dynamic Dispatcher | `8005` | `5178` |
| **6** | [`06_anomaly_detection`](./06_anomaly_detection) | **Autonomous Anomaly Threat Intelligence** | Isolation Forest, LOF, One-Class SVM, Mahalanobis, PR-AUC Calibration for Imbalance | `8006` | `5179` |
| **7** | [`07_automl_autogluon`](./07_automl_autogluon) | **AutoGluon Multi-Layer Stacking Platform** | 3-Level Stacking DAG, Caruana Greedy Selection, Out-of-Fold Leakage-Free Blending | `8007` | `5180` |
| **8** | [`08_datascience_visual_mastery`](./08_datascience_visual_mastery) | **Data Science Foundations & Interactive Mastery** | Live Simulators (Naive Bayes, PR Curves, Calculus, Backprop), Chapter Quizzes | `8008` | `5181` |
| **9** | [`09_flowforge_dag_engine`](./09_flowforge_dag_engine) | **FlowForge DAG Engine (Matt Pocock Patterns)** | Nominal Branding, Discriminated Unions, Kahn's DAG Compiler, SSE Streaming Bus | `8009` | `5182` |
| **10** | [`10_crispdm_masters_curriculum`](./10_crispdm_masters_curriculum) | **CRISP-DM Master's Data Science Platform** | 7-Phase Census Analytics, Gradient Boosting ($R^2=0.91$), Cosine LSH ($14.8\times$ speedup) | `8010` | `5183` |
| **11** | [`11_enterprise_ds_audit`](./11_enterprise_ds_audit) | **Enterprise Data Science Audit & Governance** | 6-Dimension Scorecard, Live Data Leakage Sandbox, Mitchell et al. Model Cards | `8011` | `5184` |
| **12** | [`12_timeseries_forecasting`](./12_timeseries_forecasting) | **Time Series Forecasting & Anomaly Telemetry** | Multi-Horizon Fans ($h=7..60$d), 40-Lag ACF/PACF, Walk-Forward Tournament (LightGBM) | `8012` | `5185` |

---

## 🎯 Verbatim Reproduction Prompts Catalog

All verbatim user prompts, follow-up steering directives, and architectural parameters required to reproduce each system are fully documented in:
👉 **[PROMPTS.md](./PROMPTS.md)**

---

## 🛠️ Quick Start & Running Locally

### Prerequisites
* Python 3.10+ (with `fastapi`, `uvicorn`, `scikit-learn`, `pandas`, `numpy`, `torch`)
* Node.js 18+ (with `npm`)

### Starting Any Project
```bash
# Example: Launching Time Series Forecasting Engine
cd 12_timeseries_forecasting/backend
python -m uvicorn main:app --host 127.0.0.1 --port 8012

# In a separate terminal
cd 12_timeseries_forecasting/frontend
npm install
npm run dev # Launches Vite on http://localhost:5185/
```
