# 🌟 Enterprise Data Science, Machine Learning & TypeScript Platform Repository

A comprehensive, production-grade portfolio of **12 Data Science, Machine Learning, Deep Learning, and Full-Stack TypeScript Systems**, engineered adhering to the **CRISP-DM standard**, rigorous mathematical foundations, and state-of-the-art interactive UX.

---

## 🏛️ Comprehensive Systems Portfolio Index

| # | System Title | Domain & Methodology | Backend Port | Frontend Port | Primary Screenshot |
|---|---|---|:---:|:---:|:---:|
| **1** | [**NYC Taxi Trip Prediction**](./01_nyc_taxi_trip_prediction) | CRISP-DM Spatial Regression & AutoResearch | `8000` | `5174` | ![NYC Taxi](./docs/screenshots/nyc_estimator_view.png) |
| **2** | [**NanoLlama SFT LLM**](./02_nano_llm_transformer) | PyTorch Autoregressive Transformer (RoPE, SwiGLU, SFT) | `8002` | `5175` | ![NanoLlama](./docs/screenshots/nanollama_chat_studio.png) |
| **3** | [**Customer Clustering**](./03_customer_segmentation_clustering) | Topological Partitioning & AutoResearch | `8003` | `5176` | ![Clustering](./docs/screenshots/clustering_explorer.png) |
| **4** | [**Market Basket Mining**](./04_associative_pattern_mining) | Apriori & FP-Growth Pattern Affinity | `8004` | `5177` | ![Market Basket](./docs/screenshots/market_basket_graph.png) |
| **5** | [**DS Skills Mastery Lab**](./05_data_science_skills_lab) | 54 Analytical Skills & 5 Kaggle Benchmarks | `8005` | `5178` | ![Skills Lab](./docs/screenshots/skills_lab_catalog.png) |
| **6** | [**Anomaly Threat Intelligence**](./06_anomaly_detection) | Multi-Backbone Telemetry Scoring (PR-AUC 0.94) | `8006` | `5179` | ![Anomaly Threat](./docs/screenshots/anomaly_threat_scorer.png) |
| **7** | [**AutoML AutoGluon Stacking**](./07_automl_autogluon) | 3-Level Stacking DAG & Caruana Ensembling | `8007` | `5180` | ![AutoML](./docs/screenshots/automl_predictor.png) |
| **8** | [**DS Visual Foundations**](./08_datascience_visual_mastery) | Interactive Math Simulators & Quizzes | `8008` | `5181` | ![DS Mastery](./docs/screenshots/mastery_probabilistic_bayes.png) |
| **9** | [**FlowForge DAG Engine**](./09_flowforge_dag_engine) | Matt Pocock TypeScript Architecture & Kahn's DAG | `8009` | `5182` | ![FlowForge](./docs/screenshots/flowforge_dag_canvas.png) |
| **10** | [**CRISP-DM Master's Platform**](./10_crispdm_masters_curriculum) | 7-Phase Census Analytics & Cosine LSH | `8010` | `5183` | ![CRISP-DM](./docs/screenshots/crispdm_eda_correlations.png) |
| **11** | [**Enterprise DS Audit**](./11_enterprise_ds_audit) | 6-Dimension Quality & Leakage Scorecard | `8011` | `5184` | ![DS Audit](./docs/screenshots/ds_audit_scorecard.png) |
| **12** | [**TimePulse Forecasting**](./12_timeseries_forecasting) | Multi-Horizon Forecast Fans & 40-Lag ACF/PACF | `8012` | `5185` | ![Time Series](./docs/screenshots/timeseries_forecast_studio.png) |

---

## 🎯 Verbatim Reproduction Prompt Catalog

Every prompt used to generate, iterate, and verify these applications is cataloged in:
👉 **[PROMPTS.md](./PROMPTS.md)**

---

## 🛠️ Global Quick Start

### 1. Prerequisites
* Python 3.10+ (with `torch`, `fastapi`, `uvicorn`, `scikit-learn`, `pandas`, `numpy`)
* Node.js 18+ (with `npm`)

### 2. Launching Any Application
Navigate to any project directory to launch its backend API and Vite React frontend:
```bash
# Example: Launching Time Series Forecasting Platform
cd 12_timeseries_forecasting/backend
python -m uvicorn main:app --host 127.0.0.1 --port 8012

# In a separate terminal
cd 12_timeseries_forecasting/frontend
npm install
npm run dev # Opens http://localhost:5185/
```
