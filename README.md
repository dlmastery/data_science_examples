# Data Science & Machine Learning Repository — DLMastery Examples

Enterprise-grade end-to-end Data Science, Machine Learning, and Modern Deep Learning platforms featuring AutoResearch Tabular Hill-Climbing, state-of-the-art architectures, interactive visualization dashboards, and publication-standard CRISP-DM research documentation.

---

## 📂 Repository Structure

```
data_science_examples/
├── 01_nyc_taxi_trip_prediction/             # NYC Taxi Fare & Duration ML Platform (LightGBM, XGBoost, AutoResearch Tabular)
│   ├── ml/                                  # Spatial Feature Engineering & Training Pipelines
│   ├── server/                              # FastAPI Microservice (Port 8000)
│   ├── client/                              # React 18 + Vite Interactive Route Estimator (Port 5174)
│   └── DESIGN_DOC.md                        # Full System Design & Mathematical Formulation
│
├── 02_nano_llm_transformer/                 # NanoLlama — Modern Transformer LLM & Chatbot from Scratch
│   ├── core/                                # RoPE, SwiGLU, RMSNorm, KV-Cache, Subword Tokenizer
│   ├── server/                              # FastAPI SSE Streaming Microservice (Port 8002)
│   ├── client/                              # React 18 + Vite Quantum Neural Web App (Port 5175)
│   └── DESIGN_DOC.md                        # Transformer Primitives & Attention Heatmap Extraction
│
├── 03_customer_segmentation_clustering/     # Customer Intelligence & Unsupervised Segmentation Platform
│   ├── ml/                                  # Multi-Backbone Clustering (K-Means++, GMM, Agglomerative, DBSCAN, Spectral, 2D PCA & t-SNE)
│   ├── server/                              # FastAPI REST Microservice (Port 8003)
│   ├── client/                              # React 18 + Vite Manifold Explorer & Admin Dashboard (Port 5176)
│   └── DESIGN_DOC.md                        # Persona Taxonomy, Silhouette Optimization & CRISP-DM
│
├── 04_associative_pattern_mining/           # Market Basket Intelligence & Associative Pattern Mining Platform
│   ├── ml/                                  # Apriori, FP-Growth, ECLAT, 2D Association Network Graph, AutoResearch Tabular
│   ├── server/                              # FastAPI REST Microservice (Port 8004)
│   ├── client/                              # React 18 + Vite Basket Builder & 2D Graph Visualizer (Port 5177)
│   └── DESIGN_DOC.md                        # High-Order Metric Formulations & CRISP-DM Report
│
├── 05_data_science_skills_lab/              # Data Science & Analytics Interactive Skills Mastery Lab (46 Agent Skills)
│   ├── core/                                # Titanic, House Prices, Fraud, E-Commerce Analytics, Data Quality Profiling
│   ├── server/                              # FastAPI REST Microservice (Port 8005)
│   ├── client/                              # React 18 + Vite Interactive Skills Workbench & Encyclopedia (Port 5178)
│   └── README.md                            # Comprehensive Skills Documentation
│
└── .agents/skills/                          # Autonomous Reproduction Skills for Antigravity Agents
    ├── nyc-taxi-autoresearch/SKILL.md
    ├── nano-llm-transformer/SKILL.md
    ├── customer-segmentation-clustering/SKILL.md
    ├── associative-pattern-mining/SKILL.md
    └── data-science-skills-lab/SKILL.md
```

---

## 🌟 Projects Summary & Benchmark Comparison

| Project | Domain & Dataset | SOTA Champion Algorithm | Primary Metric | Kaggle SOTA Baseline | AutoResearch Gain | Live UI | Backend API |
|---|---|---|---|---|---|---|---|
| **01. NYC Taxi Trip Predictor** | Spatial-Temporal GPS ($N=10\text{k}$) | **XGBoost Regressor** | **RMSLE: 0.1531** ($R^2 = 96.97\%$) | RMSLE: 0.3680 | **+58.4% improvement** | [Port 5174](http://localhost:5174) | [Port 8000](http://127.0.0.1:8000/docs) |
| **02. NanoLlama SOTA LLM** | Conversational QA & Code | **3-Layer RoPE + SwiGLU + KV-Cache** | **Perplexity: 1.05** (Loss: 0.0531) | Transformer Baseline PPL: 16.34 | **93.6% PPL reduction** | [Port 5175](http://localhost:5175) | [Port 8002](http://127.0.0.1:8002/docs) |
| **03. Customer Segmentation** | Retail Behavior ($N=10\text{k}$) | **K-Means++ & AutoResearch Consensus** | **Silhouette: 0.4180** (DB: 1.0193) | Top 1% SOTA: 0.3850 | **+21.0% gain** | [Port 5176](http://localhost:5176) | [Port 8003](http://127.0.0.1:8003/docs) |
| **04. Market Basket Mining** | Kaggle Instacart ($N=10\text{k}$) | **FP-Growth (Prefix Tree)** | **Mean Lift: 3.793** (Peak: 4.48x) | Top 1% SOTA Lift: 4.850 | **+7.9% lift gain** | [Port 5177](http://localhost:5177) | [Port 8004](http://127.0.0.1:8004/docs) |
| **05. Skills Mastery Lab** | 5 Kaggle Datasets (Titanic, Housing, Fraud, SaaS) | **46 Production Agent Skills** | **Multi-Domain SOTA** | Kaggle Master Benchmarks | **100% Skill Automation** | [Port 5178](http://localhost:5178) | [Port 8005](http://127.0.0.1:8005/docs) |

---

## 🛠️ Autonomous Reproduction via Skills

Each project includes an autonomous runbook in `.agents/skills/<skill_name>/SKILL.md`:
- **NYC Taxi**: Trigger `skills/nyc-taxi-autoresearch`
- **NanoLlama**: Trigger `skills/nano-llm-transformer`
- **Customer Clustering**: Trigger `skills/customer-segmentation-clustering`
- **Market Basket Mining**: Trigger `skills/associative-pattern-mining`
- **Data Science Skills Lab**: Trigger `skills/data-science-skills-lab`
