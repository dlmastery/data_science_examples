# 🧠 Autonomous AI Agent Skills Catalog

This repository includes a full suite of **54+ Antigravity Agent Skills** located in `.agents/skills/` and embedded directly within each project subfolder.

These skills allow any AI coding agent or developer using **Google Antigravity**, **Gemini CLI**, or **Claude Code** to autonomously execute, replicate, verify, and extend all 12 platforms in this portfolio.

---

## 🧭 Project-to-Skill Mapping Index

| # | Project Directory | Embedded Agent Skills | Skill Purpose & Capabilities |
|---|---|---|---|
| **1** | [`01_nyc_taxi_trip_prediction`](./01_nyc_taxi_trip_prediction) | `nyc-taxi-autoresearch`<br>`exploratory-data-analysis`<br>`feature-engineering`<br>`pandas-patterns` | Spatial Haversine/Manhattan distance engineering, XGBoost duration modeling, Karpathy-style autonomous hyperparameter hill-climbing. |
| **2** | [`02_nano_llm_transformer`](./02_nano_llm_transformer) | `nano-llm-transformer`<br>`pytorch-training-loop`<br>`llm-finetuning` | Pure PyTorch autoregressive LLM construction from scratch, RoPE relative embeddings, SwiGLU feed-forward networks, and Supervised Fine-Tuning (SFT). |
| **3** | [`03_customer_segmentation_clustering`](./03_customer_segmentation_clustering) | `customer-segmentation-clustering`<br>`segmentation-analysis`<br>`sklearn-pipelines` | K-Means & GMM silhouette optimization, 2D PCA & t-SNE manifold projections, customer persona profiling, and automated cluster naming. |
| **4** | [`04_associative_pattern_mining`](./04_associative_pattern_mining) | `associative-pattern-mining`<br>`funnel-analysis` | Frequent itemset pattern mining, Apriori candidate generation, FP-Growth tree discovery, Lift/Conviction calculations, and real-time cross-sell recommender. |
| **5** | [`05_data_science_skills_lab`](./05_data_science_skills_lab) | `exploratory-data-analysis`<br>`feature-engineering`<br>`data-cleaning`<br>`imbalanced-data`<br>`model-evaluation`<br>`data-quality-audit` | Complete modular analytics execution across Kaggle Titanic, House Prices, Credit Card Fraud, and E-Commerce benchmarks. |
| **6** | [`06_anomaly_detection`](./06_anomaly_detection) | `anomaly-detection`<br>`imbalanced-data`<br>`model-evaluation`<br>`ml-debugging` | High-dimensional cloud telemetry threat scoring, multi-backbone ensembling (Isolation Forest, Autoencoder, LOF, One-Class SVM), and IQR attribution. |
| **7** | [`07_automl_autogluon`](./07_automl_autogluon) | `automl-autogluon`<br>`hyperparameter-tuning`<br>`experiment-tracking` | 3-Level stacking DAG orchestration, Caruana greedy forward ensemble selection, and multi-task Kaggle tournament optimization. |
| **8** | [`08_datascience_visual_mastery`](./08_datascience_visual_mastery) | `data-science-mastery`<br>`visualization-builder`<br>`reproducible-ml` | Interactive visual curriculum for Naive Bayes, Confusion Matrix/PR trade-offs, Differential Calculus/Gradients, and Chain Rule Backpropagation. |
| **9** | [`09_flowforge_dag_engine`](./09_flowforge_dag_engine) | `matt-pocock-typescript-patterns`<br>`matt-pocock-to-spec`<br>`matt-pocock-to-tickets`<br>`matt-pocock-grill-me` | Matt Pocock TypeScript architecture, nominal branded types, discriminated unions, `assertNever()` exhaustiveness, and Kahn's topological sort. |
| **10** | [`10_crispdm_masters_curriculum`](./10_crispdm_masters_curriculum) | `data-cleaning`<br>`data-narrative-builder`<br>`methodology-explainer`<br>`analysis-planning` | End-to-end 7-phase CRISP-DM master's curriculum, Pearson correlation heatmap, and Sub-Linear Cosine LSH nearest neighbor search. |
| **11** | [`11_enterprise_ds_audit`](./11_enterprise_ds_audit) | `data-quality-audit`<br>`analysis-qa-checklist`<br>`peer-review-template`<br>`metric-reconciliation` | 6-dimension governance certification, data leakage simulation sandbox, and Mitchell et al. Model Card generation. |
| **12** | [`12_timeseries_forecasting`](./12_timeseries_forecasting) | `time-series-analysis`<br>`business-metrics-calculator`<br>`dashboard-specification` | Multi-horizon forecast fans ($h=7..60$d), classical additive decomposition, ADF/KPSS stationarity tests, and 40-lag ACF/PACF bar charts. |

---

## 🤖 How AI Agents Execute These Skills

When an AI Agent is paired with this workspace, it discovers skills via:
* **Global Customizations**: `~/.gemini/config/skills/`
* **Workspace Customizations**: `.agents/skills/` (and within each project folder)

Each skill contains a `SKILL.md` instruction file defining:
1. **YAML Frontmatter**: Name and description triggering auto-discovery.
2. **Methodological Directives**: Step-by-step guidance, mathematical foundations, and leakage prevention rules.
3. **Reference Implementations**: Scripts, schemas, and verification checklists.
