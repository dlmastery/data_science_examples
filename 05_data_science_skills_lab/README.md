# Data Science & Analytics Interactive Skills Mastery Lab

An interactive visual educational workbench demonstrating **46 production AI Agent Skills** from [`param087/agent-ml-skills`](https://github.com/param087/agent-ml-skills) (15 ML & MLOps skills) and [`nimrodfisher/data-analytics-skills`](https://github.com/nimrodfisher/data-analytics-skills) (31 Data Analytics skills) across 5 popular Kaggle datasets.

---

## 🌟 Key Features

1. **46-Skill Interactive Encyclopedia & Runner**:
   - Every skill explained clearly for beginners: **Purpose & Real-World Motivation**, **Statistical / Mathematical Intuition**, **Input / Output Schemas**, **Common Beginner Pitfalls**, and **Live Execution Runner**.
2. **Kaggle Benchmark Interactive Workbenches**:
   - 🚢 **Titanic Disaster Survival (Classification)**: Feature engineering, pipeline DAG, ROC-AUC curve (0.7267), Confusion Matrix, and Feature Importance bar chart.
   - 🏡 **House Prices: Advanced Regression**: Actual vs Predicted scatter plot, Log-target transform ($y \to \log(1 + y)$), R² (0.8105), RMSE, and Residuals.
   - 💳 **Credit Card Fraud (Imbalanced Data)**: 0.17% rare target, baseline vs balanced cost-matrix comparison (Recall jumps from 54% to 96%), and **Live Decision Threshold Slider ($\tau$)**.
   - 📈 **E-Commerce & SaaS Analytics**: 12-Month User Retention Cohort Heatmap, 5-Stage Checkout Funnel Drop-off Waterfall, and **Live A/B Testing Z-Test Calculator**.
   - 🔍 **Automated Data Quality Audit**: 87.8 / 100 Data Quality Scorecard, duplicate row detection, null density mapping, and anomaly assertion checking.
3. **CRISP-DM 6-Phase Publication Research Report**.

---

## 🚀 Quickstart

### 1. Run FastAPI Backend (Port 8005)
```bash
cd server
python -m uvicorn main:app --host 127.0.0.1 --port 8005
```

### 2. Run React + Vite Frontend (Port 5178)
```bash
cd client
npm install
npm run dev
```

### 3. Run Automated Unit Tests
```bash
python server/test_api.py
```
*(8 / 8 tests pass in < 0.70s)*.
