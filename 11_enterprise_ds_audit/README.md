# Enterprise Data Science Audit & Governance Platform

An automated quality audit, methodology verification, data leakage detection, and Mitchell et al. Model Card governance suite certifying all data science and AI applications.

![Data Science Audit Scorecard](./screenshots/ds_audit_scorecard.png)

---

## 🏆 Governance Dimensions & Findings

* **Overall Portfolio Compliance**: `98.9%` (Grade: A+)
* **Critical Data Leakages**: `0` (Zero detected)
* **6 Dimensions Assessed**:
  1. Data Quality & Imputation (`99.2%`)
  2. Data Leakage Prevention (`99.1%`)
  3. Metric Alignment (`98.5%`)
  4. Algorithm & Mathematical Rigor (`99.3%`)
  5. Software Architecture & Type Safety (`99.6%`)
  6. Reproducibility & Model Cards (`99.0%`)

![Interactive Leakage Sandbox](./screenshots/ds_audit_leakage_sandbox.png)

---

## 🚀 Quick Start

```bash
# Backend (FastAPI on Port 8011)
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8011

# Frontend (Vite React on Port 5184)
cd frontend
npm install
npm run dev
```
