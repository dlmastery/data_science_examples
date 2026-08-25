# 🚕 NYC Taxi ML Intelligence — End-to-End Data Science Project

A production-grade, state-of-the-art Machine Learning system for NYC Taxi Trip Duration & Fare Estimation with real-time inference, interactive spatial map simulation, and a transparent Data Science Admin lifecycle dashboard.

---

## 🌟 Key Features

### 1. State-of-the-Art ML Pipeline
- **Dataset**: Modeled on Kaggle's NYC Taxi Trip Duration competition.
- **Feature Engineering**: 20 spatial & temporal features including Haversine & Manhattan distances, bearing, transit hub proximity (JFK, LGA, EWR, Times Sq, Wall St), rush hour & night traffic factors.
- **Algorithms**: Tuned XGBoost gradient boosting optimizing for Root Mean Squared Logarithmic Error (RMSLE).
- **Benchmark Performance**:
  - SOTA XGBoost: **RMSLE = 0.1479**, **R² = 0.9679** (96.8% variance explained).

### 2. Interactive Rider & Fare Estimator UI
- **Interactive NYC Map Canvas**: Visualizes route trajectories across Manhattan, Brooklyn, Queens, Bronx, and airports with animated cruising taxi!
- **Landmark Presets**: Quick route selection (Times Square, JFK Airport, LGA, Central Park, Wall Street, Brooklyn Bridge).
- **Fare Breakdown**: Official NYC taxi rate formula (base fare, distance rate, congestion fee, peak rush hour surge, airport fee).
- **Confidence Intervals**: 95% upper and lower duration estimates.

### 3. Data Science & ML Admin Dashboard
- **Model Experiment Matrix**: Benchmark comparison between Ridge Baseline, Random Forest, and SOTA XGBoost.
- **Feature Importance Chart**: Ranked predictive power across engineered features.
- **Validation Loss Curves**: Loss vs boosting iterations plot.
- **Dataset Explorer**: 24-hour pickup density and duration frequency distributions.
- **Live Retraining Studio**: Interactive hyperparameter tuner to launch live model retraining with progress feedback.

---

## 🚀 Quickstart

### 1. Run Backend & Frontend Simultaneously
From `secondtest-nyc`:
```bash
npm run dev
```

Or run each independently:
```bash
# Terminal 1: Python FastAPI Server (Port 8000)
cd server
python -m uvicorn main:app --host 127.0.0.1 --port 8000

# Terminal 2: React Vite Client (Port 5174)
cd client
npm run dev
```

Open [http://localhost:5174](http://localhost:5174) in your browser.

---

## 🧪 Testing

Run all automated API tests:
```bash
python server/test_api.py
```

Run Chrome DevTools Protocol visual audit:
```bash
node devtools_audit.mjs
```
