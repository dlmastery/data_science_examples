# Time Series Forecasting & Anomaly Telemetry Engine

An end-to-end CRISP-DM time series forecasting platform featuring multi-horizon predictions ($h=7..60$ days), expanding 95% confidence interval fans, classical additive signal decomposition, and walk-forward backtesting tournaments.

![Time Series Forecast Studio](./screenshots/timeseries_forecast_studio.png)

---

## 📈 Platform Capabilities

1. **Live Forecast Studio**:
   - Multi-step horizon fans ($h=7..60$ days) with weather surge scenario sliders and anomaly flag markers ($> 3\sigma$).
2. **Classical Signal Decomposition & Autocorrelation**:
   - Orthogonal additive decomposition: $Y_t = \text{Trend}_t + \text{Seasonal}_t + \text{Residual}_t$.
   - Stationarity tests (ADF $p < 0.0001$, KPSS).
   - 40-lag interactive ACF & PACF bar charts.
3. **Walk-Forward Tournament**:
   - Model tournament comparing LightGBM Lag GBDT (MAPE: `2.84%`, MASE: `0.42`), Deep N-BEATS (`3.12%`), Prophet (`3.65%`), SARIMAX (`4.18%`), and Seasonal Naive (`7.42%`).
4. **AutoResearch Hill-Climbing**:
   - Iterative lag search beating Kaggle SOTA by $-0.11\%$.

![Time Series Decomposition & ACF](./screenshots/timeseries_decomposition_acf.png)
![Time Series Tournament](./screenshots/timeseries_tournament_leaderboard.png)

---

## 🚀 Quick Start

```bash
# Backend (FastAPI on Port 8012)
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8012

# Frontend (Vite React on Port 5185)
cd frontend
npm install
npm run dev
```
