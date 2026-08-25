# NanoLlama — Modern Transformer Language Model & Chatbot from Scratch

A lightweight, from-scratch modern Transformer Language Model designed to run and train seamlessly on laptop CPUs and edge GPUs, built with exact state-of-the-art primitives (**RoPE**, **SwiGLU**, **RMSNorm**, **KV-Cache**).

---

## 🌟 Key Highlights & Features
- 🧠 **Pure PyTorch SOTA Architecture**:
  - **Rotary Position Embeddings (RoPE)**: Relative position encoding via 2D rotational vector geometry.
  - **SwiGLU Gated Feed-Forward**: High-expressivity SiLU activation gating.
  - **RMSNorm Pre-Normalization**: Fast and stable normalization without mean-centering.
  - **Autoregressive Key-Value Cache (KV-Cache)**: Stateful $O(1)$ token decoding.
- 💬 **Interactive Streaming Chatbot**:
  - Server-Sent Events (SSE) token streamer with live generation telemetry (tokens/sec, TTFT).
  - Decoding controls: Temperature, Top-P Nucleus, Top-K Filtering, Repetition Penalty, Max Tokens.
  - Curated prompt presets (Baby Dragon Bedtime Story, Python Fibonacci, RoPE Explanation, Math Reasoning).
- 🔬 **Multi-Head Attention Heatmap Visualizer**:
  - Interactive 2D attention matrices across all 3 layers and 4 heads with token-to-token hover tooltips.
- 🧩 **Tokenizer & Subword Studio**:
  - Interactive subword visualizer with alternating color-coded tags, token IDs, and Top-5 next token probability distribution bar charts.
- 📈 **Training Telemetry & Perplexity Dashboard**:
  - Step-by-step training loss curve, validation perplexity progression (achieving **1.05 Perplexity**), and learning rate schedule.
- 🧪 **100% Verified**:
  - 6/6 Automated API tests passing.
  - Chrome DevTools Protocol visual audit passing with 0 network failures.

---

## 🚀 Quickstart Guide

### 1. Requirements
- Python 3.10+
- PyTorch (`torch`)
- FastAPI (`fastapi`, `uvicorn`)
- Node.js 18+

### 2. Training the Model
```bash
cd core
python train.py
```

### 3. Running the Backend Server (Port 8002)
```bash
cd server
python -m uvicorn main:app --host 127.0.0.1 --port 8002
```

### 4. Running the Frontend Web Application (Port 5175)
```bash
cd client
npm install
npm run dev
```

Open your browser at **`http://localhost:5175`**!
