# Data Science & Machine Learning Visual Mastery Platform

An interactive visual textbook and learning platform teaching foundational Data Science and Machine Learning mathematical principles across 4 comprehensive curriculum modules (**Probabilistic Classification & Bayes**, **True Model Evaluation & Threshold Dynamics**, **Differential Calculus & Gradient Optimization**, and **The Chain Rule & Backpropagation Mechanics**). Features live interactive parameter sliders, chapter-by-chapter mastery quizzes with instant explanations, data science technical interview flashcards, and static GitHub Pages (`github.io`) deployment artifacts.

---

## 🌟 Key Features

1. **Module 1: Probabilistic Classification & The Naive-Bayes Mechanism (`content/naive_bayes_module.json`)**:
   - The combinatorial explosion of language ($10,000^{20}$ possibilities).
   - Conditional independence factorization ($P(x_1 \dots x_n | c) = \prod P(x_i | c)$).
   - Live Napkin Bayes Calculator (Prior base rate and word likelihood sliders with instant posterior updating).
   - Laplace add-one smoothing ($\frac{\text{count}+1}{\text{total}+V}$) and log-space underflow protection.

2. **Module 2: Model Evaluation, Confusion Matrices & Threshold Dynamics (`content/evaluation_metrics_module.json`)**:
   - The 99% accuracy trap on class-imbalanced datasets.
   - 2x2 Contingency Matrix ($TP, TN, FP, FN$).
   - Precision vs Recall tradeoffs with interactive decision threshold slider ($T \in [0.1, 0.9]$).
   - $F_1$ Harmonic Mean formulation and business cost matrix optimization.

3. **Module 3: Differential Calculus, Derivatives & Gradient Descent (`content/calculus_gradients_module.json`)**:
   - The optimization landscape in the fog.
   - Derivative as local tangent slope and multivariable Gradient vector ($\nabla f$).
   - Live Loss Bowl Simulator ($f(x, y) = x^2 + 3y^2$) with customizable learning rate $\eta$, demonstrating stable convergence vs explosive overshoot when $\eta > \frac{1}{3}$.
   - 10-line vectorized NumPy loop.

4. **Module 4: The Chain Rule & Backpropagation Mechanics (`content/backprop_chainrule_module.json`)**:
   - Nested composite functions $L(y(h(w)))$.
   - Chain rule and multivariable fan-out gradient summation.
   - Interactive single-neuron forward activation caching ($x, z, a, L$) and backward $\delta$ error signal routing.
   - 4-line vectorized matrix backpropagation and vanishing gradient dynamics.

5. **Chapter Mastery Quizzes & Technical Interview Flashcard Deck (`content/quizzes_and_interviews.json`)**:
   - 16 module quiz questions with instant feedback and score tracking with celebratory confetti.
   - Technical interview flashcards with question categorization, answer reveals, and key takeaways.

6. **Static GitHub Pages (`github.io`) Deployment**:
   - Production-ready zero-dependency static build in `docs/` and `static_gh_pages/`.

---

## 🚀 Quickstart

### 1. Run FastAPI Microservice (Port 8008)
```bash
cd server
python -m uvicorn main:app --host 127.0.0.1 --port 8008
```

### 2. Run React 18 + Vite Interactive Textbook (Port 5181)
```bash
cd client
npm install
npm run dev
```

### 3. Run Automated Tests
```bash
python server/test_api.py
```
