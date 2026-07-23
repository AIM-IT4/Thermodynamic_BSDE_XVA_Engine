# Thermodynamic BSDE XVA Engine ⚡📊

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_BSDE-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Quantitative Finance](https://img.shields.io/badge/Domain-XVA_%26_Derivatives-blue?style=flat-square)](https://github.com/AIM-IT4)

A novel, production-grade Quantitative Finance Engine implementing **Stochastic Thermodynamics**, **High-Dimensional Deep Backward Stochastic Differential Equations (Deep BSDEs)**, and **Optimal Transport Volatility Surface Repair** for nonlinear multi-curve XVA (CVA, DVA, FVA, MVA, KVA) valuation.

---

## ⚡ Why This Is NOT a Generic / Traditional Quant Project

Most online quantitative finance projects rely on simple Black-Scholes formulas or basic Monte Carlo loops without credit, funding, or margin frictions. This repository builds a **non-traditional, research-grade front-office lab**:

| Dimension | Generic / Textbook Quant Project | ⚡ Thermodynamic BSDE XVA Engine |
| :--- | :--- | :--- |
| **Market Microstructure** | Assumes frictionless markets & constant volatility. | **Stochastic Thermodynamics**: Evaluates path-wise entropy production rates $\sigma_S(t) \ge 0$ to verify compliance with the 2nd law of quantitative thermodynamics (No-Arbitrage). |
| **Valuation Framework** | Linear expectation $\mathbb{E}[e^{-rT} g(S_T)]$. | **Nonlinear BSDE Solver**: Solves coupled Forward-Backward Stochastic Differential Equations using PyTorch neural networks for uncollateralized funding spreads & margin drag. |
| **Volatility Dynamics** | Constant or static implied volatility. | **3D Dupire Surface & Optimal Transport Repair**: Projects raw volatility matrices onto non-arbitrage local variance manifolds via KL-divergence optimization. |
| **Credit & Risk Coupling** | Independent constant default probabilities. | **Wrong-Way Risk (WWR)**: Nonlinearly couples default hazard rate $\lambda(S, V)$ to asset drawdowns and stochastic volatility. |
| **XVA Coverage** | Basic CVA only (or none). | **Full Front-Office Suite**: CVA, DVA, FVA, MVA (ISDA SIMM proxy), and KVA (Regulatory SA-CCR proxy). |

---

## 🔬 Mathematical Architecture

Under counterparty credit risk, margin requirements (VM & IM), and uncollateralized borrowing spreads ($r^b > r^l$), portfolio value $Y_t$ satisfies a **Coupled Nonlinear BSDE**:

$$-dY_t = f(t, X_t, Y_t, Z_t) \, dt - Z_t \cdot dW_t, \quad Y_T = g(X_T)$$

### 1. Coupled Nonlinear XVA Driver $f(t, X_t, Y_t, Z_t)$
$$f(t, X_t, Y_t, Z_t) = -r_t Y_t + \lambda_t^C (1-R_C) (Y_t - M_t)^+ - \lambda_t^I (1-R_I) (Y_t - M_t)^- + s_F (Y_t - M_t)^+ + s_{\text{IM}} \text{IM}_t$$

### 2. Stochastic Thermodynamics Entropy Metric
Following non-equilibrium thermodynamics, financial arbitrage opportunities correspond to negative entropy production:
$$\sigma_S(t) = \frac{(\mu(t) - r(t))^2}{\sigma^2(t)} \ge 0$$

---

## 🎨 3D Manifolds & Surface Visualizations

The engine generates stunning 3D surface diagnostic plots (`xva_3d_surfaces.png`):

1. **3D Dupire Local Volatility Surface $\sigma_{\text{loc}}(K, T)$**: Visualizes the volatility smile and Optimal Transport KL-repair manifold.
2. **3D Deep BSDE Portfolio Value Surface $Y(S, V)$**: Captures the nonlinear valuation curvature induced by asymmetric borrowing spreads across Spot Price $S$ and Heston Variance $V$.
3. **3D Potential Future Exposure (PFE) Density**: Time evolution of uncollateralized exposure across all confidence quantiles $\alpha \in [50\%, 99\%]$.
4. **3D Wrong-Way Risk (WWR) Hazard Surface $\lambda(S, V)$**: Non-linear coupling between market drawdowns and default intensity.

---

## 🛠️ Project Structure

```
Thermodynamic_BSDE_XVA_Engine/
│
├── config.py                     # Simulation, market, credit & neural net hyperparams
├── models/
│   └── stochastic_processes.py   # Heston, Sinusoidal Hull-White, and WWR default processes
├── solvers/
│   ├── deep_bsde_solver.py       # PyTorch Deep BSDE Neural Net & LSMC solvers
│   └── thermodynamic_arbitrage.py# Entropy production & KL volatility surface repair
├── xva/
│   └── engine.py                 # CVA, DVA, FVA, MVA, KVA & Exposure Profiling (EE, PFE)
├── tests/
│   └── test_engine.py            # Automated unit test suite (100% Passed)
├── run_simulation.py             # Main execution entry point & 2D/3D diagnostic generator
└── requirements.txt              # Dependencies
```

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/AIM-IT4/Thermodynamic_BSDE_XVA_Engine.git
cd Thermodynamic_BSDE_XVA_Engine
pip install -r requirements.txt
```

### 2. Execute Engine & Generate 3D Visualizations
```bash
python run_simulation.py
```

### 3. Run Automated Tests
```bash
python -m unittest discover tests
```

---

## 📜 License
MIT License. Created by [Amit Kumar Jha](https://github.com/AIM-IT4) (Founder @ [Desk2Quant](https://desk2quant.vercel.app)).
