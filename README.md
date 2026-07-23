# Thermodynamic BSDE XVA Engine ⚡📊

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_BSDE-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Quantitative Finance](https://img.shields.io/badge/Domain-XVA_%26_Derivatives-blue?style=flat-square)](https://github.com/AIM-IT4)

A novel, production-grade Quantitative Finance Engine implementing **Stochastic Thermodynamics**, **High-Dimensional Deep Backward Stochastic Differential Equations (Deep BSDEs)**, and **Optimal Transport Volatility Surface Repair** for nonlinear multi-curve XVA (CVA, DVA, FVA, MVA, KVA) valuation.

---

## 🔬 Mathematical Architecture & Novelty

Classical valuation frameworks assume linear, frictionless markets where Black-Scholes risk-neutral expectations hold. In modern front-office trading desks, counterparty credit risk, asymmetric borrowing/lending rates, margin requirements (VM/IM), and regulatory capital break linearity, yielding a **coupled nonlinear Backward Stochastic Differential Equation (BSDE)**:

$$-dY_t = f(t, X_t, Y_t, Z_t) \, dt - Z_t \cdot dW_t, \quad Y_T = g(X_T)$$

### 1. Coupled Nonlinear XVA Driver $f(t, X_t, Y_t, Z_t)$
The generator function incorporates nonlinear credit and funding friction costs:
$$f(t, X_t, Y_t, Z_t) = -r_t Y_t + \lambda_t^C (1-R_C) (Y_t - M_t)^+ - \lambda_t^I (1-R_I) (Y_t - M_t)^- + s_F (Y_t - M_t)^+ + s_{\text{IM}} \text{IM}_t$$

### 2. Stochastic Thermodynamics Entropy Metric (No-Arbitrage)
Following non-equilibrium stochastic thermodynamics, financial arbitrage opportunities correspond to negative entropy production. We compute the path-wise entropy production rate:
$$\sigma_S(t) = \frac{(\mu(t) - r(t))^2}{\sigma^2(t)} \ge 0$$
Strict positivity $\sigma_S(t) > 0$ verifies compliance with the 2nd law of quantitative thermodynamics across path simulations.

### 3. Optimal Transport Volatility Surface Repair
Raw implied volatility surfaces often exhibit local butterfly or calendar arbitrage ($\sigma_{\text{loc}}^2(K, T) < 0$). We project raw surfaces onto non-arbitrage manifolds via Kullback-Leibler (KL) divergence minimization subject to Dupire non-negativity constraints:
$$\min_{\sigma_{\text{loc}}} \int D_{\text{KL}}(\sigma_{\text{loc}} \| \sigma_{\text{mkt}}) \, dK \, dT \quad \text{s.t. } \frac{\partial C}{\partial T} + r K \frac{\partial C}{\partial K} - \frac{1}{2} \sigma_{\text{loc}}^2 K^2 \frac{\partial^2 C}{\partial K^2} = 0$$

---

## 🛠️ Project Structure

```
Thermodynamic_BSDE_XVA_Engine/
│
├── config.py                     # Simulation, market, credit & neural net parameters
├── models/
│   └── stochastic_processes.py   # Heston, Sinusoidal Hull-White, and WWR default processes
├── solvers/
│   ├── deep_bsde_solver.py       # PyTorch Deep BSDE Neural Net & LSMC solvers
│   └── thermodynamic_arbitrage.py# Entropy production & KL volatility surface repair
├── xva/
│   └── engine.py                 # CVA, DVA, FVA, MVA, KVA & Exposure Profiling (EE, PFE)
├── tests/
│   └── test_engine.py            # Automated unit test suite
├── run_simulation.py             # Main execution entry point & visual diagnostic plot generator
└── requirements.txt              # Dependencies
```

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/AIM-IT4/Thermodynamic_BSDE_XVA_Engine.git
cd Thermodynamic_BSDE_XVA_Engine
pip install -r requirements.txt
```

### 2. Run Main Simulation & Diagnostics
Execute the full engine pipeline to run Monte Carlo simulations, solve the nonlinear BSDE, calculate XVA metrics, and generate visual diagnostic plots:
```bash
python run_simulation.py
```

### 3. Run Unit Tests
```bash
python -m unittest discover tests
```

---

## 📊 Sample Visual Output

The engine automatically generates `xva_thermodynamic_diagnostics.png` containing:
1. **Asset Path Trajectories**: Correlated Heston stochastic volatility & Sinusoidal short-rate paths.
2. **Exposure Profiles**: Time evolution of Expected Exposure (EE), Negative Expected Exposure (NEE), and 95%/99% Potential Future Exposure (PFE).
3. **XVA Component Breakdown**: Bar chart breakdown of CVA, DVA, FVA, MVA, and KVA.
4. **Thermodynamic Entropy Rate**: Path-wise entropy production confirming non-negative arbitrage dynamics.

---

## 📄 References & Theoretical Foundation
1. **Burgard, C., & Kjaer, M. (2013)**. *In the Balance: Funding and Valuation Adjustments*. Risk Magazine.
2. **E, W., Han, J., & Jentzen, A. (2017)**. *Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations*. Communications in Mathematics and Statistics.
3. **Jha, A. K. (2025)**. *A Stochastic Thermodynamics Approach to Price Impact and Arbitrage*. arXiv:2512.03123.

---

## 📜 License
MIT License. Created by [Amit Kumar Jha](https://github.com/AIM-IT4) (Founder @ [Desk2Quant](https://desk2quant.vercel.app)).
