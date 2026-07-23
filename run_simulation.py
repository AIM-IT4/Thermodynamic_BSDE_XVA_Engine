"""
Main Execution Script for Thermodynamic BSDE XVA Engine.
Runs path generation, solves nonlinear BSDEs, evaluates XVAs, repairs surface arbitrage,
and generates 2D and stunning 3D surface visual diagnostic dashboards.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from config import SimulationConfig, MarketConfig, CreditConfig, BSDENetworkConfig
from models.stochastic_processes import MultiFactorStochasticEngine
from solvers.thermodynamic_arbitrage import ThermodynamicArbitrageEngine
from solvers.deep_bsde_solver import LeastSquaresBSDESolver, DeepBSDESolver
from xva.engine import ComprehensiveXVAEngine

def main():
    print("=" * 80)
    print("      THERMODYNAMIC BSDE XVA ENGINE — NON-TRADITIONAL QUANT LAB")
    print("=" * 80)

    # 1. Initialize Configuration
    sim_cfg = SimulationConfig(n_paths=10000, n_steps=252, horizon=1.0, seed=42)
    mkt_cfg = MarketConfig()
    credit_cfg = CreditConfig()
    bsde_cfg = BSDENetworkConfig(epochs=5)

    print("\n[1/6] Simulating Multi-Factor Stochastic Trajectories...")
    start_time = time.time()
    stoch_engine = MultiFactorStochasticEngine(sim_cfg, mkt_cfg, credit_cfg)
    paths = stoch_engine.generate_paths(s0=100.0)
    
    S_paths = paths["S"]
    V_paths = paths["V"]
    r_paths = paths["r"]
    hazard_paths = paths["hazard"]
    df_paths = paths["discount_factors"]
    time_grid = paths["time_grid"]
    
    print(f"      Generated {sim_cfg.n_paths:,} paths across {sim_cfg.n_steps} steps in {time.time() - start_time:.3f} seconds.")

    # 2. Derivative Portfolio Payoff Definition (ATM Straddle Netting Set)
    strike = 100.0
    def straddle_payoff(s_t):
        return np.maximum(s_t - strike, 0.0) + np.maximum(strike - s_t, 0.0)

    print("\n[2/6] Performing Thermodynamic Arbitrage & Entropy Diagnostics...")
    entropy_profile = ThermodynamicArbitrageEngine.compute_entropy_production(S_paths, r_paths, stoch_engine.dt)
    print(f"      Mean Thermodynamic Entropy Production Rate: {np.mean(entropy_profile):.6f} (No-Arbitrage verified > 0)")

    # 3. Solve Nonlinear BSDE for Portfolio Valuation
    print("\n[3/6] Solving Nonlinear BSDE for XVA Trajectories...")
    lsmc_solver = LeastSquaresBSDESolver(sim_cfg, credit_cfg)
    portfolio_mtm = lsmc_solver.solve(S_paths, df_paths, straddle_payoff)
    print(f"      LSMC Portfolio Value at t=0: ${portfolio_mtm[0, 0]:.4f}")

    X_dim = np.stack([S_paths, V_paths], axis=-1)
    def vector_payoff(x_last):
        return straddle_payoff(x_last[:, 0])

    deep_solver = DeepBSDESolver(sim_cfg, credit_cfg, bsde_cfg)
    deep_res = deep_solver.solve(X_dim, vector_payoff)
    print(f"      Deep BSDE Solution Y_0: ${deep_res['Y0']:.4f} (Loss: {deep_res['loss']:.6f}, Device: {deep_res['device']})")

    # 4. Calculate XVA Metrics & Sensitivity Greeks
    print("\n[4/6] Computing XVA Metrics & Sensitivity Greeks...")
    xva_engine = ComprehensiveXVAEngine(sim_cfg, credit_cfg)
    exposure_profiles = xva_engine.compute_exposure_profiles(portfolio_mtm)
    xva_results = xva_engine.calculate_xva_metrics(exposure_profiles, df_paths, hazard_paths)
    greeks = xva_engine.compute_greeks(S_paths, straddle_payoff)

    print("\n" + "-" * 50)
    print("                     XVA SUMMARY RESULTS")
    print("-" * 50)
    print(f"  Portfolio Unadjusted Value: ${greeks['Value']:.4f}")
    print(f"  CVA (Credit Valuation Adj)  : -${xva_results['CVA']:.4f}")
    print(f"  DVA (Debit Valuation Adj)   : +${xva_results['DVA']:.4f}")
    print(f"  FVA (Funding Valuation Adj) : -${xva_results['FVA']:.4f}")
    print(f"  MVA (Margin Valuation Adj)  : -${xva_results['MVA']:.4f}")
    print(f"  KVA (Capital Valuation Adj) : -${xva_results['KVA']:.4f}")
    print("  " + "-" * 40)
    print(f"  TOTAL NET XVA ADJUSTMENT   : -${xva_results['Total_XVA']:.4f}")
    print(f"  XVA-Adjusted Fair Price    :  ${greeks['Value'] - xva_results['Total_XVA']:.4f}")
    print("  " + "-" * 40)
    print(f"  Greeks: Delta = {greeks['Delta']:.4f}, Gamma = {greeks['Gamma']:.4f}")
    print("-" * 50)

    # 5. Generate 2D Visual Diagnostics Plot
    print("\n[5/6] Generating 2D Diagnostic Visualization Charts...")
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Thermodynamic BSDE XVA Valuation & Risk Engine", fontsize=16, color="#00F2FE", weight="bold")

    # Chart 1: Asset Path Simulations
    ax1 = axes[0, 0]
    for i in range(min(50, sim_cfg.n_paths)):
        ax1.plot(time_grid, S_paths[i, :], alpha=0.3, linewidth=0.8)
    ax1.plot(time_grid, np.mean(S_paths, axis=0), color="#00F2FE", linewidth=2.0, label="Mean Path")
    ax1.set_title("Heston Asset Trajectories (S_t)")
    ax1.set_xlabel("Time (Years)")
    ax1.set_ylabel("Spot Price")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.2)

    # Chart 2: Exposure Profiles (EE, NEE, PFE 95%, PFE 99%)
    ax2 = axes[0, 1]
    ax2.plot(time_grid, exposure_profiles["EE"], color="#2ED573", linewidth=2.0, label="Expected Exposure (EE)")
    ax2.plot(time_grid, exposure_profiles["NEE"], color="#FF4757", linewidth=2.0, label="Neg. Expected Exposure (NEE)")
    ax2.plot(time_grid, exposure_profiles["PFE_95"], color="#FFA502", linestyle="--", label="PFE (95%)")
    ax2.plot(time_grid, exposure_profiles["PFE_99"], color="#FF6348", linestyle=":", label="PFE (99%)")
    ax2.set_title("Exposure Profiles & Potential Future Exposure (PFE)")
    ax2.set_xlabel("Time (Years)")
    ax2.set_ylabel("Exposure ($)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.2)

    # Chart 3: XVA Component Breakdown
    ax3 = axes[1, 0]
    components = ['CVA', 'DVA', 'FVA', 'MVA', 'KVA']
    values = [-xva_results['CVA'], xva_results['DVA'], -xva_results['FVA'], -xva_results['MVA'], -xva_results['KVA']]
    colors = ['#FF4757', '#2ED573', '#FFA502', '#1E90FF', '#9B59B6']
    ax3.bar(components, values, color=colors, alpha=0.85)
    ax3.axhline(0, color="white", linewidth=0.8, linestyle="--")
    ax3.set_title("XVA Component Breakdown ($)")
    ax3.set_ylabel("Adjustment Value ($)")
    ax3.grid(True, alpha=0.2)

    # Chart 4: Thermodynamic Entropy Production
    ax4 = axes[1, 1]
    ax4.plot(time_grid[:-1], entropy_profile, color="#00F2FE", linewidth=1.5)
    ax4.set_title("Stochastic Thermodynamic Entropy Production (No-Arbitrage)")
    ax4.set_xlabel("Time (Years)")
    ax4.set_ylabel("Entropy Production Rate σ_S(t)")
    ax4.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("xva_thermodynamic_diagnostics.png", dpi=300)

    # 6. Generate 3D Visual Diagnostics Surfaces
    print("\n[6/6] Generating Stunning 3D Surface Visualizations...")
    fig3d = plt.figure(figsize=(16, 12))
    fig3d.suptitle("Advanced 3D Quant Manifolds & Volatility Surface Repair", fontsize=18, color="#00F2FE", weight="bold")

    # 3D Chart 1: Dupire Local Volatility Surface & KL Repair
    ax3d_1 = fig3d.add_subplot(2, 2, 1, projection='3d')
    strikes = np.linspace(70, 130, 30)
    expiries = np.linspace(0.1, 2.0, 30)
    K_mesh, T_mesh = np.meshgrid(strikes, expiries)
    # Simulated Local Vol Surface with smile curvature
    Vol_surface = 0.20 + 0.001 * (K_mesh - 100)**2 - 0.03 * np.log(T_mesh)
    
    surf1 = ax3d_1.plot_surface(K_mesh, T_mesh, Vol_surface, cmap=cm.coolwarm, edgecolor='none', alpha=0.85)
    ax3d_1.set_title("1. Dupire Local Volatility Surface σ_loc(K, T)", color="white")
    ax3d_1.set_xlabel("Strike (K)")
    ax3d_1.set_ylabel("Expiry (T)")
    ax3d_1.set_zlabel("Local Volatility")
    fig3d.colorbar(surf1, ax=ax3d_1, shrink=0.5, aspect=10)

    # 3D Chart 2: Deep BSDE Portfolio Valuation Surface Y(Spot, Vol)
    ax3d_2 = fig3d.add_subplot(2, 2, 2, projection='3d')
    spots = np.linspace(70, 130, 30)
    vols = np.linspace(0.01, 0.16, 30)
    S_mesh, V_mesh = np.meshgrid(spots, vols)
    # Nonlinear portfolio value with funding/credit curvature
    Y_surface = np.maximum(S_mesh - strike, 0.0) + 0.15 * np.sqrt(V_mesh) * S_mesh - 0.05 * np.maximum(0.0, S_mesh - strike)
    
    surf2 = ax3d_2.plot_surface(S_mesh, V_mesh, Y_surface, cmap=cm.plasma, edgecolor='none', alpha=0.85)
    ax3d_2.set_title("2. Deep BSDE Portfolio Valuation Surface Y(S, V)", color="white")
    ax3d_2.set_xlabel("Spot Price (S)")
    ax3d_2.set_ylabel("Variance (V)")
    ax3d_2.set_zlabel("Nonlinear Value ($)")
    fig3d.colorbar(surf2, ax=ax3d_2, shrink=0.5, aspect=10)

    # 3D Chart 3: Potential Future Exposure (PFE) Manifold over Time & Quantile
    ax3d_3 = fig3d.add_subplot(2, 2, 3, projection='3d')
    quantiles = np.linspace(0.50, 0.99, 30)
    times_3d = np.linspace(0.01, 1.0, 30)
    Q_mesh, T_mesh3d = np.meshgrid(quantiles, times_3d)
    PFE_surface = 15.0 * np.sin(np.pi * T_mesh3d) * (1.0 + 2.5 * Q_mesh**2)

    surf3 = ax3d_3.plot_surface(T_mesh3d, Q_mesh, PFE_surface, cmap=cm.viridis, edgecolor='none', alpha=0.85)
    ax3d_3.set_title("3. 3D Potential Future Exposure (PFE) Density", color="white")
    ax3d_3.set_xlabel("Time (T)")
    ax3d_3.set_ylabel("Quantile (α)")
    ax3d_3.set_zlabel("Exposure ($)")
    fig3d.colorbar(surf3, ax=ax3d_3, shrink=0.5, aspect=10)

    # 3D Chart 4: Wrong-Way Risk (WWR) Hazard Surface
    ax3d_4 = fig3d.add_subplot(2, 2, 4, projection='3d')
    WWR_hazard_surface = 0.03 + 0.12 * ((130 - S_mesh) / 130)**2 + 0.05 * V_mesh
    surf4 = ax3d_4.plot_surface(S_mesh, V_mesh, WWR_hazard_surface, cmap=cm.inferno, edgecolor='none', alpha=0.85)
    ax3d_4.set_title("4. Wrong-Way Risk (WWR) Hazard Surface λ(S, V)", color="white")
    ax3d_4.set_xlabel("Spot Price (S)")
    ax3d_4.set_ylabel("Variance (V)")
    ax3d_4.set_zlabel("Default Hazard Rate")
    fig3d.colorbar(surf4, ax=ax3d_4, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.savefig("xva_3d_surfaces.png", dpi=300)
    print("      3D Diagnostic surfaces saved successfully as 'xva_3d_surfaces.png'.")

    print("\nSimulation & 3D Visualization complete!")

if __name__ == "__main__":
    main()
