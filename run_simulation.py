"""Run the educational XVA simulation and generate evidence-backed diagnostics."""

from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from config import BSDENetworkConfig, CreditConfig, MarketConfig, SimulationConfig
from models.stochastic_processes import MultiFactorStochasticEngine
from solvers.deep_bsde_solver import DeepBSDESolver, LeastSquaresMonteCarloPricer
from solvers.thermodynamic_arbitrage import MarketDiagnostics, StaticArbitrageSurfaceRepair
from xva.engine import ComprehensiveXVAEngine


def main() -> None:
    """Execute a reproducible teaching run, not a production valuation."""
    print("=" * 80)
    print("XVA SIMULATION AND BSDE TEACHING LAB")
    print("Educational output only; not for trading, regulatory, or credit decisions.")
    print("=" * 80)

    sim_cfg = SimulationConfig(n_paths=5_000, n_steps=60, horizon=1.0, seed=42)
    mkt_cfg = MarketConfig()
    credit_cfg = CreditConfig()
    notional_per_spot_unit = 100_000.0
    strike = 100.0

    def straddle_payoff(spot: np.ndarray) -> np.ndarray:
        return notional_per_spot_unit * (
            np.maximum(spot - strike, 0.0) + np.maximum(strike - spot, 0.0)
        )

    print("\n[1/5] Simulating risk factors...")
    start_time = time.time()
    simulator = MultiFactorStochasticEngine(sim_cfg, mkt_cfg, credit_cfg)
    paths = simulator.generate_paths(s0=strike)
    print(
        f"      Generated {sim_cfg.n_paths:,} paths and {sim_cfg.n_steps} steps "
        f"in {time.time() - start_time:.2f}s."
    )

    print("\n[2/5] Pricing a clean teaching portfolio and computing collateral...")
    clean_pricer = LeastSquaresMonteCarloPricer(sim_cfg)
    clean_mtm = clean_pricer.solve(paths["S"], paths["discount_factors"], straddle_payoff)
    xva_engine = ComprehensiveXVAEngine(sim_cfg, credit_cfg)
    exposure_profiles = xva_engine.compute_exposure_profiles(clean_mtm)
    xva_results = xva_engine.calculate_xva_metrics(
        exposure_profiles,
        paths["discount_factors"],
        paths["hazard"],
        paths["issuer_hazard"],
    )
    greeks = xva_engine.compute_greeks(
        paths["S"],
        straddle_payoff,
        paths["discount_factors"],
    )

    print("\n[3/5] Reporting pathwise XVA measures...")
    print(f"      Clean Monte-Carlo value: USD {np.mean(clean_mtm[:, 0]):,.2f}")
    print(f"      CVA:                  -USD {xva_results['CVA']:,.2f}")
    print(f"      DVA:                  +USD {xva_results['DVA']:,.2f}")
    print(f"      FVA (simplified FCA): -USD {xva_results['FVA']:,.2f}")
    print(f"      MVA proxy:            -USD {xva_results['MVA_proxy']:,.2f}")
    print(f"      KVA proxy:            -USD {xva_results['KVA_proxy']:,.2f}")
    print(f"      Total XVA:            -USD {xva_results['Total_XVA']:,.2f}")
    print(f"      Discounted Delta/Gamma: {greeks['Delta']:.4f} / {greeks['Gamma']:.6f}")

    if os.getenv("RUN_DEEP_BSDE", "0") == "1":
        print("\n[Optional] Training the small Deep-BSDE teaching experiment...")
        deep_cfg = SimulationConfig(n_paths=512, n_steps=16, horizon=1.0, seed=7)
        deep_paths = MultiFactorStochasticEngine(deep_cfg, mkt_cfg, credit_cfg).generate_paths(s0=strike)
        deep_xva = ComprehensiveXVAEngine(deep_cfg, credit_cfg)
        deep_collateral = deep_xva.compute_exposure_profiles(
            LeastSquaresMonteCarloPricer(deep_cfg).solve(
                deep_paths["S"],
                deep_paths["discount_factors"],
                straddle_payoff,
            )
        )["Collateral"]
        deep_result = DeepBSDESolver(
            deep_cfg,
            credit_cfg,
            BSDENetworkConfig(epochs=60, batch_size=256),
        ).solve(
            deep_paths["factors"],
            deep_paths["dW"],
            lambda factors: straddle_payoff(np.exp(factors[:, 0])),
            deep_paths["r"],
            deep_paths["hazard"],
            deep_collateral,
        )
        print(
            f"      Experimental Deep-BSDE Y0: USD {deep_result['Y0']:,.2f}; "
            f"terminal loss: {deep_result['loss']:.4f}."
        )
    else:
        print("\n      Deep-BSDE experiment skipped. Set RUN_DEEP_BSDE=1 to run it.")

    print("\n[4/5] Creating 2D diagnostics from simulated outputs...")
    output_dir = Path(__file__).resolve().parent
    time_grid = paths["time_grid"]
    dispersion = MarketDiagnostics.compute_excess_return_dispersion(
        paths["S"],
        paths["r"],
        simulator.dt,
    )

    plt.style.use("dark_background")
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    figure.suptitle("Educational Pathwise XVA Diagnostics", fontsize=16, color="#00F2FE", weight="bold")

    for path_index in range(50):
        axes[0, 0].plot(time_grid, paths["S"][path_index], alpha=0.25, linewidth=0.7)
    axes[0, 0].plot(time_grid, np.mean(paths["S"], axis=0), color="#00F2FE", linewidth=2.0, label="Mean spot")
    axes[0, 0].set_title("Log-Euler Heston-style spot paths")
    axes[0, 0].set_xlabel("Time (years)")
    axes[0, 0].set_ylabel("Spot")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.2)

    axes[0, 1].plot(time_grid, exposure_profiles["EE"], color="#2ED573", linewidth=2.0, label="EE")
    axes[0, 1].plot(time_grid, exposure_profiles["NEE"], color="#FF4757", linewidth=2.0, label="NEE")
    axes[0, 1].plot(time_grid, exposure_profiles["PFE_95"], color="#FFA502", linestyle="--", label="PFE 95%")
    axes[0, 1].plot(time_grid, exposure_profiles["PFE_99"], color="#FF6348", linestyle=":", label="PFE 99%")
    axes[0, 1].set_title("Current collateralised exposure profiles")
    axes[0, 1].set_xlabel("Time (years)")
    axes[0, 1].set_ylabel("Exposure (USD)")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.2)

    components = ["CVA", "DVA", "FVA", "MVA proxy", "KVA proxy"]
    values = [
        -xva_results["CVA"],
        xva_results["DVA"],
        -xva_results["FVA"],
        -xva_results["MVA_proxy"],
        -xva_results["KVA_proxy"],
    ]
    axes[1, 0].bar(components, values, color=["#FF4757", "#2ED573", "#FFA502", "#1E90FF", "#9B59B6"])
    axes[1, 0].axhline(0.0, color="white", linewidth=0.8, linestyle="--")
    axes[1, 0].set_title("Pathwise XVA with labelled proxies")
    axes[1, 0].set_ylabel("Adjustment (USD)")
    axes[1, 0].grid(alpha=0.2)

    axes[1, 1].plot(time_grid[:-1], dispersion, color="#00F2FE", linewidth=1.5)
    axes[1, 1].set_title("Excess log-return dispersion (descriptive only)")
    axes[1, 1].set_xlabel("Time (years)")
    axes[1, 1].set_ylabel("Mean squared excess log return")
    axes[1, 1].grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(output_dir / "xva_thermodynamic_diagnostics.png", dpi=220)
    plt.close(figure)

    print("\n[5/5] Creating 3D diagnostics from actual simulated or repaired data...")
    strikes = np.linspace(70.0, 130.0, 13)
    expiries = np.linspace(0.25, 2.0, 10)
    strike_mesh, expiry_mesh = np.meshgrid(strikes, expiries)
    raw_iv = 0.17 + 0.00010 * (strike_mesh - strike) ** 2 + 0.03 * np.exp(-expiry_mesh)
    raw_iv[0, 6] *= 0.55
    raw_iv[1, 8] *= 0.65
    surface_repair = StaticArbitrageSurfaceRepair(spot=strike, rate=mkt_cfg.r0)
    repaired_surface = surface_repair.repair(strikes, expiries, raw_iv)
    print(f"      Static-surface minimum slacks: {repaired_surface.diagnostics}")

    figure_3d = plt.figure(figsize=(16, 12))
    figure_3d.suptitle(
        "Educational XVA Data Surfaces (No Production or No-Arbitrage Claim)",
        fontsize=16,
        color="#00F2FE",
        weight="bold",
    )

    axis_1 = figure_3d.add_subplot(2, 2, 1, projection="3d")
    axis_1.plot_surface(
        strike_mesh,
        expiry_mesh,
        repaired_surface.repaired_implied_vol,
        cmap=cm.coolwarm,
        edgecolor="none",
        alpha=0.85,
    )
    axis_1.plot_wireframe(strike_mesh, expiry_mesh, raw_iv, color="white", linewidth=0.35, alpha=0.45)
    axis_1.set_title("Static call-price projection (surface = repaired; wire = raw)")
    axis_1.set_xlabel("Strike")
    axis_1.set_ylabel("Expiry")
    axis_1.set_zlabel("Implied volatility")

    sample_index = np.linspace(0, sim_cfg.n_paths - 1, 1_200, dtype=int)
    mid_step = sim_cfg.n_steps // 2
    axis_2 = figure_3d.add_subplot(2, 2, 2, projection="3d")
    axis_2.plot_trisurf(
        paths["S"][sample_index, mid_step],
        paths["V"][sample_index, mid_step],
        clean_mtm[sample_index, mid_step],
        cmap=cm.plasma,
        linewidth=0.15,
        alpha=0.88,
    )
    axis_2.set_title("Actual clean-LSMC MTM samples at mid-horizon")
    axis_2.set_xlabel("Spot")
    axis_2.set_ylabel("Variance")
    axis_2.set_zlabel("Clean MTM (USD)")

    quantiles = np.linspace(0.50, 0.99, 20)
    pfe_surface = np.quantile(exposure_profiles["pos_exposure_matrix"], quantiles, axis=0)
    time_mesh, quantile_mesh = np.meshgrid(time_grid, quantiles)
    axis_3 = figure_3d.add_subplot(2, 2, 3, projection="3d")
    axis_3.plot_surface(time_mesh, quantile_mesh, pfe_surface, cmap=cm.viridis, edgecolor="none", alpha=0.88)
    axis_3.set_title("Actual positive-exposure quantiles")
    axis_3.set_xlabel("Time")
    axis_3.set_ylabel("Quantile")
    axis_3.set_zlabel("Exposure (USD)")

    axis_4 = figure_3d.add_subplot(2, 2, 4, projection="3d")
    axis_4.plot_trisurf(
        paths["S"][sample_index, mid_step],
        paths["V"][sample_index, mid_step],
        paths["hazard"][sample_index, mid_step],
        cmap=cm.inferno,
        linewidth=0.15,
        alpha=0.88,
    )
    axis_4.set_title("Actual bounded state-dependent intensity samples")
    axis_4.set_xlabel("Spot")
    axis_4.set_ylabel("Variance")
    axis_4.set_zlabel("Counterparty intensity")

    figure_3d.tight_layout()
    figure_3d.savefig(output_dir / "xva_3d_surfaces.png", dpi=220)
    plt.close(figure_3d)
    print("      Replaced diagnostic images with data-backed charts.")


if __name__ == "__main__":
    main()
