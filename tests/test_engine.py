"""Regression and benchmark tests for the educational XVA lab."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from config import BSDENetworkConfig, CreditConfig, MarketConfig, SimulationConfig
from models.stochastic_processes import MultiFactorStochasticEngine
from solvers.deep_bsde_solver import DeepBSDESolver, LeastSquaresMonteCarloPricer
from solvers.surface_repair import MarketDiagnostics, StaticArbitrageSurfaceRepair
from xva.engine import ComprehensiveXVAEngine


class TestEducationalXVALab(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = SimulationConfig(n_paths=256, n_steps=8, horizon=0.5, seed=123)
        self.market = MarketConfig()
        self.credit = CreditConfig(
            threshold=5.0,
            minimum_transfer_amount=0.0,
            margin_period_of_risk=1.0 / 252.0,
        )
        self.paths = MultiFactorStochasticEngine(
            self.sim,
            self.market,
            self.credit,
        ).generate_paths(s0=100.0)

    @staticmethod
    def _profiles(
        positive: np.ndarray,
        negative: np.ndarray | None = None,
        mpor_positive: np.ndarray | None = None,
        mpor_negative: np.ndarray | None = None,
    ) -> dict[str, np.ndarray | float]:
        if negative is None:
            negative = np.zeros_like(positive)
        if mpor_positive is None:
            mpor_positive = positive
        if mpor_negative is None:
            mpor_negative = negative
        return {
            "EE": np.mean(positive, axis=0),
            "NEE": np.mean(negative, axis=0),
            "PFE_95": np.quantile(positive, 0.95, axis=0),
            "PFE_99": np.quantile(positive, 0.99, axis=0),
            "EPE": 0.0,
            "pos_exposure_matrix": positive,
            "neg_exposure_matrix": negative,
            "mpor_pos_exposure_matrix": mpor_positive,
            "mpor_neg_exposure_matrix": mpor_negative,
        }

    def test_paths_are_reproducible_positive_and_discounted_from_one(self) -> None:
        repeat = MultiFactorStochasticEngine(
            self.sim,
            self.market,
            self.credit,
        ).generate_paths(s0=100.0)

        self.assertEqual(self.paths["S"].shape, (256, 9))
        self.assertEqual(self.paths["dW"].shape, (256, 8, 3))
        self.assertTrue(np.all(self.paths["S"] > 0.0))
        self.assertTrue(np.allclose(self.paths["discount_factors"][:, 0], 1.0))
        self.assertTrue(np.allclose(self.paths["S"], repeat["S"]))
        self.assertTrue(np.allclose(self.paths["dW"], repeat["dW"]))

    def test_log_spot_scheme_remains_positive_under_stress(self) -> None:
        stressed_paths = MultiFactorStochasticEngine(
            SimulationConfig(n_paths=10_000, n_steps=1, horizon=1.0, seed=11),
            MarketConfig(v0=4.0, theta_v=4.0, sigma_v=0.0),
            CreditConfig(),
        ).generate_paths()
        self.assertTrue(np.all(stressed_paths["S"] > 0.0))

    def test_collateral_call_frequency_mta_and_mpor(self) -> None:
        sim = SimulationConfig(n_paths=2, n_steps=4, horizon=4.0 / 252.0, seed=1)
        credit = CreditConfig(
            threshold=10.0,
            minimum_transfer_amount=5.0,
            collateral_call_frequency_days=2,
            margin_period_of_risk=2.0 / 252.0,
        )
        engine = ComprehensiveXVAEngine(sim, credit)
        mtm = np.array(
            [
                [0.0, 20.0, 12.0, 30.0, 25.0],
                [0.0, -20.0, -12.0, -30.0, -25.0],
            ]
        )
        collateral = engine.compute_collateral_margin(mtm)
        profiles = engine.compute_exposure_profiles(mtm, collateral)

        self.assertTrue(np.allclose(collateral[:, :4], 0.0))
        self.assertTrue(np.allclose(collateral[:, 4], [15.0, -15.0]))
        self.assertAlmostEqual(profiles["mpor_pos_exposure_matrix"][0, 0], 12.0)
        self.assertAlmostEqual(profiles["mpor_neg_exposure_matrix"][1, 0], 12.0)

    def test_one_period_cva_matches_first_to_default_benchmark(self) -> None:
        sim = SimulationConfig(n_paths=1, n_steps=1, horizon=1.0, seed=1)
        credit = CreditConfig(
            hazard_rate_c=0.03,
            hazard_rate_i=0.0,
            lgd_c=0.60,
            funding_spread=0.0,
            im_funding_spread=0.0,
            cost_of_capital=0.0,
        )
        engine = ComprehensiveXVAEngine(sim, credit)
        positive = np.ones((1, 2))
        results = engine.calculate_xva_metrics(
            self._profiles(positive),
            np.ones((1, 2)),
        )
        expected_cva = credit.lgd_c * (1.0 - np.exp(-credit.hazard_rate_c))
        self.assertAlmostEqual(results["CVA"], expected_cva, places=12)

    def test_pathwise_wwr_is_preserved(self) -> None:
        sim = SimulationConfig(n_paths=2, n_steps=1, horizon=1.0, seed=1)
        credit = CreditConfig(
            hazard_rate_c=0.0,
            hazard_rate_i=0.0,
            funding_spread=0.0,
            im_funding_spread=0.0,
            cost_of_capital=0.0,
        )
        engine = ComprehensiveXVAEngine(sim, credit)
        positive = np.array([[10.0, 10.0], [0.0, 0.0]])
        profiles = self._profiles(positive)
        aligned_hazard = np.array([[0.50, 0.50], [0.01, 0.01]])
        anti_aligned_hazard = np.array([[0.01, 0.01], [0.50, 0.50]])

        aligned = engine.calculate_xva_metrics(profiles, np.ones((2, 2)), aligned_hazard)
        anti_aligned = engine.calculate_xva_metrics(profiles, np.ones((2, 2)), anti_aligned_hazard)
        self.assertGreater(aligned["CVA"], anti_aligned["CVA"])

    def test_lsmc_returns_clean_terminal_mtm(self) -> None:
        payoff = lambda spot: np.maximum(spot - 100.0, 0.0)
        values = LeastSquaresMonteCarloPricer(self.sim).solve(
            self.paths["S"],
            self.paths["discount_factors"],
            payoff,
        )
        self.assertTrue(np.allclose(values[:, -1], payoff(self.paths["S"][:, -1])))
        self.assertTrue(np.all(np.isfinite(values)))

    def test_spot_greeks_are_initial_spot_sensitivities(self) -> None:
        engine = ComprehensiveXVAEngine(self.sim, self.credit)
        linear_payoff = lambda spot: 2.5 * spot
        greeks = engine.compute_greeks(
            self.paths["S"],
            linear_payoff,
            self.paths["discount_factors"],
            h=1.0,
        )
        initial_spot = self.paths["S"][0, 0]
        # A linear claim has Delta = V / S0 and zero Gamma under an initial-spot
        # bump; an additive terminal-spot bump would not satisfy this identity.
        self.assertAlmostEqual(greeks["Delta"], greeks["Value"] / initial_spot, places=6)
        self.assertAlmostEqual(greeks["Gamma"], 0.0, places=6)

    def test_lsmc_state_augmented_basis_changes_fit(self) -> None:
        payoff = lambda spot: np.maximum(spot - 100.0, 0.0)
        pricer = LeastSquaresMonteCarloPricer(self.sim)
        spot_only = pricer.solve(self.paths["S"], self.paths["discount_factors"], payoff)
        state = np.stack([self.paths["V"], self.paths["r"]], axis=-1)
        augmented = pricer.solve(
            self.paths["S"],
            self.paths["discount_factors"],
            payoff,
            state_paths=state,
        )
        self.assertTrue(np.allclose(augmented[:, -1], payoff(self.paths["S"][:, -1])))
        self.assertFalse(np.allclose(spot_only, augmented))

    def test_market_diagnostic_is_finite_and_descriptive(self) -> None:
        diagnostic = MarketDiagnostics.compute_excess_return_dispersion(
            self.paths["S"],
            self.paths["r"],
            self.sim.horizon / self.sim.n_steps,
        )
        self.assertEqual(diagnostic.shape, (self.sim.n_steps,))
        self.assertTrue(np.all(np.isfinite(diagnostic)))
        self.assertTrue(np.all(diagnostic >= 0.0))

    def test_static_surface_repair_enforces_constraints_and_uses_strikes(self) -> None:
        expiries = np.array([0.5, 1.0])
        raw_iv = np.array([[0.60, 0.05, 0.60], [0.05, 0.70, 0.05]])
        repair = StaticArbitrageSurfaceRepair(spot=100.0, rate=0.02)

        result = repair.repair(np.array([80.0, 100.0, 120.0]), expiries, raw_iv)
        diagnostics = result.diagnostics
        self.assertGreaterEqual(diagnostics["min_monotonicity_slack"], -1e-7)
        self.assertGreaterEqual(diagnostics["min_convexity_slack"], -1e-7)
        self.assertGreaterEqual(diagnostics["min_calendar_slack"], -1e-7)
        self.assertGreaterEqual(diagnostics["min_vertical_spread_upper_slack"], -1e-7)
        vertical_spread = (
            result.repaired_call_prices[:, :-1] - result.repaired_call_prices[:, 1:]
        )
        vertical_upper = np.exp(-0.02 * expiries)[:, np.newaxis] * 20.0
        self.assertTrue(np.all(vertical_spread <= vertical_upper + 1e-7))

        shifted = repair.repair(np.array([70.0, 100.0, 140.0]), expiries, raw_iv)
        self.assertFalse(np.allclose(result.repaired_call_prices, shifted.repaired_call_prices))

    def test_deep_bsde_driver_uses_credit_configuration(self) -> None:
        sim = SimulationConfig(n_paths=4, n_steps=1, horizon=1.0, seed=2)
        low_credit = DeepBSDESolver(
            sim,
            CreditConfig(hazard_rate_c=0.0, funding_spread=0.0),
            BSDENetworkConfig(epochs=1, batch_size=4, hidden_dim=4),
        )
        high_credit = DeepBSDESolver(
            sim,
            CreditConfig(hazard_rate_c=0.50, funding_spread=0.50),
            BSDENetworkConfig(epochs=1, batch_size=4, hidden_dim=4),
        )
        value = torch.tensor([1.0])
        rate = torch.tensor([0.0])
        hazard = torch.tensor([0.50])
        collateral = torch.tensor([0.0])
        self.assertLess(
            high_credit.driver(value, rate, hazard, collateral).item(),
            low_credit.driver(value, rate, hazard, collateral).item(),
        )

    def test_small_deep_bsde_run_consumes_simulated_brownian_increments(self) -> None:
        sim = SimulationConfig(n_paths=32, n_steps=1, horizon=0.25, seed=4)
        paths = MultiFactorStochasticEngine(sim, MarketConfig(), CreditConfig()).generate_paths()
        solver = DeepBSDESolver(
            sim,
            CreditConfig(),
            BSDENetworkConfig(hidden_dim=4, n_layers=1, epochs=2, batch_size=32),
        )
        result = solver.solve(
            paths["factors"],
            paths["dW"],
            lambda factors: np.maximum(np.exp(factors[:, 0]) - 100.0, 0.0),
            paths["r"],
            paths["hazard"],
        )
        self.assertTrue(np.isfinite(result["Y0"]))
        self.assertTrue(np.isfinite(result["loss"]))
        self.assertTrue(np.isfinite(result["rmse"]))
        self.assertGreaterEqual(result["relative_rmse"], 0.0)
        self.assertIsInstance(result["converged"], bool)


if __name__ == "__main__":
    unittest.main()
