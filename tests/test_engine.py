"""
Unit Test Suite for Thermodynamic BSDE XVA Engine.
Verifies stochastic process bounds, martingale properties, and XVA metrics.
"""

import unittest
import numpy as np

from config import SimulationConfig, MarketConfig, CreditConfig
from models.stochastic_processes import MultiFactorStochasticEngine
from solvers.thermodynamic_arbitrage import ThermodynamicArbitrageEngine
from xva.engine import ComprehensiveXVAEngine

class TestThermodynamicXVAEngine(unittest.TestCase):

    def setUp(self):
        self.sim_cfg = SimulationConfig(n_paths=1000, n_steps=50, horizon=0.5, seed=123)
        self.mkt_cfg = MarketConfig()
        self.credit_cfg = CreditConfig()
        self.stoch_engine = MultiFactorStochasticEngine(self.sim_cfg, self.mkt_cfg, self.credit_cfg)
        self.paths = self.stoch_engine.generate_paths(s0=100.0)

    def test_stochastic_path_shapes(self):
        """Verify output matrices have correct shape (n_paths, n_steps + 1)"""
        S = self.paths["S"]
        self.assertEqual(S.shape, (1000, 51))
        self.assertTrue(np.all(S > 0), "Asset price S_t must be strictly positive.")

    def test_variance_positivity(self):
        """Verify Heston variance paths remain non-negative under full truncation"""
        V = self.paths["V"]
        self.assertTrue(np.all(V >= 0), "Heston variance V_t must be non-negative.")

    def test_entropy_production_positivity(self):
        """Verify thermodynamic entropy production rate is non-negative (No-Arbitrage)"""
        entropy = ThermodynamicArbitrageEngine.compute_entropy_production(self.paths["S"], self.paths["r"], self.stoch_engine.dt)
        self.assertTrue(np.all(entropy >= 0.0), "Entropy production rate must be non-negative under no-arbitrage.")

    def test_xva_metrics_non_negative(self):
        """Verify CVA and FVA metrics are non-negative for uncollateralized positive exposure"""
        portfolio_mtm = np.maximum(0.0, self.paths["S"] - 100.0)
        xva_engine = ComprehensiveXVAEngine(self.sim_cfg, self.credit_cfg)
        profiles = xva_engine.compute_exposure_profiles(portfolio_mtm)
        xva_res = xva_engine.calculate_xva_metrics(profiles, self.paths["discount_factors"])
        
        self.assertGreaterEqual(xva_res["CVA"], 0.0, "CVA must be non-negative.")
        self.assertGreaterEqual(xva_res["FVA"], 0.0, "FVA must be non-negative.")

if __name__ == '__main__':
    unittest.main()
