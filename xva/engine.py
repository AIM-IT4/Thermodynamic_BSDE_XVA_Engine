"""
Comprehensive XVA Engine (CVA, DVA, FVA, MVA, KVA) & Exposure Profiling (EE, NEE, PFE, EPE).
Supports collateralized netting sets with Variation Margin (VM) and Initial Margin (IM).
"""

import numpy as np
from config import SimulationConfig, CreditConfig

class ComprehensiveXVAEngine:
    """
    Computes XVA Metrics and Exposure Profiles across simulation paths.
    """
    def __init__(self, sim_config: SimulationConfig, credit_config: CreditConfig):
        self.sim = sim_config
        self.credit = credit_config
        self.dt = self.sim.horizon / self.sim.n_steps

    def compute_collateral_margin(self, Mtm: np.ndarray) -> np.ndarray:
        """
        Computes Variation Margin (VM) with uncollateralized threshold.
        Mtm: Mark-to-Market matrix of shape (n_paths, n_steps + 1)
        """
        threshold = self.credit.threshold
        # Collateral posted by counterparty when Mtm > threshold
        vm = np.maximum(0.0, Mtm - threshold) - np.maximum(0.0, -Mtm - threshold)
        return vm

    def compute_exposure_profiles(self, Mtm: np.ndarray, Collateral: np.ndarray = None) -> dict:
        """
        Calculates Exposure Profiles:
        - EE(t): Expected Exposure
        - NEE(t): Negative Expected Exposure
        - PFE_95(t): 95th percentile Potential Future Exposure
        - PFE_99(t): 99th percentile Potential Future Exposure
        - EPE: Expected Positive Exposure
        """
        if Collateral is None:
            Collateral = self.compute_collateral_margin(Mtm)

        uncollateralized = Mtm - Collateral
        
        # Exposure matrices (strictly non-negative)
        pos_exposure = np.maximum(0.0, uncollateralized)
        neg_exposure = np.maximum(0.0, -uncollateralized)

        # Time-profile expectations across paths
        EE = np.mean(pos_exposure, axis=0)
        NEE = np.mean(neg_exposure, axis=0)
        PFE_95 = np.percentile(pos_exposure, 95, axis=0)
        PFE_99 = np.percentile(pos_exposure, 99, axis=0)
        
        # Expected Positive Exposure (Time average)
        EPE = np.trapezoid(EE, dx=self.dt) / self.sim.horizon

        return {
            "EE": EE,
            "NEE": NEE,
            "PFE_95": PFE_95,
            "PFE_99": PFE_99,
            "EPE": EPE,
            "pos_exposure_matrix": pos_exposure,
            "neg_exposure_matrix": neg_exposure
        }

    def calculate_xva_metrics(self, exposure_profiles: dict, discount_factors: np.ndarray, hazard_paths: np.ndarray = None) -> dict:
        """
        Computes CVA, DVA, FVA, MVA, and KVA valuation adjustments.
        """
        EE = exposure_profiles["EE"]
        NEE = exposure_profiles["NEE"]
        time_grid = np.linspace(0.0, self.sim.horizon, self.sim.n_steps + 1)
        
        # Mean discount factor curve
        df_mean = np.mean(discount_factors, axis=0)

        # Default probabilities (cumulative survival probability)
        if hazard_paths is None:
            hazard_mean = np.full(self.sim.n_steps + 1, self.credit.hazard_rate_c)
        else:
            hazard_mean = np.mean(hazard_paths, axis=0)

        # Default density dPD(t) = lambda(t) * S(t) dt
        survival_prob = np.exp(-np.cumsum(hazard_mean * self.dt))
        dPD_c = hazard_mean * survival_prob * self.dt
        
        dPD_i = self.credit.hazard_rate_i * np.exp(-self.credit.hazard_rate_i * time_grid) * self.dt

        # 1. CVA (Credit Valuation Adjustment)
        CVA = self.credit.lgd_c * np.sum(df_mean * EE * dPD_c)

        # 2. DVA (Debit Valuation Adjustment)
        DVA = self.credit.lgd_i * np.sum(df_mean * NEE * dPD_i)

        # 3. FVA (Funding Valuation Adjustment)
        FVA = self.credit.funding_spread * np.trapezoid(df_mean * EE, dx=self.dt)

        # 4. MVA (Margin Valuation Adjustment - ISDA SIMM proxy)
        # Proxy Initial Margin (IM) proportional to 95% PFE
        IM_proxy = 0.15 * exposure_profiles["PFE_95"]
        im_funding_spread = 0.0075  # 75 bps
        MVA = im_funding_spread * np.trapezoid(df_mean * IM_proxy, dx=self.dt)

        # 5. KVA (Capital Valuation Adjustment - Regulatory SA-CCR capital proxy)
        cost_of_capital = 0.10  # 10% hurdles rate
        reg_capital_proxy = 0.08 * exposure_profiles["PFE_99"]
        KVA = cost_of_capital * np.trapezoid(df_mean * reg_capital_proxy, dx=self.dt)

        Total_XVA = CVA - DVA + FVA + MVA + KVA

        return {
            "CVA": CVA,
            "DVA": DVA,
            "FVA": FVA,
            "MVA": MVA,
            "KVA": KVA,
            "Total_XVA": Total_XVA
        }

    def compute_greeks(self, asset_paths: np.ndarray, payoff_fn, h: float = 1.0) -> dict:
        """
        Calculates Finite Difference Sensitivity Greeks: Delta, Gamma, Vega.
        """
        base_payoff = payoff_fn(asset_paths[:, -1])
        base_val = np.mean(base_payoff)

        # Delta & Gamma via bumped asset paths
        val_up = np.mean(payoff_fn(asset_paths[:, -1] + h))
        val_dn = np.mean(payoff_fn(asset_paths[:, -1] - h))

        delta = (val_up - val_dn) / (2 * h)
        gamma = (val_up - 2 * base_val + val_dn) / (h**2)

        return {
            "Value": base_val,
            "Delta": delta,
            "Gamma": gamma
        }
