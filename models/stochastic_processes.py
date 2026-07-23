"""
Stochastic Process Generators for Multi-Asset, Short Rate, and Default Intensity dynamics.
Implements Heston Stochastic Volatility, Sinusoidal Hull-White Short Rate, and Wrong-Way Risk (WWR).
"""

import numpy as np
from config import SimulationConfig, MarketConfig, CreditConfig

class MultiFactorStochasticEngine:
    """
    Simulates joint paths for:
    1. Asset price process S_t (Heston Stochastic Volatility)
    2. Variance process V_t
    3. Short rate process r_t (Extended Hull-White with Sinusoidal Mean Reversion)
    4. Default intensity process lambda_t (Wrong-Way Risk coupled to asset drawdown)
    """
    
    def __init__(self, sim_config: SimulationConfig, mkt_config: MarketConfig, credit_config: CreditConfig):
        self.sim = sim_config
        self.mkt = mkt_config
        self.credit = credit_config
        
        self.dt = self.sim.horizon / self.sim.n_steps
        self.time_grid = np.linspace(0.0, self.sim.horizon, self.sim.n_steps + 1)
        
    def generate_paths(self, s0: float = 100.0) -> dict:
        """
        Generates correlated simulation paths for all risk factors.
        Returns dictionary of path matrices of shape (n_paths, n_steps + 1).
        """
        np.random.seed(self.sim.seed)
        n_paths = self.sim.n_paths
        n_steps = self.sim.n_steps
        dt = self.dt
        sqrt_dt = np.sqrt(dt)
        
        # Initialize arrays
        S = np.zeros((n_paths, n_steps + 1))
        V = np.zeros((n_paths, n_steps + 1))
        r = np.zeros((n_paths, n_steps + 1))
        hazard = np.zeros((n_paths, n_steps + 1))
        
        S[:, 0] = s0
        V[:, 0] = self.mkt.v0
        r[:, 0] = self.mkt.r0
        hazard[:, 0] = self.credit.hazard_rate_c
        
        # Correlation matrix construction (S, V, r)
        # rho_SV = leverage effect, rho_Sr = equity-rate correlation
        rho_sv = self.mkt.rho_sv
        rho_sr = 0.15
        rho_vr = -0.10
        
        corr_matrix = np.array([
            [1.0,     rho_sv,  rho_sr],
            [rho_sv,  1.0,     rho_vr],
            [rho_sr,  rho_vr,  1.0]
        ])
        
        chol = np.linalg.cholesky(corr_matrix)
        
        for t in range(n_steps):
            t_curr = self.time_grid[t]
            
            # Independent Gaussian draws
            Z_ind = np.random.normal(0.0, 1.0, (n_paths, 3))
            # Correlated Gaussian draws
            Z = Z_ind @ chol.T
            
            Z_s = Z[:, 0]
            Z_v = Z[:, 1]
            Z_r = Z[:, 2]
            
            # 1. Variance process (Heston full truncation)
            v_curr = np.maximum(V[:, t], 0.0)
            dV = self.mkt.kappa_v * (self.mkt.theta_v - v_curr) * dt + \
                 self.mkt.sigma_v * np.sqrt(v_curr) * sqrt_dt * Z_v
            V[:, t+1] = np.maximum(V[:, t] + dV, 0.0)
            
            # 2. Short rate process (Sinusoidal Hull-White)
            # theta_t incorporates cyclical long-term macroeconomic oscillation
            theta_t = self.mkt.theta_r + 0.005 * np.sin(2.0 * np.pi * t_curr)
            dr = self.mkt.kappa_r * (theta_t - r[:, t]) * dt + self.mkt.sigma_r * sqrt_dt * Z_r
            r[:, t+1] = r[:, t] + dr
            
            # 3. Asset price process S_t
            r_curr = r[:, t]
            dS = r_curr * S[:, t] * dt + np.sqrt(v_curr) * S[:, t] * sqrt_dt * Z_s
            S[:, t+1] = S[:, t] + dS
            
            # 4. Wrong-Way Risk (WWR) hazard rate coupling
            # Default intensity spikes when asset price drops below 80% of initial spot (Drawdown-driven default)
            drawdown = np.maximum(0.0, (s0 - S[:, t+1]) / s0)
            hazard[:, t+1] = self.credit.hazard_rate_c + 0.10 * drawdown**2
            
        discount_factors = np.exp(-np.cumsum(r * dt, axis=1))
        
        return {
            "S": S,
            "V": V,
            "r": r,
            "hazard": hazard,
            "discount_factors": discount_factors,
            "time_grid": self.time_grid
        }
