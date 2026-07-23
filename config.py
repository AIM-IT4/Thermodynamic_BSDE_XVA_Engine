"""
Configuration parameters for the Thermodynamic BSDE XVA Engine.
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class SimulationConfig:
    """Parameters for Monte Carlo & BSDE Simulation"""
    n_paths: int = 50000          # Number of Monte Carlo sample paths
    n_steps: int = 252            # Number of time steps (1 year = 252 steps)
    horizon: float = 1.0          # Maturity horizon in years
    seed: int = 42                # Random seed for reproducibility

@dataclass
class MarketConfig:
    """Market & Interest Rate parameters"""
    r0: float = 0.04              # Initial short rate
    kappa_r: float = 0.5          # Mean reversion speed for Hull-White
    theta_r: float = 0.04         # Long-term mean short rate
    sigma_r: float = 0.015        # Short rate volatility
    
    # Heston Volatility parameters
    v0: float = 0.04              # Initial variance (20% vol)
    kappa_v: float = 2.0          # Mean reversion speed for variance
    theta_v: float = 0.04         # Long-term variance
    sigma_v: float = 0.3          # Volatility of volatility
    rho_sv: float = -0.7          # Spot-vol correlation (equity leverage effect)

@dataclass
class CreditConfig:
    """Counterparty Credit Risk & Collateral parameters"""
    hazard_rate_c: float = 0.03    # Counterparty default intensity (300 bps)
    hazard_rate_i: float = 0.01    # Investor default intensity (100 bps)
    lgd_c: float = 0.6            # Loss Given Default for counterparty (40% recovery)
    lgd_i: float = 0.6            # Loss Given Default for investor
    funding_spread: float = 0.015  # Uncollateralized borrowing spread (150 bps)
    margin_period_of_risk: float = 10 / 252.0  # MPOR in years (10 business days)
    threshold: float = 1e6        # Uncollateralized threshold amount ($1M)

@dataclass
class BSDENetworkConfig:
    """Deep BSDE Neural Network Hyperparameters"""
    hidden_dim: int = 64
    n_layers: int = 3
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 256
