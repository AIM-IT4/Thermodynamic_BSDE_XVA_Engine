"""Configuration for the educational XVA simulation lab.

The defaults intentionally favour a fast, reproducible demonstration. They are
not calibrated market inputs and must not be used for trading, regulatory, or
counterparty-credit decisions.
"""

from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Monte-Carlo grid and reproducibility settings."""

    n_paths: int = 20_000
    n_steps: int = 252
    horizon: float = 1.0
    seed: int = 42


@dataclass
class MarketConfig:
    """Risk-neutral teaching parameters for spot, variance, and short rate."""

    r0: float = 0.04
    kappa_r: float = 0.50
    theta_r: float = 0.04
    sigma_r: float = 0.015
    dividend_yield: float = 0.0

    v0: float = 0.04
    kappa_v: float = 2.0
    theta_v: float = 0.04
    sigma_v: float = 0.30

    rho_sv: float = -0.70
    rho_sr: float = 0.15
    rho_vr: float = -0.10


@dataclass
class CreditConfig:
    """Simplified counterparty, collateral, and proxy-capital assumptions."""

    hazard_rate_c: float = 0.03
    hazard_rate_i: float = 0.01
    lgd_c: float = 0.60
    lgd_i: float = 0.60
    funding_spread: float = 0.015

    threshold: float = 1_000_000.0
    minimum_transfer_amount: float = 100_000.0
    collateral_call_frequency_days: int = 1
    margin_period_of_risk: float = 10.0 / 252.0

    # The following are deliberately labelled as teaching proxies.
    im_quantile: float = 0.99
    im_multiplier: float = 0.15
    im_funding_spread: float = 0.0075
    regulatory_capital_factor: float = 0.08
    cost_of_capital: float = 0.10

    # State-dependent intensity stress. This is a bounded illustrative WWR
    # specification, not a calibrated credit model.
    wwr_drawdown_beta: float = 1.50
    wwr_variance_beta: float = 0.25
    hazard_cap: float = 5.0


@dataclass
class BSDENetworkConfig:
    """Hyperparameters for the optional educational Deep-BSDE experiment."""

    hidden_dim: int = 32
    n_layers: int = 2
    learning_rate: float = 2e-3
    epochs: int = 100
    batch_size: int = 512
