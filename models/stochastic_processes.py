"""Joint risk-factor simulation used by the educational XVA lab.

The variance uses a full-truncation Euler scheme and the spot is evolved in
log space. The latter keeps spot paths strictly positive, unlike additive
Euler updates of the price itself. The returned Brownian increments are the
actual correlated increments used to evolve the factors; they are exposed for
the optional Deep-BSDE experiment.
"""

from __future__ import annotations

import numpy as np

from config import CreditConfig, MarketConfig, SimulationConfig


class MultiFactorStochasticEngine:
    """Simulate a Heston-style spot/variance process and an OU short rate.

    This is a reproducible teaching discretisation, not a calibrated market
    model. Counterparty intensity is a bounded state-dependent stress so that
    pathwise WWR can be demonstrated without claiming credit calibration.
    """

    def __init__(
        self,
        sim_config: SimulationConfig,
        mkt_config: MarketConfig,
        credit_config: CreditConfig,
    ) -> None:
        if sim_config.n_paths <= 0 or sim_config.n_steps <= 0:
            raise ValueError("n_paths and n_steps must be positive")
        if sim_config.horizon <= 0.0:
            raise ValueError("horizon must be positive")

        self.sim = sim_config
        self.mkt = mkt_config
        self.credit = credit_config
        self.dt = self.sim.horizon / self.sim.n_steps
        self.time_grid = np.linspace(0.0, self.sim.horizon, self.sim.n_steps + 1)

    def _cholesky(self) -> np.ndarray:
        correlation = np.array(
            [
                [1.0, self.mkt.rho_sv, self.mkt.rho_sr],
                [self.mkt.rho_sv, 1.0, self.mkt.rho_vr],
                [self.mkt.rho_sr, self.mkt.rho_vr, 1.0],
            ]
        )
        try:
            return np.linalg.cholesky(correlation)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Configured factor correlation matrix is not positive definite") from exc

    def generate_paths(self, s0: float = 100.0) -> dict[str, np.ndarray]:
        """Generate joint factor, discount, and intensity paths.

        dW has shape (n_paths, n_steps, 3) and contains the correlated Brownian
        increments used for log spot, variance, and short rate, respectively.
        discount_factors[:, 0] is exactly one.
        """
        if s0 <= 0.0:
            raise ValueError("s0 must be strictly positive")

        n_paths, n_steps = self.sim.n_paths, self.sim.n_steps
        dt, sqrt_dt = self.dt, np.sqrt(self.dt)
        rng = np.random.default_rng(self.sim.seed)

        log_s = np.empty((n_paths, n_steps + 1))
        variance = np.empty_like(log_s)
        short_rate = np.empty_like(log_s)
        hazard_c = np.empty_like(log_s)
        hazard_i = np.full_like(log_s, self.credit.hazard_rate_i)
        discount_factors = np.empty_like(log_s)
        d_w = np.empty((n_paths, n_steps, 3))

        log_s[:, 0] = np.log(s0)
        variance[:, 0] = self.mkt.v0
        short_rate[:, 0] = self.mkt.r0
        hazard_c[:, 0] = self.credit.hazard_rate_c
        discount_factors[:, 0] = 1.0

        cholesky = self._cholesky()
        variance_scale = max(self.mkt.v0, 1e-10)

        for step in range(n_steps):
            shocks = rng.standard_normal((n_paths, 3)) @ cholesky.T
            d_w[:, step, :] = shocks * sqrt_dt
            d_w_s, d_w_v, d_w_r = d_w[:, step, :].T

            v_t = np.maximum(variance[:, step], 0.0)
            r_t = short_rate[:, step]

            variance[:, step + 1] = np.maximum(
                variance[:, step]
                + self.mkt.kappa_v * (self.mkt.theta_v - v_t) * dt
                + self.mkt.sigma_v * np.sqrt(v_t) * d_w_v,
                0.0,
            )
            short_rate[:, step + 1] = (
                r_t
                + self.mkt.kappa_r * (self.mkt.theta_r - r_t) * dt
                + self.mkt.sigma_r * d_w_r
            )

            log_s[:, step + 1] = (
                log_s[:, step]
                + (r_t - self.mkt.dividend_yield - 0.5 * v_t) * dt
                + np.sqrt(v_t) * d_w_s
            )
            discount_factors[:, step + 1] = discount_factors[:, step] * np.exp(-r_t * dt)

            drawdown = np.maximum(0.0, np.log(s0) - log_s[:, step + 1])
            variance_excess = np.maximum(variance[:, step + 1] - self.mkt.v0, 0.0) / variance_scale
            intensity = self.credit.hazard_rate_c * np.exp(
                self.credit.wwr_drawdown_beta * drawdown
                + self.credit.wwr_variance_beta * variance_excess
            )
            hazard_c[:, step + 1] = np.clip(intensity, 0.0, self.credit.hazard_cap)

        spot = np.exp(np.clip(log_s, -700.0, 700.0))
        factors = np.stack([log_s, variance, short_rate], axis=-1)

        return {
            "S": spot,
            "log_S": log_s,
            "V": variance,
            "r": short_rate,
            "hazard": hazard_c,
            "issuer_hazard": hazard_i,
            "discount_factors": discount_factors,
            "dW": d_w,
            "factors": factors,
            "time_grid": self.time_grid,
        }
