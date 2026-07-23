"""Descriptive market diagnostics and static option-surface repair.

No path statistic can certify dynamic no-arbitrage. The surface routine below
therefore makes only a narrow claim: it projects Black-Scholes call prices
onto basic static no-arbitrage constraints in strike and expiry space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, brentq, minimize
from scipy.stats import norm


class MarketDiagnostics:
    """Descriptive diagnostics only; these are not no-arbitrage tests."""

    @staticmethod
    def compute_excess_return_dispersion(
        asset_paths: np.ndarray,
        rate_paths: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Return cross-sectional mean squared excess log returns by time step."""
        asset_paths = np.asarray(asset_paths, dtype=float)
        rate_paths = np.asarray(rate_paths, dtype=float)
        if asset_paths.shape != rate_paths.shape or asset_paths.ndim != 2:
            raise ValueError("asset_paths and rate_paths must be same-shaped matrices")
        if np.any(asset_paths <= 0.0) or dt <= 0.0:
            raise ValueError("asset paths and dt must be strictly positive")

        log_returns = np.diff(np.log(asset_paths), axis=1)
        midpoint_rates = 0.5 * (rate_paths[:, :-1] + rate_paths[:, 1:])
        excess_log_returns = log_returns - midpoint_rates * dt
        return np.mean(excess_log_returns**2, axis=0)


@dataclass(frozen=True)
class SurfaceRepairResult:
    """Result of a static call-price projection."""

    raw_implied_vol: np.ndarray
    repaired_implied_vol: np.ndarray
    raw_call_prices: np.ndarray
    repaired_call_prices: np.ndarray
    diagnostics: dict[str, float]


class StaticArbitrageSurfaceRepair:
    """Repair a small Black-Scholes surface under explicit linear constraints.

    The routine enforces bounds, call-price monotonicity and convexity in
    strike, plus non-decreasing call prices across increasing expiries. It is
    not optimal transport, not a KL projection, and not a Dupire calibration.
    """

    def __init__(self, spot: float, rate: float = 0.0) -> None:
        if spot <= 0.0:
            raise ValueError("spot must be positive")
        self.spot = float(spot)
        self.rate = float(rate)

    def _call_price(
        self,
        strikes: np.ndarray,
        expiries: np.ndarray,
        implied_vol: np.ndarray,
    ) -> np.ndarray:
        strike_grid, expiry_grid = np.meshgrid(strikes, expiries)
        volatility = np.asarray(implied_vol, dtype=float)
        sqrt_time = np.sqrt(expiry_grid)
        discounted_strike = strike_grid * np.exp(-self.rate * expiry_grid)
        intrinsic = np.maximum(self.spot - discounted_strike, 0.0)

        result = intrinsic.copy()
        positive_vol = volatility > 1e-10
        if np.any(positive_vol):
            sigma_sqrt_t = volatility[positive_vol] * sqrt_time[positive_vol]
            d1 = (
                np.log(self.spot / strike_grid[positive_vol])
                + (self.rate + 0.5 * volatility[positive_vol] ** 2)
                * expiry_grid[positive_vol]
            ) / sigma_sqrt_t
            d2 = d1 - sigma_sqrt_t
            result[positive_vol] = (
                self.spot * norm.cdf(d1)
                - discounted_strike[positive_vol] * norm.cdf(d2)
            )
        return result

    def _implied_volatility(
        self,
        strike: float,
        expiry: float,
        call_price: float,
    ) -> float:
        discounted_strike = strike * np.exp(-self.rate * expiry)
        lower_bound = max(self.spot - discounted_strike, 0.0)
        upper_bound = self.spot
        clipped_price = float(np.clip(call_price, lower_bound, upper_bound))
        if clipped_price <= lower_bound + 1e-10:
            return 0.0
        if clipped_price >= upper_bound - 1e-10:
            return 5.0

        def residual(volatility: float) -> float:
            return float(
                self._call_price(
                    np.array([strike]),
                    np.array([expiry]),
                    np.array([[volatility]]),
                )[0, 0]
                - clipped_price
            )

        return float(brentq(residual, 1e-8, 5.0))

    @staticmethod
    def validate_call_surface(
        strikes: np.ndarray,
        call_prices: np.ndarray,
    ) -> dict[str, float]:
        """Return minimum slacks for the repaired static constraints."""
        strike_diffs = np.diff(strikes)
        monotonicity = call_prices[:, :-1] - call_prices[:, 1:]
        slopes = np.diff(call_prices, axis=1) / strike_diffs[np.newaxis, :]
        convexity = np.diff(slopes, axis=1)
        calendar = np.diff(call_prices, axis=0)

        return {
            "min_monotonicity_slack": float(np.min(monotonicity)),
            "min_convexity_slack": float(np.min(convexity)),
            "min_calendar_slack": float(np.min(calendar)),
        }

    def repair(
        self,
        strikes: np.ndarray,
        expiries: np.ndarray,
        raw_implied_vol: np.ndarray,
    ) -> SurfaceRepairResult:
        """Project raw Black-Scholes call prices onto static constraints."""
        strikes = np.asarray(strikes, dtype=float)
        expiries = np.asarray(expiries, dtype=float)
        raw_implied_vol = np.asarray(raw_implied_vol, dtype=float)
        if strikes.ndim != 1 or expiries.ndim != 1:
            raise ValueError("strikes and expiries must be one-dimensional")
        if len(strikes) < 3 or len(expiries) < 2:
            raise ValueError("at least three strikes and two expiries are required")
        if np.any(np.diff(strikes) <= 0.0) or np.any(np.diff(expiries) <= 0.0):
            raise ValueError("strikes and expiries must be strictly increasing")
        if raw_implied_vol.shape != (len(expiries), len(strikes)):
            raise ValueError("raw_implied_vol shape must be (len(expiries), len(strikes))")
        if np.any(raw_implied_vol <= 0.0) or not np.all(np.isfinite(raw_implied_vol)):
            raise ValueError("raw_implied_vol must be finite and strictly positive")

        n_expiries, n_strikes = raw_implied_vol.shape
        raw_calls = self._call_price(strikes, expiries, raw_implied_vol)
        variable_count = n_expiries * n_strikes

        def index(expiry_index: int, strike_index: int) -> int:
            return expiry_index * n_strikes + strike_index

        constraints: list[tuple[np.ndarray, float, float]] = []

        # Calls decrease with strike.
        for expiry_index in range(n_expiries):
            for strike_index in range(n_strikes - 1):
                row = np.zeros(variable_count)
                row[index(expiry_index, strike_index)] = 1.0
                row[index(expiry_index, strike_index + 1)] = -1.0
                constraints.append(
                    (
                        row,
                        0.0,
                        np.exp(-self.rate * expiries[expiry_index]) * np.diff(strikes)[strike_index],
                    )
                )

        # Calls are convex in strike, including non-uniform strike spacing.
        strike_diffs = np.diff(strikes)
        for expiry_index in range(n_expiries):
            for strike_index in range(1, n_strikes - 1):
                left_width = strike_diffs[strike_index - 1]
                right_width = strike_diffs[strike_index]
                row = np.zeros(variable_count)
                row[index(expiry_index, strike_index - 1)] = 1.0 / left_width
                row[index(expiry_index, strike_index)] = -1.0 / left_width - 1.0 / right_width
                row[index(expiry_index, strike_index + 1)] = 1.0 / right_width
                constraints.append((row, 0.0, np.inf))

        # Under a non-negative carry assumption, calls are non-decreasing in expiry.
        for expiry_index in range(n_expiries - 1):
            for strike_index in range(n_strikes):
                row = np.zeros(variable_count)
                row[index(expiry_index, strike_index)] = -1.0
                row[index(expiry_index + 1, strike_index)] = 1.0
                constraints.append((row, 0.0, np.inf))

        strike_grid, expiry_grid = np.meshgrid(strikes, expiries)
        lower_bounds = np.maximum(
            self.spot - strike_grid * np.exp(-self.rate * expiry_grid),
            0.0,
        ).ravel()
        upper_bounds = np.full(variable_count, self.spot)
        constraint_matrix = np.vstack([entry[0] for entry in constraints])
        constraint_lower = np.asarray([entry[1] for entry in constraints])
        constraint_upper = np.asarray([entry[2] for entry in constraints])
        raw_vector = raw_calls.ravel()

        result = minimize(
            fun=lambda values: float(np.sum((values - raw_vector) ** 2)),
            x0=np.clip(raw_vector, lower_bounds, upper_bounds),
            jac=lambda values: 2.0 * (values - raw_vector),
            method="SLSQP",
            bounds=Bounds(lower_bounds, upper_bounds),
            constraints=[LinearConstraint(constraint_matrix, constraint_lower, constraint_upper)],
            options={"ftol": 1e-11, "maxiter": 1_000},
        )
        if not result.success:
            raise RuntimeError(f"Static surface repair failed: {result.message}")

        repaired_calls = result.x.reshape(n_expiries, n_strikes)
        diagnostics = self.validate_call_surface(strikes, repaired_calls)
        vertical_spread = repaired_calls[:, :-1] - repaired_calls[:, 1:]
        vertical_upper = (
            np.exp(-self.rate * expiries)[:, np.newaxis] * np.diff(strikes)[np.newaxis, :]
        )
        diagnostics["min_vertical_spread_upper_slack"] = float(np.min(vertical_upper - vertical_spread))
        if min(diagnostics.values()) < -1e-7:
            raise RuntimeError("Static surface repair returned an infeasible surface")

        repaired_iv = np.empty_like(raw_implied_vol)
        for expiry_index, expiry in enumerate(expiries):
            for strike_index, strike in enumerate(strikes):
                repaired_iv[expiry_index, strike_index] = self._implied_volatility(
                    strike,
                    expiry,
                    repaired_calls[expiry_index, strike_index],
                )

        return SurfaceRepairResult(
            raw_implied_vol=raw_implied_vol,
            repaired_implied_vol=repaired_iv,
            raw_call_prices=raw_calls,
            repaired_call_prices=repaired_calls,
            diagnostics=diagnostics,
        )

    def repair_volatility_surface(
        self,
        strikes: np.ndarray,
        expiries: np.ndarray,
        raw_implied_vol: np.ndarray,
    ) -> np.ndarray:
        """Compatibility wrapper returning only repaired implied volatilities."""
        return self.repair(strikes, expiries, raw_implied_vol).repaired_implied_vol


class ThermodynamicArbitrageEngine(MarketDiagnostics):
    """Deprecated compatibility name for descriptive diagnostics only."""
