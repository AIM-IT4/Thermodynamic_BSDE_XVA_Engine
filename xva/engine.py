"""Pathwise XVA and collateral calculations for the educational lab.

The implementation keeps discounting, exposure, and default intensities on the
same Monte-Carlo paths. It models first-to-default probabilities over each
time interval, rather than assigning default mass to time zero. MVA and KVA
remain explicitly documented teaching proxies.
"""

from __future__ import annotations

import numpy as np

from config import CreditConfig, SimulationConfig


class ComprehensiveXVAEngine:
    """Compute simplified, pathwise XVA measures for a single netting set."""

    def __init__(self, sim_config: SimulationConfig, credit_config: CreditConfig) -> None:
        self.sim = sim_config
        self.credit = credit_config
        self.dt = self.sim.horizon / self.sim.n_steps

    def _validate_time_matrix(self, values: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        expected_times = self.sim.n_steps + 1
        if array.ndim != 2 or array.shape[1] != expected_times:
            raise ValueError(f"{name} must have shape (n_paths, {expected_times})")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")
        return array

    def _collateral_call_steps(self) -> int:
        frequency_years = self.credit.collateral_call_frequency_days / 252.0
        return max(1, int(round(frequency_years / self.dt)))

    def compute_collateral_margin(self, mtm: np.ndarray) -> np.ndarray:
        """Apply symmetric VM threshold, MTA, and call-frequency mechanics.

        Positive collateral denotes collateral held by the bank; negative
        collateral denotes collateral posted by the bank.
        """
        mtm = self._validate_time_matrix(mtm, "mtm")
        if self.credit.threshold < 0.0 or self.credit.minimum_transfer_amount < 0.0:
            raise ValueError("threshold and minimum_transfer_amount must be non-negative")

        collateral = np.zeros_like(mtm)
        held = np.zeros(mtm.shape[0])
        call_steps = self._collateral_call_steps()

        for step in range(mtm.shape[1]):
            if step % call_steps == 0:
                target = np.sign(mtm[:, step]) * np.maximum(
                    np.abs(mtm[:, step]) - self.credit.threshold, 0.0
                )
                transfer = target - held
                held = np.where(
                    np.abs(transfer) >= self.credit.minimum_transfer_amount,
                    target,
                    held,
                )
            collateral[:, step] = held

        return collateral

    def _mpor_closeout(self, mtm: np.ndarray) -> np.ndarray:
        mpor_steps = max(0, int(np.ceil(self.credit.margin_period_of_risk / self.dt)))
        time_indices = np.minimum(
            np.arange(mtm.shape[1]) + mpor_steps,
            mtm.shape[1] - 1,
        )
        return mtm[:, time_indices]

    def compute_exposure_profiles(
        self,
        mtm: np.ndarray,
        collateral: np.ndarray | None = None,
    ) -> dict[str, np.ndarray | float]:
        """Return current and MPOR-shifted exposure profiles.

        Default exposure uses future MTM at the end of MPOR against collateral
        held at the default time. Funding exposure uses current uncollateralised
        MTM. This distinction prevents MPOR from being silently ignored.
        """
        mtm = self._validate_time_matrix(mtm, "mtm")
        if collateral is None:
            collateral = self.compute_collateral_margin(mtm)
        else:
            collateral = self._validate_time_matrix(collateral, "collateral")
            if collateral.shape != mtm.shape:
                raise ValueError("collateral must have the same shape as mtm")

        current_uncollateralised = mtm - collateral
        closeout_uncollateralised = self._mpor_closeout(mtm) - collateral

        pos_exposure = np.maximum(current_uncollateralised, 0.0)
        neg_exposure = np.maximum(-current_uncollateralised, 0.0)
        mpor_pos_exposure = np.maximum(closeout_uncollateralised, 0.0)
        mpor_neg_exposure = np.maximum(-closeout_uncollateralised, 0.0)

        ee = np.mean(pos_exposure, axis=0)
        nee = np.mean(neg_exposure, axis=0)
        pfe_95 = np.quantile(pos_exposure, 0.95, axis=0)
        pfe_99 = np.quantile(pos_exposure, 0.99, axis=0)
        epe = float(
            np.sum(0.5 * (ee[:-1] + ee[1:]) * self.dt) / self.sim.horizon
        )

        return {
            "EE": ee,
            "NEE": nee,
            "PFE_95": pfe_95,
            "PFE_99": pfe_99,
            "EPE": epe,
            "Collateral": collateral,
            "pos_exposure_matrix": pos_exposure,
            "neg_exposure_matrix": neg_exposure,
            "mpor_pos_exposure_matrix": mpor_pos_exposure,
            "mpor_neg_exposure_matrix": mpor_neg_exposure,
        }

    def _hazard_matrix(
        self,
        hazard_paths: np.ndarray | None,
        n_paths: int,
        default_hazard: float,
        name: str,
    ) -> np.ndarray:
        if hazard_paths is None:
            return np.full((n_paths, self.sim.n_steps + 1), default_hazard)
        hazard = self._validate_time_matrix(hazard_paths, name)
        if hazard.shape[0] != n_paths:
            raise ValueError(f"{name} and exposure paths must have the same number of paths")
        if np.any(hazard < 0.0):
            raise ValueError(f"{name} must be non-negative")
        return hazard

    def _first_to_default_probabilities(
        self,
        counterparty_hazard: np.ndarray,
        issuer_hazard: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return pathwise interval probabilities of each name defaulting first."""
        n_paths, n_steps = counterparty_hazard.shape[0], self.sim.n_steps
        q_counterparty = np.empty((n_paths, n_steps))
        q_issuer = np.empty((n_paths, n_steps))
        joint_survival = np.ones(n_paths)

        for step in range(n_steps):
            lambda_c = counterparty_hazard[:, step]
            lambda_i = issuer_hazard[:, step]
            total_hazard = lambda_c + lambda_i
            interval_default = -np.expm1(-total_hazard * self.dt)
            counterparty_share = np.divide(
                lambda_c,
                total_hazard,
                out=np.zeros_like(lambda_c),
                where=total_hazard > 0.0,
            )
            issuer_share = np.divide(
                lambda_i,
                total_hazard,
                out=np.zeros_like(lambda_i),
                where=total_hazard > 0.0,
            )
            q_counterparty[:, step] = joint_survival * counterparty_share * interval_default
            q_issuer[:, step] = joint_survival * issuer_share * interval_default
            joint_survival *= np.exp(-total_hazard * self.dt)

        return q_counterparty, q_issuer

    def calculate_xva_metrics(
        self,
        exposure_profiles: dict[str, np.ndarray | float],
        discount_factors: np.ndarray,
        hazard_paths: np.ndarray | None = None,
        issuer_hazard_paths: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Calculate pathwise CVA, DVA, FCA, and clearly-labelled proxies.

        CVA/DVA preserve the simulated dependence among discount factors,
        exposure, and intensities. FVA is a simplified funding-cost adjustment;
        MVA and KVA are proxy calculations, not regulatory measures.
        """
        discount_factors = self._validate_time_matrix(discount_factors, "discount_factors")
        if not np.allclose(discount_factors[:, 0], 1.0):
            raise ValueError("discount_factors[:, 0] must equal one")

        pos = self._validate_time_matrix(
            np.asarray(exposure_profiles["pos_exposure_matrix"]),
            "pos_exposure_matrix",
        )
        neg = self._validate_time_matrix(
            np.asarray(exposure_profiles["neg_exposure_matrix"]),
            "neg_exposure_matrix",
        )
        mpor_pos = self._validate_time_matrix(
            np.asarray(exposure_profiles["mpor_pos_exposure_matrix"]),
            "mpor_pos_exposure_matrix",
        )
        mpor_neg = self._validate_time_matrix(
            np.asarray(exposure_profiles["mpor_neg_exposure_matrix"]),
            "mpor_neg_exposure_matrix",
        )
        if pos.shape[0] != discount_factors.shape[0]:
            raise ValueError("exposure and discount-factor path counts differ")

        n_paths = pos.shape[0]
        lambda_c = self._hazard_matrix(
            hazard_paths,
            n_paths,
            self.credit.hazard_rate_c,
            "hazard_paths",
        )
        lambda_i = self._hazard_matrix(
            issuer_hazard_paths,
            n_paths,
            self.credit.hazard_rate_i,
            "issuer_hazard_paths",
        )
        q_c, q_i = self._first_to_default_probabilities(lambda_c, lambda_i)
        interval_df = discount_factors[:, :-1]

        cva = float(
            np.mean(
                np.sum(
                    interval_df * mpor_pos[:, :-1] * self.credit.lgd_c * q_c,
                    axis=1,
                )
            )
        )
        dva = float(
            np.mean(
                np.sum(
                    interval_df * mpor_neg[:, :-1] * self.credit.lgd_i * q_i,
                    axis=1,
                )
            )
        )
        fva = float(
            np.mean(
                np.sum(
                    interval_df * pos[:, :-1] * self.credit.funding_spread * self.dt,
                    axis=1,
                )
            )
        )

        im_profile = self.credit.im_multiplier * np.quantile(
            pos,
            self.credit.im_quantile,
            axis=0,
        )
        mean_df = np.mean(interval_df, axis=0)
        mva = float(
            np.sum(mean_df * im_profile[:-1] * self.credit.im_funding_spread * self.dt)
        )
        capital_profile = self.credit.regulatory_capital_factor * np.asarray(
            exposure_profiles["PFE_99"],
            dtype=float,
        )
        kva = float(
            np.sum(mean_df * capital_profile[:-1] * self.credit.cost_of_capital * self.dt)
        )

        total_xva = cva - dva + fva + mva + kva
        return {
            "CVA": cva,
            "DVA": dva,
            "FVA": fva,
            "MVA_proxy": mva,
            "KVA_proxy": kva,
            "Total_XVA": total_xva,
        }

    def compute_greeks(
        self,
        asset_paths: np.ndarray,
        payoff_fn,
        discount_factors: np.ndarray,
        h: float = 1.0,
    ) -> dict[str, float]:
        """Calculate discounted bump-and-revalue spot Delta and Gamma.

        The bump is applied to the initial spot and propagated to the terminal
        spot by the homogeneity of the simulated dynamics: log-Euler paths
        scale S_T linearly with S_0, so bumping S_0 by h with the Brownian
        increments held fixed multiplies each terminal spot by (S_0 +/- h)/S_0.
        This yields dV/dS_0. An additive shift of the terminal spot instead
        measures sensitivity to a parallel move of the terminal distribution,
        which materially understates the spot Delta of a convex payoff.
        """
        if h <= 0.0:
            raise ValueError("h must be positive")
        asset_paths = self._validate_time_matrix(asset_paths, "asset_paths")
        discount_factors = self._validate_time_matrix(discount_factors, "discount_factors")
        if asset_paths.shape != discount_factors.shape:
            raise ValueError("asset_paths and discount_factors must have the same shape")

        initial_spot = asset_paths[:, 0]
        if np.any(initial_spot <= 0.0):
            raise ValueError("initial spot must be strictly positive")
        if h >= float(np.min(initial_spot)):
            raise ValueError("h must be smaller than the initial spot")

        terminal_spot = asset_paths[:, -1]
        terminal_df = discount_factors[:, -1]
        scale_up = (initial_spot + h) / initial_spot
        scale_down = (initial_spot - h) / initial_spot
        base_value = float(np.mean(terminal_df * payoff_fn(terminal_spot)))
        value_up = float(np.mean(terminal_df * payoff_fn(terminal_spot * scale_up)))
        value_down = float(np.mean(terminal_df * payoff_fn(terminal_spot * scale_down)))

        return {
            "Value": base_value,
            "Delta": (value_up - value_down) / (2.0 * h),
            "Gamma": (value_up - 2.0 * base_value + value_down) / (h**2),
        }
