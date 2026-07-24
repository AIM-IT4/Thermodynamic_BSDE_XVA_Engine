"""Clean-price LSMC and an optional educational Deep-BSDE solver.

The Deep-BSDE experiment consumes the Brownian increments actually used by the
forward simulation. It is deliberately small and transparent; it is not a
production XVA solver or an independent model-validation result.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import BSDENetworkConfig, CreditConfig, SimulationConfig


class SubNetwork(nn.Module):
    """Approximate the Brownian integrand Z from the current factor state."""

    def __init__(
        self,
        state_dim: int,
        brownian_dim: int,
        hidden_dim: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for _ in range(max(1, n_layers)):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, brownian_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DeepBSDESolver:
    """Train a small neural BSDE experiment using genuine Brownian increments."""

    def __init__(
        self,
        sim_config: SimulationConfig,
        credit_config: CreditConfig,
        net_config: BSDENetworkConfig,
    ) -> None:
        self.sim = sim_config
        self.credit = credit_config
        self.net_cfg = net_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def driver(
        self,
        value: torch.Tensor,
        short_rate: torch.Tensor,
        counterparty_hazard: torch.Tensor,
        collateral: torch.Tensor,
    ) -> torch.Tensor:
        """Simplified nonlinear XVA driver used only by this experiment.

        The sign convention is -dY = f dt - Z dW. Positive unsecured value
        attracts credit and funding cost; negative unsecured value receives an
        issuer-default benefit. All coefficients come from CreditConfig.
        """
        unsecured = value - collateral
        positive = torch.relu(unsecured)
        negative = torch.relu(-unsecured)
        return (
            -short_rate * value
            - counterparty_hazard * self.credit.lgd_c * positive
            - self.credit.funding_spread * positive
            + self.credit.hazard_rate_i * self.credit.lgd_i * negative
        )

    @staticmethod
    def _validate_paths(
        factor_paths: np.ndarray,
        brownian_increments: np.ndarray,
        rate_paths: np.ndarray,
        hazard_paths: np.ndarray,
        collateral: np.ndarray | None,
    ) -> tuple[int, int, int, int]:
        if factor_paths.ndim != 3:
            raise ValueError("factor_paths must have shape (n_paths, n_steps + 1, state_dim)")
        n_paths, n_times, state_dim = factor_paths.shape
        n_steps = n_times - 1
        if n_steps <= 0 or state_dim <= 0:
            raise ValueError("factor_paths must contain at least one step and one state")
        if brownian_increments.ndim != 3 or brownian_increments.shape[:2] != (n_paths, n_steps):
            raise ValueError("brownian_increments must have shape (n_paths, n_steps, brownian_dim)")
        if brownian_increments.shape[2] <= 0:
            raise ValueError("brownian_increments must contain at least one Brownian factor")
        expected_time_shape = (n_paths, n_times)
        for name, matrix in (("rate_paths", rate_paths), ("hazard_paths", hazard_paths)):
            if matrix.shape != expected_time_shape:
                raise ValueError(f"{name} must have shape {expected_time_shape}")
        if collateral is not None and collateral.shape != expected_time_shape:
            raise ValueError(f"collateral must have shape {expected_time_shape}")
        return n_paths, n_steps, state_dim, brownian_increments.shape[2]

    def solve(
        self,
        factor_paths: np.ndarray,
        brownian_increments: np.ndarray,
        payoff_fn,
        rate_paths: np.ndarray,
        hazard_paths: np.ndarray,
        collateral: np.ndarray | None = None,
    ) -> dict[str, float | str]:
        """Fit Y0 and Z networks to a terminal payoff over simulated paths."""
        factor_paths = np.asarray(factor_paths, dtype=np.float32)
        brownian_increments = np.asarray(brownian_increments, dtype=np.float32)
        rate_paths = np.asarray(rate_paths, dtype=np.float32)
        hazard_paths = np.asarray(hazard_paths, dtype=np.float32)
        if collateral is None:
            collateral = np.zeros(rate_paths.shape, dtype=np.float32)
        else:
            collateral = np.asarray(collateral, dtype=np.float32)

        n_paths, n_steps, state_dim, brownian_dim = self._validate_paths(
            factor_paths,
            brownian_increments,
            rate_paths,
            hazard_paths,
            collateral,
        )
        if n_steps != self.sim.n_steps:
            raise ValueError("factor_paths time steps must match SimulationConfig")
        if self.net_cfg.epochs <= 0 or self.net_cfg.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")

        terminal_values = np.asarray(payoff_fn(factor_paths[:, -1, :]), dtype=np.float32)
        if terminal_values.shape != (n_paths,) or not np.all(np.isfinite(terminal_values)):
            raise ValueError("payoff_fn must return one finite value per path")

        torch.manual_seed(self.sim.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.sim.seed)

        x_tensor = torch.as_tensor(factor_paths, device=self.device)
        dw_tensor = torch.as_tensor(brownian_increments, device=self.device)
        rate_tensor = torch.as_tensor(rate_paths, device=self.device)
        hazard_tensor = torch.as_tensor(hazard_paths, device=self.device)
        collateral_tensor = torch.as_tensor(collateral, device=self.device)
        terminal_tensor = torch.as_tensor(terminal_values, device=self.device)

        feature_mean = x_tensor.mean(dim=(0, 1), keepdim=True)
        feature_std = x_tensor.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
        x_scaled = (x_tensor - feature_mean) / feature_std

        initial_guess = float(np.mean(terminal_values * np.exp(-rate_paths[:, 0] * self.sim.horizon)))
        y0 = nn.Parameter(torch.tensor(initial_guess, dtype=torch.float32, device=self.device))
        networks = nn.ModuleList(
            [
                SubNetwork(
                    state_dim,
                    brownian_dim,
                    self.net_cfg.hidden_dim,
                    self.net_cfg.n_layers,
                )
                for _ in range(n_steps)
            ]
        ).to(self.device)
        optimizer = optim.Adam(
            [y0, *networks.parameters()],
            lr=self.net_cfg.learning_rate,
        )
        dt = self.sim.horizon / n_steps
        batch_size = min(self.net_cfg.batch_size, n_paths)

        networks.train()
        for _ in range(self.net_cfg.epochs):
            batch_index = torch.randperm(n_paths, device=self.device)[:batch_size]
            value = y0.expand(batch_size)
            for step in range(n_steps):
                z_value = networks[step](x_scaled[batch_index, step, :])
                driver_value = self.driver(
                    value,
                    rate_tensor[batch_index, step],
                    hazard_tensor[batch_index, step],
                    collateral_tensor[batch_index, step],
                )
                value = (
                    value
                    - driver_value * dt
                    + torch.sum(z_value * dw_tensor[batch_index, step, :], dim=1)
                )
            loss = torch.mean((value - terminal_tensor[batch_index]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([y0, *networks.parameters()], max_norm=10.0)
            optimizer.step()

        networks.eval()
        with torch.no_grad():
            full_value = y0.expand(n_paths)
            for step in range(n_steps):
                z_value = networks[step](x_scaled[:, step, :])
                driver_value = self.driver(
                    full_value,
                    rate_tensor[:, step],
                    hazard_tensor[:, step],
                    collateral_tensor[:, step],
                )
                full_value = (
                    full_value
                    - driver_value * dt
                    + torch.sum(z_value * dw_tensor[:, step, :], dim=1)
                )
            final_loss = torch.mean((full_value - terminal_tensor) ** 2).item()
            terminal_spread = float(terminal_tensor.std(unbiased=False).item())

        rmse = float(np.sqrt(max(final_loss, 0.0)))
        # Fraction of the terminal payoff dispersion left unhedged by the Z
        # networks. A value near one means the experiment has not converged and
        # Y0 is close to its discounted-mean-payoff initialisation rather than a
        # learned price; a value near zero means the terminal payoff is matched.
        relative_rmse = rmse / terminal_spread if terminal_spread > 0.0 else float("nan")

        return {
            "Y0": float(y0.detach().cpu().item()),
            "loss": float(final_loss),
            "rmse": rmse,
            "relative_rmse": relative_rmse,
            "converged": bool(relative_rmse < 0.10),
            "device": str(self.device),
        }


class LeastSquaresMonteCarloPricer:
    """One-factor clean-price LSMC benchmark for the demonstration portfolio."""

    def __init__(self, sim_config: SimulationConfig) -> None:
        self.sim = sim_config

    def solve(
        self,
        asset_paths: np.ndarray,
        discount_factors: np.ndarray,
        payoff_fn,
        state_paths: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a clean MTM matrix without CVA or FVA deductions.

        The conditional-expectation regression uses a quadratic spot basis. When
        ``state_paths`` is supplied (shape ``(n_paths, n_steps + 1, n_state)``,
        e.g. variance and short rate), those states enter the basis linearly and
        as a spot cross term, so the fitted MTM depends on the stochastic state
        rather than on spot alone.
        """
        asset_paths = np.asarray(asset_paths, dtype=float)
        discount_factors = np.asarray(discount_factors, dtype=float)
        expected_shape = (self.sim.n_paths, self.sim.n_steps + 1)
        if asset_paths.shape != expected_shape or discount_factors.shape != expected_shape:
            raise ValueError(f"asset_paths and discount_factors must have shape {expected_shape}")
        if not np.allclose(discount_factors[:, 0], 1.0):
            raise ValueError("discount_factors[:, 0] must equal one")

        n_paths, n_times = asset_paths.shape
        if state_paths is not None:
            state_paths = np.asarray(state_paths, dtype=float)
            if state_paths.ndim != 3 or state_paths.shape[:2] != (n_paths, n_times):
                raise ValueError(
                    f"state_paths must have shape ({n_paths}, {n_times}, n_state)"
                )

        values = np.zeros((n_paths, n_times))
        values[:, -1] = payoff_fn(asset_paths[:, -1])

        for step in range(n_times - 2, -1, -1):
            spot = asset_paths[:, step]
            step_discount = discount_factors[:, step + 1] / discount_factors[:, step]
            continuation = values[:, step + 1] * step_discount
            columns = [np.ones(n_paths), spot, spot**2]
            if state_paths is not None:
                for state_index in range(state_paths.shape[2]):
                    state = state_paths[:, step, state_index]
                    columns.extend([state, spot * state])
            features = np.column_stack(columns)
            coefficients, _, _, _ = np.linalg.lstsq(features, continuation, rcond=None)
            values[:, step] = features @ coefficients

        return values


LeastSquaresBSDESolver = LeastSquaresMonteCarloPricer
