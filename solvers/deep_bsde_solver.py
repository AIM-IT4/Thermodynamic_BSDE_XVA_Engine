"""
Nonlinear Backward Stochastic Differential Equation (BSDE) Solvers for XVA Valuation.
Implements PyTorch Deep BSDE Neural Network and Least-Squares Monte Carlo (LSMC) solvers.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import SimulationConfig, CreditConfig, BSDENetworkConfig

class SubNetwork(nn.Module):
    """Deep Neural Network approximating gradient Z_t = grad_x Y(t, X_t) at time step t"""
    def __init__(self, dim: int, hidden_dim: int):
        super(SubNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DeepBSDESolver:
    """
    High-Dimensional Deep BSDE Neural Network Solver.
    Solves nonlinear XVA coupled FBSDEs where funding and credit spreads depend nonlinearly on portfolio value.
    """
    def __init__(self, sim_config: SimulationConfig, credit_config: CreditConfig, net_config: BSDENetworkConfig):
        self.sim = sim_config
        self.credit = credit_config
        self.net_cfg = net_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def solve(self, X_paths: np.ndarray, payoff_fn) -> dict:
        """
        Solves the BSDE for terminal payoff g(X_T).
        X_paths: shape (n_paths, n_steps + 1, dim)
        """
        n_paths, n_steps_p1, dim = X_paths.shape
        n_steps = n_steps_p1 - 1
        dt = self.sim.horizon / n_steps
        sqrt_dt = np.sqrt(dt)

        # Convert paths to torch tensors
        X_tensor = torch.tensor(X_paths, dtype=torch.float32, device=self.device)

        # Initial value Y_0 (trainable parameter)
        Y0 = nn.Parameter(torch.tensor([np.mean(payoff_fn(X_paths[:, -1, :]))], dtype=torch.float32, device=self.device))
        
        # Sub-networks for Z_t at each step t
        subnets = nn.ModuleList([SubNetwork(dim, self.net_cfg.hidden_dim) for _ in range(n_steps)]).to(self.device)

        optimizer = optim.Adam([Y0] + list(subnets.parameters()), lr=self.net_cfg.learning_rate)

        # Driver function f(t, X, Y, Z) representing nonlinear XVA drag
        def driver(y_val, hazard=0.03, lgd=0.6, funding_spread=0.015):
            cva_drag = hazard * lgd * torch.relu(y_val)
            fva_drag = funding_spread * torch.relu(y_val)
            return -(cva_drag + fva_drag)

        # Training loop
        for epoch in range(self.net_cfg.epochs):
            optimizer.zero_grad()
            
            # Forward simulation of Y_t
            Y = Y0.repeat(n_paths)
            
            for t in range(n_steps):
                X_t = X_tensor[:, t, :]
                X_next = X_tensor[:, t+1, :]
                dW = (X_next - X_t) / sqrt_dt  # Proxy Brownian increment
                
                Z_t = subnets[t](X_t)
                
                # BSDE Forward step: Y_{t+1} = Y_t - f(t, X_t, Y_t, Z_t)*dt + Z_t . dW_t
                f_val = driver(Y)
                Y = Y - f_val * dt + torch.sum(Z_t * dW, dim=1)

            # Terminal loss matching payoff g(X_T)
            terminal_payoff = torch.tensor(payoff_fn(X_paths[:, -1, :]), dtype=torch.float32, device=self.device)
            loss = torch.mean((Y - terminal_payoff)**2)

            loss.backward()
            optimizer.step()

        return {
            "Y0": Y0.item(),
            "loss": loss.item(),
            "device": str(self.device)
        }

class LeastSquaresBSDESolver:
    """
    Fast Vectorized Least-Squares Monte Carlo (LSMC) Solver for BSDE XVA.
    Computes backward induction valuation over simulation trajectories.
    """
    def __init__(self, sim_config: SimulationConfig, credit_config: CreditConfig):
        self.sim = sim_config
        self.credit = credit_config

    def solve(self, asset_paths: np.ndarray, discount_factors: np.ndarray, payoff_fn) -> np.ndarray:
        """
        Backward induction for portfolio valuation matrix (n_paths, n_steps + 1).
        """
        n_paths, n_steps_p1 = asset_paths.shape
        n_steps = n_steps_p1 - 1
        dt = self.sim.horizon / n_steps

        portfolio_values = np.zeros((n_paths, n_steps + 1))
        portfolio_values[:, -1] = payoff_fn(asset_paths[:, -1])

        for t in range(n_steps - 1, -1, -1):
            S_t = asset_paths[:, t]
            df_step = discount_factors[:, t+1] / discount_factors[:, t]
            continuation_val = portfolio_values[:, t+1] * df_step

            # Quadratic basis polynomial regression
            poly_features = np.column_stack([np.ones(n_paths), S_t, S_t**2])
            coeff, _, _, _ = np.linalg.lstsq(poly_features, continuation_val, rcond=None)
            
            # Expected continuation value
            expected_val = poly_features @ coeff
            
            # Nonlinear XVA Driver Correction
            uncollateralized_exposure = np.maximum(0.0, expected_val)
            cva_penalty = self.credit.hazard_rate_c * self.credit.lgd_c * uncollateralized_exposure * dt
            fva_penalty = self.credit.funding_spread * uncollateralized_exposure * dt
            
            portfolio_values[:, t] = expected_val - (cva_penalty + fva_penalty)

        return portfolio_values
