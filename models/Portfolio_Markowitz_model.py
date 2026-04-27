import numpy as np
import gurobipy as gp
from pyepo.model.grb import optGrbModel

from utils.seed_manager import SeedManager


class PortfolioMarkowitzModel(optGrbModel):
    """
    Markowitz portfolio model with turnover fee penalty.

    Objective:
        min cost^T w + lambda_risk * w^T Sigma w
            + fee_rate * sum_i |w_i - prev_weight_i|

    Sigma is estimated from the latest ``cov_history`` rows of historical
    asset log returns passed as ``scenario_returns``.
    """

    def __init__(
        self,
        n_assets,
        lambda_risk=1.0,
        cov_history=252,
        cov_reg=1e-6,
        fee_rate=0.0,
        budget=1.0,
        lb=0.0,
        ub=1.0,
        threads=None,
        seed=42,
    ):
        if n_assets <= 0:
            raise ValueError("n_assets must be positive")
        if budget <= 0:
            raise ValueError("budget must be positive")
        if lb > ub:
            raise ValueError("lb cannot be greater than ub")
        if n_assets * lb > budget:
            raise ValueError("Infeasible model: n_assets * lb > budget")
        if n_assets * ub < budget:
            raise ValueError("Infeasible model: n_assets * ub < budget")
        if lambda_risk < 0:
            raise ValueError("lambda_risk must be non-negative")
        if cov_history < 2:
            raise ValueError("cov_history must be at least 2")
        if cov_reg < 0:
            raise ValueError("cov_reg must be non-negative")
        if fee_rate < 0:
            raise ValueError("fee_rate must be non-negative")

        self.n_assets = n_assets
        self.lambda_risk = float(lambda_risk)
        self.cov_history = int(cov_history)
        self.scenario_history = int(cov_history)
        self.cov_reg = float(cov_reg)
        self.fee_rate = float(fee_rate)
        self.budget = budget
        self.lb = lb
        self.ub = ub
        self.threads = threads
        self.seed = seed

        self.requires_scenarios = True
        self.supports_turnover = True
        self._current_cost = None
        self._current_scenarios = None
        self._current_prev_weight = None

        super().__init__()

    def _getModel(self):
        m = gp.Model("PortfolioMarkowitz")
        SeedManager.set_gurobi_seed(m, self.seed)
        m.setParam("OutputFlag", 0)
        if self.threads is not None:
            m.setParam("Threads", self.threads)

        w = m.addVars(
            self.n_assets,
            lb=self.lb,
            ub=self.ub,
            vtype=gp.GRB.CONTINUOUS,
            name="w",
        )
        m.addConstr(gp.quicksum(w[i] for i in range(self.n_assets)) == self.budget)

        self.w = w
        self.z = None
        if self.fee_rate > 0:
            self.z = m.addVars(
                self.n_assets,
                lb=0.0,
                vtype=gp.GRB.CONTINUOUS,
                name="z",
            )
        self.turnover_constrs = []
        return m, w

    def _to_numpy_1d(self, arr, name):
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr, dtype=float).reshape(-1)
        if len(arr) != self.n_assets:
            raise ValueError(f"{name} length {len(arr)} != n_assets {self.n_assets}")
        return arr

    def _to_numpy_2d(self, arr, name):
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != self.n_assets:
            raise ValueError(
                f"{name} shape {arr.shape} must be (n_samples, {self.n_assets})"
            )
        if arr.shape[0] < 2:
            raise ValueError(f"{name} must contain at least 2 historical rows")
        return arr

    def _estimate_covariance(self, scenario_returns):
        returns = self._to_numpy_2d(scenario_returns, "scenario_returns")
        returns = returns[-self.cov_history :]
        cov = np.cov(returns, rowvar=False, ddof=1)
        cov = np.asarray(cov, dtype=float)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        cov = 0.5 * (cov + cov.T)
        if self.cov_reg > 0:
            cov = cov + self.cov_reg * np.eye(self.n_assets)
        return cov

    def _clear_turnover_constraints(self):
        if self.turnover_constrs:
            self._model.remove(self.turnover_constrs)
            self.turnover_constrs = []
            self._model.update()

    def setObj(self, cost=None, scenario_returns=None, prev_weight=None):
        if cost is not None:
            self._current_cost = self._to_numpy_1d(cost, "cost")
        if scenario_returns is not None:
            self._current_scenarios = self._to_numpy_2d(
                scenario_returns, "scenario_returns"
            )
        if prev_weight is not None:
            self._current_prev_weight = self._to_numpy_1d(prev_weight, "prev_weight")
        elif self.fee_rate > 0:
            self._current_prev_weight = np.zeros(self.n_assets, dtype=float)

        if self._current_cost is None:
            raise ValueError("Markowitz model requires a cost vector")
        if self._current_scenarios is None:
            raise ValueError(
                "Markowitz model requires scenario_returns to estimate covariance"
            )

        self._clear_turnover_constraints()

        cov = self._estimate_covariance(self._current_scenarios)
        linear_term = gp.quicksum(
            self._current_cost[i] * self.w[i] for i in range(self.n_assets)
        )
        risk_term = gp.QuadExpr()
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                if cov[i, j] != 0.0:
                    risk_term.add(cov[i, j] * self.w[i] * self.w[j])

        obj = linear_term + self.lambda_risk * risk_term

        if self.fee_rate > 0:
            if self._current_prev_weight is None:
                self._current_prev_weight = np.zeros(self.n_assets, dtype=float)
            for i in range(self.n_assets):
                c1 = self._model.addConstr(
                    self.z[i] >= self.w[i] - self._current_prev_weight[i]
                )
                c2 = self._model.addConstr(
                    self.z[i] >= -(self.w[i] - self._current_prev_weight[i])
                )
                self.turnover_constrs.extend([c1, c2])
            obj += self.fee_rate * gp.quicksum(
                self.z[i] for i in range(self.n_assets)
            )

        self._model.setObjective(obj, gp.GRB.MINIMIZE)
        self._model.update()

    def solve(self, cost=None, scenario_returns=None, prev_weight=None):
        if cost is not None or scenario_returns is not None or prev_weight is not None:
            self.setObj(
                cost=cost,
                scenario_returns=scenario_returns,
                prev_weight=prev_weight,
            )

        self._model.optimize()
        if self._model.status != gp.GRB.OPTIMAL:
            raise RuntimeError(
                f"Gurobi failed to solve Markowitz model: {self._model.status}"
            )

        sol = np.array([self.w[i].X for i in range(self.n_assets)], dtype=float)
        obj = float(self._model.ObjVal)
        return sol, obj
