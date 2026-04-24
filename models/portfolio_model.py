import numpy as np
import gurobipy as gp
from pyepo.model.grb.grbmodel import optGrbModel


from utils.seed_manager import SeedManager


class PortfolioModel(optGrbModel):
    """
    长仓投资组合优化模型

    当 fee_rate = 0 时：
        min c^T x

    当 fee_rate > 0 时：
        min c^T x + fee_rate * sum(|x - prev_weight|)
    """

    def __init__(
        self,
        n_assets,
        budget=1.0,
        lb=0.0,
        ub=1.0,
        fee_rate=0.0,
        threads=None,
        seed=42,
    ):
        self.n_assets = n_assets
        self.fee_rate = fee_rate
        self.seed = seed

        # ---------- 参数默认值处理 ----------
        if lb is None:
            lb = 0.0
        if ub is None:
            ub = 1.0

        # ---------- 参数检查 ----------
        if n_assets <= 0:
            raise ValueError("n_assets must be positive")
        if budget <= 0:
            raise ValueError("budget must be positive")
        if lb > ub:
            raise ValueError("lb cannot be greater than ub")
        if fee_rate < 0:
            raise ValueError("fee_rate must be non-negative")

        # 可行性检查
        if n_assets * lb > budget:
            raise ValueError("Infeasible model: n_assets * lb > budget")
        if n_assets * ub < budget:
            raise ValueError("Infeasible model: n_assets * ub < budget")

        # ---------- 创建 Gurobi 模型 ----------
        m = gp.Model()
        # 使用工具类设置 Gurobi 内部种子
        SeedManager.set_gurobi_seed(m, self.seed)
        # ...
        m.setParam("OutputFlag", 0)

        if threads is not None:
            m.setParam("Threads", threads)

        # ---------- 决策变量：资产权重 ----------
        x = m.addVars(
            n_assets,
            lb=lb,
            ub=ub,
            vtype=gp.GRB.CONTINUOUS,
            name="x",
        )

        # ---------- 预算约束 ----------
        m.addConstr(gp.quicksum(x[i] for i in range(n_assets)) == budget, name="budget")

        # ---------- 若考虑手续费，则创建 turnover 辅助变量 z ----------
        z = None
        if fee_rate > 0:
            z = m.addVars(
                n_assets,
                lb=0.0,
                vtype=gp.GRB.CONTINUOUS,
                name="z",
            )

        m.modelSense = gp.GRB.MINIMIZE

        self.m = m
        self.x = x
        self.z = z

        # 用来保存每次 setObj 动态添加的 turnover 约束
        self.turnover_constrs = []

        super().__init__()

    def _getModel(self):
        return self.m, self.x

    def _to_numpy_1d(self, arr, name):
        """把输入统一转成一维 numpy 数组，并检查长度。"""
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()

        arr = np.asarray(arr, dtype=float).reshape(-1)

        if len(arr) != self.n_assets:
            raise ValueError(f"{name} length {len(arr)} != n_assets {self.n_assets}")

        return arr

    def setObj(self, cost_vec, prev_weight=None):
        """
        设置目标函数。

        参数：
            cost_vec: 一维 cost 向量
            prev_weight: 上一期权重，仅在 fee_rate > 0 时需要
        """
        cost_vec = self._to_numpy_1d(cost_vec, "cost_vec")

        # ---------- 先删除旧的 turnover 约束 ----------
        if self.turnover_constrs:
            for constr in self.turnover_constrs:
                self.m.remove(constr)
            self.turnover_constrs.clear()
            self.m.update()

        # ---------- 基础目标：min c^T x ----------
        obj = gp.quicksum(cost_vec[i] * self.x[i] for i in range(self.n_assets))

        # ---------- 若考虑手续费，则加入 turnover 项 ----------
        if self.fee_rate > 0:
            if prev_weight is None:
                # 兼容 PyEPO 在 surrogate 内部直接调用 setObj(cp[i]) 的场景
                # 训练阶段缺失上一期权重时，使用零向量作为换手基准
                prev_weight = np.zeros(self.n_assets, dtype=float)
            else:
                prev_weight = self._to_numpy_1d(prev_weight, "prev_weight")

            # 线性化 |x_i - prev_weight_i|
            for i in range(self.n_assets):
                c1 = self.m.addConstr(self.z[i] >= self.x[i] - prev_weight[i])
                c2 = self.m.addConstr(self.z[i] >= -(self.x[i] - prev_weight[i]))
                self.turnover_constrs.extend([c1, c2])

            obj += self.fee_rate * gp.quicksum(self.z[i] for i in range(self.n_assets))

        self.m.setObjective(obj, gp.GRB.MINIMIZE)
        self.m.update()

    def solve(self, cost_vec=None, prev_weight=None):
        """
        求解模型。

        参数：
            cost_vec: 若提供，则先更新目标函数
            prev_weight: 当 fee_rate > 0 时需要

        返回：
            sol: 最优权重
            obj: 最优目标值
        """
        if cost_vec is not None:
            self.setObj(cost_vec, prev_weight=prev_weight)

        self.m.optimize()

        if self.m.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"Optimization failed. Gurobi status: {self.m.Status}")

        sol = np.array([self.x[i].X for i in range(self.n_assets)], dtype=float)
        obj = float(self.m.ObjVal)

        return sol, obj


# 实例化
# model = PortfolioModel(n_assets=8, fee_rate=0.003)
# w, obj = model.solve(cost_vec=c_hat, prev_weight=w_prev)
