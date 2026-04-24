import numpy as np
import gurobipy as gp
from pyepo.model.grb import optGrbModel
from utils.seed_manager import SeedManager


class PortfolioCVaRModel(optGrbModel):
    """
    基于条件风险价值 (CVaR) 的投资组合优化模型。
    该模型继承自 PyEPO 的 optGrbModel，使其能够作为 SPO+ Loss 的下游优化器。

    优化目标： minimize (预测成本 * 权重) + lambda * CVaR风险项
    """

    def __init__(
        self, n_assets, alpha=0.95, lambda_cvar=1.0, budget=1.0, lb=0.0, ub=1.0, seed=42
    ):
        """
        参数:
            n_assets: 资产数量 (决策变量的数量)
            alpha: CVaR 的置信水平 (例如 0.95 表示计算最差 5% 情况的平均损失)
            lambda_cvar: 风险厌恶系数 (权重越大，模型越看重控制风险)
            budget: 总预算约束 (资产权重之和)
            lb: 每个资产权重的下限
            ub: 每个资产权重的上限
            seed: 随机种子，确保实验可重复
        """
        self.n_assets = n_assets
        self.alpha = alpha
        self.lambda_cvar = lambda_cvar
        self.budget = budget
        self.lb = lb
        self.ub = ub
        self.seed = seed

        # 标志位：告知 SPO 框架此模型需要历史收益率“场景”数据来计算风险
        self.requires_scenarios = True

        # === 核心状态暂存 ===
        # 由于 PyEPO 在计算 SPO+ 梯度时会多次调用 setObj 且仅传递 cost，
        # 我们需要暂存当前的场景数据和 Cost，以保证优化结构完整。
        self._current_scenarios = None  # 存储用于计算风险的历史收益率矩阵 (S x N)
        self._current_cost = None  # 存储当前的成本向量 (通常是预测的负收益率)

        # 调用父类初始化，这会自动触发 _getModel() 的调用
        super().__init__()

    def _getModel(self):
        """
        构建 Gurobi 静态模型结构。定义变量和基础约束。
        """
        m = gp.Model("PortfolioCVaR")
        SeedManager.set_gurobi_seed(m, self.seed)  # 设置 Gurobi 内部随机种子
        m.setParam("OutputFlag", 0)  # 禁用 Gurobi 控制台输出

        # 1. 定义决策变量：每个资产的投资权重 w
        w = m.addVars(self.n_assets, lb=self.lb, ub=self.ub, name="w")

        # 2. 添加基础约束：所有资产权重之和等于总预算
        m.addConstr(gp.quicksum(w[i] for i in range(self.n_assets)) == self.budget)

        # 3. 定义 CVaR 辅助变量：eta (VaR 值)
        # VaR 是风险阈值，其取值范围可以是任意实数
        self.eta = m.addVar(lb=-gp.GRB.INFINITY, name="eta")

        # 初始化存储动态约束和变量的列表
        self.w = w
        self.u_vars = []  # 用于存储每个场景超额损失的辅助变量
        self.cvar_constrs = []  # 用于存储 CVaR 线性化约束

        return m, w

    def setObj(self, cost=None, scenario_returns=None):
        """
        动态更新模型的目标函数和约束。
        在 SPO 训练过程中，预测值 (cost) 改变时会频繁调用此函数。
        """
        # 1. 优先更新场景数据：如果有新的场景传入（例如进入新的 Batch），更新暂存
        if scenario_returns is not None:
            if hasattr(scenario_returns, "detach"):
                scenario_returns = scenario_returns.detach().cpu().numpy()
            self._current_scenarios = scenario_returns

        # 2. 更新成本向量：如果是 PyEPO 内部计算梯度，会传入新的预测 cost
        if cost is not None:
            if hasattr(cost, "detach"):
                cost = cost.detach().cpu().numpy()
            self._current_cost = cost

        # 3. 完整性检查：确保在执行优化前，所有必要数据都已到位
        if self._current_scenarios is None:
            raise ValueError("CVaR 模型运行需要 scenario_returns (历史场景数据)。")
        if self._current_cost is None:
            raise ValueError("CVaR 模型运行需要 cost vector (成本向量)。")

        # 4. 动态构建 CVaR 线性化约束
        # 每次 cost 改变时，为了保持模型纯净，先移除旧的场景变量和约束
        if self.cvar_constrs:
            self._model.remove(self.cvar_constrs)
        if hasattr(self, "u_vars") and self.u_vars:
            self._model.remove(self.u_vars)

        S = self._current_scenarios.shape[0]  # 获取场景数量 (行数)
        # 定义辅助变量 u_s：表示第 s 个场景下的超额损失
        self.u_vars = self._model.addVars(S, lb=0.0, name="u")
        self.cvar_constrs = []

        # 遍历每个历史场景，添加线性化约束：u_s >= 损失 - eta
        for s in range(S):
            # 场景损失 = -(场景收益率 * 权重)
            scenario_loss = -gp.quicksum(
                self._current_scenarios[s, i] * self.w[i] for i in range(self.n_assets)
            )
            # 添加约束：u_s >= scenario_loss - eta
            c = self._model.addConstr(self.u_vars[s] >= scenario_loss - self.eta)
            self.cvar_constrs.append(c)

        # 5. 定义并设置复合目标函数
        # 部分 A: 线性预期成本 (预期收益的负值)
        linear_term = gp.quicksum(
            self._current_cost[i] * self.w[i] for i in range(self.n_assets)
        )

        # 部分 B: CVaR 风险度量公式
        # CVaR = eta + 1/((1-alpha)*S) * sum(u_s)
        cvar_term = self.eta + (1.0 / ((1.0 - self.alpha) * S)) * gp.quicksum(
            self.u_vars
        )

        # 最终目标：最小化 (预期成本 + 风险系数 * CVaR)
        self._model.setObjective(
            linear_term + self.lambda_cvar * cvar_term, gp.GRB.MINIMIZE
        )

        # 更新模型，使修改生效
        self._model.update()

    def solve(self, cost=None, scenario_returns=None):
        """
        统一的求解接口，适配 PyEPO 自动调用和用户手动调用。
        """
        # 如果手动传入了数据（如在训练脚本中计算 Oracle 解时），先更新状态
        if cost is not None or scenario_returns is not None:
            self.setObj(cost=cost, scenario_returns=scenario_returns)

        # 调用 Gurobi 求解器执行优化
        self._model.optimize()

        # 检查是否找到最优解
        if self._model.status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi 未找到最优解，状态码={self._model.status}")

        # 提取最优决策变量值 (资产权重)
        sol = np.array([self.w[i].X for i in range(self.n_assets)])
        # 获取最优目标函数值
        obj = self._model.ObjVal

        return sol, obj
