import torch
import numpy as np
from pyepo.func import SPOPlus


class SPOPlusLoss(torch.nn.Module):
    """
    PyTorch 封装的 SPO+ loss 模块。

    作用：
        1. 接收模型输出的预测 cost 向量 pred_cost
        2. 接收真实 cost 向量 true_cost
        3. 对每个样本的真实 cost 调用优化器，求出 oracle 解 true_sols 和对应目标值 true_objs
        4. 将这些信息传给 PyEPO 提供的 SPOPlus loss 进行计算

    说明：
        PyEPO 中的 SPOPlus 通常需要以下几个输入：
            - pred_cost : 预测的 cost
            - true_cost : 真实的 cost
            - true_sols : 在真实 cost 下求得的最优解 x*(c)
            - true_objs : 在真实 cost 下的最优目标值 z*(c)

        这里为了方便，直接在 forward 中根据 true_cost 逐个求 oracle 解。
    """

    def __init__(self, opt_model):
        """
        初始化 SPO+ loss 模块。

        Args:
            opt_model:
                下游优化模型对象，通常应当实现 solve(cost) 方法。
                solve(cost) 需要返回：
                    - sol : 对应 cost 下的最优决策解
                    - obj : 对应最优目标值
        """
        super().__init__()

        # 保存下游优化模型
        self.opt_model = opt_model

        # PyEPO 提供的 SPO+ 损失函数对象
        # 它会利用优化模型结构来构造 surrogate loss
        self.loss_fn = SPOPlus(opt_model)

    def forward(self, pred_cost, true_cost, scenario_returns=None):
        """
        计算一个 batch 的 SPO+ loss。

        Args:
            pred_cost (torch.Tensor):
                模型预测的 cost 张量，形状为 (B, N)
                - B: batch size
                - N: 资产数 / 决策变量维度

            true_cost (torch.Tensor):
                真实的 cost 张量，形状为 (B, N)

        Returns:
            torch.Tensor:
                一个标量张量，表示当前 batch 的 SPO+ loss
        """

        # B: batch size
        # N: 每个样本的 cost 向量维度（例如资产数）
        B, N = pred_cost.shape

        # 用于保存 batch 中每个样本在真实 cost 下的最优解 x*(c)
        true_sols = []

        # 用于保存 batch 中每个样本在真实 cost 下的最优目标值 z*(c)
        true_objs = []

        # 对 batch 中每个样本分别求 oracle 解
        for i in range(B):
            # 取出第 i 个样本的真实 cost
            # 先从计算图中 detach，避免无意义的梯度传播
            # 再搬到 CPU 并转成 numpy，因为很多优化器 / Gurobi 接口通常吃 numpy
            ct_np = true_cost[i].detach().cpu().numpy()

            # 调用下游优化模型求解：
            # sol: 在真实 cost 下的最优决策
            # obj: 对应的最优目标值
            if getattr(self.opt_model, "requires_scenarios", False):
                if scenario_returns is None:
                    raise ValueError("CVaR 模式需要 scenario_returns，但当前为 None")
                sc_np = scenario_returns[i].detach().cpu().numpy()
                # 这里的调用现在会被 PortfolioCVaRModel.solve(cost=..., scenario_returns=...) 正确处理
                sol, obj = self.opt_model.solve(cost=ct_np, scenario_returns=sc_np)
            elif getattr(self.opt_model, "fee_rate", 0) > 0:
                # 训练阶段无法追踪每个历史样本对应的实际前期持仓，
                # 使用零向量作为 prev_weight 是已知的近似处理（SPO+ with tx cost 的常见做法）。
                # 这会低估换仓成本，导致 oracle 解偏向较激进的换仓策略，但训练信号仍然有效。
                prev_weight = torch.zeros_like(true_cost[i]).detach().cpu().numpy()
                sol, obj = self.opt_model.solve(ct_np, prev_weight=prev_weight)
            else:
                sol, obj = self.opt_model.solve(ct_np)

            # 记录该样本的 oracle 解和目标值
            true_sols.append(sol)
            true_objs.append(obj)

        # 将 list 转成 torch.Tensor，并放到和 pred_cost 相同的 device 上
        # true_sols 的形状应为 (B, N)
        true_sols = torch.as_tensor(
            np.asarray(true_sols), dtype=torch.float32, device=pred_cost.device
        )

        # true_objs 的形状应为 (B,)
        true_objs = torch.as_tensor(
            np.asarray(true_objs), dtype=torch.float32, device=pred_cost.device
        )

        # 调用 PyEPO 的 SPO+ 损失函数
        # 输入：
        #   pred_cost : 预测 cost
        #   true_cost : 真实 cost
        #   true_sols : 真实 cost 对应的最优解
        #   true_objs : 真实 cost 对应的最优目标值
        loss = self.loss_fn(pred_cost, true_cost, true_sols, true_objs)

        # 返回当前 batch 的标量 loss
        return loss
