import torch
from torch.utils.data import Dataset
import numpy as np


class SPODataset(Dataset):
    """
    通用型 SPO 数据集加载器。
    适配基础组合优化 (Standard) 和带风险约束的优化 (CVaR)。
    """

    def __init__(self, df, context_history=0, target_horizon=1):
        self.context_history = context_history
        self.target_horizon = int(target_horizon)
        if self.target_horizon < 1:
            raise ValueError("target_horizon 必须 >= 1")
        self.tickers = sorted(df["ticker"].unique())
        self.num_assets = len(self.tickers)

        # 1. 数据透视与对齐
        feature_cols = [
            c for c in df.columns if c not in ["Date", "ticker", "log_return"]
        ]
        self.input_dim = len(feature_cols)

        # 将长表转为宽表（这里是产生 NaN 的地方）
        pivot_df = df.pivot(index="Date", columns="ticker")

        if pivot_df.isnull().any().any():
            pivot_df = pivot_df.dropna()

        # 转换为 numpy 矩阵: (Dates, Assets, Features)
        self.X_all = pivot_df[feature_cols].values.reshape(
            -1, self.num_assets, self.input_dim
        )
        # 收益率矩阵: (Dates, Assets)
        self.R_all = pivot_df["log_return"].values.astype(float)

        # 未来 target_horizon 天累计收益标签（不含当日，预测 t+1...t+h）
        n_dates = self.R_all.shape[0]
        self.R_targets = np.full_like(self.R_all, np.nan, dtype=float)
        h = self.target_horizon
        for t in range(n_dates - h):
            self.R_targets[t] = self.R_all[t + 1 : t + 1 + h].sum(axis=0)

        # 2. 确定有效索引范围
        self.start_idx = context_history
        max_valid_exclusive = len(pivot_df) - self.target_horizon
        if max_valid_exclusive <= self.start_idx:
            self.valid_indices = np.array([], dtype=int)
        else:
            self.valid_indices = np.arange(self.start_idx, max_valid_exclusive)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        x = torch.FloatTensor(self.X_all[t])
        c = torch.FloatTensor(-self.R_targets[t])  # 预测目标：未来 horizon 天负累计收益（成本）

        if self.context_history > 0:
            # 严格逻辑：scenarios = [t-history, t-1] 的真实收益
            # 这样在训练 SPO+ 计算 oracle 解时，风险评估完全基于历史观测值
            scenarios = torch.FloatTensor(self.R_all[t - self.context_history : t])
            return x, c, scenarios

        return x, c
