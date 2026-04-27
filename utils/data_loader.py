import numpy as np
import torch
from torch.utils.data import Dataset


class SPODataset(Dataset):
    """Dataset for SPO training.

    Features are daily cross-sectional asset features. The true cost is the
    negative cumulative log return over the next ``label_window`` trading days.
    """

    def __init__(self, df, context_history=0, label_window=21):
        self.context_history = int(context_history)
        self.label_window = int(label_window)
        if self.context_history < 0:
            raise ValueError("context_history must be non-negative")
        if self.label_window <= 0:
            raise ValueError("label_window must be a positive integer")

        self.tickers = sorted(df["ticker"].unique())
        self.num_assets = len(self.tickers)

        feature_cols = [
            c for c in df.columns if c not in ["Date", "ticker", "log_return"]
        ]
        self.input_dim = len(feature_cols)

        pivot_df = df.pivot(index="Date", columns="ticker").sort_index()
        if pivot_df.isnull().any().any():
            pivot_df = pivot_df.dropna()

        self.X_all = pivot_df[feature_cols].values.reshape(
            -1, self.num_assets, self.input_dim
        )
        self.R_all = pivot_df["log_return"].values
        self.C_all = self._build_forward_costs(self.R_all, self.label_window)

        self.start_idx = self.context_history
        self.end_idx = len(pivot_df) - self.label_window
        self.valid_indices = np.arange(self.start_idx, self.end_idx)

    @staticmethod
    def _build_forward_costs(returns, label_window):
        """Build costs from t+1 through t+label_window cumulative log returns."""
        costs = np.full_like(returns, np.nan, dtype=float)
        for i in range(len(returns) - label_window):
            forward_log_return = returns[i + 1 : i + 1 + label_window].sum(axis=0)
            costs[i] = -forward_log_return
        return costs

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        x = torch.FloatTensor(self.X_all[t])
        c = torch.FloatTensor(self.C_all[t])
        if self.context_history > 0:
            scenarios = torch.FloatTensor(self.R_all[t - self.context_history : t])
            return x, c, scenarios

        return x, c
