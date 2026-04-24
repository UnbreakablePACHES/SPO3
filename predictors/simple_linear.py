import torch.nn as nn
import pandas as pd


class SimpleLinear(nn.Module):
    """Simple linear predictor producing per-asset scores."""

    def __init__(self, num_assets, input_dim):
        super().__init__()
        self.num_assets = num_assets
        self.input_dim = input_dim

        self.linear = nn.Linear(num_assets * input_dim, num_assets)

    def forward(self, x):
        """Run the linear layer on flattened features.

        Args:
            x: Tensor of shape ``(batch_size, num_assets, input_dim)``.

        Returns:
            Tensor of shape ``(batch_size, num_assets)`` containing predicted
            returns or costs per asset.
        """
        batch_size = x.size(0)
        x = x.reshape(batch_size, self.num_assets * self.input_dim)
        return self.linear(x)

    def get_feature_contributions(self, x_step, tickers, feature_cols, rebalance_date):
        """导出单次调仓时每个输出资产的输入特征贡献度。"""
        if x_step.ndim == 2:
            x_step = x_step.unsqueeze(0)
        x_step = x_step.detach().cpu()

        weight = self.linear.weight.detach().cpu()  # (num_assets, num_assets*input_dim)
        x_flat = x_step.reshape(-1)  # (num_assets*input_dim,)
        contrib = weight * x_flat.unsqueeze(0)  # (num_assets, num_assets*input_dim)

        rows = []
        for target_idx, target_ticker in enumerate(tickers):
            for source_idx, source_ticker in enumerate(tickers):
                base = source_idx * self.input_dim
                for f_idx, f_name in enumerate(feature_cols):
                    rows.append(
                        {
                            "rebalance_date": pd.to_datetime(rebalance_date),
                            "target_ticker": target_ticker,
                            "source_ticker": source_ticker,
                            "feature": f_name,
                            "contribution": float(
                                contrib[target_idx, base + f_idx].item()
                            ),
                        }
                    )
        return pd.DataFrame(rows)
