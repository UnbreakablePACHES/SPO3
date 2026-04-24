import numpy as np
import pandas as pd
import torch
import gurobipy as gp
from torch.utils.data import DataLoader, TensorDataset

from utils.window_generator import RollingWindowGenerator
from utils.trading_days import TradingDayShifter
from predictors.simple_linear import SimpleLinear


class BaselineRunner:
    """Run baseline strategies under the same rolling-window protocol."""

    def __init__(self, device=None, trading_days_path=None, seed=42):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seed = seed
        self.trading_day_shifter = (
            TradingDayShifter(trading_days_path) if trading_days_path else None
        )
        self.pto_predictions = None

    def _infer_backtest_range(self, df, window_months, test_start_date=None):
        first_date = pd.to_datetime(df["Date"].min())
        default_test_start = (first_date + pd.DateOffset(months=window_months)).replace(
            day=1
        )
        if test_start_date is not None:
            test_start = pd.to_datetime(test_start_date).replace(day=1)
            if test_start < default_test_start:
                raise ValueError(
                    "test_start_date 早于最早可回测日期。"
                    f"给定: {test_start.strftime('%Y-%m-%d')}，"
                    f"最早可用: {default_test_start.strftime('%Y-%m-%d')}"
                )
        else:
            test_start = default_test_start
        test_end = pd.to_datetime(df["Date"].max())
        return test_start, test_end

    def _build_rebalance_snapshot(
        self, test_data, rebalance_date, tickers, feature_cols, target_horizon
    ):
        test_data = test_data.copy()
        test_data["Date"] = pd.to_datetime(test_data["Date"])
        rebalance_date = pd.to_datetime(rebalance_date)

        if self.trading_day_shifter is not None:
            try:
                effective_date = self.trading_day_shifter.prev_or_same(rebalance_date)
            except IndexError:
                effective_date = None
        else:
            effective_date = test_data.loc[
                test_data["Date"] <= rebalance_date, "Date"
            ].max()

        if pd.isna(effective_date):
            raise ValueError(
                f"调仓日 {rebalance_date.date()} 之前无可用特征数据，无法预测。"
            )

        step_df = test_data[test_data["Date"] == effective_date]
        step_df = step_df.set_index("ticker").reindex(tickers)

        if step_df[feature_cols].isnull().any().any():
            raise ValueError(f"有效日 {effective_date.date()} 特征缺失。")

        x_step = step_df[feature_cols].values.astype(float)
        future_rows = test_data[test_data["Date"] > effective_date].copy()
        future_rows = future_rows.sort_values(["ticker", "Date"])
        true_r_vals = []
        for ticker in tickers:
            one = future_rows[future_rows["ticker"] == ticker]["log_return"].head(
                target_horizon
            )
            if len(one) < target_horizon:
                true_r_vals.append(np.nan)
            else:
                true_r_vals.append(float(one.sum()))
        true_r = np.array(true_r_vals, dtype=float)
        return torch.FloatTensor(x_step), effective_date, true_r

    def _build_return_stats(self, train_data, tickers):
        returns = (
            train_data.pivot(index="Date", columns="ticker", values="log_return")
            .sort_index()
            .reindex(columns=tickers)
            .dropna(how="any")
        )
        if returns.empty:
            return None, None
        mu = returns.mean(axis=0).values.astype(float)
        cov = returns.cov().values.astype(float)
        # 数值稳定性：对角线加微小抖动
        cov = cov + np.eye(len(tickers)) * 1e-6
        return mu, cov

    def _solve_markowitz(self, mu, cov, risk_aversion=10.0, lb=0.0, ub=1.0, budget=1.0):
        n_assets = len(mu)
        m = gp.Model("markowitz_baseline")
        m.setParam("OutputFlag", 0)

        w = m.addVars(n_assets, lb=lb, ub=ub, vtype=gp.GRB.CONTINUOUS, name="w")
        m.addConstr(gp.quicksum(w[i] for i in range(n_assets)) == budget, name="budget")

        quad_term = gp.quicksum(
            cov[i, j] * w[i] * w[j] for i in range(n_assets) for j in range(n_assets)
        )
        linear_term = gp.quicksum(mu[i] * w[i] for i in range(n_assets))

        # maximize(mu^T w - lambda * w^T Sigma w)
        m.setObjective(risk_aversion * quad_term - linear_term, sense=gp.GRB.MINIMIZE)
        m.optimize()

        if m.status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"Markowitz baseline 求解失败，状态码={m.status}")

        return np.array([w[i].X for i in range(n_assets)], dtype=float)

    def _normalize_window_features(self, train_data, test_data, feature_cols):
        """按 ticker 在训练窗口内做 z-score 标准化，与 SPOBacktester 保持一致。"""
        train_data = train_data.copy()
        test_data = test_data.copy()

        stat_frames = []
        for col in feature_cols:
            grouped = (
                train_data.groupby("ticker")[col].agg(["mean", "std"]).reset_index()
            )
            grouped = grouped.rename(
                columns={"mean": f"{col}__mean", "std": f"{col}__std"}
            )
            stat_frames.append(grouped)

        stats = stat_frames[0]
        for g in stat_frames[1:]:
            stats = stats.merge(g, on="ticker", how="inner")

        for col in feature_cols:
            std_col = f"{col}__std"
            stats[std_col] = stats[std_col].replace(0, 1.0).fillna(1.0)

        train_data = train_data.merge(stats, on="ticker", how="left")
        test_data = test_data.merge(stats, on="ticker", how="left")

        for col in feature_cols:
            mean_col = f"{col}__mean"
            std_col = f"{col}__std"
            train_data[col] = (train_data[col] - train_data[mean_col]) / train_data[
                std_col
            ]
            test_data[col] = (test_data[col] - test_data[mean_col]) / test_data[std_col]

        drop_cols = [c for c in stats.columns if c != "ticker"]
        train_data = train_data.drop(columns=drop_cols)
        test_data = test_data.drop(columns=drop_cols)
        return train_data, test_data

    def _fit_simple_linear_predictor(
        self, train_data, tickers, feature_cols, epochs=30, lr=1e-3, target_horizon=20
    ):
        pivot = train_data.pivot(index="Date", columns="ticker").sort_index()
        x_all = pivot[feature_cols].values.reshape(-1, len(tickers), len(feature_cols))
        y_raw = pivot["log_return"].values.astype(float)
        y_all = np.full_like(y_raw, np.nan, dtype=float)
        for t in range(len(y_raw) - target_horizon):
            y_all[t] = y_raw[t + 1 : t + 1 + target_horizon].sum(axis=0)
        valid_mask = ~np.isnan(y_all).any(axis=1)
        x_all = x_all[valid_mask]
        y_all = y_all[valid_mask]
        if len(x_all) == 0:
            raise ValueError("训练窗口不足以构造目标 horizon 标签。")

        dataset = TensorDataset(
            torch.tensor(x_all, dtype=torch.float32),
            torch.tensor(y_all, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=min(64, len(dataset)), shuffle=True)

        model = SimpleLinear(num_assets=len(tickers), input_dim=len(feature_cols)).to(
            self.device
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        model.train()
        for _ in range(epochs):
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return model

    def run_markowitz(
        self,
        df,
        window_months=12,
        freq="MS",
        test_start_date=None,
        risk_aversion=10.0,
    ):
        tickers = sorted(df["ticker"].unique())
        test_start, test_end = self._infer_backtest_range(
            df, window_months=window_months, test_start_date=test_start_date
        )

        generator = RollingWindowGenerator(
            test_start=test_start.strftime("%Y-%m-%d"),
            test_end=test_end.strftime("%Y-%m-%d"),
            window_months=window_months,
            freq=freq,
        )

        all_weights = []
        rebalance_dates = []
        holding_periods = []

        for window in generator:
            train_data = df[
                (df["Date"] >= window.train_start) & (df["Date"] <= window.train_end)
            ]
            if train_data.empty:
                continue

            mu, cov = self._build_return_stats(train_data=train_data, tickers=tickers)
            if mu is None:
                continue

            weights = self._solve_markowitz(mu=mu, cov=cov, risk_aversion=risk_aversion)
            rebalance_dt = pd.to_datetime(window.test_start)

            all_weights.append(weights)
            rebalance_dates.append(rebalance_dt)
            holding_periods.append((rebalance_dt, pd.to_datetime(window.test_end)))

        weights_df = pd.DataFrame(all_weights, index=rebalance_dates, columns=tickers)
        return weights_df, holding_periods

    def run_simplelinear_po_markowitz(
        self,
        df,
        window_months=12,
        freq="MS",
        test_start_date=None,
        risk_aversion=10.0,
        pred_epochs=30,
        pred_lr=1e-3,
        target_horizon=20,
    ):
        tickers = sorted(df["ticker"].unique())
        feature_cols = [
            c for c in df.columns if c not in ["Date", "ticker", "log_return"]
        ]

        test_start, test_end = self._infer_backtest_range(
            df, window_months=window_months, test_start_date=test_start_date
        )
        generator = RollingWindowGenerator(
            test_start=test_start.strftime("%Y-%m-%d"),
            test_end=test_end.strftime("%Y-%m-%d"),
            window_months=window_months,
            freq=freq,
        )

        all_weights = []
        rebalance_dates = []
        holding_periods = []
        prediction_rows = []

        for window in generator:
            train_data = df[
                (df["Date"] >= window.train_start) & (df["Date"] <= window.train_end)
            ]
            test_data = df[
                (df["Date"] >= pd.to_datetime(window.test_start) - pd.Timedelta(days=7))
                & (df["Date"] <= pd.to_datetime(window.test_end))
            ]
            if train_data.empty or test_data.empty:
                continue

            mu_hist, cov = self._build_return_stats(
                train_data=train_data, tickers=tickers
            )
            if mu_hist is None:
                continue

            train_data, test_data = self._normalize_window_features(
                train_data=train_data,
                test_data=test_data,
                feature_cols=feature_cols,
            )

            predictor = self._fit_simple_linear_predictor(
                train_data=train_data,
                tickers=tickers,
                feature_cols=feature_cols,
                epochs=pred_epochs,
                lr=pred_lr,
                target_horizon=target_horizon,
            )
            x_step, effective_date, true_r = self._build_rebalance_snapshot(
                test_data=test_data,
                rebalance_date=pd.to_datetime(window.test_start),
                tickers=tickers,
                feature_cols=feature_cols,
                target_horizon=target_horizon,
            )
            x_step = x_step.unsqueeze(0)

            predictor.eval()
            with torch.no_grad():
                mu_pred = predictor(x_step.to(self.device)).cpu().numpy().reshape(-1)
            prediction_rows.extend(
                [
                    {
                        "rebalance_date": pd.to_datetime(window.test_start),
                        "effective_date": effective_date,
                        "ticker": ticker,
                        "pto_pred": float(mu_pred[i]),
                        "true_r": float(true_r[i]),
                    }
                    for i, ticker in enumerate(tickers)
                ]
            )

            weights = self._solve_markowitz(
                mu=mu_pred, cov=cov, risk_aversion=risk_aversion
            )
            rebalance_dt = pd.to_datetime(window.test_start)

            all_weights.append(weights)
            rebalance_dates.append(rebalance_dt)
            holding_periods.append((rebalance_dt, pd.to_datetime(window.test_end)))

        weights_df = pd.DataFrame(all_weights, index=rebalance_dates, columns=tickers)
        if prediction_rows:
            self.pto_predictions = pd.DataFrame(prediction_rows)
        else:
            self.pto_predictions = pd.DataFrame(
                columns=["rebalance_date", "effective_date", "ticker", "pto_pred", "true_r"]
            )
        return weights_df, holding_periods
