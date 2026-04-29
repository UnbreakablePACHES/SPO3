import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from utils.window_generator import RollingWindowGenerator
from utils.data_loader import SPODataset
from utils.metrics import StrategyEvaluator
from utils.trading_days import TradingDayShifter
from utils.prediction_transforms import rescale_to_range
from predictors.simple_linear import SimpleLinear
from losses.SPOplus_loss import SPOPlusLoss


class SPOBacktester:
    """SPO 滚动窗口回测引擎（按月调仓 + 窗口滚动 + 区间持有）。"""

    def __init__(self, opt_model, device=None, trading_days_path=None):
        self.opt_model = opt_model
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.results = None
        self.holding_periods = []
        self.trading_day_shifter = (
            TradingDayShifter(trading_days_path) if trading_days_path else None
        )
        self.seed = 42
        self.feature_contributions = None
        self.predicted_returns = None
        self.target_results = None

    def run(
        self,
        df,
        trainer_cls,
        window_months=12,
        epochs=20,
        lr=1e-3,
        batch_size=32,
        freq="MS",
        test_start_date=None,
        seed=42,
        normalize_features=True,
        context_history=20,
        label_window=21,
        prediction_return_clip=None,
        prediction_return_rescale_range=None,
        weight_adjust_delta=None,
    ):
        """
        执行滚动回测。

        逻辑与论文 4.2 对齐：
        - 每个调仓月仅使用调仓日前可见历史数据训练（滚动窗口）
        - 每次滚动都从头训练预测器
        - 在调仓日生成权重并持有到下次调仓日前一日
        """
        tickers = sorted(df["ticker"].unique())
        num_assets = len(tickers)
        feature_cols = [
            c for c in df.columns if c not in ["Date", "ticker", "log_return"]
        ]
        input_dim = len(feature_cols)

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

        generator = RollingWindowGenerator(
            test_start=test_start.strftime("%Y-%m-%d"),
            test_end=test_end.strftime("%Y-%m-%d"),
            window_months=window_months,
            freq=freq,
        )

        all_weights = []
        rebalance_dates = []
        holding_periods = []
        contribution_rows = []
        predicted_return_rows = []
        target_weights_rows = []
        requires_scenarios = getattr(self.opt_model, "requires_scenarios", False)
        scenario_history = int(
            getattr(self.opt_model, "scenario_history", context_history)
        )
        self.seed = seed

        for window_idx, window in enumerate(generator):
            print(
                f"==> 回测区间: {window.test_start} ~ {window.test_end} | 训练窗口: {window.train_start} ~ {window.train_end}"
            )

            # 确保在提取训练和测试数据时，范围是足够宽的
            train_data = df[
                (df["Date"] >= window.train_start) & (df["Date"] <= window.train_end)
            ]
            # 修改测试数据的提取，使其允许包含寻找历史特征所需的 buffer
            # buffer=20天，确保技术指标（如20日MA）能完整计算
            test_data = df[
                (df["Date"] >= pd.to_datetime(window.test_start) - pd.Timedelta(days=20))
                & (df["Date"] <= pd.to_datetime(window.test_end))
            ]

            if train_data.empty or test_data.empty:
                continue
            if normalize_features:
                train_data, test_data = self._normalize_window_features(
                    train_data=train_data,
                    test_data=test_data,
                    feature_cols=feature_cols,
                )

            train_ds = SPODataset(
                train_data,
                context_history=scenario_history if requires_scenarios else 0,
                label_window=label_window,
            )
            if len(train_ds) == 0:
                if requires_scenarios:
                    print(
                        "[Skip] Not enough training history for scenario model: "
                        f"need more than scenario_history({scenario_history}) + "
                        f"label_window({label_window}) observations, "
                        f"got {len(train_data['Date'].unique())} dates."
                    )
                continue
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(self.seed + window_idx),
            )

            predictor = SimpleLinear(num_assets=num_assets, input_dim=input_dim).to(
                self.device
            )
            loss_fn = SPOPlusLoss(self.opt_model).to(self.device)
            trainer = trainer_cls(
                model=predictor,
                loss_fn=loss_fn,
                lr=lr,
                device=self.device,
            )
            trainer.fit(train_loader, epochs=epochs)

            rebalance_date = pd.to_datetime(window.test_start)
            x_step = self._build_rebalance_input(
                test_data=test_data,
                rebalance_date=rebalance_date,
                tickers=tickers,
                feature_cols=feature_cols,
            )
            pred_cost = trainer.predict(x_step).flatten()
            pred_return = -pred_cost
            if (
                prediction_return_clip is not None
                and prediction_return_rescale_range is not None
            ):
                raise ValueError(
                    "Use either prediction_return_clip or "
                    "prediction_return_rescale_range, not both"
                )
            if prediction_return_clip is not None:
                clip = abs(float(prediction_return_clip))
                pred_return = np.clip(pred_return, -clip, clip)
                pred_cost = -pred_return
            elif prediction_return_rescale_range is not None:
                pred_return = rescale_to_range(
                    pred_return, prediction_return_rescale_range
                )
                pred_cost = -pred_return
            contrib_df = predictor.get_feature_contributions(
                x_step=x_step,
                tickers=tickers,
                feature_cols=feature_cols,
                rebalance_date=rebalance_date,
            )
            contribution_rows.append(contrib_df)

            initial_w = (
                np.ones(num_assets, dtype=float) / num_assets
                if weight_adjust_delta is not None
                else np.zeros(num_assets, dtype=float)
            )
            prev_w = all_weights[-1] if all_weights else initial_w
            if requires_scenarios:
                scenario_returns = self._build_scenarios(
                    train_data=train_data,
                    tickers=tickers,
                    context_history=scenario_history,
                )
                if scenario_returns is None:
                    continue
                solve_kwargs = {
                    "cost": pred_cost,
                    "scenario_returns": scenario_returns,
                }
                if getattr(self.opt_model, "supports_turnover", False):
                    solve_kwargs["prev_weight"] = prev_w
                target_weights, _ = self.opt_model.solve(**solve_kwargs)
            else:
                target_weights, _ = self.opt_model.solve(
                    cost_vec=pred_cost, prev_weight=prev_w
                )

            target_weights = np.asarray(target_weights, dtype=float)
            if weight_adjust_delta is not None:
                delta = float(weight_adjust_delta)
                if not 0 < delta <= 1:
                    raise ValueError("weight_adjust_delta must satisfy 0 < delta <= 1")
                weights = prev_w + delta * (target_weights - prev_w)
            else:
                weights = target_weights

            all_weights.append(weights)
            target_weights_rows.append(target_weights)
            rebalance_dt = pd.to_datetime(window.test_start)
            rebalance_dates.append(rebalance_dt)
            holding_periods.append((rebalance_dt, pd.to_datetime(window.test_end)))
            # SPO is trained on cost = -forward return.
            predicted_return_rows.append(pred_return)

        self.results = pd.DataFrame(all_weights, index=rebalance_dates, columns=tickers)
        if self.results.empty:
            if requires_scenarios:
                raise ValueError(
                    "No rebalance weights were generated. Scenario-based models need "
                    f"more than scenario_history({scenario_history}) + "
                    f"label_window({label_window}) trading days inside each training "
                    "window. Increase hyperparams.window_months, reduce cov_history/"
                    "context_history, or move backtest_start_date later."
                )
            raise ValueError(
                "No rebalance weights were generated. Check the date range, "
                "backtest_start_date, and rolling window settings."
            )
        self.predicted_returns = pd.DataFrame(
            predicted_return_rows, index=rebalance_dates, columns=tickers
        )
        self.target_results = pd.DataFrame(
            target_weights_rows, index=rebalance_dates, columns=tickers
        )
        self.holding_periods = holding_periods
        if contribution_rows:
            self.feature_contributions = pd.concat(
                contribution_rows, axis=0, ignore_index=True
            )
        else:
            self.feature_contributions = pd.DataFrame(
                columns=[
                    "rebalance_date",
                    "target_ticker",
                    "source_ticker",
                    "feature",
                    "contribution",
                ]
            )
        return self.results

    def build_feature_contribution_timeseries(
        self, use_abs=True, aggfunc="mean", source_ticker=None, target_ticker=None
    ):
        """
        将逐次调仓的贡献明细聚合成“时间 x 特征”矩阵。

        Args:
            use_abs: 是否使用绝对值贡献度，默认 True（更能反映贡献强度）。
            aggfunc: 聚合方式，支持 "mean" / "sum" / "median"。
            source_ticker: 若提供，仅聚合该 source_ticker 的贡献。
            target_ticker: 若提供，仅聚合该 target_ticker 的贡献。
        """
        if self.feature_contributions is None or self.feature_contributions.empty:
            return pd.DataFrame()

        contrib = self.feature_contributions.copy()
        contrib["rebalance_date"] = pd.to_datetime(contrib["rebalance_date"])

        if source_ticker is not None:
            contrib = contrib[contrib["source_ticker"] == source_ticker]
        if target_ticker is not None:
            contrib = contrib[contrib["target_ticker"] == target_ticker]
        if contrib.empty:
            return pd.DataFrame()

        value_col = "contribution"
        if use_abs:
            contrib["contrib_value"] = contrib["contribution"].abs()
            value_col = "contrib_value"

        pivot = contrib.pivot_table(
            index="feature",
            columns="rebalance_date",
            values=value_col,
            aggfunc=aggfunc,
        ).sort_index(axis=1)
        return pivot

    def plot_feature_contribution_heatmap(
        self,
        save_path,
        use_abs=True,
        aggfunc="mean",
        source_ticker=None,
        target_ticker=None,
    ):
        """
        绘制 SPO 的时间序列特征贡献热图（纵轴特征，横轴调仓时间）。
        """
        heatmap_data = self.build_feature_contribution_timeseries(
            use_abs=use_abs,
            aggfunc=aggfunc,
            source_ticker=source_ticker,
            target_ticker=target_ticker,
        )
        if heatmap_data.empty:
            return heatmap_data

        # 仅显示年月，避免横轴太拥挤
        x_labels = [dt.strftime("%Y-%m") for dt in heatmap_data.columns]

        sns.set_theme(style="white")
        plt.figure(
            figsize=(max(12, len(x_labels) * 0.45), max(6, len(heatmap_data) * 0.4))
        )
        ax = sns.heatmap(
            heatmap_data,
            cmap="RdYlBu_r" if not use_abs else "YlOrRd",
            cbar_kws={"label": "Feature Contribution"},
            linewidths=0.2,
            linecolor="white",
        )
        ax.set_title(
            "SPO Feature Contribution Heatmap (Time Series)", fontsize=13, pad=12
        )
        ax.set_xlabel("Rebalance Date")
        ax.set_ylabel("Feature")
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return heatmap_data

    def _normalize_window_features(self, train_data, test_data, feature_cols):
        """按 ticker 在滚动窗口内做 z-score 标准化，避免未来信息泄漏。"""
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
            mean_col = f"{col}__mean"
            std_col = f"{col}__std"
            std_safe = stats[std_col].replace(0, 1.0).fillna(1.0)
            stats[std_col] = std_safe

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

    def _build_rebalance_input(self, test_data, rebalance_date, tickers, feature_cols):
        test_data = test_data.copy()
        test_data["Date"] = pd.to_datetime(test_data["Date"])
        rebalance_date = pd.to_datetime(rebalance_date)

        # 核心修复：寻找“截止到调仓日”最新的特征，而不是“调仓日当天或之后”
        if self.trading_day_shifter is not None:
            try:
                # 寻找 rebalance_date 当天或之前的最后一个交易日
                effective_date = self.trading_day_shifter.prev_or_same(rebalance_date)
            except IndexError:
                effective_date = None
        else:
            # 如果没有交易日历，从数据集中找不大于调仓日的最大日期
            effective_date = test_data.loc[
                test_data["Date"] <= rebalance_date, "Date"
            ].max()

        if pd.isna(effective_date):
            raise ValueError(
                f"调仓日 {rebalance_date.date()} 之前无可用特征数据，无法预测。"
            )

        # 打印警告，方便追踪非交易日调仓的情况
        if effective_date != rebalance_date:
            print(
                f"[Safe] 调仓日 {rebalance_date.date()} 采用历史最近交易日 {effective_date.date()} 的特征做预测。"
            )

        # 提取该有效日期的特征
        step_df = test_data[test_data["Date"] == effective_date]

        # 保持资产顺序一致并转为张量
        step_df = step_df.set_index("ticker").reindex(tickers)
        if step_df[feature_cols].isnull().any().any():
            raise ValueError(f"有效日 {effective_date.date()} 特征缺失。")

        x_step = step_df[feature_cols].values.astype(float)
        return torch.FloatTensor(x_step)

    def _build_scenarios(self, train_data, tickers, context_history):
        pivot_returns = (
            train_data.pivot(index="Date", columns="ticker", values="log_return")
            .sort_index()
            .reindex(columns=tickers)
        )
        if len(pivot_returns) < context_history:
            return None
        return pivot_returns.tail(context_history).values.astype(float)

    def evaluate(self, df, fee_rate=0.005):
        if self.results is None:
            raise ValueError("请先运行 run() 执行回测")

        returns_df = df.pivot(
            index="Date", columns="ticker", values="log_return"
        ).sort_index()
        evaluator = StrategyEvaluator(fee_rate=fee_rate)
        metrics = evaluator.calculate_tearsheet(
            weights_df=self.results,
            returns_df=returns_df,
            holding_periods=self.holding_periods,
        )
        evaluator.print_summary(metrics)
        return metrics
