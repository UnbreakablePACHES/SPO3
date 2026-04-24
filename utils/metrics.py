import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils.logger import ProjectLogger


class StrategyEvaluator:
    """策略性能评估器。"""

    def __init__(
        self, fee_rate=0.0, risk_free_rate=0.0, annual_periods=252, returns_type="log"
    ):
        self.fee_rate = fee_rate
        self.rf = risk_free_rate
        self.annual_periods = annual_periods
        self.returns_type = returns_type

    def _expand_weights_to_daily(self, weights_df, returns_df, holding_periods=None):
        """将调仓日的权重映射到每日。"""
        daily = pd.DataFrame(
            index=returns_df.index, columns=weights_df.columns, dtype=float
        )

        if holding_periods is None or len(holding_periods) == 0:
            for i, dt in enumerate(weights_df.index):
                end = (
                    weights_df.index[i + 1] - pd.Timedelta(days=1)
                    if i < len(weights_df.index) - 1
                    else returns_df.index.max()
                )
                mask = (daily.index >= dt) & (daily.index <= end)
                daily.loc[mask] = weights_df.iloc[i].values
        else:
            for i, (start_dt, end_dt) in enumerate(holding_periods):
                mask = (daily.index >= pd.to_datetime(start_dt)) & (
                    daily.index <= pd.to_datetime(end_dt)
                )
                if i < len(weights_df):
                    daily.loc[mask] = weights_df.iloc[i].values

        return daily.dropna(how="all")

    def calculate_tearsheet(self, weights_df, returns_df, holding_periods=None):
        """计算核心回测指标并记录累计收益。"""
        weights_df = weights_df.copy()
        returns_df = returns_df.copy()
        weights_df.index = pd.to_datetime(weights_df.index)
        returns_df.index = pd.to_datetime(returns_df.index)
        weights_df = weights_df.sort_index()
        returns_df = returns_df.sort_index()
        daily_weights = self._expand_weights_to_daily(
            weights_df, returns_df, holding_periods
        )

        common_dates = daily_weights.index.intersection(returns_df.index)
        if len(common_dates) == 0:
            raise ValueError("weights_df 与 returns_df 无重叠日期，无法计算绩效指标。")
        w = daily_weights.loc[common_dates].values
        r = returns_df.loc[common_dates, daily_weights.columns].values

        rebalance_dates = pd.Index(pd.to_datetime(weights_df.index)).intersection(
            common_dates
        )

        turnover_series = np.zeros(len(common_dates))
        prev_w = np.zeros(len(weights_df.columns))
        for dt in rebalance_dates:
            cur_w = weights_df.loc[dt].values
            turnover_series[common_dates.get_loc(dt)] = np.sum(np.abs(cur_w - prev_w))
            prev_w = cur_w

        # 计算收益
        if self.returns_type == "log":
            # 修改点：先转简单收益率再加权，以更准确反映投资组合表现
            raw_portfolio_returns = np.sum(w * np.expm1(r), axis=1)
        else:
            raw_portfolio_returns = np.sum(w * r, axis=1)

        fees = turnover_series * self.fee_rate
        net_returns = raw_portfolio_returns - fees

        # 1. 记录投资组合的累计收益 (净值曲线)
        equity_curve = np.cumprod(1 + net_returns)

        # ... (计算 Sharpe, Max Drawdown 等指标)
        total_return = equity_curve[-1] - 1
        ann_return = (1 + total_return) ** (self.annual_periods / len(net_returns)) - 1
        ann_vol = np.std(net_returns) * np.sqrt(self.annual_periods)
        sharpe = (ann_return - self.rf) / ann_vol if ann_vol != 0 else 0
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = np.min(drawdown)
        avg_turnover = (
            np.mean(turnover_series[turnover_series > 0])
            if rebalance_dates.size > 1
            else 0.0
        )

        return {
            "Total Return": total_return,
            "Annualized Return": ann_return,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_drawdown,
            "Average Turnover": avg_turnover / 2,  # 转换为单边换手率
            "Equity Curve": pd.Series(equity_curve, index=common_dates),
            "Net Returns": pd.Series(net_returns, index=common_dates),
        }

    def print_summary(self, metrics):
        """打印策略表现摘要并输出至日志。"""
        # 获取项目统一的 logger
        logger = ProjectLogger.get_logger()

        summary_lines = [
            "-" * 35,
            f"{'Strategy Performance Summary':^35}",
            "-" * 35,
            f"Cumulative Return:     {metrics['Total Return']:>10.2%}",
            f"Annualized Return:     {metrics['Annualized Return']:>10.2%}",
            f"Sharpe Ratio:          {metrics['Sharpe Ratio']:>10.2f}",
            f"Max Drawdown:          {metrics['Max Drawdown']:>10.2%}",
            f"Avg Rebal Turnover:    {metrics['Average Turnover']:>10.2%}",
            "-" * 35,
        ]

        # 1. 打印到控制台 (保持原有功能)
        for line in summary_lines:
            print(line)

        # 2. 输出到日志文件
        logger.info("回测结果摘要:")
        for line in summary_lines:
            logger.info(line)

    def save_metrics(self, metrics, save_dir):
        """功能 1: 记录投资组合的累计收益和月度收益到 CSV。"""
        os.makedirs(save_dir, exist_ok=True)
        # 保存每日累计净值 (累计收益)
        metrics["Equity Curve"].to_csv(
            os.path.join(save_dir, "cumulative_returns.csv"), header=["Net_Value"]
        )
        # 计算并保存月度收益
        monthly = (
            metrics["Net Returns"].resample("MS").apply(lambda x: (1 + x).prod() - 1)
        )
        monthly.to_csv(
            os.path.join(save_dir, "monthly_returns.csv"), header=["Monthly_Return"]
        )

    def plot_performance(self, metrics, save_path):
        """
        功能 2: 绘制美化后的净值曲线和年度/月度收益图。
        X轴修改为仅显示年份。
        """
        # 设置全局绘图风格和更漂亮的字体
        sns.set_theme(style="white", context="notebook")
        plt.rcParams["font.sans-serif"] = [
            "DejaVu Sans",
            "Arial",
            "Microsoft YaHei",
        ]  # 优先使用矢量平滑字体
        plt.rcParams["axes.unicode_minus"] = False

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 10), gridspec_kw={"hspace": 0.4}
        )

        # --- 1. 累计净值曲线 (美化) ---
        equity_curve = metrics["Equity Curve"]
        ax1.plot(
            equity_curve.index,
            equity_curve.values,
            label="Strategy Net Value",
            color="#1f77b4",
            lw=2.5,
            alpha=0.9,
        )

        # 填充曲线下方区域
        ax1.fill_between(
            equity_curve.index, 1, equity_curve.values, color="#1f77b4", alpha=0.1
        )

        ax1.set_title(
            "Portfolio Cumulative Returns", fontsize=16, fontweight="bold", pad=15
        )
        ax1.set_ylabel("Net Value", fontsize=12)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.legend(frameon=False)

        # --- 2. 月度收益柱状图 (X轴仅显示年份) ---
        net_returns = metrics["Net Returns"]
        monthly_returns = net_returns.resample("MS").apply(lambda x: (1 + x).prod() - 1)

        # 准备数据绘图
        plot_df = monthly_returns.to_frame(name="Returns").reset_index()
        colors = ["#55a868" if x > 0 else "#c44e52" for x in plot_df["Returns"]]

        # 绘图
        sns.barplot(
            data=plot_df,
            x="Date",
            y="Returns",
            ax=ax2,
            palette=colors,
            hue="Date",
            legend=False,
        )

        ax2.set_title(
            "Monthly Performance Overview", fontsize=16, fontweight="bold", pad=15
        )
        ax2.set_ylabel("Return Rate", fontsize=12)
        ax2.axhline(0, color="black", lw=1, alpha=0.5)  # 零基准线

        # --- 核心修改：X 轴日期格式化 ---
        # 设置 X 轴标签仅显示年份
        # 由于 barplot 的 X 轴本质上是类别型的，我们需要手动控制 tick 标签
        tick_labels = []
        last_year = None
        for i, dt in enumerate(plot_df["Date"]):
            current_year = dt.strftime("%Y")
            if current_year != last_year:
                tick_labels.append(current_year)
                last_year = current_year
            else:
                tick_labels.append("")  # 同一年份仅显示第一个标签，保持简洁

        ax2.set_xticks(range(len(tick_labels)))
        ax2.set_xticklabels(
            tick_labels, rotation=0, horizontalalignment="center", fontsize=11
        )

        # 移除冗余边框
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(axis="y", linestyle="--", alpha=0.5)

        # 保存图表
        _dir = os.path.dirname(save_path)
        if _dir:
            os.makedirs(_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white")
        plt.close()

    def save_comparison_table(self, metrics_dict, save_path):
        """保存多策略核心指标对比表。"""
        rows = []
        for name, m in metrics_dict.items():
            rows.append(
                {
                    "Strategy": name,
                    "Total Return": m["Total Return"],
                    "Annualized Return": m["Annualized Return"],
                    "Annualized Volatility": m["Annualized Volatility"],
                    "Sharpe Ratio": m["Sharpe Ratio"],
                    "Max Drawdown": m["Max Drawdown"],
                    "Average Turnover": m["Average Turnover"],
                }
            )

        table = pd.DataFrame(rows).set_index("Strategy")
        _dir = os.path.dirname(save_path)
        if _dir:
            os.makedirs(_dir, exist_ok=True)
        table.to_csv(save_path)
        return table

    def plot_comparison_equity(self, metrics_dict, save_path):
        """绘制本模型 + Baselines 的净值对比曲线。"""
        sns.set_theme(style="white", context="notebook")
        plt.figure(figsize=(12, 6))

        for strategy_name, metrics in metrics_dict.items():
            eq = metrics["Equity Curve"]
            plt.plot(eq.index, eq.values, lw=2, label=strategy_name)

        plt.title("Model vs Baselines: Equity Curve", fontsize=15, fontweight="bold")
        plt.ylabel("Net Value")
        plt.xlabel("Date")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.legend(frameon=False)
        plt.tight_layout()

        _dir = os.path.dirname(save_path)
        if _dir:
            os.makedirs(_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

    def save_prediction_comparison(
        self,
        spo_pred_df,
        pto_pred_df,
        save_csv_path,
        save_plot_path=None,
    ):
        if spo_pred_df is None:
            spo_pred_df = pd.DataFrame(
                columns=["rebalance_date", "effective_date", "ticker", "spo_pred", "true_r"]
            )
        if pto_pred_df is None:
            pto_pred_df = pd.DataFrame(
                columns=["rebalance_date", "effective_date", "ticker", "pto_pred", "true_r"]
            )

        pred_compare = spo_pred_df.merge(
            pto_pred_df,
            on=["rebalance_date", "effective_date", "ticker"],
            how="outer",
            suffixes=("_spo", "_pto"),
        )

        if "true_r_spo" in pred_compare.columns and "true_r_pto" in pred_compare.columns:
            pred_compare["true_r"] = pred_compare["true_r_spo"].combine_first(
                pred_compare["true_r_pto"]
            )
        elif "true_r_spo" in pred_compare.columns:
            pred_compare["true_r"] = pred_compare["true_r_spo"]
        elif "true_r_pto" in pred_compare.columns:
            pred_compare["true_r"] = pred_compare["true_r_pto"]

        pred_compare = pred_compare.drop(
            columns=[c for c in ["true_r_spo", "true_r_pto"] if c in pred_compare.columns]
        ).sort_values(["rebalance_date", "ticker"])

        _dir = os.path.dirname(save_csv_path)
        if _dir:
            os.makedirs(_dir, exist_ok=True)
        pred_compare.to_csv(save_csv_path, index=False)

        if save_plot_path is not None and not pred_compare.empty:
            pred_plot = pred_compare.copy()
            pred_plot["rebalance_date"] = pd.to_datetime(pred_plot["rebalance_date"])
            tickers = sorted(pred_plot["ticker"].dropna().unique().tolist())

            fig, axes = plt.subplots(
                len(tickers),
                1,
                figsize=(14, max(4, 3.2 * len(tickers))),
                sharex=True,
            )
            if len(tickers) == 1:
                axes = [axes]

            for ax, ticker in zip(axes, tickers):
                one = pred_plot[pred_plot["ticker"] == ticker].sort_values("rebalance_date")
                ax.plot(one["rebalance_date"], one["spo_pred"], label="SPO Pred", linewidth=1.2)
                ax.plot(one["rebalance_date"], one["pto_pred"], label="PTO Pred", linewidth=1.2)
                ax.plot(one["rebalance_date"], one["true_r"], label="True r", linewidth=1.2)
                ax.set_title(ticker)
                ax.grid(alpha=0.25)
                ax.legend(loc="upper right", fontsize=8)

            plt.tight_layout()
            plot_dir = os.path.dirname(save_plot_path)
            if plot_dir:
                os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(save_plot_path, dpi=250, bbox_inches="tight")
            plt.close()

        return pred_compare
