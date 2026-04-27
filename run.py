import os
import yaml
import argparse
import pandas as pd
from datetime import datetime
from typing import Optional
from scripts.data_preprocess import preprocess_etf_features
from utils.trainer import SPOTrainer
from utils.backtester import SPOBacktester
from utils.seed_manager import SeedManager
from utils.factories import ModelFactory
from utils.logger import ProjectLogger
from utils.metrics import StrategyEvaluator
from utils.baselines import BaselineRunner


def _load_config(config_path: str, add_vix_arg: Optional[str]):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if add_vix_arg is not None:
        cfg["add_vix"] = add_vix_arg.lower() == "true"
    return cfg


def _build_experiment_dir(output_dir: str, config_name: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}_{config_name}"
    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--add_vix", type=str, default=None)
    parser.add_argument("--lambda_cvar", type=float, default=None)
    parser.add_argument("--context_history", type=int, default=None)
    args = parser.parse_args()

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    cfg = _load_config(args.config, args.add_vix)
    exp_dir = _build_experiment_dir(cfg["output_dir"], config_name)

    logger = ProjectLogger.get_logger()
    SeedManager.set_seed(cfg["seed"])

    with open(os.path.join(exp_dir, "exp_config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    logger.info(f"实验目录已创建: {exp_dir}")

    tickers = cfg["tickers"]
    etf_data = {
        t: pd.read_csv(os.path.join(cfg["data_dir"], f"{t}.csv")) for t in tickers
    }

    if args.lambda_cvar is not None:
        cfg["model_args"]["lambda_cvar"] = args.lambda_cvar
    if args.context_history is not None:
        cfg["hyperparams"]["context_history"] = args.context_history

    vix_df = None
    if cfg["add_vix"]:
        vix_path = os.path.join(cfg["data_dir"], "^VIX.csv")
        if os.path.exists(vix_path):
            vix_df = pd.read_csv(vix_path)
        else:
            logger.error("Missing VIX data")
            raise FileNotFoundError(f"VIX 数据文件不存在: {vix_path}")

    feat_df = preprocess_etf_features(
        etf_data=etf_data,
        vix_df=vix_df,
        etf_universe=tickers,
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        add_vix=cfg["add_vix"],
    )

    model_params = {**cfg["hyperparams"], **cfg["model_args"], "seed": cfg["seed"]}
    opt_model = ModelFactory.get_opt_model(
        cfg["model_type"], n_assets=len(tickers), **model_params
    )

    backtester = SPOBacktester(
        opt_model=opt_model, trading_days_path=cfg.get("trading_days_path")
    )

    logger.info("开始执行训练与滚动回测...")
    backtester.run(
        df=feat_df,
        trainer_cls=SPOTrainer,
        window_months=cfg["hyperparams"]["window_months"],
        epochs=cfg["hyperparams"]["epochs"],
        lr=cfg["hyperparams"]["lr"],
        batch_size=cfg["hyperparams"]["batch_size"],
        freq=cfg["hyperparams"]["rebalance_freq"],
        test_start_date=cfg.get("backtest_start_date"),
        seed=cfg["seed"],
        normalize_features=cfg.get("feature_normalization", True),
        context_history=cfg["hyperparams"].get("context_history", 20),
        label_window=cfg["hyperparams"].get("label_window", 1),
    )
    eval_fee_rate = cfg["hyperparams"]["fee_rate"]
    evaluator = StrategyEvaluator(fee_rate=eval_fee_rate)
    weights_path, weights_plot_path = evaluator.save_weight_outputs(
        backtester.results,
        exp_dir,
        weights_filename="spo_weights.csv",
        plot_filename="spo_weights_timeseries.png",
        title="SPO Portfolio Weights Over Time",
    )
    if weights_path is not None:
        logger.info(f"Saved SPO weights: {weights_path}")
        logger.info(f"Saved SPO weight time series plot: {weights_plot_path}")
    else:
        logger.warning("SPO weights are empty; skipped weight outputs.")

    model_metrics = backtester.evaluate(feat_df, fee_rate=eval_fee_rate)

    evaluator.save_metrics(model_metrics, exp_dir)
    evaluator.plot_performance(
        model_metrics, os.path.join(exp_dir, "performance_charts.png")
    )

    # ===== 新增：两个 baseline（Markowitz / SimpleLinear+Markowitz PO） =====
    baseline_runner = BaselineRunner(
        trading_days_path=cfg.get("trading_days_path"),
        seed=cfg["seed"],
    )

    risk_aversion = cfg.get("baseline_args", {}).get("risk_aversion", 10.0)
    po_pred_epochs = cfg.get("baseline_args", {}).get("po_pred_epochs", 30)
    po_pred_lr = cfg.get("baseline_args", {}).get("po_pred_lr", 1e-3)

    markowitz_weights, markowitz_holding = baseline_runner.run_markowitz(
        df=feat_df,
        window_months=cfg["hyperparams"]["window_months"],
        freq=cfg["hyperparams"]["rebalance_freq"],
        test_start_date=cfg.get("backtest_start_date"),
        risk_aversion=risk_aversion,
    )

    po_weights, po_holding = baseline_runner.run_simplelinear_po_markowitz(
        df=feat_df,
        window_months=cfg["hyperparams"]["window_months"],
        freq=cfg["hyperparams"]["rebalance_freq"],
        test_start_date=cfg.get("backtest_start_date"),
        risk_aversion=risk_aversion,
        pred_epochs=po_pred_epochs,
        pred_lr=po_pred_lr,
        label_window=cfg["hyperparams"].get("label_window", 1),
    )

    returns_df = feat_df.pivot(
        index="Date", columns="ticker", values="log_return"
    ).sort_index()
    markowitz_metrics = evaluator.calculate_tearsheet(
        weights_df=markowitz_weights,
        returns_df=returns_df,
        holding_periods=markowitz_holding,
    )
    po_metrics = evaluator.calculate_tearsheet(
        weights_df=po_weights,
        returns_df=returns_df,
        holding_periods=po_holding,
    )

    all_metrics = {
        "SPO_Model": model_metrics,
        "Baseline_Markowitz": markowitz_metrics,
        "Baseline_PO_SimpleLinear_Markowitz": po_metrics,
    }

    # 输出统一图和表（适用于任意数据集/实验）
    evaluator.plot_comparison_equity(
        all_metrics,
        os.path.join(exp_dir, "comparison_equity_curve.png"),
    )
    evaluator.save_comparison_table(
        all_metrics,
        os.path.join(exp_dir, "comparison_metrics.csv"),
    )

    # 输出 baseline 结果文件夹（weights/equity/net_returns）
    baseline_dir = os.path.join(exp_dir, "baselines")
    os.makedirs(baseline_dir, exist_ok=True)

    markowitz_weights.to_csv(os.path.join(baseline_dir, "markowitz_weights.csv"))
    po_weights.to_csv(
        os.path.join(baseline_dir, "po_simplelinear_markowitz_weights.csv")
    )

    markowitz_metrics["Equity Curve"].to_csv(
        os.path.join(baseline_dir, "markowitz_equity_curve.csv"), header=["Net_Value"]
    )
    markowitz_metrics["Net Returns"].to_csv(
        os.path.join(baseline_dir, "markowitz_net_returns.csv"), header=["Net_Return"]
    )
    po_metrics["Equity Curve"].to_csv(
        os.path.join(baseline_dir, "po_simplelinear_markowitz_equity_curve.csv"),
        header=["Net_Value"],
    )
    po_metrics["Net Returns"].to_csv(
        os.path.join(baseline_dir, "po_simplelinear_markowitz_net_returns.csv"),
        header=["Net_Return"],
    )

    if cfg.get("save_feature_contribution", True):
        feature_contrib_path = os.path.join(exp_dir, "feature_contributions.csv")
        backtester.feature_contributions.to_csv(feature_contrib_path, index=False)
        logger.info(f"已保存特征贡献度: {feature_contrib_path}")

        heatmap_path = os.path.join(exp_dir, "spo_feature_contribution_heatmap.png")
        heatmap_matrix = backtester.plot_feature_contribution_heatmap(
            save_path=heatmap_path,
            use_abs=True,
            aggfunc="mean",
        )
        if not heatmap_matrix.empty:
            heatmap_csv_path = os.path.join(
                exp_dir, "spo_feature_contribution_timeseries.csv"
            )
            heatmap_matrix.to_csv(heatmap_csv_path)
            logger.info(f"已保存 SPO 特征贡献时间序列矩阵: {heatmap_csv_path}")
            logger.info(f"已保存 SPO 特征贡献热图: {heatmap_path}")
        else:
            logger.warning("特征贡献为空，跳过 SPO 特征贡献热图输出。")

    logger.info("已输出 Baseline 对比图表与结果表格。")


if __name__ == "__main__":
    main()
