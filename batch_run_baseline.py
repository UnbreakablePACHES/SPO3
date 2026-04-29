import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from scripts.data_preprocess import preprocess_etf_features
from utils.baselines import BaselineRunner
from utils.metrics import StrategyEvaluator
from utils.seed_manager import SeedManager


DATASET_CONFIGS = {
    "ETF_A": Path(
        "configs/_lambda_sweep_by_dataset/noclip_noadjust/ETF_A/"
        "mvo_ETF_A_noclip_noadjust_lambda_20.yaml"
    ),
    "ETF_B": Path(
        "configs/_lambda_sweep_by_dataset/noclip_noadjust/ETF_B/"
        "mvo_ETF_B_noclip_noadjust_lambda_20.yaml"
    ),
    "DOW": Path(
        "configs/_lambda_sweep_by_dataset/noclip_noadjust/DOW/"
        "mvo_DOW_noclip_noadjust_lambda_20.yaml"
    ),
}
DATASET_ORDER = ["ETF_A", "ETF_B", "DOW"]
RISK_AVERSIONS = [0.1, 1, 10, 20, 50]
OUTPUT_ROOT = Path("outputs/_baseline_lambda_sweep_by_dataset")


def lambda_tag(value):
    return str(value).replace(".", "p")


def load_config(config_path):
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_experiment_dir(output_root, dataset, risk_aversion):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}_baseline_pto_mvo_lambda_{lambda_tag(risk_aversion)}"
    exp_dir = output_root / dataset / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_baseline_artifacts(exp_dir, markowitz_weights, po_weights, metrics):
    baseline_dir = exp_dir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    markowitz_weights.to_csv(baseline_dir / "markowitz_weights.csv")
    po_weights.to_csv(baseline_dir / "po_simplelinear_markowitz_weights.csv")

    metrics["Baseline_Markowitz"]["Equity Curve"].to_csv(
        baseline_dir / "markowitz_equity_curve.csv", header=["Net_Value"]
    )
    metrics["Baseline_Markowitz"]["Net Returns"].to_csv(
        baseline_dir / "markowitz_net_returns.csv", header=["Net_Return"]
    )
    metrics["Baseline_PO_SimpleLinear_Markowitz"]["Equity Curve"].to_csv(
        baseline_dir / "po_simplelinear_markowitz_equity_curve.csv",
        header=["Net_Value"],
    )
    metrics["Baseline_PO_SimpleLinear_Markowitz"]["Net Returns"].to_csv(
        baseline_dir / "po_simplelinear_markowitz_net_returns.csv",
        header=["Net_Return"],
    )


def run_one(dataset, config_path, risk_aversion, output_root):
    cfg = load_config(config_path)
    cfg.setdefault("baseline_args", {})
    cfg["baseline_args"]["risk_aversion"] = float(risk_aversion)
    cfg["output_dir"] = str(output_root / dataset)

    exp_dir = build_experiment_dir(output_root, dataset, risk_aversion)
    with (exp_dir / "exp_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    SeedManager.set_seed(cfg["seed"])
    tickers = cfg["tickers"]
    etf_data = {
        ticker: pd.read_csv(Path(cfg["data_dir"]) / f"{ticker}.csv")
        for ticker in tickers
    }

    vix_df = None
    if cfg.get("add_vix"):
        vix_path = Path(cfg["data_dir"]) / "^VIX.csv"
        if not vix_path.exists():
            raise FileNotFoundError(f"Missing VIX data: {vix_path}")
        vix_df = pd.read_csv(vix_path)

    feat_df = preprocess_etf_features(
        etf_data=etf_data,
        vix_df=vix_df,
        etf_universe=tickers,
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        add_vix=cfg.get("add_vix", False),
    )

    baseline_runner = BaselineRunner(
        trading_days_path=cfg.get("trading_days_path"),
        seed=cfg["seed"],
    )
    hp = cfg["hyperparams"]
    baseline_args = cfg.get("baseline_args", {})
    prediction_return_clip = cfg.get(
        "prediction_return_clip", cfg.pop("prediction_daily_return_clip", None)
    )
    prediction_return_rescale_range = cfg.get("prediction_return_rescale_range")

    markowitz_weights, markowitz_holding = baseline_runner.run_markowitz(
        df=feat_df,
        window_months=hp["window_months"],
        freq=hp["rebalance_freq"],
        test_start_date=cfg.get("backtest_start_date"),
        risk_aversion=risk_aversion,
    )
    po_weights, po_holding = baseline_runner.run_simplelinear_po_markowitz(
        df=feat_df,
        window_months=hp["window_months"],
        freq=hp["rebalance_freq"],
        test_start_date=cfg.get("backtest_start_date"),
        risk_aversion=risk_aversion,
        pred_epochs=baseline_args.get("po_pred_epochs", 30),
        pred_lr=baseline_args.get("po_pred_lr", 1e-3),
        label_window=int(hp.get("label_window", 21)),
        prediction_return_clip=prediction_return_clip,
        prediction_return_rescale_range=prediction_return_rescale_range,
    )

    returns_df = feat_df.pivot(
        index="Date", columns="ticker", values="log_return"
    ).sort_index()
    evaluator = StrategyEvaluator(fee_rate=hp["fee_rate"])
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
        "Baseline_Markowitz": markowitz_metrics,
        "Baseline_PO_SimpleLinear_Markowitz": po_metrics,
    }

    evaluator.save_comparison_table(all_metrics, exp_dir / "comparison_metrics.csv")
    evaluator.plot_comparison_equity(
        all_metrics, exp_dir / "comparison_equity_curve.png"
    )
    save_baseline_artifacts(exp_dir, markowitz_weights, po_weights, all_metrics)
    return exp_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline-only PTO/MVO risk_aversion sweep."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASET_ORDER,
        help="Run only this dataset. Can be passed multiple times.",
    )
    parser.add_argument(
        "--lambda-risk",
        dest="risk_aversions",
        action="append",
        type=float,
        help="Run only this baseline risk_aversion. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT),
        help="Output root directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print tasks without running them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining tasks if one task fails.",
    )
    args = parser.parse_args()

    datasets = args.dataset or DATASET_ORDER
    risk_aversions = args.risk_aversions or RISK_AVERSIONS
    output_root = Path(args.output_root)

    tasks = [
        (dataset, DATASET_CONFIGS[dataset], risk_aversion)
        for dataset in datasets
        for risk_aversion in risk_aversions
    ]

    for i, (dataset, config_path, risk_aversion) in enumerate(tasks, start=1):
        print(
            f"\n[{i}/{len(tasks)}] dataset={dataset} "
            f"baseline risk_aversion={risk_aversion}"
        )
        print(f"config={config_path}")
        print(f"output_root={output_root}")
        if args.dry_run:
            continue

        try:
            exp_dir = run_one(dataset, config_path, risk_aversion, output_root)
            print(f"saved: {exp_dir}")
        except Exception as exc:
            message = (
                f"failed: dataset={dataset} risk_aversion={risk_aversion}: {exc}"
            )
            if args.continue_on_error:
                print(message, file=sys.stderr)
            else:
                raise


if __name__ == "__main__":
    main()
