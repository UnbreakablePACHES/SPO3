import argparse
from pathlib import Path

import pandas as pd


STRATEGY_WEIGHT_FILES = {
    "SPO_Model": Path("spo_weights.csv"),
    "Baseline_Markowitz": Path("baselines/markowitz_weights.csv"),
    "Baseline_PO_SimpleLinear_Markowitz": Path(
        "baselines/po_simplelinear_markowitz_weights.csv"
    ),
}


def average_rebalance_turnover(weights_path: Path) -> float:
    weights = pd.read_csv(weights_path, index_col=0)
    if len(weights) <= 1:
        return 0.0

    weights = weights.apply(pd.to_numeric, errors="raise")
    one_way_turnover = weights.diff().abs().sum(axis=1).iloc[1:] / 2.0
    one_way_turnover = one_way_turnover[one_way_turnover > 0]
    if one_way_turnover.empty:
        return 0.0
    return float(one_way_turnover.mean())


def update_metrics_file(metrics_path: Path, dry_run: bool = False) -> tuple[int, list[str]]:
    exp_dir = metrics_path.parent
    metrics = pd.read_csv(metrics_path)
    if "Strategy" not in metrics.columns or "Average Turnover" not in metrics.columns:
        return 0, [f"skip {metrics_path}: missing Strategy/Average Turnover column"]

    updated = 0
    warnings = []
    for idx, row in metrics.iterrows():
        strategy = row["Strategy"]
        rel_weights = STRATEGY_WEIGHT_FILES.get(strategy)
        if rel_weights is None:
            warnings.append(f"skip {metrics_path}: unknown strategy {strategy}")
            continue

        weights_path = exp_dir / rel_weights
        if not weights_path.exists():
            warnings.append(f"skip {strategy} in {metrics_path}: missing {rel_weights}")
            continue

        old_value = metrics.at[idx, "Average Turnover"]
        new_value = average_rebalance_turnover(weights_path)
        if pd.isna(old_value) or abs(float(old_value) - new_value) > 1e-12:
            metrics.at[idx, "Average Turnover"] = new_value
            updated += 1

    if updated and not dry_run:
        metrics.to_csv(metrics_path, index=False)

    return updated, warnings


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Recompute Average Turnover in outputs/**/comparison_metrics.csv from "
            "saved weight CSVs. Turnover is one-way average rebalance turnover and "
            "excludes initial portfolio construction."
        )
    )
    parser.add_argument(
        "--outputs-root",
        default="outputs",
        help="Root directory to scan. Defaults to outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many rows would change without writing files.",
    )
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    metrics_paths = sorted(outputs_root.rglob("comparison_metrics.csv"))
    total_files_changed = 0
    total_rows_changed = 0
    all_warnings = []

    for metrics_path in metrics_paths:
        updated, warnings = update_metrics_file(metrics_path, dry_run=args.dry_run)
        all_warnings.extend(warnings)
        if updated:
            total_files_changed += 1
            total_rows_changed += updated
            action = "would update" if args.dry_run else "updated"
            print(f"{action}: {metrics_path} ({updated} rows)")

    print(
        f"scanned {len(metrics_paths)} comparison_metrics.csv files; "
        f"{total_files_changed} files, {total_rows_changed} rows "
        f"{'would change' if args.dry_run else 'changed'}."
    )

    if all_warnings:
        print("\nwarnings:")
        for warning in all_warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
