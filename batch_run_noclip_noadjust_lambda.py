import argparse
import subprocess
import sys
from pathlib import Path

import yaml


CONFIGS = [
    Path("configs/mvo_ETF_A_noclip_noadjust.yaml"),
    Path("configs/mvo_ETF_B_noclip_noadjust.yaml"),
    Path("configs/mvo_DOW_noclip_noadjust.yaml"),
]

LAMBDA_RISKS = [0.1, 1, 10, 20, 50]
BACKTEST_START_DATE = "2020-01-01"
WINDOW_MONTHS = 12
COV_HISTORY = 220


def lambda_tag(value):
    return str(value).replace(".", "p")


def write_sweep_config(source_config, lambda_risk, sweep_dir):
    with source_config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("model_args", {})
    cfg.setdefault("hyperparams", {})
    cfg["backtest_start_date"] = BACKTEST_START_DATE
    cfg["hyperparams"]["window_months"] = WINDOW_MONTHS
    cfg["model_args"]["lambda_risk"] = float(lambda_risk)
    cfg["model_args"]["cov_history"] = COV_HISTORY

    target = sweep_dir / (
        f"{source_config.stem}_lambda_{lambda_tag(lambda_risk)}{source_config.suffix}"
    )
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return target


def main():
    parser = argparse.ArgumentParser(
        description="Run noclip_noadjust lambda_risk sweep for ETF_A, ETF_B, and DOW."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch run.py. Defaults to current interpreter.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining tasks if one task fails.",
    )
    args = parser.parse_args()

    sweep_dir = Path("configs/_lambda_sweep_noclip_noadjust")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for config in CONFIGS:
        for lambda_risk in LAMBDA_RISKS:
            sweep_config = write_sweep_config(config, lambda_risk, sweep_dir)
            tasks.append((config, lambda_risk, sweep_config))

    for i, (source_config, lambda_risk, sweep_config) in enumerate(tasks, start=1):
        cmd = [
            args.python,
            "run.py",
            "--config",
            str(sweep_config),
        ]
        print(
            f"\n[{i}/{len(tasks)}] {source_config.stem} "
            f"lambda_risk={lambda_risk}"
        )
        print(" ".join(cmd))

        if args.dry_run:
            continue

        result = subprocess.run(cmd)
        if result.returncode != 0:
            message = (
                f"failed: {source_config} lambda_risk={lambda_risk} "
                f"(exit code {result.returncode})"
            )
            if args.continue_on_error:
                print(message)
            else:
                raise SystemExit(message)


if __name__ == "__main__":
    main()
