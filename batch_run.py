import argparse
import subprocess
import sys
from pathlib import Path


SWEEP_ROOT = Path("configs/_lambda_sweep_by_dataset")
OUTPUT_ROOT = Path("outputs/_lambda_sweep_by_dataset")
GROUP_ORDER = ["noclip_noadjust", "noclip_adjust", "clip_noadjust", "clip_adjust"]
DATASET_ORDER = ["ETF_A", "ETF_B", "DOW"]


def lambda_sort_key(config_path):
    tag = config_path.stem.rsplit("_lambda_", 1)[-1]
    return float(tag.replace("p", "."))


def collect_configs(groups=None, datasets=None):
    groups = groups or GROUP_ORDER
    datasets = datasets or DATASET_ORDER

    configs = []
    for group in groups:
        for dataset in datasets:
            dataset_dir = SWEEP_ROOT / group / dataset
            if not dataset_dir.exists():
                raise FileNotFoundError(f"Missing directory: {dataset_dir}")
            configs.extend(sorted(dataset_dir.glob("*.yaml"), key=lambda_sort_key))
    return configs


def config_group_dataset(config_path):
    rel = config_path.relative_to(SWEEP_ROOT)
    group = rel.parts[0]
    dataset = rel.parts[1]
    return group, dataset


def main():
    parser = argparse.ArgumentParser(
        description="Run all lambda sweep configs organized by group and dataset."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch run.py. Defaults to current interpreter.",
    )
    parser.add_argument(
        "--group",
        action="append",
        choices=GROUP_ORDER,
        help="Run only this group. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASET_ORDER,
        help="Run only this dataset. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining configs if one config fails.",
    )
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT),
        help="Root output directory. Results are written to output-root/dataset/group.",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Use each yaml's output_dir instead of organizing outputs by dataset/group.",
    )
    args = parser.parse_args()

    configs = collect_configs(groups=args.group, datasets=args.dataset)
    if not configs:
        raise SystemExit("No configs found.")

    for i, config in enumerate(configs, start=1):
        cmd = [args.python, "run.py", "--config", str(config)]
        if not args.flat_output:
            group, dataset = config_group_dataset(config)
            output_dir = Path(args.output_root) / dataset / group
            cmd.extend(["--output_dir", str(output_dir)])
        print(f"\n[{i}/{len(configs)}] running {config}")
        print(" ".join(cmd))

        if args.dry_run:
            continue

        result = subprocess.run(cmd)
        if result.returncode != 0:
            message = f"failed: {config} (exit code {result.returncode})"
            if args.continue_on_error:
                print(message)
            else:
                raise SystemExit(message)


if __name__ == "__main__":
    main()
