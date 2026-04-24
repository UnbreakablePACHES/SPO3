import sys
import argparse
import yaml
import pandas as pd
from pathlib import Path

sys.path.append(str(Path.cwd()))

try:
    from utils.data_downloader import StockDataDownloader
    from scripts.data_preprocess import preprocess_etf_features
except ImportError as e:
    print(f"❌ 导入错误: {e}。请确保你在项目根目录下运行此脚本。")
    sys.exit(1)


def _load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="量化回测数据下载 + 预处理一体化工具")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="读取 config.yaml，自动填充所有参数。示例: --config configs/config.yaml",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="覆盖 config 中的 tickers。",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="覆盖 config 中的 start_date (YYYY-MM-DD)。",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="覆盖 config 中的 end_date (YYYY-MM-DD)。不填则下载到今天。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="原始数据保存目录，覆盖 config 中的 data_dir。",
    )
    parser.add_argument(
        "--processed_path",
        type=str,
        default=None,
        help="预处理结果保存路径（CSV）。不填则自动生成到 data/processed/ 下。",
    )
    parser.add_argument(
        "--add_vix",
        type=lambda x: x.lower() == "true",
        default=None,
        help="是否下载并加入 VIX 特征，覆盖 config 中的 add_vix。示例: --add_vix True",
    )
    parser.add_argument(
        "--no_preprocess",
        action="store_true",
        help="只下载，跳过预处理步骤。",
    )

    args = parser.parse_args()

    # 从 config 读取默认值，CLI 参数优先级更高
    cfg = {}
    if args.config:
        cfg = _load_config(args.config)

    tickers = args.tickers or cfg.get("tickers")
    start_date = args.start_date or cfg.get("start_date")
    end_date = args.end_date or cfg.get("end_date")
    output_dir = args.output_dir or cfg.get("data_dir")
    add_vix = args.add_vix if args.add_vix is not None else cfg.get("add_vix", False)

    # 若启用 VIX，自动追加到下载列表
    download_tickers = list(tickers) if tickers else []
    if add_vix and "^VIX" not in download_tickers:
        download_tickers.append("^VIX")

    # 校验必填项
    missing = [
        name
        for name, val in [
            ("tickers", tickers),
            ("start_date", start_date),
            ("output_dir", output_dir),
        ]
        if not val
    ]
    if missing:
        parser.error(
            f"以下参数未提供（可通过 --config 或 CLI 指定）: {', '.join('--' + m for m in missing)}"
        )

    # ===== 步骤 1：下载 =====
    print("\n" + "=" * 50)
    print("🚀 步骤 1/2：批量下载原始数据")
    print(f"   资产池  : {download_tickers}")
    print(f"   时间区间: {start_date} 至 {end_date if end_date else '今日'}")
    print(f"   输出目录: {output_dir}")
    print("=" * 50)

    downloader = StockDataDownloader(output_dir=output_dir)
    download_report = downloader.batch_download(
        tickers=download_tickers,
        start_date=start_date,
        end_date=end_date,
    )

    success_count = sum(1 for s in download_report.values() if s)
    fail_count = len(download_report) - success_count
    print(f"\n✅ 成功: {success_count} 支  ❌ 失败: {fail_count} 支")
    if fail_count > 0:
        failed = [t for t, s in download_report.items() if not s]
        print(f"   失败列表: {failed}")

    if args.no_preprocess:
        print("\n已跳过预处理（--no_preprocess）。\n")
        return

    # ===== 步骤 2：预处理 =====
    print("\n" + "=" * 50)
    print("⚙️  步骤 2/2：特征预处理")
    print("=" * 50)

    raw_dir = Path(output_dir)
    etf_data = {}
    for t in tickers:
        csv_path = raw_dir / f"{t}.csv"
        if csv_path.exists():
            etf_data[t] = pd.read_csv(csv_path)
        else:
            print(f"⚠️  找不到 {csv_path}，跳过该 ticker。")

    vix_df = None
    if add_vix:
        vix_path = raw_dir / "^VIX.csv"
        if vix_path.exists():
            vix_df = pd.read_csv(vix_path)
        else:
            print("⚠️  add_vix=True 但找不到 ^VIX.csv，已忽略 VIX 特征。")
            add_vix = False

    feat_df = preprocess_etf_features(
        etf_data=etf_data,
        vix_df=vix_df,
        etf_universe=list(etf_data.keys()),
        start_date=start_date,
        end_date=end_date,
        add_vix=add_vix,
    )

    # 决定输出路径
    if args.processed_path:
        save_path = Path(args.processed_path)
    else:
        config_stem = Path(args.config).stem if args.config else "data"
        save_path = Path("data/processed") / f"{config_stem}_features.csv"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(save_path, index=False)

    print(f"\n✅ 预处理完成，共 {len(feat_df)} 行")
    print(f"   已保存至: {save_path}\n")


if __name__ == "__main__":
    main()
