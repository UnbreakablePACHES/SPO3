import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from tqdm import tqdm


class StockDataDownloader:
    """
    股票数据下载器 (tqdm 进度条版)
    """

    def __init__(self, output_dir: str = "stock_data"):
        """
        初始化下载器

        :param output_dir: 数据保存的文件夹路径
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_single(
        self, ticker: str, start_date: str, end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        下载单个股票的数据（静默执行，清理多层表头）
        """
        if not end_date:
            end_date = datetime.today().strftime("%Y-%m-%d")

        try:
            # 下载数据
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if df.empty:
                return None

            if isinstance(df.columns, pd.MultiIndex):
                # 删掉第二层表头（即 Ticker 名字，比如 'EEM'）
                df.columns = df.columns.droplevel(1)
                # 清除第一层表头的名字（默认叫 'Price'），避免在 CSV 留下一行空白
                df.columns.name = None

            return df

        except Exception:
            return None

    def save_to_csv(self, df: pd.DataFrame, ticker: str) -> bool:
        """将 DataFrame 保存为 CSV 文件（静默执行，无输出）"""
        if df is None or df.empty:
            return False

        file_path = self.output_dir / f"{ticker}.csv"
        try:
            df.to_csv(file_path)
            return True
        except Exception:
            return False

    def batch_download(
        self, tickers: List[str], start_date: str, end_date: Optional[str] = None
    ) -> dict:
        """
        批量下载多个股票并自动保存，使用 tqdm 动态显示每个 ticker
        """
        results = {}

        # 1. 实例化 tqdm 对象，而不是直接放在 for 循环里
        pbar = tqdm(tickers, unit="支")

        for ticker in pbar:
            # 2. 动态改变进度条左侧的文字，提示当前正在下载的股票
            pbar.set_description(
                f"正在下载: {ticker: <6}"
            )  # <6 表示左对齐占6个字符，防止文字抖动

            # 执行下载
            df = self.download_single(ticker, start_date, end_date)

            if df is not None:
                success = self.save_to_csv(df, ticker)
                results[ticker] = success
                # 3. 下载成功后，可以在进度条右侧附加状态信息
                pbar.set_postfix({"状态": "成功"})
            else:
                results[ticker] = False
                # 3. 下载失败时的状态
                pbar.set_postfix({"状态": "失败"})

        # 循环结束后，把进度条文字改为完成
        pbar.set_description("全部完成      ")
        return results


# ==========================================
# 使用示例 (入口点)
# ==========================================
if __name__ == "__main__":
    # 1. 配置参数
    TARGET_STOCKS = ["AAPL", "MSFT", "GOOG", "TSLA", "NVDA", "INVALID_TICKER"]
    START_DATE = "2023-01-01"
    END_DATE = "2023-12-31"
    SAVE_FOLDER = "my_financial_data"

    # 2. 实例化下载器
    downloader = StockDataDownloader(output_dir=SAVE_FOLDER)

    # 3. 执行批量下载（此时屏幕上只会显示一行刷新变化的 tqdm 进度条）
    download_report = downloader.batch_download(
        tickers=TARGET_STOCKS, start_date=START_DATE, end_date=END_DATE
    )

    # 4. 打印最终报告（等进度条 100% 走完后输出）
    print("\n" + "=" * 30)
    print("下载任务总结报告:")
    for ticker, status in download_report.items():
        status_text = "✅ 成功" if status else "❌ 失败"
        print(f"[{ticker}]: {status_text}")
    print("=" * 30)
