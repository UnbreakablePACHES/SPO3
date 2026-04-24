import numpy as np
import pandas as pd
from typing import Dict, Iterable, Optional


# 最终保留的特征列
FEATURE_COLUMNS = [
    "Date",
    "Close",
    "log_return",
    "SMA_10",
    "price_bias",
    "RSI_14",
    "MACD_diff",
    "bollinger_width",
    "volume_bias",
]


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """计算 RSI（相对强弱指数）"""
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # 正常计算 RS
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask(avg_loss == 0, 100.0)

    return rsi.astype(float)


def _compute_macd_diff(close: pd.Series) -> pd.Series:
    """计算 MACD 柱状图（MACD线 - Signal线）"""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    return macd_line - signal_line


def _add_vix_feature(
    etf_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    vix_col_name: str = "VIX_shift1_OCavg",
) -> pd.DataFrame:
    """按日期将 VIX 特征合并到 ETF 数据中"""
    vix = vix_df.copy()
    vix["Date"] = pd.to_datetime(vix["Date"])
    vix = vix.sort_values("Date")

    vix[vix_col_name] = ((vix["Open"] + vix["Close"]) / 2.0).shift(1)

    merged = etf_df.merge(vix[["Date", vix_col_name]], on="Date", how="left")
    return merged


def preprocess_etf_features(
    etf_data: Dict[str, pd.DataFrame],
    vix_df: Optional[pd.DataFrame],  # 改为可选
    etf_universe: Iterable[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    dropna: bool = True,
    add_vix: bool = False,  # 1. 新增布尔判断逻辑
) -> pd.DataFrame:
    """ETF 特征预处理主函数

    设计原则：
    1. 不要在算特征前就裁掉历史数据，否则 rolling 特征前面一定是 NaN。
    2. 先用完整原始数据算特征。
    3. 再裁到目标区间。
    4. 最后 dropna，得到“可直接使用”的结果。

    参数：
        etf_data: ticker -> DataFrame，至少包含 Date / Close / Volume
        vix_df: VIX 数据，至少包含 Date / Open / Close
        etf_universe: 需要处理的 ETF 列表
        start_date: 最终输出的起始日期，例如 "2023-01-01"
        end_date: 最终输出的结束日期，例如 "2025-12-31"
        dropna: 是否删除特征缺失行，默认 True

    返回：
        一个 long format 的特征表，若 dropna=True，则输出中目标特征列不含 NaN
    """
    frames = []

    if add_vix and vix_df is not None:
        vix_df = vix_df.copy()
        vix_df["Date"] = pd.to_datetime(vix_df["Date"])
        vix_df = vix_df.sort_values("Date")
    elif add_vix and vix_df is None:
        raise ValueError("add_vix 为 True 但未提供 vix_df 数据")

    start_ts = pd.to_datetime(start_date) if start_date is not None else None
    end_ts = pd.to_datetime(end_date) if end_date is not None else None

    # 3. 动态确定需要 dropna 的列
    check_subset = FEATURE_COLUMNS.copy()
    if add_vix:
        check_subset.append("VIX_shift1_OCavg")

    for ticker in etf_universe:
        if ticker not in etf_data:
            raise KeyError(f"Ticker {ticker} not found in etf_data.")

        df = etf_data[ticker].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        # 4) 根据判断加不加 VIX 特征
        if add_vix:
            df = _add_vix_feature(df, vix_df)

        # 收益/趋势/动量/波动率特征计算保持不变...
        ratio = df["Close"] / df["Close"].shift(1)
        df["log_return"] = np.log(ratio)
        df["SMA_10"] = df["Close"].rolling(window=10, min_periods=10).mean()
        df["price_bias"] = (df["Close"] / df["SMA_10"]) - 1.0
        df["RSI_14"] = _compute_rsi(df["Close"], window=14)
        df["MACD_diff"] = _compute_macd_diff(df["Close"])

        mid20 = df["Close"].rolling(window=20, min_periods=20).mean()
        std20 = df["Close"].rolling(window=20, min_periods=20).std()
        upper = mid20 + 2 * std20
        lower = mid20 - 2 * std20
        df["bollinger_width"] = (upper - lower) / mid20

        vol_ma20 = df["Volume"].rolling(window=20, min_periods=20).mean()
        df["volume_bias"] = (df["Volume"] / vol_ma20) - 1.0

        df["ticker"] = ticker

        if start_ts is not None:
            df = df[df["Date"] >= start_ts]
        if end_ts is not None:
            df = df[df["Date"] <= end_ts]

        # 5) 动态清理缺失行
        if dropna:
            df = df.dropna(subset=check_subset)
        # 核心修复：确保数值列绝对干净
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=numeric_cols)  # 只要数值列有 NaN 就删除该行

        # 6) 动态导出列
        keep_cols = FEATURE_COLUMNS + ["ticker"]
        if add_vix:
            keep_cols.insert(-1, "VIX_shift1_OCavg")  # 插入到 ticker 之前

        frames.append(df[keep_cols])

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.sort_values(["Date", "ticker"]).reset_index(drop=True)
    # 在函数最后 return 前，再次全局检查
    if out.isnull().any().any():
        print("Warning: Final DataFrame still contains NaNs. Dropping them...")
        out = out.dropna()
    return out


if __name__ == "__main__":
    import os

    # ===== 配置 =====
    TICKERS = ["EEM", "EFA", "JPXN", "SPY", "XLK", "VTI", "AGG", "DBC"]
    DATA_DIR = "data/raw"
    START = "2014-01-01"
    END = "2025-12-31"

    # ===== 读取ETF数据 =====
    etf_data = {}
    for t in TICKERS:
        path = os.path.join(DATA_DIR, f"{t}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")

        df = pd.read_csv(path)
        etf_data[t] = df

    # ===== 读取VIX =====
    vix_path = os.path.join(DATA_DIR, "^VIX.csv")
    if not os.path.exists(vix_path):
        raise FileNotFoundError(f"{vix_path} not found.")

    vix_df = pd.read_csv(vix_path)

    # ===== 特征预处理 =====
    feat_df = preprocess_etf_features(
        etf_data=etf_data,
        vix_df=vix_df,
        etf_universe=TICKERS,
        start_date=START,
        end_date=END,
        dropna=True,
    )

    # ===== 输出检查 =====
    print("Feature shape:", feat_df.shape)
    print(feat_df.head())

    # ===== 可选：保存 =====
    save_path = "data/processed/FeaturedData_8ETFsickers.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    feat_df.to_csv(save_path, index=False)

    print(f"Saved to {save_path}")
