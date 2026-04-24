import pandas as pd
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from typing import Iterator, NamedTuple


class BacktestWindow(NamedTuple):
    train_start: str
    train_end: str
    test_start: str
    test_end: str


class RollingWindowGenerator:
    """
    支持日、周、月三种逻辑的滚动窗口生成器。

    常用频率 (freq):
    - 'D'  : 每日调仓
    - 'W'  : 每周调仓 (默认周日，可用 'W-MON' 指定周一)
    - 'MS' : 每月月初调仓
    """

    def __init__(
        self, test_start: str, test_end: str, window_months: int = 12, freq: str = "MS"
    ):
        self.test_start = pd.Timestamp(test_start)
        self.test_end = pd.Timestamp(test_end)
        self.window_months = window_months
        self.freq = freq.upper()

        # 1. 生成调仓日期序列
        self.rebalance_dates = pd.date_range(
            start=self.test_start, end=self.test_end, freq=self.freq
        )

    def __len__(self) -> int:
        return len(self.rebalance_dates)

    def __iter__(self) -> Iterator[BacktestWindow]:
        for i in range(len(self.rebalance_dates)):
            yield self._get_window_at_index(i)

    def _get_window_at_index(self, index: int) -> BacktestWindow:
        current_date = self.rebalance_dates[index]

        # --- 训练集逻辑 (统一由 window_months 决定) ---
        train_end_dt = current_date - timedelta(days=1)
        train_start_dt = (
            train_end_dt - relativedelta(months=self.window_months) + timedelta(days=1)
        )

        # --- 测试集逻辑 (受 freq 影响) ---
        test_start_dt = current_date

        # 如果不是最后一个调仓点，测试结束于下一次调仓的前一天
        if index < len(self.rebalance_dates) - 1:
            test_end_dt = self.rebalance_dates[index + 1] - timedelta(days=1)
        else:
            # 最后一个窗口的边界处理
            if "D" in self.freq:
                test_end_dt = current_date  # 日频，当天即结束
            elif "W" in self.freq:
                test_end_dt = current_date + timedelta(days=6)  # 周频，覆盖一周
            else:
                test_end_dt = current_date + pd.offsets.MonthEnd(0)  # 月频，到月末

        return BacktestWindow(
            train_start=train_start_dt.strftime("%Y-%m-%d"),
            train_end=train_end_dt.strftime("%Y-%m-%d"),
            test_start=test_start_dt.strftime("%Y-%m-%d"),
            test_end=test_end_dt.strftime("%Y-%m-%d"),
        )
