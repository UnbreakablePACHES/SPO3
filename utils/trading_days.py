import pandas as pd


class TradingDayShifter:
    """
    Utility class for trading day operations.

    Supports:
    - load trading days from CSV
    - shift forward/backward by trading days
    """

    def __init__(self, filepath: str):
        """
        Args:
            filepath: path to csv containing trading days
        """
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)

        self.days = pd.DatetimeIndex(df.index).sort_values()

    def shift(self, date, n: int):
        """
        Shift a date by n trading days.

        Args:
            date: str or Timestamp
            n: int (positive → forward, negative → backward)

        Returns:
            pd.Timestamp
        """
        date = pd.to_datetime(date)

        # 找到插入位置
        idx = self.days.searchsorted(date)

        target_idx = idx + n

        if target_idx < 0 or target_idx >= len(self.days):
            raise IndexError("Shift out of trading day range.")

        return self.days[target_idx]

    def next(self, date):
        """Next trading day"""
        return self.shift(date, 1)

    def prev(self, date):
        """Previous trading day"""
        return self.shift(date, -1)

    def is_trading_day(self, date):
        """Check if date is trading day"""
        date = pd.to_datetime(date)
        return date in self.days

    def next_or_same(self, date):
        """
        Return the next trading day on or after `date`.
        """
        date = pd.to_datetime(date)
        idx = self.days.searchsorted(date, side="left")
        if idx >= len(self.days):
            raise IndexError("Date out of trading day range.")
        return self.days[idx]

    def prev_or_same(self, date):
        """
        返回 `date` 当天或之前的最后一个交易日。
        用于确保预测时只使用历史已知的特征，避免数据泄露。
        """
        date = pd.to_datetime(date)
        # searchsorted(side='right') 找插入位置，-1 即为不大于该日期的最大索引
        idx = self.days.searchsorted(date, side="right") - 1
        if idx < 0:
            raise IndexError(f"日期 {date.date()} 之前无可用交易日数据。")
        return self.days[idx]
