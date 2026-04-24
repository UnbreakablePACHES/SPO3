import logging
import os
from datetime import datetime


class ProjectLogger:
    """
    项目统一日志管理器
    """

    @staticmethod
    def get_logger(name="SPO-CVaR", log_dir="outputs/logs"):
        os.makedirs(log_dir, exist_ok=True)

        logger = logging.getLogger(name)
        if logger.hasHandlers():
            return logger

        logger.setLevel(logging.INFO)

        # 格式设定
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 1. 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. 文件处理器 (按日期命名)
        log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger
