import logging


class ProjectLogger:
    """Project-wide logger factory."""

    @staticmethod
    def get_logger(name="SPO-CVaR", log_dir=None):
        logger = logging.getLogger(name)
        if logger.hasHandlers():
            return logger

        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger
