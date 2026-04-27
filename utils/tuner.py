import optuna
from utils.backtester import SPOBacktester  #
from utils.trainer import SPOTrainer  #


class SPOHyperTuner:
    """
    使用 Optuna 为 SPO 框架进行自动调参
    """

    def __init__(self, df, opt_model, n_trials=20, label_window=21):
        """
        Args:
            df: 预处理后的数据集
            opt_model: 初始化的投资组合模型
            n_trials: 自动调参的尝试次数
        """
        self.df = df
        self.opt_model = opt_model
        self.n_trials = n_trials
        self.label_window = label_window

    def objective(self, trial):
        # 1. 定义想要优化的超参数搜索空间
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        epochs = trial.suggest_int("epochs", 5, 30)
        window_months = trial.suggest_categorical("window_months", [6, 12, 18])

        # 2. 实例化回测引擎
        backtester = SPOBacktester(opt_model=self.opt_model)

        # 3. 运行回测（建议在调参阶段缩短回测时间范围，以节省计算量）
        try:
            backtester.run(
                self.df,
                trainer_cls=SPOTrainer,  #
                window_months=window_months,
                epochs=epochs,
                lr=lr,
                label_window=self.label_window,
            )

            # 4. 获取评估指标
            # 我们以夏普比率（Sharpe Ratio）作为优化目标
            metrics = backtester.evaluate(self.df, fee_rate=self.opt_model.fee_rate)
            return metrics["Sharpe Ratio"]  # 目标是最大化夏普比

        except Exception as e:
            # 某些参数组合可能导致模型不收敛或报错，返回负无穷
            print(f"Trial failed: {e}")
            return -float("inf")

    def tune(self):
        # 创建研究对象，方向为最大化
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)

        print("\n" + "=" * 30)
        print("最佳超参数组合:")
        print(study.best_params)
        print(f"最佳夏普比率: {study.best_value:.4f}")
        print("=" * 30)

        return study.best_params
