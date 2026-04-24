import random
import os
import numpy as np
import torch


class SeedManager:
    """
    统一管理项目的随机种子，确保实验可复现。
    """

    @staticmethod
    def set_seed(seed: int = 42):
        """
        设置全局随机种子。

        Args:
            seed (int): 随机种子数值，默认为 42。
        """
        # 1. Python 原生随机库
        random.seed(seed)

        # 2. 操作系统环境变量 (影响一些底层哈希操作)
        os.environ["PYTHONHASHSEED"] = str(seed)

        # 3. NumPy
        np.random.seed(seed)

        # 4. PyTorch (CPU & GPU)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU

        # 5. PyTorch 确定性设置
        # 保证每次卷积、矩阵乘法等操作使用相同的算法实现
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f"[SeedManager] 全局随机种子已设置为: {seed}")

    @staticmethod
    def set_gurobi_seed(model, seed: int = 42):
        """
        专门为 Gurobi 模型设置种子。

        Args:
            model (gp.Model): Gurobi 模型实例。
            seed (int): 随机种子。
        """
        # Gurobi 的 Seed 参数会影响其内部算法（如单纯形法）的选择
        model.setParam("Seed", seed)
