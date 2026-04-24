import torch
from tqdm import tqdm


class SPOTrainer:
    """
    用于训练 SPO+ 预测器的训练器。

    该类整合了：
    1. 线性预测模型 (SimpleLinear)
    2. SPO+ 损失函数 (SPOPlusLoss)
    3. 优化器 (Optimizer)
    """

    def __init__(self, model, loss_fn, optimizer=None, lr=1e-3, device=None):
        """
        Args:
            model: SimpleLinear 实例
            loss_fn: SPOPlusLoss 实例
            optimizer: PyTorch 优化器，若为 None 则默认使用 Adam
            lr: 学习率
            device: 运行设备 (cpu/cuda)
        """
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)

        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer

    def train_epoch(self, train_loader):
        """
        执行一个 Epoch 的训练。

        Args:
            train_loader: DataLoader 实例，生成 (features, true_costs)
        """
        self.model.train()
        total_loss = 0

        for batch in train_loader:
            if len(batch) == 3:
                x, c, scenarios = batch
                scenarios = scenarios.to(self.device)
            else:
                x, c = batch
                scenarios = None

            x, c = x.to(self.device), c.to(self.device)

            # 1. 前向传播：预测 cost
            pred_c = self.model(x)

            # 2. 计算 SPO+ Loss
            loss = self.loss_fn(pred_c, c, scenarios)

            # 3. 反向传播与优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def fit(self, train_loader, epochs=10):
        """
        循环训练多个 Epoch。
        """
        pbar = tqdm(range(epochs), desc="SPO Training")
        for epoch in pbar:
            avg_loss = self.train_epoch(train_loader)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        return self.model

    def predict(self, x):
        """
        使用训练好的模型进行推理。
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32).to(self.device)
            # 期望输入形状为 (num_assets, input_dim)，需要增加 batch 维度变为 (1, num_assets, input_dim)
            if x.ndim == 2:
                x = x.unsqueeze(0)
            pred_c = self.model(x)
        return pred_c.cpu().numpy()
