import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from mlp_app import mlp_app
from utils import set_model_params, get_model_params

# 设置种子
torch.manual_seed(42)
np.random.seed(42)

# 创建模型
model = nn.Sequential(
    nn.Linear(2, 8, bias=True),
    nn.ReLU(),
    nn.Linear(8, 2, bias=False)
).double()

# 初始化参数
num_params = len(get_model_params(model))
params = np.random.randn(num_params)
params = np.sqrt(num_params) * params / np.sqrt(np.sum(params ** 2))
set_model_params(model, params)

# 生成数据
num_samples = 200
X = torch.randn(num_samples, 2, dtype=torch.float64)
y = torch.stack([
    torch.tanh(X[:, 0] * X[:, 1]),
    0.5 * torch.sin(X[:, 0]) + 0.5 * torch.cos(X[:, 1])
], dim=1)

# 优化参数
K = 400
lambda_ = 1 / np.sqrt(num_params)
rho = 0.96
n = 40

# 运行优化
print(f"开始优化，参数数量: {len(get_model_params(model))}")
loss_history = mlp_app(model, X, y, K, lambda_, rho, n)

# 绘制损失曲线
plt.plot(range(len(loss_history)), loss_history, label="APP")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("APP Optimization Process")
plt.legend()
plt.grid(True)
plt.show()
