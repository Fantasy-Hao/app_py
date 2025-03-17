import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from mlp_adam import mlp_adam
from mlp_app import mlp_app
from utils import set_model_params, get_model_params

# 设置种子
torch.manual_seed(42)
np.random.seed(42)

# 创建两个相同的模型
model_app = nn.Sequential(
    nn.Linear(2, 16, bias=True),
    nn.ReLU(),
    nn.Linear(16, 2, bias=False)
).double()

model_adam = nn.Sequential(
    nn.Linear(2, 16, bias=True),
    nn.ReLU(),
    nn.Linear(16, 2, bias=False)
).double()

# 确保两个模型具有相同的初始参数
num_params = len(get_model_params(model_app))
params = np.random.randn(num_params)
# params = np.sqrt(num_params) * params / np.sqrt(np.sum(params ** 2))
set_model_params(model_app, params)
set_model_params(model_adam, params.copy())

# 生成数据
num_samples = 200
X = torch.randn(num_samples, 2, dtype=torch.float64)
y = torch.stack([
    torch.tanh(X[:, 0] * X[:, 1]),
    0.5 * torch.sin(X[:, 0]) + 0.5 * torch.cos(X[:, 1])
], dim=1)

# APP优化参数
K = 400
lambda_ = 1 / np.sqrt(num_params)
rho = 0.96
n = 40

# 运行两种优化方法
print(f"开始优化，参数数量: {len(get_model_params(model_app))}")
loss_history_app = mlp_app(model_app, X, y, K, lambda_, rho, n)
loss_history_adam = mlp_adam(model_adam, X, y, epochs=K, lr=1e-1)

# 绘制损失曲线对比
plt.figure(figsize=(10, 6))
plt.plot(range(len(loss_history_app)), loss_history_app, label="APP")
plt.plot(range(len(loss_history_adam)), loss_history_adam, label="Adam")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("APP vs Adam Optimization")
plt.legend()
plt.grid(True)
plt.yscale('log')  # 使用对数刻度更好地显示损失变化
plt.show()
