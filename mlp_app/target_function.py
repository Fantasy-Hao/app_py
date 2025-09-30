import matplotlib.pyplot as plt
import numpy as np
import torch

# 定义目标函数
def target_function(X):
    y1 = torch.sin(2 * X[:, 0]) + 0.5 * X[:, 1]
    y2 = torch.cos(3 * X[:, 1]) - 0.3 * X[:, 0]
    y = torch.stack([y1, y2], dim=1)
    return torch.nn.functional.normalize(y, dim=0)

# 生成网格数据
x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(x1, x2)
X = torch.tensor(np.c_[X1.ravel(), X2.ravel()], dtype=torch.float64)

# 计算目标函数值
y = target_function(X).detach().numpy()
Y1 = y[:, 0].reshape(X1.shape)
Y2 = y[:, 1].reshape(X2.shape)

# 绘制目标函数 y1
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.contourf(X1, X2, Y1, levels=50, cmap="viridis")
plt.colorbar()
plt.title("Target Function y1")
plt.xlabel("x1")
plt.ylabel("x2")

# 绘制目标函数 y2
plt.subplot(1, 2, 2)
plt.contourf(X1, X2, Y2, levels=50, cmap="plasma")
plt.colorbar()
plt.title("Target Function y2")
plt.xlabel("x1")
plt.ylabel("x2")

plt.tight_layout()
plt.savefig("./mlp_app/target_function.png", dpi=300, bbox_inches='tight')
plt.show()