import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import qr
from scipy.stats import truncnorm, norm, qmc
from subspace_app import subspace_app


# 参数设置
d = 200  # 总维度
a, b = 1, 2  # 均值和标准差
c = 1

# 创建并截断正态分布对象，然后生成随机向量r
np.random.seed(100)  # 固定随机数种子
r = truncnorm.rvs(a=0, b=np.inf, loc=a, scale=b, size=d)
R = np.diag(r)

# 生成正交矩阵
np.random.seed(100)  # 固定随机数种子
U = qr(np.random.randn(d, d))[0]
A = U.T @ R @ U
A = A / np.linalg.norm(A, axis=1, keepdims=True)

# 目标函数
func = lambda x: c * d - c * np.sum(np.cos(3 * np.pi * x), axis=1) + np.diag(x @ A @ x.T)

# 生成初始点
np.random.seed(42)
x_init = 2 * np.random.rand(d) - 1
x_init = np.sqrt(d) * x_init / np.linalg.norm(x_init)

# 子空间优化参数
max_epoch = 160  # 外循环迭代次数（选择子空间的次数）
lambda_epoch = 1 / np.sqrt(d)
rho_epoch = 0.8
subspace_dim = 50  # 固定的子空间维度

# 内部APP算法参数
max_iter = 1000  # 内部APP迭代次数（比原始APP多，因为子空间维度更高）
lambda_iter = lambda_epoch
rho_iter = 0.98
sample_size = 50

# 运行子空间APP算法
print(f"\n运行子空间维度: {subspace_dim}")
x_trace, _, _ = subspace_app(
    func, x_init,
    max_epoch, lambda_epoch, rho_epoch, subspace_dim,
    max_iter, lambda_iter, rho_iter, sample_size,
    verbose=True
)

# 绘图
plt.figure(figsize=(10, 8))
# 计算距离平方的变化（参考app_run.py）
distance_squared = np.sum(x_trace ** 2, axis=1)

plt.plot(np.log10(distance_squared), 'k-', label=f'子空间维度={subspace_dim}')

plt.ylim([-12, 4])
plt.yticks([-12, -8, -4, 0, 4])
plt.title(f'子空间APP算法 - 总维度d={d}')
plt.xlabel('迭代次数 (epoch)')
plt.ylabel(r'$\log_{10}\|x_k-x_*\|_2^2$', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('./app/subspace_app.png', dpi=300, bbox_inches='tight')
plt.show()

# 输出最终结果
print("\n最终结果:")
final_objective = func(np.array([x_trace[-1]]))[0]
print(f"子空间维度 {subspace_dim}: 最终目标函数值 = {final_objective:.8e}")
