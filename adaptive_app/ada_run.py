import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import qr
from scipy.stats import truncnorm, norm, qmc

log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
log_filename = os.path.join(log_dir, f"ada_log_{timestamp}.txt")


def ada_app(fun, x1, K, lambda_, rho, n):
    d = len(x1)
    xk = np.array(x1, dtype=np.float64)
    XTrace = np.zeros((K, d))
    YTrace = np.zeros(K)
    alpha = lambda_

    halton = qmc.Halton(d, scramble=True, seed=42)
    fc = np.inf
    with open(log_filename, 'a', encoding='utf-8') as log_file:
        for i in range(K):
            # Generate n random vector from halton sequence
            x = halton.random(n)
            t = np.vstack([xk, norm.ppf(x, loc=xk, scale=1 / alpha)])

            # Compute function value sequence
            f = fun(t)
            fk = f[0]
            f_min = min(f)
            fc = min(fc, f_min)
            f = f - f_min

            # Use averaged asymptotic formula
            f_mean = np.mean(f)
            if f_mean > 0:
                f /= f_mean

            # Compute weights and new xk
            weights = np.exp(-f)
            xk = np.average(t, axis=0, weights=weights)

            # Update and record
            if f_mean / fk < 1 / (i + 1):
                alpha *= rho ** i
            else:
                alpha /= rho

            XTrace[i] = xk
            YTrace[i] = fk
            # print(f'APP - Iter: {i + 1} alpha: {alpha:.8e} f_mean: {f_mean:.8e} Objective: {fk:.8e};')

            log_msg = f'APP - Iter: {i + 1} alpha: {alpha:.8e} f_mean: {f_mean:.8e} Objective: {fk:.8e};'
            print(log_msg)
            log_file.write(log_msg + '\n')
    return XTrace, YTrace, fc


# 参数设置
d = 100
a, b = 1, 2  # 均值和标准差

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
c = 1
fun = lambda x: c * d - c * np.sum(np.cos(3 * np.pi * x), axis=1) + np.diag(x @ A @ x.T)

# 算法参数
K = 8000
lambda_ = 1 / np.sqrt(d)
rhos = [0.993, 0.995, 0.997]
n = 40

# 生成初始点
np.random.seed(42)
x1three = 2 * np.random.rand(3, d) - 1
x1three = np.sqrt(d) * x1three / np.linalg.norm(x1three, axis=1, keepdims=True)

# 运行算法
XTrace1 = ada_app(fun, x1three[0, :], K, lambda_, rhos[0], n)[0]
XTrace2 = ada_app(fun, x1three[1, :], K, lambda_, rhos[1], n)[0]
XTrace3 = ada_app(fun, x1three[2, :], K, lambda_, rhos[2], n)[0]

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(np.log10(np.sum(XTrace1 ** 2, axis=1)), 'k-', label=f'ρ={rhos[0]}, n={n}')
plt.plot(np.log10(np.sum(XTrace2 ** 2, axis=1)), 'b--', label=f'ρ={rhos[1]}, n={n}')
plt.plot(np.log10(np.sum(XTrace3 ** 2, axis=1)), 'm-.', label=f'ρ={rhos[2]}, n={n}')
plt.ylim([-12, 4])
plt.yticks([-12, -8, -4, 0, 4])
plt.title(f'd={d}')
plt.xlabel('iteration (k)')
plt.ylabel(r'$\log_{10}\|x_k-x_*\|_2^2$', fontsize=10)
plt.legend()
plt.savefig(f"{log_dir}/ada_log_{timestamp}.png")
plt.show()
