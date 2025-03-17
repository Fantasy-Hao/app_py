import numpy as np
from scipy.stats import norm, qmc


def ada_app(fun, x1, K, lambda_, rho, n):
    d = len(x1)
    xk = np.array(x1, dtype=np.float64)
    XTrace = np.zeros((K, d))
    YTrace = np.zeros(K)
    alpha = lambda_

    # 记录历史最优值和连续未改进次数
    best_f = np.inf
    no_improve_count = 0
    max_no_improve = 10  # 连续未改进的最大次数

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
