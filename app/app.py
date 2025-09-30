import numpy as np
from scipy.stats import norm, qmc


def app(fun, x1, K, lambda_, rho, n):
    d = len(x1)
    xk = np.array(x1, dtype=np.float64)
    XTrace = np.zeros((K, d))
    YTrace = np.zeros(K)
    alpha = lambda_

    halton = qmc.Halton(d, scramble=True, seed=42)
    fc = np.inf
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
        alpha /= rho
        XTrace[i] = xk
        YTrace[i] = fk
        print(f'APP - Iter: {i + 1} Objective: {fk:.8e}')
    return XTrace, YTrace, fc
