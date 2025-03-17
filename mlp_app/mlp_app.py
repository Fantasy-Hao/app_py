import numpy as np
import torch.nn.functional as F
from scipy.stats import qmc, norm

from utils import get_model_params, set_model_params


def loss_fun(model, params, inputs, target):
    set_model_params(model, params)
    return F.mse_loss(model(inputs), target).item()


def mlp_app(model, inputs, target, K, lambda_, rho, n):
    params = get_model_params(model)
    d = len(params)
    xk = params.detach().numpy().astype(np.float64)
    loss_history = []
    alpha = lambda_

    halton = qmc.Halton(d=d, scramble=True, seed=42)
    fc = np.inf
    for i in range(K):
        # Generate n random vector from halton sequence
        x = halton.random(n)
        t = np.vstack([xk, norm.ppf(x, loc=xk, scale=1 / alpha)])

        # Compute function value sequence
        f = [loss_fun(model, t[k], inputs, target) for k in range(n + 1)]
        fk = f[0]
        f_min = min(f)
        fc = min(fc, f_min)
        f = np.array(f) - f_min

        # Use averaged asymptotic formula
        f_mean = np.mean(f)
        if f_mean > 0:
            f /= f_mean

        # Compute weights and new xk
        weights = np.exp(-f)
        xk = np.average(t, axis=0, weights=weights)

        # Update and record
        alpha /= rho
        loss_history.append(fk)
        print(f'MLP_APP - Iter: {i + 1}, Loss: {fk:.12e}')
    return loss_history
