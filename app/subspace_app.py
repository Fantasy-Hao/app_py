import numpy as np
from scipy.stats import norm, qmc


def subspace_app(
        func, x_init,
        max_epoch, lambda_epoch, rho_epoch, subspace_dim,
        max_iter, lambda_iter, rho_iter, sample_size,
        verbose=True
):
    total_dim = len(x_init)
    x_full = np.array(x_init, dtype=np.float64)
    x_trace_full = [x_full.copy()]
    y_trace_full = [func(np.array([x_full]))[0]]
    fc_full = np.inf

    if total_dim < subspace_dim:
        raise ValueError("s.t. total_dim > subspace_dim")
    if total_dim % subspace_dim != 0:
        raise ValueError("s.t. total_dim % subspace_dim")

    num_subspaces = total_dim // subspace_dim
    for epoch in range(max_epoch):
        subspace_idx = epoch % num_subspaces
        start_idx = subspace_dim * subspace_idx
        subspace_indices = list(range(start_idx, start_idx + subspace_dim))

        def subspace_func(x_sub_batch):
            x_full_batch = np.tile(x_full, (x_sub_batch.shape[0], 1))
            x_full_batch[:, subspace_indices] = x_sub_batch
            return func(x_full_batch)

        x_sub_init = x_full[subspace_indices]
        x_sub_trace, y_sub_trace, fc_sub = app(subspace_func, x_sub_init, max_iter, lambda_iter, rho_iter, sample_size, verbose=False)
        x_sub_optim = x_sub_trace[-1]
        x_full[subspace_indices] = x_sub_optim
        lambda_epoch *= rho_epoch
        lambda_iter = lambda_epoch

        x_trace_full.append(x_full.copy())
        global_objective = func(np.array([x_full]))[0]
        y_trace_full.append(global_objective)
        fc_full = min(fc_full, fc_sub)

        if verbose:
            print(f'Subspace APP - Epoch: {epoch + 1} Global Objective: {global_objective:.8e}')

    return np.array(x_trace_full), np.array(y_trace_full), fc_full


def app(fun, x1, K, lambda_, rho, n, verbose=True):
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
        if verbose:
            print(f'APP - Iter: {i + 1} Objective: {fk:.8e}')
    return XTrace, YTrace, fc