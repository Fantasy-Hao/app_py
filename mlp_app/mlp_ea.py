import torch
from pyswarm import pso
from scipy.optimize import dual_annealing, differential_evolution
from utils import get_model_params, set_model_params


def mlp_sa(model, X, y, epochs=400, initial_temp=1.0, restart_temp_ratio=2e-5, bounds=None):
    """使用模拟退火算法优化MLP模型."""
    loss_fn = torch.nn.MSELoss()
    x0 = get_model_params(model).detach().numpy()
    n_params = len(x0)
    if bounds is None:
        bounds = [(-1, 1)] * n_params

    def objective_function(params):
        set_model_params(model, params)
        with torch.no_grad():
            return loss_fn(model(X), y).item()

    loss_history = []
    best_loss = float('inf')

    def callback(x, f, context):
        nonlocal best_loss
        if f < best_loss:
            best_loss = f
        loss_history.append(best_loss)
        if len(loss_history) >= epochs:
            return True
        return False

    result = dual_annealing(
        objective_function,
        bounds,
        maxiter=epochs,
        initial_temp=initial_temp,
        restart_temp_ratio=restart_temp_ratio,
        callback=callback,
        no_local_search=False
    )

    set_model_params(model, result.x)
    if len(loss_history) < epochs:
        loss_history.extend([loss_history[-1]] * (epochs - len(loss_history)))
    return loss_history


def mlp_de(model, X, y, epochs=400, population_size=50, bounds=None):
    """使用差分进化算法优化MLP模型."""
    loss_fn = torch.nn.MSELoss()
    x0 = get_model_params(model).detach().numpy()
    n_params = len(x0)
    if bounds is None:
        bounds = [(-1, 1)] * n_params

    loss_history = []
    best_loss = float('inf')

    def objective_function(params):
        set_model_params(model, params)
        with torch.no_grad():
            return loss_fn(model(X), y).item()

    def callback(xk, convergence):
        nonlocal best_loss
        current_loss = objective_function(xk)
        best_loss = min(best_loss, current_loss)
        loss_history.append(best_loss)

    result = differential_evolution(
        objective_function,
        bounds,
        maxiter=epochs,
        popsize=population_size,
        callback=callback,
        updating='deferred',
        workers=1
    )

    set_model_params(model, result.x)
    return loss_history


def mlp_pso(model, X, y, epochs=400, swarmsize=50, omega=0.5, phip=0.5, phig=0.5, bounds=None):
    """使用粒子群算法优化MLP模型."""
    loss_fn = torch.nn.MSELoss()
    x0 = get_model_params(model).detach().numpy()
    n_params = len(x0)

    if bounds is None:
        lb = [-1] * n_params
        ub = [1] * n_params
    else:
        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds]

    loss_history = []
    best_loss = float('inf')
    iteration_counter = [0]

    def objective_function(params):
        set_model_params(model, params)
        with torch.no_grad():
            loss = loss_fn(model(X), y).item()
        nonlocal best_loss
        if loss < best_loss:
            best_loss = loss
        loss_history.append(best_loss)
        iteration_counter[0] += 1
        if iteration_counter[0] >= epochs:
            return float('inf')
        return loss

    xopt, fopt = pso(
        objective_function,
        lb,
        ub,
        swarmsize=swarmsize,
        omega=omega,
        phip=phip,
        phig=phig,
        maxiter=epochs,
        debug=False
    )

    set_model_params(model, xopt)
    if len(loss_history) < epochs:
        loss_history.extend([loss_history[-1]] * (epochs - len(loss_history)))
    return loss_history