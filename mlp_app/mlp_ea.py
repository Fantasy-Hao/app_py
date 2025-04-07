import torch
from pyswarm import pso
from scipy.optimize import dual_annealing, differential_evolution

from utils import get_model_params, set_model_params


def mlp_sa(model, X, y, epochs=400, initial_temp=1.0, restart_temp_ratio=2e-5, bounds=None):
    """使用模拟退火算法优化MLP模型. """
    loss_fn = torch.nn.MSELoss()

    # 获取初始参数
    x0 = get_model_params(model).detach().numpy()
    n_params = len(x0)

    # 设置参数边界
    if bounds is None:
        bounds = [(-1, 1)] * n_params

    # 定义目标函数
    def objective_function(params):
        set_model_params(model, params)
        with torch.no_grad():
            y_pred = model(X)
            loss = loss_fn(y_pred, y).item()
        return loss

    # 用于记录每次迭代的损失
    loss_history = []
    best_loss = float('inf')

    # 回调函数，用于记录优化过程
    def callback(x, f, context):
        nonlocal best_loss
        if f < best_loss:
            best_loss = f
        loss_history.append(best_loss)

        iteration = len(loss_history)
        if iteration % 10 == 0:
            print(f'MLP_SA - Iter: {iteration}, Loss: {best_loss:.12e}')

        # 如果达到最大迭代次数，停止优化
        if len(loss_history) >= epochs:
            return True
        return False

    # 运行模拟退火优化
    result = dual_annealing(
        objective_function,
        bounds,
        maxiter=epochs,
        initial_temp=initial_temp,
        restart_temp_ratio=restart_temp_ratio,
        callback=callback,
        no_local_search=False  # 启用局部搜索以提高精度
    )

    # 设置最佳参数
    set_model_params(model, result.x)

    # 如果迭代次数不足，补齐损失历史
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
            y_pred = model(X)
            loss = loss_fn(y_pred, y).item()
        return loss

    def callback(xk, convergence):
        nonlocal best_loss
        current_loss = objective_function(xk)
        best_loss = min(best_loss, current_loss)
        loss_history.append(best_loss)

        iteration = len(loss_history)
        if iteration % 10 == 0:
            print(f'MLP_DE - Iter: {iteration}, Loss: {best_loss:.12e}')

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
        lb = [-1] * n_params  # 下界
        ub = [1] * n_params  # 上界
    else:
        lb = [b[0] for b in bounds]  # 下界
        ub = [b[1] for b in bounds]  # 上界

    loss_history = []
    best_loss = float('inf')
    iteration_counter = [0]  # 使用列表以便在回调中修改

    def objective_function(params):
        set_model_params(model, params)
        with torch.no_grad():
            y_pred = model(X)
            loss = loss_fn(y_pred, y).item()

        # 记录当前最佳损失
        nonlocal best_loss
        if loss < best_loss:
            best_loss = loss
        loss_history.append(best_loss)

        # 打印进度
        iteration_counter[0] += 1
        if iteration_counter[0] % 10 == 0:
            print(f'MLP_PSO - Iter: {iteration_counter[0]}, Loss: {best_loss:.12e}')

        # 如果达到最大迭代次数，停止优化
        if iteration_counter[0] >= epochs:
            return float('inf')  # 返回一个大值，使优化器认为这是一个不好的解

        return loss

    # 运行粒子群优化
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

    # 设置最佳参数
    set_model_params(model, xopt)

    # 如果迭代次数不足，补齐损失历史
    if len(loss_history) < epochs:
        loss_history.extend([loss_history[-1]] * (epochs - len(loss_history)))

    return loss_history
