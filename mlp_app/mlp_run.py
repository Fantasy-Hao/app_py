import matplotlib.pyplot as plt
import numpy as np
import torch
from mlp_app import mlp_app
from mlp_ea import mlp_sa, mlp_de, mlp_pso
from mlp_grad import mlp_adam, mlp_lbfgs
from utils import create_model, initialize_model, get_model_params, set_model_params

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

layer_dims = [2, 8, 2]
model_app = create_model(layer_dims)
model_sa = create_model(layer_dims)
model_de = create_model(layer_dims)
model_pso = create_model(layer_dims)
model_adam = create_model(layer_dims)
model_lbfgs = create_model(layer_dims)

initialize_model(model_app)
initial_params = get_model_params(model_app).detach().numpy()
set_model_params(model_app, initial_params.copy())
# set_model_params(model_sa, initial_params.copy())
# set_model_params(model_de, initial_params.copy())
# set_model_params(model_pso, initial_params.copy())
set_model_params(model_adam, initial_params.copy())
set_model_params(model_lbfgs, initial_params.copy())

num_samples = 200
X = torch.randn(num_samples, 2, dtype=torch.float64)
y = torch.stack([
    torch.sin(2 * X[:, 0]) + 0.5 * X[:, 1],
    torch.cos(3 * X[:, 1]) - 0.3 * X[:, 0]
], dim=1)
y = torch.nn.functional.normalize(y, dim=0)

max_iters = 400

loss_history_app = mlp_app(model_app, X, y, K=max_iters, lambda_=1 / np.sqrt(len(initial_params)), rho=0.96, n=len(initial_params))
# loss_history_sa = mlp_sa(model_sa, X, y, epochs=max_iters, initial_temp=1.0, restart_temp_ratio=2e-5, bounds=None)
# loss_history_de = mlp_de(model_de, X, y, epochs=max_iters, population_size=len(initial_params), bounds=None)
# loss_history_pso = mlp_pso(model_pso, X, y, epochs=max_iters, swarmsize=len(initial_params), omega=0.5, phip=0.5, phig=0.5, bounds=None)
loss_history_adam = mlp_adam(model_adam, X, y, epochs=max_iters, lr=1e-3)
loss_history_lbfgs = mlp_lbfgs(model_lbfgs, X, y, epochs=max_iters)

plt.figure(figsize=(10, 6))
plt.plot(range(max_iters), loss_history_app[:max_iters], label="APP", color='blue', linewidth=2)
# plt.plot(range(max_iters), loss_history_sa[:max_iters], label="SA", color='red', linewidth=2)
# plt.plot(range(max_iters), loss_history_de[:max_iters], label="DE", color='green', linewidth=2)
# plt.plot(range(max_iters), loss_history_pso[:max_iters], label="PSO", color='purple', linewidth=2)
plt.plot(range(max_iters), loss_history_adam[:max_iters], label="Adam", color='orange', linewidth=2)
plt.plot(range(max_iters), loss_history_lbfgs[:max_iters], label="LBFGS", color='brown', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.title("Optimization Methods Comparison on MLP")
plt.savefig("./mlp_run.png", dpi=300, bbox_inches='tight')

with torch.no_grad():
    loss_app = torch.nn.functional.mse_loss(model_app(X), y).item()
    # sa_loss = torch.nn.functional.mse_loss(model_sa(X), y).item()
    # de_loss = torch.nn.functional.mse_loss(model_de(X), y).item()
    # pso_loss = torch.nn.functional.mse_loss(model_pso(X), y).item()
    adam_loss = torch.nn.functional.mse_loss(model_adam(X), y).item()
    lbfgs_loss = torch.nn.functional.mse_loss(model_lbfgs(X), y).item()
# print(f"最终损失 - APP: {loss_app:.2e}, SA: {sa_loss:.2e}, DE: {de_loss:.2e}, PSO: {pso_loss:.2e}, Adam: {adam_loss:.2e}, LBFGS: {lbfgs_loss:.2e}")
print(f"最终损失 - APP: {loss_app:.2e}, Adam: {adam_loss:.2e}, LBFGS: {lbfgs_loss:.2e}")
print(f"参数数量: {len(initial_params)}")