import torch.nn as nn
from torch.optim import Adam, LBFGS


def mlp_adam(model, X, y, epochs=400, lr=1e-3):
    """使用 Adam 优化器训练模型. """
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loss_history = []

    for i in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        print(f'MLP_Adam - Iter: {i + 1}, Loss: {loss.item():.12e}')

    return loss_history


def mlp_lbfgs(model, X, y, epochs=400, max_iter=20, tolerance_grad=1e-7, tolerance_change=1e-9):
    """使用L-BFGS优化MLP模型参数. """
    model.train()
    loss_fn = nn.MSELoss()
    optimizer = LBFGS(model.parameters(),
                      max_iter=max_iter,
                      tolerance_grad=tolerance_grad,
                      tolerance_change=tolerance_change)

    loss_history = []

    def closure():
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        return loss

    for epoch in range(epochs):
        loss = optimizer.step(closure)
        loss_history.append(loss.detach().item())
        print(f'MLP_LBFGS - Iter: {epoch + 1}, Loss: {loss:.12e}')

    return loss_history
