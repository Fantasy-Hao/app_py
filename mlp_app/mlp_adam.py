import torch.nn as nn
from torch.optim import Adam


def mlp_adam(model, X, y, epochs, lr=1e-3):
    """使用 Adam 优化器训练模型"""
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
