import torch


def create_model(layer_dims):
    """创建一个简单的MLP模型结构."""
    layers = []
    for i in range(len(layer_dims) - 1):
        layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i + 1], bias=(i < len(layer_dims) - 2)))
        if i < len(layer_dims) - 2:
            layers.append(torch.nn.Tanh())
    return torch.nn.Sequential(*layers).double()


def initialize_model(model):
    """初始化模型参数."""
    for layer in model:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
    return model


def get_model_params(model):
    """将模型参数展平成一个向量."""
    return torch.cat([param.flatten() for param in model.parameters()])


def set_model_params(model, numpy_vector):
    """从一个向量恢复模型参数."""
    vector = torch.tensor(numpy_vector, dtype=torch.float64)
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.data = vector[offset:offset + numel].reshape(param.shape)
        offset += numel
    return model
