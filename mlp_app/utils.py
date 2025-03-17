import torch


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
