import torch
from torch import nn
from torch.nn import functional as func


class Ridge(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super(Ridge, self).__init__()
        self.weight = nn.Parameter(torch.ones((input_dim, output_dim)), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros((1, output_dim)), requires_grad=True)

    def loss(self, x: torch.Tensor, y: torch.Tensor, weight_decay: float = 0.) -> torch.Tensor:
        return func.mse_loss(self.forward(x), y) + weight_decay * self.bias.pow(2).sum()

    def backward(self, x: torch.Tensor, y: torch.Tensor, weight_decay: float = 0.):
        return self.loss(x, y, weight_decay).backward()

    def forward(self, x: torch):
        return x @ self.weight + self.bias
