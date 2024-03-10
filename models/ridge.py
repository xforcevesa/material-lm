import torch
from torch import nn
from torch.nn import functional as func
import numpy as np


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

    @classmethod
    def test(cls):
        batch_size = np.random.randint(10, 2000)
        model = cls(input_dim=5 + 1, output_dim=4)
        from utils.test import test_model2
        test_model2('ridge_test', model, batch_size)
