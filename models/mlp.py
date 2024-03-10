import torch
from torch import nn
import numpy as np


class MLP(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, hidden_depth: int, output_size: int):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            *[
                nn.Sequential(
                    nn.ELU(),
                    nn.Linear(in_features=hidden_size, out_features=hidden_size)
                ) for _ in range(hidden_depth)
            ],
            nn.ELU(),
            nn.Linear(in_features=hidden_size, out_features=output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MLPRegressor(nn.Module):

    def __init__(self, input_src: int, input_trg: int, hidden_size: int, hidden_depth: int,
                 inter_size: int, output_size: int):
        super(MLPRegressor, self).__init__()
        self.src_mlp = MLP(input_size=input_src, hidden_size=hidden_size,
                           hidden_depth=hidden_depth, output_size=inter_size)
        self.trg_mlp = MLP(input_size=input_trg, hidden_size=hidden_size,
                           hidden_depth=hidden_depth, output_size=inter_size)
        self.out_mlp = MLP(input_size=inter_size * 2, hidden_size=hidden_size,
                           hidden_depth=hidden_depth, output_size=output_size)

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        inter = torch.cat([self.src_mlp(src), self.trg_mlp(trg)], dim=1)
        return self.out_mlp(inter)

    @classmethod
    def test(cls):
        from test import test_model
        batch_size = np.random.randint(10, 2000)
        model = MLPRegressor(input_src=5, input_trg=14, hidden_size=5, hidden_depth=3, inter_size=20, output_size=4)
        test_model("mlp_regressor_test", model, batch_size)
