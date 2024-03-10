import numpy as np
from torch import nn


class CnnBlock(nn.Module):

    def __init__(self, depth: int, in_channels, out_channels, kernel_size, embed_dim):
        super(CnnBlock, self).__init__()
        self.model = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=(kernel_size - 1) // 2,
                        bias=True
                    ),
                    nn.BatchNorm1d(
                        num_features=out_channels
                    ),
                    nn.ELU(),
                    nn.Linear(in_features=embed_dim, out_features=embed_dim)
                ) for _ in range(depth)
            ]
        )

    def forward(self, x):
        return self.model(x) + x


class ResNet(nn.Module):

    def __init__(self, input_size: int, output_size: int, embed_dim: int, hidden_width: int,
                 n_layers_a: int, n_layers_b: int, depth: int, kernel_size: int):
        super(ResNet, self).__init__()
        self.layers_a = nn.ModuleList([
            CnnBlock(depth, input_size, input_size, kernel_size, embed_dim) for _ in range(n_layers_a)
        ])
        self.layers_b = nn.ModuleList([
            CnnBlock(depth, output_size, output_size, kernel_size, embed_dim) for _ in range(n_layers_b)
        ])
        from models.decoder import Embedding, DeEmbedding
        self.input_linear = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_width),
            nn.ELU(),
            nn.Linear(in_features=hidden_width, out_features=output_size)
        )
        self.embedding = Embedding(embed_dim=embed_dim, embed_depth=hidden_width)
        self.de_embedding = DeEmbedding(embed_dim=embed_dim, embed_depth=hidden_width)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers_a:
            x = layer(x)
        x = self.input_linear(x.transpose(1, 2)).transpose(1, 2)
        for layer in self.layers_b:
            x = layer(x)
        x = self.de_embedding(x)
        return x

    @classmethod
    def test(cls):
        batch_size = np.random.randint(10, 2000)
        model = cls(input_size=6, output_size=4, embed_dim=20, hidden_width=20,
                    n_layers_a=2, n_layers_b=2, depth=3, kernel_size=3)
        from utils.test import test_model2
        test_model2('resnet_test', model, batch_size)
