import torch
from torch import nn
import numpy as np


class TimeMixing(nn.Module):

    def __init__(self, embed_dim: int):
        super(TimeMixing, self).__init__()
        self.shift = torch.nn.ZeroPad2d((0, 0, 1, -1))
        self.act = nn.SiLU()
        self.output_linear = nn.Linear(in_features=embed_dim, out_features=embed_dim)

        self.wk = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.wv = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.wr = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.tmk = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.tmv = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.tmr = nn.Parameter(torch.ones(1, 1, embed_dim))

        self.u = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.w = nn.Parameter(torch.ones(1, 1, embed_dim))

    def forward(self, x):
        xs = self.shift(x)
        k = x * self.tmk + xs * (1 - self.tmk)
        v = x * self.tmv + xs * (1 - self.tmv)
        r = x * self.tmr + xs * (1 - self.tmr)

        r = self.act(r)

        k, v, r = self.wk(k), self.wv(v), self.wr(r)

        wkv = torch.zeros_like(k)
        a_t = torch.zeros_like(k[:, 0, :])
        b_t = torch.zeros_like(k[:, 0, :])

        for i in range(x.shape[1]):
            t = torch.max(self.u + k[:, i, :], self.w)
            a_t = torch.exp(-self.w - t) * a_t + torch.exp(self.u + k[:, i, :] - t) * v[:, i, :]
            b_t = torch.exp(-self.w - t) * b_t + torch.exp(self.u + k[:, i, :] - t)
            wkv[:, i, :] = a_t / b_t

        return self.output_linear(wkv * r)


class ChannelMixing(nn.Module):

    def __init__(self, embed_dim: int):
        super(ChannelMixing, self).__init__()
        self.shift = torch.nn.ZeroPad2d((0, 0, 1, -1))
        self.act = nn.SiLU()
        self.relu = nn.ReLU()
        self.output_linear = nn.Linear(in_features=embed_dim, out_features=embed_dim)

        self.wk = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.wv = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.wr = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.cmr = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.cmk = nn.Parameter(torch.ones(1, 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.shift(x)

        k = x * self.cmk + xs * (1 - self.cmk)
        r = x * self.cmr + xs * (1 - self.cmr)

        k, r = self.wk(k), self.wr(r)
        k, r = self.relu(k).square(), self.act(r)
        v = self.wv(k)

        return self.output_linear(r * v)


class RWKVBlock(nn.Module):

    def __init__(self, embed_dim: int):
        super(RWKVBlock, self).__init__()
        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_b = nn.LayerNorm(embed_dim)
        self.tm = TimeMixing(embed_dim)
        self.cm = ChannelMixing(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x += self.tm(self.norm_a(x))
        x += self.cm(self.norm_b(x))
        return x


class RWKV(nn.Module):

    def __init__(self, embed_dim: int, input_size: int, output_size: int, n_layers: int,
                 hidden_width: int, hidden_depth: int):
        super(RWKV, self).__init__()
        self.layers = nn.ModuleList([RWKVBlock(embed_dim) for _ in range(n_layers)])
        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_b = nn.LayerNorm(embed_dim)
        from models.decoder import Embedding, DeEmbedding
        self.input_linear = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_width),
            nn.ELU(),
            nn.Linear(in_features=hidden_width, out_features=output_size)
        )
        self.embed = Embedding(embed_dim, hidden_depth)
        self.de_embed = DeEmbedding(embed_dim, hidden_depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_linear(x)
        x = self.embed(x)
        x = self.norm_a(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_b(x)
        x = self.de_embed(x)
        return x

    @classmethod
    def test(cls):
        batch_size = np.random.randint(10, 2000)
        model = cls(embed_dim=10, input_size=6, output_size=4, n_layers=2,
                    hidden_width=10, hidden_depth=10)
        from utils.test import test_model2
        test_model2('rwkv_test', model, batch_size)

