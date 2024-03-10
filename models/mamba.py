import torch
from torch import nn
import einops
import numpy as np


class SelectiveSSM(nn.Module):

    def __init__(self, inner_dim: int, dt_size: int, state_dim: int):
        super(SelectiveSSM, self).__init__()
        self.dt_size = dt_size
        self.state_dim = state_dim
        self.inner_dim = inner_dim
        self.inner_linear = nn.Linear(in_features=inner_dim, out_features=dt_size + state_dim * 2, bias=False)
        self.dt_linear = nn.Linear(in_features=dt_size, out_features=inner_dim)
        alpha = einops.repeat(torch.arange(1, state_dim + 1), 'n -> d n', d=inner_dim)
        self.alpha_log = nn.Parameter(torch.log(alpha), requires_grad=True)
        self.delta = nn.Parameter(torch.ones(inner_dim), requires_grad=True)
        self.act = nn.Softplus()

    @staticmethod
    def selective_scan(input_tensor: torch.Tensor, dt, alpha, beta, gamma, delta) -> torch.Tensor:
        batch_size, seq_len, embed_dim = input_tensor.shape
        alpha_size = alpha.shape[1]
        dt_a = torch.exp(einops.einsum(dt, alpha, 'b l d, d n -> b l d n'))
        dt_b_i = einops.einsum(dt, beta, input_tensor, 'b l d, b l n, b l d -> b l d n')
        x = torch.zeros((batch_size, embed_dim, alpha_size))
        ys = torch.zeros((batch_size, seq_len, embed_dim))
        for i in range(seq_len):
            x = dt_a[:, i] * x + dt_b_i[:, i]
            ys[:, i, :] = einops.einsum(x, gamma[:, i, :], 'b d n, b n -> b d')
        ys = ys + input_tensor * delta
        return ys

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        alpha = -torch.exp(self.alpha_log.float())
        delta = self.delta.float()
        dt, beta, gamma = self.inner_linear(input_tensor).split([self.dt_size, self.inner_dim, self.inner_dim], dim=-1)
        dt = self.act(self.dt_linear(dt))
        return self.selective_scan(input_tensor, dt, alpha, beta, gamma, delta)


class MambaBlock(nn.Module):

    def __init__(self, embed_dim: int, inner_dim: int, kernel_size: int, dt_size: int, state_dim: int):
        super(MambaBlock, self).__init__()
        self.input_linear = nn.Linear(in_features=embed_dim, out_features=inner_dim * 2)
        self.conv = nn.Conv1d(
            in_channels=inner_dim,
            out_channels=inner_dim,
            kernel_size=kernel_size,
            groups=inner_dim,
            padding=kernel_size - 1
        )
        self.output_linear = nn.Linear(in_features=inner_dim, out_features=embed_dim)
        self.act = nn.SiLU()
        self.selective_ssm = SelectiveSSM(
            inner_dim=inner_dim,
            dt_size=dt_size,
            state_dim=state_dim
        )
        from models.decoder import RMSNorm
        self.norm = RMSNorm(embed_size=embed_dim)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = input_tensor.shape
        input_tensor = self.norm(input_tensor)
        x, zeta = self.input_linear(input_tensor).chunk(2, dim=-1)
        x = einops.rearrange(x, 'b l d -> b d l')
        # print(x.shape, [batch_size, seq_len, embed_dim])
        x = self.conv(x)[:, :, :seq_len]
        x = einops.rearrange(x, 'b d l -> b l d')
        x = self.act(x)
        output_tensor = self.selective_ssm(x)
        return self.output_linear(output_tensor) + input_tensor


class Mamba(nn.Module):

    def __init__(self, embed_dim: int, inner_dim: int, kernel_size: int, dt_size: int, state_dim: int,
                 n_layers: int, hidden_size: int, input_size: int, output_size: int, embed_depth: int) -> None:
        super(Mamba, self).__init__()
        self.input_linear = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.SiLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size)
        )
        self.layers = nn.ModuleList([
            MambaBlock(embed_dim, inner_dim, kernel_size, dt_size, state_dim)
            for _ in range(n_layers)
        ])
        from models.decoder import Embedding, DeEmbedding, RMSNorm
        self.embed = Embedding(embed_dim=embed_dim, embed_depth=embed_depth)
        self.de_embed = DeEmbedding(embed_dim=embed_dim, embed_depth=embed_depth)
        self.norm = RMSNorm(embed_size=embed_dim)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.embed(self.input_linear(input_tensor))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.de_embed(x)

    @classmethod
    def test(cls):
        batch_size = np.random.randint(10, 2000)
        model = cls(embed_dim=10, hidden_size=20, input_size=6, dt_size=10, inner_dim=10, kernel_size=3,
                    output_size=4, embed_depth=10, n_layers=10, state_dim=10)
        input_src = torch.randn((batch_size, 5 + 1))
        input_trg = torch.randn((batch_size, 14))
        input_src[:, -1] = input_trg.argmax(dim=1)
        output = model(input_src)
        assert output.shape == torch.Size([batch_size, 4]), \
            f'output shape expected: {[batch_size]}, but got {list(output.shape)}'
        print(f"mamba_test: success! output shape: {list(output.shape)}")
