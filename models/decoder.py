import torch
from torch import nn
from torch.nn import functional as func
import math
import numpy as np
import einops


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, batch_size: int):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.batch_size = batch_size
        pe = torch.zeros(batch_size, d_model)
        position = torch.arange(0, batch_size, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        return x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)


class RMSNorm(nn.Module):

    def __init__(self, embed_size: int):
        super(RMSNorm, self).__init__()
        self.alpha = nn.Parameter(torch.ones(embed_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(embed_size), requires_grad=True)
        self.epsilon = nn.Parameter(torch.zeros(embed_size) + 1e-6, requires_grad=True)

    def forward(self, x: torch.Tensor):
        return x / (x.std(keepdim=True) + self.epsilon) * self.alpha + self.beta


class Attention(nn.Module):

    def __init__(self, num_heads: int, embed_dim: int):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, query, key, value):
        # size: [batch_size, embed_size, src_size]
        query = einops.rearrange(query, 'b (h n) d -> b h n d', h=self.num_heads)
        key = einops.rearrange(key, 'b (h n) d -> b h n d', h=self.num_heads)
        value = einops.rearrange(value, 'b (h n) d -> b h n d', h=self.num_heads)
        attn_score = torch.matmul(query, key.transpose(-2, -1))
        attn_weights = attn_score / math.sqrt(self.embed_dim)
        output = func.softmax(attn_weights, dim=-1) @ value
        output = einops.rearrange(output, 'b h n d -> b (h n) d')
        return output


class FFTAttention(Attention):

    def __init__(self, num_heads: int, embed_dim: int):
        super(FFTAttention, self).__init__(num_heads, embed_dim)

    def forward(self, query, key, value):
        query = einops.rearrange(query, 'b (h n) d -> b h n d', h=self.num_heads)
        key = einops.rearrange(key, 'b (h n) d -> b h n d', h=self.num_heads)
        value = einops.rearrange(value, 'b (h n) d -> b h n d', h=self.num_heads)
        query = torch.fft.fft2(query, dim=-2)
        key = torch.fft.fft2(key, dim=-2)
        attn_score = torch.matmul(query, key.transpose(-2, -1))
        output = torch.abs(torch.fft.ifft2(attn_score, dim=-2)) @ value
        output = einops.rearrange(output, 'b h n d -> b (h n) d')
        return output


class Decoder(nn.Module):

    def __init__(self, n_heads: int, embed_dim: int, hidden_dim: int, dropout_prob: float, use_fft: bool = True):
        super(Decoder, self).__init__()
        self.attn = FFTAttention(num_heads=n_heads, embed_dim=embed_dim) if use_fft \
            else Attention(num_heads=n_heads, embed_dim=embed_dim)
        self.wk = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.wq = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.wv = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.wo = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=embed_dim)
        )
        self.dropout = nn.Dropout(p=dropout_prob)
        self.attn_norm = RMSNorm(embed_dim)
        self.mlp_norm = RMSNorm(embed_dim)

    def forward(self, x: torch.Tensor, src_output: torch.Tensor | None = None):
        s = src_output if src_output is not None else x
        key = self.wk(s).transpose(1, 2)
        query = self.wq(s).transpose(1, 2)
        value = self.wv(x).transpose(1, 2)
        attn = self.attn(query, key, value).transpose(1, 2)
        attn = self.wo(attn) + x
        attn = self.attn_norm(attn)
        attn = self.dropout(attn)
        output = self.mlp(attn) + attn
        output = self.mlp_norm(output)
        return output


class Embedding(nn.Module):

    def __init__(self, embed_dim: int, embed_depth: int) -> None:
        super(Embedding, self).__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Sequential(
            nn.Linear(in_features=1, out_features=embed_depth),
            nn.ELU(),
            nn.Linear(in_features=embed_depth, out_features=embed_dim)
        )

    def forward(self, x: torch.Tensor):
        x = x.reshape(*x.shape, 1)
        return self.embed(x)


class DeEmbedding(nn.Module):

    def __init__(self, embed_dim: int, embed_depth: int) -> None:
        super(DeEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.de_embed = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_depth),
            nn.ELU(),
            nn.Linear(in_features=embed_depth, out_features=1),
            nn.Flatten(start_dim=-2)
        )

    def forward(self, x: torch.Tensor):
        output = self.de_embed(x)
        return output


class Transformer(nn.Module):

    def __init__(self, embed_size: int, hidden_size: int, n_heads: int, embed_depth: int, batch_size: int,
                 dropout_prob: float, tgt_size: int, output_size: int, n_encoders: int = 1, n_decoders: int = 1) -> None:
        super(Transformer, self).__init__()
        self.src_embed = Embedding(embed_dim=embed_size, embed_depth=embed_depth)
        self.tgt_linear = nn.Sequential(
            nn.Linear(in_features=tgt_size, out_features=hidden_size),
            nn.SiLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size)
        )
        self.trg_embed = Embedding(embed_dim=embed_size, embed_depth=embed_depth)
        self.encoders = nn.ModuleList(
            [
                Decoder(n_heads=n_heads, embed_dim=embed_size, hidden_dim=hidden_size, dropout_prob=dropout_prob)
                for _ in range(n_encoders)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                Decoder(n_heads=n_heads, embed_dim=embed_size, hidden_dim=hidden_size, dropout_prob=dropout_prob)
                for _ in range(n_decoders)
            ]
        )
        self.pe = PositionalEncoding(d_model=embed_size, batch_size=batch_size)
        self.de_embed = DeEmbedding(embed_dim=embed_size, embed_depth=embed_depth)

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        src = self.src_embed(src)  # src: [batch_size, src_size, embed_size]
        trg = self.trg_embed(self.tgt_linear(trg))  # tgt: [batch_size, src_size, embed_size]
        src = self.pe(src)
        trg = self.pe(trg)
        output: torch.Tensor = src
        for encoder in self.encoders:
            output = encoder(output)
        src_output = output.clone()
        output = trg
        for decoder in self.decoders:
            output = decoder(output, src_output)
        output = self.de_embed(output)
        return output

    @classmethod
    def test(cls):
        from utils.test import test_model
        batch_size = np.random.randint(10, 2000)
        model = Transformer(n_heads=2, embed_size=10, hidden_size=20, dropout_prob=0, tgt_size=14,
                            output_size=4, batch_size=batch_size, embed_depth=100, n_encoders=10, n_decoders=10)
        test_model("decoder_test", model, batch_size)
