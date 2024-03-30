from torch import nn, optim
import torch
from typing import Tuple
import numpy as np
from torch.nn import functional as F
from tqdm import trange


# 下面三个函数为辅助函数，无需理解
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# 要求:每步计算需要记录Tensor形状的变化
# 例:[batch size,seqlen,embed_size] -> [batch size,seqlen,heads,head_dim]
class FFN(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()
        # 下面是FFN的两个实现方案，选择一个实现即可
        '''
        方案1:2个linear层,一个激活函数(act)
        两个linear层的名字分别为:up,down
        up:将hidden_size投影为4*hidden_size
        down:将hidden_size*4投影为hidden_size
        计算公式:down(act(up(x)))
        '''
        '''
        方案1:3个linear层,一个激活函数(act)
        三个linear层的名字分别为:up,down,gate
        up:将hidden_size投影为4*hidden_size
        gate:将hidden_size投影为4*hidden_size
        down:将hidden_size*4投影为hidden_size
        计算公式:down(act(up(x)) * gate(x))
        '''
        self.up = nn.Linear(hidden_size, 4 * hidden_size)
        self.gate = nn.Linear(hidden_size, 4 * hidden_size)
        self.down = nn.Linear(4 * hidden_size, hidden_size)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.up(x)) * self.gate(x))


class Attention(nn.Module):

    def __init__(self, head_dim: int, head: int):
        super().__init__()
        '''
        当你选择了使用绝对注意力时使用方案1
        当你选择了使用相对注意力时使用方案2
        '''
        # 以下是进入attention前的形状变换过程，下面用函数R表示
        # R(q,k,v): [batch_size,seqlen,hidden_state] -> [batch_size,seqlen,heads,head_dim] -> [batch_size,heads,seqlen,head_dim]
        # attention计算完后的形状变化过程,用函数P表示，是R的逆过程
        # P(o): [batch_size,heads,seqlen,head_dim] -> [batch_size,seqlen,heads,head_dim] -> [batch_size,seqlen,hidden_state]
        '''
        方案1:
        四个linear层名字分别为q_proj,k_proj,v_proj,o_proj
        计算公式(绝对注意力):
        输入:x,mask
        q = q_proj(x)
        k = k_proj(x)
        v = v_proj(x)
        q,k,v = R(q,k,v)
        attn = Attention(q,k,v,mask)
        attn = P(attn)
        o = o_proj(attn)
        '''
        '''
        方案2(相对注意力):
        四个linear层名字分别为q_proj,k_proj,v_proj,o_proj
        计算公式：
        输入:x,mask,freqs_cis
        q = q_proj(x)
        k = k_proj(x)
        v = v_proj(x)
        q,k,v = R(q,k,v)
        q,k = apply_rotary_emb(q,k,freqs_cis)
        attn = Attention(q,k,v,mask)
        attn = P(attn)
        o = o_proj(attn)
        '''
        self.q_proj = nn.Linear(head_dim * head, head_dim * head)
        self.k_proj = nn.Linear(head_dim * head, head_dim * head)
        self.v_proj = nn.Linear(head_dim * head, head_dim * head)
        self.o_proj = nn.Linear(head_dim * head, head_dim * head)
        self.act = nn.Softmax(dim=-1)
        self.head = head
        self.head_dim = head_dim
        # 思考题：相对位置编码和绝对位置编码有什么区别

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                freqs_cis: torch.Tensor = None) -> torch.Tensor:
        # Attention的Tensor形状变化较为繁琐，注意写出每步的形状变化
        batch_size, seqlen, hidden_state = x.size()
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        query = query.view(batch_size, seqlen, self.head, self.head_dim)
        key = key.view(batch_size, seqlen, self.head, self.head_dim)
        value = value.view(batch_size, seqlen, self.head, self.head_dim)
        query, key = apply_rotary_emb(query, key, freqs_cis)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn_score = (query @ key.transpose(2, 3)) / np.sqrt(self.head_dim)
        if mask is not None:
            attn_score = attn_score + mask
        output = self.act(attn_score.float()) @ value
        output = output.transpose(1, 2).reshape(batch_size, seqlen, hidden_state)
        return self.o_proj(output)


class TransformerBlock(nn.Module):

    def __init__(self, head_dim: int, head: int):
        super().__init__()
        '''
        方案1:
        一个Attention层、一个FFN层、两个Norm层(可以选择使用RMSNorm或者Layernorm)
        计算公式:
        输入x,mask,freqs_cis(相对注意力需要)
        h = x + attention(norm(x),mask,freqs_cis)
        out = h + ffn(norm(h))
        '''
        '''
        方案2:
        一个Attention层、一个FFN层、两个Norm层(可以选择使用RMSNorm或者Layernorm)
        计算公式:
        输入x,mask,freqs_cis(相对注意力需要)
        h = x + norm(attention(x,mask,freqs_cis))
        out = h + norm(ffn(h))
        '''
        self.norm_a = nn.LayerNorm(head_dim * head)
        self.norm_b = nn.LayerNorm(head_dim * head)
        self.attn = Attention(head_dim, head)
        self.ffn = FFN(head_dim * head)
        # 思考题：两种做法有何不同

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                freqs_cis: torch.Tensor = None) -> torch.Tensor:
        x = self.norm_a(self.attn(x, mask, freqs_cis)) + x
        x = self.norm_b(self.ffn(x)) + x
        return x


class Transformer(nn.Module):
    def __init__(self, layers, heads, head_dim, max_seq_len, vocab_size):
        super().__init__()
        '''
        提示:
        1. hidden_size = head_dim * head
        2. layers指的是要实例化多少个TransformerBlock
        3. max_seq_len指的是可以接受的最大输入长度
        4. vocab为词表的大小
        5. 也就是lm_head的最终输出结果
        6. 类的初始化在__init__中，计算在forward中
        '''

        '''
        方案1(绝对位置编码):
        两个embedding层、一个linear层和N个TransformerBlock(记得使用nn.ModuleList包装)
        embedding层的名字分别为:token_embedding,position_embedding
        linear层的名字为:lm_head
        N个TransformerBlock的名字为:trnasformer_blocks
        计算公式:
        输入:token
        hidden_state = token_embedding(token) + position_embedding(range(token.shape[2]))
        hidden_state = trnasformer_blocks(hidden_state,mask)
        out = lm_head(hidden_state)
        '''
        '''
        方案2(相对位置编码):
        一个embedding层、一个linear层和N个TransformerBlock(记得使用nn.ModuleList包装)
        embedding层的名字为:token_embedding
        linear层的名字为:lm_head
        N个TransformerBlock的名字为:trnasformer_blocks
        计算公式:
        输入:token
        hidden_state = token_embedding(token)
        hidden_state = trnasformer_blocks(hidden_state,mask,freqs_cis)
        out = lm_head(hidden_state)
        '''
        # 使用相对位置编码取消下面的注释
        '''
        self.freqs_cis = precompute_freqs_cis(
            head_dim, max_seq_len * 2
        )
        '''
        self.token_embedding = nn.Embedding(vocab_size, heads * head_dim)
        self.freqs_cis = precompute_freqs_cis(
            head_dim, max_seq_len * 2
        ).cuda(0)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(head_dim, heads) for _ in range(layers)
        ])
        self.lm_head = nn.Sequential(
            nn.Linear(head_dim * heads, vocab_size),
            nn.Softmax(dim=-1)
        )

    # @torch.inference_mode
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        for layer in self.transformer_blocks:
            x = layer(x, freqs_cis=self.freqs_cis[:x.shape[1]])
        return self.lm_head(x)


def test_model():
    # run it!
    # torch.set_default_device('cuda:0' if torch.cuda.is_available() else 'cpu')
    vocab_size = 64000  # 与Yi保持一致
    transformer = Transformer(2, 2, 4, 2048, vocab_size).cuda(0)
    tokens = torch.arange(0, 2048).view(1, -1).cuda(0)
    out = transformer(tokens)
    print(out.shape)
    print(out.shape == torch.Size([1, 2048, vocab_size]))  # 输出True为成功


def test_text_input():
    from transformers import BertTokenizer
    text = 'hello world'
    # torch.set_default_device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    encoded_input = tokenizer(text, return_tensors='pt')['input_ids'][0]
    encoded_input = encoded_input.view(1, -1)
    transformer = Transformer(2, 2, 4, 512, tokenizer.vocab_size).cuda(0)
    out = transformer(encoded_input.cuda(0))
    print(out.shape)
    print(out.shape == torch.Size([*encoded_input.shape, tokenizer.vocab_size]))


def train_model():
    from transformers import AutoTokenizer
    import json
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-tokenizer')
    with open('./openwebtext_json/openwebtext_1.json', 'r') as file:
        text_set = json.load(file)
    print("DATA LOAD SUCCESS")
    transformer = Transformer(16, 6, 240, 4096, tokenizer.vocab_size + 1).cuda(0)
    print("MODEL LOAD SUCCESS")
    print(f'Number of parameters: {sum([param.nelement() for param in transformer.parameters()])}')
    optimizer = optim.AdamW(transformer.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(200):
        print(f'Epoch: {epoch}')
        for index, text in enumerate(text_set):
            input_tensor = torch.zeros(4096)
            encoded_input = tokenizer(text, return_tensors='pt')['input_ids'][0]
            input_tensor[0: len(encoded_input)] = encoded_input
            input_tensor[len(encoded_input)] = tokenizer.eos_token_id
            input_tensor = input_tensor.view(1, -1)
            loss_total = .0
            progress = trange(1, len(encoded_input))
            for step in progress:
                optimizer.zero_grad()
                input_t = torch.concat([input_tensor[:, :step], torch.ones(1, 1) * tokenizer.pad_token_id], dim=-1)
                # print(input_t.cpu(), tokenizer.vocab_size)
                # exit(0)
                output: torch.Tensor = transformer(input_t.int().cuda(0))
                # print(input_t.shape, output.shape)

                del input_t
                pred = F.one_hot(input_tensor[:, :step + 1].long().cuda(0), num_classes=tokenizer.vocab_size + 1).float().cuda()
                # print(input_t.shape, output.shape)
                loss: torch.Tensor = criterion(pred, output)
                ll = loss.detach().cpu().numpy().mean()
                loss_total += ll
                progress.desc = f'Loss: {ll}'
                loss.backward()
                optimizer.step()
            print(f'index: {index}, loss: {loss_total / len(progress)}')


def main():
    # import os
    # os.environ['http_proxy'] = 'http://127.0.0.1:8889'
    # os.environ['https_proxy'] = 'http://127.0.0.1:8889'
    # test_model()
    # test_text_input()
    torch.autograd.set_detect_anomaly(True)
    train_model()


if __name__ == '__main__':
    main()
