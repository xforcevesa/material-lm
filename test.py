import torch
import numpy as np


def __test_model(test: str, model: torch.nn.Module, batch_size: int):
    # src: [B, 5]
    # tgt: [B, 14]
    # out: [B, 4]
    model.eval()
    input_src = torch.randn(batch_size, 5)
    input_tgt = torch.randn(batch_size, 14)
    output = model(input_src, input_tgt)
    assert output.shape == torch.Size([batch_size, 4]), \
        f'output shape expected: {[batch_size]}, but got {list(output.shape)}'
    print(f"{test}: success! output shape: {list(output.shape)}")


def decoder_test():
    from decoder import TransformerModel
    batch_size = np.random.randint(10, 2000)
    model = TransformerModel(n_heads=2, embed_size=10, hidden_size=20, dropout_prob=0, tgt_size=14,
                             output_size=4, batch_size=batch_size, embed_depth=100)
    __test_model("decoder_test", model, batch_size)


def mlp_regressor_test():
    from mlp import MLPRegressor
    batch_size = np.random.randint(10, 2000)
    model = MLPRegressor(input_src=5, input_trg=14, hidden_size=5, hidden_depth=3, inter_size=20, output_size=4)
    __test_model("mlp_regressor_test", model, batch_size)


def ridge_test():
    from ridge import Ridge
    batch_size = np.random.randint(10, 2000)
    model = Ridge(input_dim=5 + 1, output_dim=4)
    input_src = torch.randn((batch_size, 5 + 1))
    input_trg = torch.randn((batch_size, 14))
    input_src[:, -1] = input_trg.argmax(dim=1)
    output = model(input_src)
    assert output.shape == torch.Size([batch_size, 4]), \
        f'output shape expected: {[batch_size]}, but got {list(output.shape)}'
    print(f"ridge_test: success! output shape: {list(output.shape)}")
