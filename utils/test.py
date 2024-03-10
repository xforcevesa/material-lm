import torch


def test_model(test: str, model: torch.nn.Module, batch_size: int):
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


def test_model2(test: str, model: torch.nn.Module, batch_size: int):
    # src: [B, 6]
    # out: [B, 4]
    model.eval()
    input_src = torch.randn((batch_size, 5 + 1))
    input_trg = torch.randn((batch_size, 14))
    input_src[:, -1] = input_trg.argmax(dim=1)
    output = model(input_src)
    assert output.shape == torch.Size([batch_size, 4]), \
        f'output shape expected: {[batch_size, 4]}, but got {list(output.shape)}'
    print(f"{test}: success! output shape: {list(output.shape)}")


def run_tests():
    from models.ridge import Ridge
    Ridge.test()
    from models.decoder import Transformer
    Transformer.test()
    from models.mlp import MLPRegressor
    MLPRegressor.test()
    from models.mamba import Mamba
    Mamba.test()
    from models.resnet import ResNet
    ResNet.test()
    from models.rwkv import RWKV
    RWKV.test()
