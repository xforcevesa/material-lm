# material-lm

## Introduction

- This is a pytorch implementation of machine learning algorithms for material structure prediction.

- Currently, we only finished the tests and fundamentals.

- Install the requirements and run with ```python main.py``` now!

## Requirements

```requirements
python>=3.7
torch>=2.0.0
torchvision
einops
numpy
scikit-learn
matplotlib
```

## TODO Lists

1. MLP Regressor (Finished)
2. Ridge Regression (Finished)
3. Random Forests (Finished)
4. Support Vector Regression (Finished)
5. Transformer - Decoder (Finished)
6. MAMBA (Finished)
7. RWKV (Finished)
8. ResNet (Finished)
9. Markov Chains (Finished)
10. Training & Validation
11. Interpretation Work

## Reference

|           Model           |               Location                |                                                                                                                                                                                                                                                                                                                           Reference (Hyperlinks, click to open)                                                                                                                                                                                                                                                                                                                           |
|:-------------------------:|:-------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      Random Forests       |       lm_tests/random_forest.py       |                                                                                                                                                                                                       1. [Machine Learning Benchmarks and Random Forest Regression](https://escholarship.org/uc/item/35x3v9t4)<br/>2. [Approximating Prediction Uncertainty for Random Forest Regression Models](https://www.ingentaconnect.com/content/asprs/pers/2016/00000082/00000003/art00016)                                                                                                                                                                                                       |
|       Markov Chains       |       lm_tests/markov_chain.py        |                                                                                                                                                                                                                                        1. [Markov models—Markov chains](https://www.nature.com/articles/s41592-019-0476-x)<br/>2. [On Markov Chains for Independent Sets](https://www.sciencedirect.com/science/article/abs/pii/S0196677499910714)                                                                                                                                                                                                                                        |
| Support Vector Regression | lm_tests/support_vector_regression.py |                                                                                                                                                                                               1. [A comparative analysis on linear regression and support vector regression](https://ieeexplore.ieee.org/abstract/document/7916627)<br/>2. [Support Vector Regression Machines](https://proceedings.neurips.cc/paper_files/paper/1996/hash/d38901788c533e8286cb6400b40b386d-Abstract.html)                                                                                                                                                                                                |
|     Ridge Regression      |            models/ridge.py            |                                                                                                                                                                                                                                                                              [Ridge Regression: Applications to Nonorthogonal Problems](https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488635)                                                                                                                                                                                                                                                                               |
|       MLP Regressor       |             models/mlp.py             |                                                                                                                                                                                 1. [A Multilayer Perceptron (MLP) Regressor Network for Monitoring the Depth of Anesthesia](https://ieeexplore.ieee.org/abstract/document/9842242)<br/>2. [A multi-layer perceptron approach for accelerated wave forecasting in Lake Michigan](https://www.sciencedirect.com/science/article/abs/pii/S0029801820305382)                                                                                                                                                                                  |
|          ResNet           |           models/resnet.py            |                                                                                                                                                                                                                                                      1. [Resnet in Resnet: Generalizing Residual Architectures](https://arxiv.org/abs/1603.08029)<br/>2. [Convolutional Residual Memory Networks](https://arxiv.org/abs/1606.05262)                                                                                                                                                                                                                                                       |
|        Transformer        |           models/decoder.py           | 1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)<br/>2. [Fast Fourier Transform With Multihead Attention for Specific Emitter Identification](https://ieeexplore.ieee.org/abstract/document/10374078)<br/>3. [FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824)<br/>4. [Fourier Transformer: Fast Long Range Modeling by Removing Sequence Redundancy with FFT Operator](https://arxiv.org/abs/2305.15099)<br/>5. [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625)<br/>6. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) |
|           RWKV            |            models/rwkv.py             |                                                                                                                                     1. [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)<br/>2. [RRWKV: Capturing Long-range Dependencies in RWKV](https://arxiv.org/abs/2306.05176)<br/>3. [RWKV-TS: Beyond Traditional Recurrent Neural Network for Time Series Tasks](https://arxiv.org/abs/2401.09093)<br/>4. [Enhancing Transformer RNNs with Multiple Temporal Perspectives](https://arxiv.org/abs/2402.02625)                                                                                                                                     |
|           MAMBA           |            models/mamba.py            |                                                                                                        1. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)<br/>2. [Is Mamba Capable of In-Context Learning?](https://arxiv.org/abs/2402.03170)<br/>3. [Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models](https://arxiv.org/abs/2403.03900)<br/>4. [DenseMamba: State Space Models with Dense Hidden Connection for Efficient Large Language Models](https://arxiv.org/abs/2403.00818)                                                                                                         |


