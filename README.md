# Building Neural-Network Components from Scratch

**Independent implementation practice — not an MVA course project**

Small PyTorch exercises for understanding the computation inside neural networks. This is a learning repository with two source files, separate from the completed MVA projects in my [portfolio](https://github.com/PineappleBlowsnow#mva-projects).

| File | Implementation |
|---|---|
| [Encoder-decoder Transformer](encoder_decoder_transformer.py) | Multi-head self/cross-attention, layer normalization, sinusoidal positions, encoder and decoder blocks |
| [SwiGLU and optimizer updates](SwiGLU%2B%20adamW.py) | Gated feed-forward computation and explicit Adam/AdamW update equations |

## Scope

The files are implementation exercises, not a pretrained model or a validated training package. No dataset, checkpoint or benchmark is included. The optimizer exercise currently needs an explicit `import torch` before its update functions can run; its equations should be checked against a reference optimizer before use. The Transformer expects callers to supply suitable padding and causal masks.

## Environment

The code uses Python and PyTorch. There is no pinned environment or automated numerical test suite yet. Read the implementation alongside the PyTorch reference operations when using it for study.
