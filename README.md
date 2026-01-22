# LLM Engineering Projects

This repository consists of a collection of my implementations to an llm engineering project list composed by [Ahmad M. Osman](https://www.ahmadosman.com/about/)

![GitHub License](https://img.shields.io/github/license/timurci/llm-engineering?style=flat-square)

## Roadmap

The following is an incomplete version of the composed [list](https://x.com/TheAhmadOsman/status/2011033609856303464).

1. Tokenization & embeddings (**in progress**)
    - Build a byte-pair encoder to train your own subword vocabulary
    - Implement a token visualizer to map chunks to IDs
    - One-hot encoding vs learned embeddings, plot cosine distances
2. Positional embeddings
    - Implement four demos: classic sinusoidal vs learned vs RoPE vs ALiBi
    - Animate a toy sequence being position-encoded in 3D
    - Ablate positions to see the attention collapse
3. Self attention & multi-head attention
    - Hand-wire dot-product attention for one token
    - Scale to multi-head, plot per-head weight heatmaps
    - Mask out future tokens, verify causal property

