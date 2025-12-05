# FemtoGPT

A clean, refactored reimplementation of Andrej Karpathy's `nanoGPT` repository.

This project is an educational reproduction of a GPT-style Transformer, built from scratch in PyTorch. The goal of this repository is to break down the complexity of Large Language Models (LLM's).

> **Note:** This implementation is heavily based on [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) and his [Let's Build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) video series.

## Motivation

I built this project to deepen my understanding of the Transformer architecture from the first principles, such as:

* The mathematics of Self-Attention mechanisms.
* Positional embeddings.
* The residual stream and layer normalization.

## ️Features

* **From Scratch:** The model is defined purely in PyTorch `nn.Module`.
* **Training Script:** Supports training on the TinyShakespeare dataset to generate infinite Shakespeare-like text.
* **Simple:** I tried to keep it as simple as possible, without adding some extra complexity to the Karpathy's repo + tryed to ompimize some code pieces.
