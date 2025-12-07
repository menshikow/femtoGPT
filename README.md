# FemtoGPT

![FemtoGPT Art](https://github.com/user-attachments/assets/88a6dad3-771f-4b89-8823-15e69e697783)

A clean, refactored reimplementation of Andrej Karpathy's `nanoGPT` repository.

This project is an educational reproduction of a GPT-style Transformer, built from scratch in PyTorch. The goal of this repository is to break down the complexity of Large Language Models (LLMs).

> **Note:** This implementation is heavily based on [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) and his [Let's Build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) video series.

## Motivation

I built this project to deepen my understanding of the Transformer architecture from first principles, specifically:

* The mathematics of Self-Attention mechanisms.
* Positional embeddings.
* The residual stream and layer normalization.

## Features

* **From Scratch:** The model is defined purely in PyTorch `nn.Module`.
* **Training:** Trained on the three major works of Fyodor Dostoevsky, capable of generating infinite text in his style.
* **Simple:** I tried to keep it as simple as possible without adding extra complexity to Karpathy's repo, while attempting to optimize specific code pieces.

## Install

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm