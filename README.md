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
````

## Usage

### 1\. Prepare the Data

First, download the Dostoevsky dataset and tokenize it into binary files (`train.bin` and `val.bin`) that the model can read.

```bash
python data/dostoevsky/prepare.py
```

### 2\. Configuration

You can adjust hyperparameters (like `n_layer`, `batch_size`, `device`) in the `config.py` file. The project is set up by default to auto-detect your hardware (CUDA, MPS, or CPU).

### 3\. Train

Start the training loop. This will utilize your GPU (if available) to train the model and save checkpoints to the `out/` directory.

```bash
python train.py
```

*Note: On an M-series Mac or NVIDIA GPU, the "Baby GPT" config should reach a good loss (\< 3.0) in about 15 minutes.*

### 4\. Generate Text

Once training is complete (or if you have a `ckpt.pt` file), you can generate new text samples.

```bash
python sample.py
```

## Project Structure

```text
.
├── config.py           # Hyperparameters and device settings
├── model.py            # The GPT Architecture (Attention, FeedForward, Blocks)
├── train.py            # Training loop and loss estimation
├── sample.py           # Text generation script
├── data/
│   └── dostoevsky/     # Data preparation scripts and raw text
└── out/                # Saved model checkpoints
```

