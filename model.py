"""
The references that Karpathy used in this nanoGPT implementation:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: move to config file later
block_size = 1024
n_embd = 768  # embedding dimension
n_head = 12  # number of heads
n_layer = 12  # number of layers


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0, "embedding dim must be divisible by heads"

        # key parameters
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        # 1. the projecttions
        # we produce query, key and value for all heads in a single batch for efficiency
        # instead of 3 separate layers, we use one big one -> 3 * n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)

        # 2. output projection
        # to mix the results of all heads back together
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        # 3. regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 4. the casual mask
        # a matrix of 1s and 0s. 1 means "visible", 0 means "hidden (future)"
        # 'register_buffer' tells PyTorch this is not a learnable weight, just a constant
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()  # batch, time (sequence length), channels (n_embd)

        # A. calculate query, key, value
        # we pass input 'x' through the linear layer to get all Q, K, V vectors
        # result shape: (B, T, 3 * n_embd)
        qkv = self.c_attn(x)

        # we split this big tensor into three separate tensors: q, k, v
        q, k, v = qkv.split(self.n_embd, dim=2)

        # B. reshape for multi-head
        # current shape: (b, t, n_embd)
        # we want to split 'n_embd' into 'n_head' pieces of size 'head_dim'
        # new shape: (B, n_head, T, head_dim) - treating heads as a batch dimension
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # C. self-attetion math
        # 1. interaction: calculate affinity scores (query dot key)
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, k, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # 2. masking: ignore the future
        # we fill the upper triangle (where bias is 0) with -infinity
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # 3. softmax: normalize scores to probablities (sums to 1)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 4. aggregate: weight the values by the attention scores
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v

        # D. reassemble
        # concatenate all heads back together
        # (B, nh, T, hs) -> (B, T, n_embd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y


class FemtoGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        # 1. token embedding: looking up the word -> vector
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # 2. position embedding: looking up the position -> vector
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # 3. transformer blocks add later

    def forward(self, idx, targets=None):
        B, T = (
            idx.shape
        )  # batch size (how many sentences), time steps (sentence length)

        # get the token embeddings
        tok_emb = self.token_embedding_table(idx)

        # get the position embeddings
        # we create a list of positions [0, 1, 2, ..., T-1] for each element in the batch
        pos_idx = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos_idx)

        # combine the token embeddings and position embeddings
        x = tok_emb + pos_emb  # (B, T, n_embd)

        return x
