"""
The references that Karpathy used in this nanoGPT implementation:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from numpy import block
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ClassifierFreeGuidanceLogitsProcessor

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


class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """transformer block: communication followed by communication"""

    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()

        # 1. communication (self-attention)
        self.sa = CausalSelfAttention(n_embd, n_head, block_size, dropout)

        # 2. computation (feed-forward)
        self.ffwd = FeedForward(n_embd, dropout)

        # 3. layer normalization (pre-norm formulation)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # + ... is the "residual connection"
        # we apply layernorm BEFORE the transformation (pre-norm)

        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class FemtoGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout=0.1):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # 2. the transformer blocks (the body)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )

        # 3. final layer norm
        self.ln_f = nn.LayerNorm(n_embd)

        # 4. language model head (the "output")
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # A. Get Embeddings
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)

        # B. Run through Transformer Blocks
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)

        # C. Calculate Logits (Prediction scores)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
