import os
import time
import math
import torch
import numpy as np
from model import FemtoGPT
import config

# -----------------------------------------------------------------------------
# configuration
# -----------------------------------------------------------------------------

# I/O
os.makedirs(config.out_dir, exist_ok=True)

# data
# We pull these directly from config now, so we don't have duplicate variables
batch_size = config.batch_size
block_size = config.block_size

# optimizer
learning_rate = config.learning_rate
max_iters = config.max_iters

device = config.detect_divce()

# -----------------------------------------------------------------------------
# data-loader
# -----------------------------------------------------------------------------
data_dir = os.path.join("data", "dostoevsky")
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(split):
    data = train_data if split == "train" else val_data
    # generate random starting spots
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # stack into a batch
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    x, y = x.to(device), y.to(device)
    return x, y


# -----------------------------------------------------------------------------
# loss estimation helper
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # switch to evaluation mode
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # switch back to training mode
    return out


# --- initialize model ---
print("initializing model...")
# Purely using config variables now
model = FemtoGPT(
    vocab_size=config.vocab_size,
    n_embd=config.n_embd,
    block_size=config.block_size,
    n_head=config.n_head,
    n_layer=config.n_layer,
    dropout=config.dropout,
)
model = model.to(device)

# print parameter count
n_params = sum(p.numel() for p in model.parameters())
print(f"number of parameters: {n_params / 1e6:.2f} million")

# -----------------------------------------------------------------------------
# --- optimizer ---
# -----------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# -----------------------------------------------------------------------------
# --- training loop ---
# -----------------------------------------------------------------------------
print("starting training...")
start_time = time.time()

for iter in range(max_iters):
    # every once in a while, evaluate the loss on train and val sets
    if iter % config.eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)

    # zero gradients, perform backprop, and update weights
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# -----------------------------------------------------------------------------
# save the model
# -----------------------------------------------------------------------------
print("saving model checkpoint...")
torch.save(model.state_dict(), os.path.join(config.out_dir, "ckpt.pt"))
print(f"training finished in {(time.time() - start_time) / 60:.2f} minutes")
