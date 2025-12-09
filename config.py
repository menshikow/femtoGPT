import torch

# -----------------------------------------------------------------------------
# model architecture
# -----------------------------------------------------------------------------
n_layer = 6  # number of transformer blocks
n_head = 6  # number of attention heads
n_embd = 384  # embedding dimension
block_size = 256  # context length
vocab_size = 50257  # GPT-2 tokenizer vocabulary size
dropout = 0.0  # 0.0 is usually fine for small datasets/models

# -----------------------------------------------------------------------------
# training hyperparameters
# -----------------------------------------------------------------------------
batch_size = 32
learning_rate = 3e-4
max_iters = 5000
eval_interval = 200
log_interval = 10
eval_iters = 200

# -----------------------------------------------------------------------------
# system & I/O
# -----------------------------------------------------------------------------
out_dir = "out"


# device detection
def detect_divce():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"--> configured to use: {device}")

    return device
