import os
import time
import math
import numpy
from config import *
from model import FemtoGPT

# configuration
# I/O
out_dir = "out"
eval_interval = 200  # how often to check validation loss
log_interval = 10  # how often to print training stats
eval_iters = 200  # how many batches to use for estimating loss
os.makedirs(out_dir, exist_ok=True)

# data
batch_size = 32  # how many independent sequences to process in parallel
block_size = 256  # maxim context length for predictions

# model (very small but faster to train)
n_layer = 6  # layers
n_head = 6  # heads
n_embd = 384  # 384 embedding dimension
