import torch
import torch.nn.functional as F
import tiktoken
from model import FemtoGPT
import config

# setup device
device = config.detect_divce()

# sampling settings
start_prompt = "\n"
num_samples = 3
max_new_tokens = 500
top_k = 200  # keep only the top 200 likly next words
temperature = 0.8

# load the model
print(f"moding the model from {config.out_dir}...")

# initialize the model structure (must match training exactly!)
model = FemtoGPT(
    vocab_size=config.vocab_size,
    n_embd=config.n_embd,
    block_size=config.block_size,
    n_head=config.n_head,
    n_layer=config.n_layer,
    dropout=config.dropout,
)

# load the trained weights
ckpt_path = f"{config.out_dir}/ckpt.pt"
try:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    print("weight loaded successfully")
except FileNotFoundError:
    print(f"could not find {ckpt_path}")
    exit()

model.to(device)
model.eval()


# setup tokenizer
enc = tiktoken.get_encoding("gpt-2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)


# generation loop
def generate(idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # 1. crop context if it's too long
        idx_cond = idx[:, -config.block_size :]

        # 2. get the predictions
        logits, _ = model(idx_cond)

        # 3. focus only on the last time step
        logits = logits[:, -1, :]

        # 4. apply temperature
        logits = logits / temperature

        # 5. crop to top_k options (optional)
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")

        # 6. sample from the distribution
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # 7. append to the sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# run
print("tokens generation...\n")
print("------------------------------------------------")

start_ids = encode(start_prompt)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

with torch.no_grad():
    for k in range(num_samples):
        y = generate(x, max_new_tokens)
        print(decode(y[0].tolist()))
        print("------------------------------------------------")
