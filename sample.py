import torch
import torch.nn.functional as F
import tiktoken
from model import FemtoGPT
import config

# configuration
device = config.detect_divce()

# load the tokenizer
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# load the Model
print(f"Loading model from {config.out_dir}...")
model = FemtoGPT(
    vocab_size=config.vocab_size,
    n_embd=config.n_embd,
    block_size=config.block_size,
    n_head=config.n_head,
    n_layer=config.n_layer,
    dropout=config.dropout,
)

ckpt_path = f"{config.out_dir}/ckpt.pt"
try:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    print("Weights loaded successfully ✔")
except FileNotFoundError:
    print(f"could not find {ckpt_path}")
    exit()

model.to(device)
model.eval()


# generation Function
def generate(start_text, max_new_tokens=200, temperature=0.8, top_k=200):
    # encode the prompt
    start_ids = encode(start_text)
    idx = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    for _ in range(max_new_tokens):
        # crop context
        idx_cond = idx[:, -config.block_size :]
        # predict
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return decode(idx[0].tolist())


# interactive Loop
print("\n--- Interactive Mode ---")
print("type a prompt and press enter, type 'exit' to quit\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    if user_input.strip() == "":
        continue

    print("FemtoGPT: ", end="", flush=True)

    # generate and print
    response = generate(user_input, max_new_tokens=150, temperature=0.8)

    # remove the prompt from the echo
    # the simple generate() returns (prompt + new), so we slice it for display
    print(response[len(user_input) :])
    print("-" * 50)
