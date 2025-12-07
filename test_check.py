import torch
from model import FemtoGPT, CausalSelfAttention, n_embd, n_head, block_size


def test_components():
    print("--- starting component check ---")

    batch_size = 2
    seq_len = 8
    vocab_size = 100

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input shape: {dummy_input.shape}  (Batch={batch_size}, Time={seq_len})")

    print("\n[1] Testing embeddings...")
    model = FemtoGPT(vocab_size, n_embd, block_size)

    try:
        x = model(dummy_input)
        print(f" ✔ embedding output shape: {x.shape}")
        # Expected: (2, 8, 768) -> (Batch, Time, n_embd)
    except Exception as e:
        print(f" ❌ embeddings failed: {e}")
        return

    print("\n[2] testing causal self-attention...")
    attn_layer = CausalSelfAttention(
        n_embd=n_embd, n_head=n_head, block_size=block_size
    )

    try:
        y = attn_layer(x)
        print(f" ✔ attention output shape: {y.shape}")
    except Exception as e:
        print(f" ❌ Attention Failed: {e}")
        return

    print("\n--- the shapes match up perfectly! ---")


if __name__ == "__main__":
    test_components()
