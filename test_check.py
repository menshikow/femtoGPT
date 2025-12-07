import torch
from model import FemtoGPT


def test_full_model():
    print("--- starting full model check ---")

    # 1. setup configuration (standard gpt-2 small params)
    vocab_size = 50257
    block_size = 1024
    n_embd = 768
    n_head = 12
    n_layer = 12
    dropout = 0.1

    # 2. setup dummy data
    batch_size = 2
    seq_len = 32

    # create fake inputs (random tokens) and fake targets
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(
        f"config: layers={n_layer}, heads={n_head}, embed={n_embd}, vocab={vocab_size}"
    )

    # 3. initialize the full model
    try:
        model = FemtoGPT(vocab_size, n_embd, block_size, n_head, n_layer, dropout)
        print("✔ model initialized successfully")
    except Exception as e:
        print(f"❌ init failed: {e}")
        return

    # 4. calculate parameter count
    # we sum up the number of elements in every weight matrix
    n_params = sum(p.numel() for p in model.parameters())
    # subtract position embeddings to get the "non-embedding" count (often reported in papers)
    n_params_non_emb = n_params - (block_size * n_embd) - (vocab_size * n_embd)

    print(f"✔ total parameters: {n_params/1e6:.2f} million")

    # 5. run forward pass (with targets to get loss)
    try:
        logits, loss = model(dummy_input, dummy_targets)
        print(f"✔ forward pass successful")
        print(
            f"   logits shape: {logits.shape} (expected: {batch_size}, {seq_len}, {vocab_size})"
        )
        print(f"   loss value:   {loss.item():.4f}")
    except Exception as e:
        print(f"❌ forward pass failed: {e}")
        return

    print("\n--- system all green! ready for training ---")


if __name__ == "__main__":
    test_full_model()
