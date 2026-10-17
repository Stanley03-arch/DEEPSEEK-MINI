"""
Mini DeepSeek – a tiny character-level transformer you can train on a laptop.
Educational demo inspired by the DeepSeek models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -------------------------------------------------
# Training text (small but richer than the original)
# -------------------------------------------------
text = """
DeepSeek is an artificial intelligence company that builds powerful open language models.
Artificial intelligence learns patterns from large amounts of text data.
Python helps developers build machine learning systems quickly and clearly.
A transformer uses self-attention to process sequences of information.
Machine learning models learn by predicting the next token in a sequence.
I am building my own mini AI model called Mini DeepSeek.
Attention is all you need – this idea changed modern deep learning.
Large language models can reason, write code, and answer questions.
Training a neural network means adjusting weights to reduce prediction error.
Embeddings turn discrete tokens into continuous vectors the model can use.
Residual connections and layer normalization help deep networks train stably.
Mini DeepSeek demonstrates how a simple transformer works from scratch.
Open source AI research helps everyone learn and innovate together.
"""

# Character-level tokenizer
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda ids: "".join(itos[i] for i in ids if i in itos)

data = torch.tensor(encode(text), dtype=torch.long)
print(f"Corpus length: {len(data)} characters | Vocab size: {len(chars)}")

# Hyperparameters
block_size = 48
vocab_size = len(chars)
n_embd = 64
n_head = 4
n_layer = 2
dropout = 0.1
batch_size = 32
max_iters = 800
learning_rate = 3e-3
eval_interval = 100


class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # Scaled attention scores
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
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
    """Transformer block: communication then computation."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniDeepSeek(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Better initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def get_batch():
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def train():
    model = MiniDeepSeek().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e3:.1f}k")
    print("Training Mini DeepSeek...\n")

    for step in range(max_iters):
        xb, yb = get_batch()
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0 or step == max_iters - 1:
            print(f"Step {step:4d} | loss = {loss.item():.4f}")

    # Save everything needed for later inference
    torch.save(
        {
            "model_state": model.state_dict(),
            "chars": chars,
            "stoi": stoi,
            "itos": itos,
            "config": {
                "block_size": block_size,
                "vocab_size": vocab_size,
                "n_embd": n_embd,
                "n_head": n_head,
                "n_layer": n_layer,
            },
        },
        "mini_deepseek.pt",
    )
    print("\nTraining finished. Model saved to mini_deepseek.pt")
    return model


def load_model(path="mini_deepseek.pt"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    # Rebuild model with saved config
    global block_size, vocab_size, n_embd, n_head, n_layer, chars, stoi, itos
    cfg = checkpoint["config"]
    block_size = cfg["block_size"]
    vocab_size = cfg["vocab_size"]
    n_embd = cfg["n_embd"]
    n_head = cfg["n_head"]
    n_layer = cfg["n_layer"]
    chars = checkpoint["chars"]
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    model = MiniDeepSeek().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def generate_text(model, prompt="DeepSeek", max_new_tokens=180, temperature=0.8):
    prompt_ids = encode(prompt)
    if not prompt_ids:
        prompt_ids = [0]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
    return decode(out[0].tolist())


if __name__ == "__main__":
    model = train()

    print("\n--- Sample generation ---")
    sample = generate_text(model, prompt="DeepSeek", max_new_tokens=200)
    print(sample)
