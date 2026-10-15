
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
device = "cpu"  # Safe starting point for your laptop

text = """
DeepSeek is an artificial intelligence model.
Artificial intelligence learns patterns from data.
Python helps developers build machine learning systems.
A transformer uses attention to process information.
Machine learning models learn by predicting the next token.
I am building my own mini AI model.
"""

# Character tokenizer
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.t
ensor(encode(text), dtype=torch.long)

block_size = 32
vocab_size = len(chars)
embedding_size = 32

class MiniDeepSeek(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(block_size, embedding_size)
        self.attention = nn.MultiheadAttention(
            embedding_size, num_heads=4, batch_first=True
        )
        self.norm = nn.LayerNorm(embedding_size)
        self.output = nn.Linear(embedding_size, vocab_size)

    def forward(self, x, targets=None):
        _, length = x.shape
        positions = torch.arange(length, device=x.device)
        h = self.token_embedding(x) + self.position_embedding(positions)

        mask = torch.triu(
            torch.ones(length, length, device=x.device),
            diagonal=1
        ).bool()

        attended, _ = self.attention(
            h, h, h, attn_mask=mask, need_weights=False
        )
        h = self.norm(h + attended)
        logits = self.output(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, x, count=150):
        self.eval()
        for _ in range(count):
            x_cond = x[:, -block_size:]
            logits, _ = self(x_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_char = torch.multinomial(probs, 1)
            x = torch.cat((x, next_char), dim=1)
        return x

model = MiniDeepSeek().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

print("Training Mini DeepSeek...")

for step in range(300):
    ix = torch.randint(0, len(data) - block_size - 1, (16,))
    x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)

    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")

torch.save({
    "model_state": model.state_dict(),
    "chars": chars,
    "embedding_size": embedding_size,
    "block_size": block_size
}, "mini_deepseek.pt")

print("\nTraining finished. Model saved!")

prompt = "DeepSeek"
prompt_ids = encode(prompt)
if not prompt_ids:
    prompt_ids = [0]

x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
result = model.generate(x, 150)

print("\nAI output:")
print(decode(result[0].tolist()))
