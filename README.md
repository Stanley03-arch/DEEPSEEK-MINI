# 🧠 Mini DeepSeek

A small educational project that combines:

1. **A tiny character-level transformer** you can train from scratch on a laptop (`mini_deepseek.py`)
2. **A Streamlit chat UI** that talks to a real local DeepSeek model via Ollama (`app.py`)
3. **A polished static landing page** with a demo chat (`index.html`)

Perfect for learning how transformers work and for showcasing a local AI assistant.

---

## Project Structure

```
DEEPSEEK-MINI/
├── mini_deepseek.py      # Train & generate with the tiny transformer
├── mini_deepseek.pt      # Saved weights (created after training)
├── app.py                # Streamlit chat → Ollama DeepSeek
├── index.html            # Beautiful static landing + demo chat
├── requirements.txt
└── README.md
```

---

## 1. Train the Mini Transformer

```bash
pip install -r requirements.txt
python mini_deepseek.py
```

What it does:
- Builds a 2-layer transformer (~ few thousand parameters)
- Trains on a short AI-themed text corpus (character level)
- Saves `mini_deepseek.pt` with weights + vocabulary + config
- Prints a sample generation at the end

You can later load the model and generate more text:

```python
from mini_deepseek import load_model, generate_text

model = load_model()
print(generate_text(model, prompt="Artificial intelligence", max_new_tokens=150))
```

---

## 2. Run the Streamlit Chat (Real DeepSeek via Ollama)

First install and start [Ollama](https://ollama.com), then pull a small DeepSeek model:

```bash
ollama pull deepseek-r1:1.5b
# or: ollama pull deepseek-r1:7b   (if you have more RAM/GPU)
```

Then launch the app:

```bash
streamlit run app.py
```

The app talks to `http://localhost:11434` (default Ollama endpoint).

---

## 3. Open the Static Landing Page

Simply open `index.html` in a browser.

The embedded chat also tries to reach your local Ollama instance.  
If Ollama is not running it shows a friendly fallback message.

---

## Features of the Tiny Model

| Feature              | Details                          |
|----------------------|----------------------------------|
| Architecture         | 2-layer Transformer              |
| Attention            | Multi-head causal self-attention |
| Feed-forward         | GELU + residual + LayerNorm      |
| Tokenizer            | Character-level                  |
| Context length       | 48 characters                    |
| Trainable on         | CPU or CUDA                      |

This is intentionally tiny so you can understand every line and experiment freely.

---

## Ideas for Next Improvements

- [ ] Load the trained `.pt` model inside Streamlit as a fallback when Ollama is offline
- [ ] Add streaming responses in the Streamlit app
- [ ] Expand the training corpus or switch to sub-word tokenization
- [ ] Add a simple web UI that serves the mini model with FastAPI + Gradio
- [ ] Deploy the static page + a small backend

---

Made for learning and experimentation. Have fun building!
