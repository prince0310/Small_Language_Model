# transformer-from-scratch

A small transformer language model built from scratch in **PyTorch**, following
the [google-deepmind/ai-foundations](https://github.com/google-deepmind/ai-foundations) curriculum.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

Edit the settings in `configs/shakespeare.toml` (or copy it to make a new config), then run:

```bash
python -m ai_foundation.train --config configs/shakespeare.toml
```

It prints the loss and a text sample after each epoch, and saves `model.pt`,
`tokenizer.pkl` and `history.json` to the `[output] dir` folder.

## Roadmap

- [x] 0. Project setup
- [ ] 1. N-gram language model
- [ ] 2. Word tokenizer
- [ ] 3. BPE tokenizer
- [ ] 4. Embeddings
- [ ] 5. MLP and training loop
- [ ] 6. Causal multi-head attention
- [ ] 7. Token and positional embeddings
- [ ] 8. Transformer block
- [ ] 9. Model, padding-aware loss, training
- [ ] 10. Text generation
