"""Train a transformer language model from a config file.

Usage (from the project folder):
    python -m ai_foundation.train --config configs/shakespeare.toml

Pipeline: read text -> train tokenizer -> encode -> cut into (input, target)
chunks -> build model -> fit -> save checkpoint.
"""

import argparse
import json
import pickle
import time
import tomllib
from pathlib import Path

import torch

from ai_foundation.tokeniztion import BPEWordTokenizer
from ai_foundation.training import CustomAccuracyPrinter
from ai_foundation.training import TextGenerator
from ai_foundation.training import create_model
from ai_foundation.training import fit
from ai_foundation.training.trainer import get_device


class CharTokenizer:
    """Character-level tokenizer: one token per character, <PAD> at ID 0."""

    PAD_TOKEN = "<PAD>"

    def __init__(self, text: str):
        self.vocabulary = [self.PAD_TOKEN] + sorted(set(text))
        self.vocabulary_size = len(self.vocabulary)
        self.token_to_index = {tok: i for i, tok in enumerate(self.vocabulary)}
        self.pad_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [self.token_to_index[c] for c in text if c in self.token_to_index]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.vocabulary[i] for i in token_ids if i != self.pad_token_id)


def build_tokenizer(cfg: dict, train_text: str):
    if cfg["type"] == "char":
        return CharTokenizer(train_text)
    if cfg["type"] == "bpe":
        return BPEWordTokenizer(train_text, num_merges=cfg["num_merges"])
    raise ValueError(f"unknown tokenizer type {cfg['type']!r}")


def encode(tokenizer, text: str) -> list[int]:
    """Encodes text. For BPE, each unique word is split only once (cached),
    because applying every merge to every word of a large text is slow."""
    if not isinstance(tokenizer, BPEWordTokenizer):
        return tokenizer.encode(text)

    cache: dict[str, list[int]] = {}
    ids: list[int] = []
    for word in text.split():
        if word not in cache:
            cache[word] = tokenizer.encode(word)
        ids.extend(cache[word])
    return ids


def make_chunks(ids: list[int], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Cuts one long ID list into (input, target) pairs; target = input shifted by one."""
    ids_tensor = torch.tensor(ids, dtype=torch.long)
    num_chunks = (len(ids_tensor) - 1) // max_length
    if num_chunks == 0:
        raise ValueError(f"not enough tokens ({len(ids_tensor)}) for max_length {max_length}")
    usable = num_chunks * max_length
    x = ids_tensor[:usable].view(num_chunks, max_length)
    y = ids_tensor[1 : usable + 1].view(num_chunks, max_length)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a transformer language model.")
    parser.add_argument("--config", required=True, help="Path to a .toml config file.")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)
    data_cfg, tok_cfg, model_cfg = config["data"], config["tokenizer"], config["model"]
    train_cfg, gen_cfg, out_cfg = config["training"], config["generation"], config["output"]

    torch.manual_seed(train_cfg["seed"])
    device = get_device() if train_cfg["device"] == "auto" else torch.device(train_cfg["device"])

    # 1. Data: read and split. The tokenizer only ever learns from the training part.
    text = Path(data_cfg["path"]).read_text(encoding="utf-8")
    if data_cfg["max_chars"]:
        text = text[: data_cfg["max_chars"]]
    split = int(len(text) * (1 - data_cfg["validation_split"]))
    train_text, val_text = text[:split], text[split:]
    print(f"Data: {len(train_text):,} train chars, {len(val_text):,} validation chars")

    # 2. Tokenizer.
    start = time.time()
    tokenizer = build_tokenizer(tok_cfg, train_text)
    print(f"Tokenizer: {tok_cfg['type']}, vocabulary size {tokenizer.vocabulary_size:,} "
          f"({time.time() - start:.1f}s)")

    # 3. Encode and cut into chunks.
    max_length = model_cfg["max_length"]
    x_train, y_train = make_chunks(encode(tokenizer, train_text), max_length)
    x_val, y_val = make_chunks(encode(tokenizer, val_text), max_length)
    print(f"Chunks: {len(x_train):,} train, {len(x_val):,} validation (length {max_length})")

    # 4. Model.
    model, optimizer, loss_fn = create_model(
        vocabulary_size=tokenizer.vocabulary_size,
        optimizer=train_cfg["optimizer"],
        learning_rate=train_cfg["learning_rate"],
        pad_token_id=tokenizer.pad_token_id,
        **model_cfg,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params:,} parameters, training on {device}\n")

    # 5. Train.
    callbacks = [
        CustomAccuracyPrinter(),
        TextGenerator(
            max_tokens=gen_cfg["max_tokens"],
            start_tokens=encode(tokenizer, gen_cfg["prompt"]),
            tokenizer=tokenizer,
            print_every=gen_cfg["print_every"],
        ),
    ]
    start = time.time()
    history = fit(
        model, optimizer, loss_fn, x_train, y_train,
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        device=device,
    )
    print(f"Training took {time.time() - start:.1f}s")

    # 6. Save everything needed to reload the model later.
    out_dir = Path(out_cfg["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config}, out_dir / "model.pt")
    with open(out_dir / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved model, tokenizer and history to {out_dir}/")


if __name__ == "__main__":
    main()
