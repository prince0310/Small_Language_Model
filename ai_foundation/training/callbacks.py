"""Callbacks that run at the end of each training epoch.

PyTorch version of DeepMind's Keras callbacks. PyTorch has no callback
system, so `Callback` below is a minimal base class that `trainer.fit` calls.
"""

from typing import Any

import torch
from torch import nn


class Callback:
    """Base class: override `on_epoch_end` to run code after each epoch."""

    def on_epoch_end(
        self, epoch: int, logs: dict[str, float], model: nn.Module
    ) -> None:
        pass


class TextGenerator(Callback):
    """Generates sample text from the model at the end of an epoch.

    1. Feed an initial prompt to the model.
    2. Predict probabilities for the next token.
    3. Sample the next token and add it to the input for the next prediction.

    Args:
        max_tokens: Number of tokens to generate.
        start_tokens: Token IDs for the initial prompt.
        tokenizer: Tokenizer with a `decode` method.
        print_every: Print the generated text every `print_every` epochs.
    """

    def __init__(
        self,
        max_tokens: int,
        start_tokens: list[int],
        tokenizer: Any,
        print_every: int = 1,
    ):
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.tokenizer = tokenizer
        self.print_every = print_every

    @torch.no_grad()
    def on_epoch_end(
        self, epoch: int, logs: dict[str, float], model: nn.Module
    ) -> None:
        if (epoch + 1) % self.print_every != 0:
            return

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()  # Turn off dropout while generating.

        tokens = list(self.start_tokens)
        for _ in range(self.max_tokens):
            # The model accepts any length up to max_length, so no padding is
            # needed. Keep only the most recent max_length tokens.
            x = torch.tensor([tokens[-model.max_length:]], device=device)
            logits = model(x)  # (1, seq, vocab)

            # Probabilities for the token after the last position.
            probabilities = torch.softmax(logits[0, -1], dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).item()
            tokens.append(next_token)

        if was_training:
            model.train()

        print("Generated text:\n", self.tokenizer.decode(tokens), "\n")


class CustomAccuracyPrinter(Callback):
    """Prints training progress every `print_every` epochs.

    Args:
        print_every: Print the training progress every `print_every` epochs.
    """

    def __init__(self, print_every: int = 1):
        self.print_every = print_every

    def on_epoch_end(
        self, epoch: int, logs: dict[str, float], model: nn.Module
    ) -> None:
        if (epoch + 1) % self.print_every != 0:
            return

        log_parts = [f"Epoch {epoch}: Training loss: {logs['loss']:.5f}"]
        if "accuracy" in logs:
            log_parts.append(f"training accuracy: {logs['accuracy'] * 100:.2f}%")
        if "val_loss" in logs:
            log_parts.append(f"validation loss: {logs['val_loss']:.5f}")
        if "val_accuracy" in logs:
            log_parts.append(f"validation accuracy: {logs['val_accuracy'] * 100:.2f}%")
        print(", ".join(log_parts))
