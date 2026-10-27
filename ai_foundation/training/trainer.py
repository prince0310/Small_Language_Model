"""A training loop: the PyTorch replacement for Keras `model.fit()`.

DeepMind's code has no file like this because Keras provides `fit`. In
PyTorch you write the loop yourself, so every step of training is visible:
forward pass -> loss -> backward pass -> optimizer step.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ai_foundation.training.callbacks import Callback


def get_device() -> torch.device:
    """Picks CUDA, Apple Silicon (MPS) or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 1,
    batch_size: int = 32,
    validation_data: tuple[torch.Tensor, torch.Tensor] | None = None,
    callbacks: list[Callback] | None = None,
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """Trains the model, like Keras `model.fit()`.

    Args:
        model: The model to train.
        optimizer: Optimizer created for this model's parameters.
        loss_fn: Loss taking (logits, targets).
        x: Input token IDs, shape (num_examples, sequence_length).
        y: Target token IDs (inputs shifted by one), same shape as x.
        epochs: Number of passes over the training data.
        batch_size: Number of examples per optimizer step.
        validation_data: Optional (x_val, y_val) evaluated after each epoch.
        callbacks: Objects whose `on_epoch_end` runs after each epoch.
        device: Where to train. Defaults to the best available device.

    Returns:
        History dict with a list of per-epoch values for 'loss' and, if
        validation data is given, 'val_loss'.
    """
    device = device or get_device()
    model.to(device)
    callbacks = callbacks or []
    history: dict[str, list[float]] = {"loss": []}

    train_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()  # Turn dropout on.
        total_loss, num_batches = 0.0, 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits = model(batch_x)            # 1. forward pass
            loss = loss_fn(logits, batch_y)    # 2. how wrong were we?

            optimizer.zero_grad()              # 3. clear old gradients
            loss.backward()                    # 4. compute new gradients
            optimizer.step()                   # 5. update the weights

            total_loss += loss.item()
            num_batches += 1

        logs = {"loss": total_loss / num_batches}
        if validation_data is not None:
            logs["val_loss"] = evaluate(model, loss_fn, *validation_data, batch_size, device)

        for name, value in logs.items():
            history.setdefault(name, []).append(value)
        for callback in callbacks:
            callback.on_epoch_end(epoch, logs, model)

    return history


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loss_fn: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> float:
    """Returns the mean loss on (x, y) without updating the model."""
    device = device or get_device()
    model.eval()  # Turn dropout off.
    total_loss, num_batches = 0.0, 0
    for batch_x, batch_y in DataLoader(TensorDataset(x, y), batch_size=batch_size):
        logits = model(batch_x.to(device))
        total_loss += loss_fn(logits, batch_y.to(device)).item()
        num_batches += 1
    return total_loss / num_batches
