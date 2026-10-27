"""A loss function for handling padded sequence data.

PyTorch version of DeepMind's `CustomMaskPadLoss`: cross-entropy over the
vocabulary that ignores padding tokens.
"""

import torch
from torch import nn
import torch.nn.functional as F


class CustomMaskPadLoss(nn.Module):
    """Cross-entropy loss that skips padding tokens.

    Padding positions are left out of both the sum and the average, so the
    loss is the mean over real tokens only.

    Args:
        pad_token_id: ID of the padding token to ignore. If None, every
            position counts.

    Forward args:
        logits: Raw model outputs, shape (batch_size, sequence_length,
            vocabulary_size).
        targets: True next-token IDs, shape (batch_size, sequence_length).

    Returns:
        Scalar tensor with the mean loss over non-padding tokens.
    """

    def __init__(self, pad_token_id: int | None = None):
        super().__init__()
        self.pad_token_id = pad_token_id

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # F.cross_entropy wants (N, vocab) and (N,), so flatten batch and time.
        vocabulary_size = logits.shape[-1]
        return F.cross_entropy(
            logits.reshape(-1, vocabulary_size),
            targets.reshape(-1),
            # -100 is PyTorch's default "ignore nothing real" value.
            ignore_index=self.pad_token_id if self.pad_token_id is not None else -100,
        )
