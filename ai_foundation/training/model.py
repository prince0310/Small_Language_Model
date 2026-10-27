"""A function to build a transformer language model.

PyTorch version of DeepMind's `create_model`. Keras bundles the model, the
optimizer and the loss together with `model.compile()`. PyTorch keeps them
separate, so `create_model` returns all three.
"""

from typing import Literal

import torch
from torch import nn

from ai_foundation.training.losses import CustomMaskPadLoss
from ai_foundation.transformer.layers import TokenAndPositionEmbedding
from ai_foundation.transformer.layers import TransformerBlock


class TransformerLanguageModel(nn.Module):
    """Decoder-only transformer that predicts the next token.

    Embedding -> N transformer blocks -> linear layer with one score (logit)
    per vocabulary token.

    Args:
        vocabulary_size: Number of unique tokens.
        max_length: Maximum input sequence length.
        embedding_dim: Size of each token vector.
        mlp_dim: Hidden size of each feed-forward network.
        num_heads: Number of attention heads.
        num_blocks: Number of transformer blocks to stack.
        dropout_rate: Dropout probability.
        activation_function: Activation in the feed-forward networks.

    Forward args:
        x: LongTensor of token IDs, shape (batch_size, sequence_length).

    Returns:
        Logits of shape (batch_size, sequence_length, vocabulary_size).
    """

    def __init__(
        self,
        vocabulary_size: int,
        max_length: int,
        embedding_dim: int = 256,
        mlp_dim: int = 256,
        num_heads: int = 2,
        num_blocks: int = 1,
        dropout_rate: float = 0.0,
        activation_function: str = "relu",
    ):
        super().__init__()
        self.max_length = max_length

        # Embedding layer that combines token and positional embeddings.
        self.embedding = TokenAndPositionEmbedding(
            max_length, vocabulary_size, embedding_dim
        )

        # A stack of transformer blocks.
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim,
                    num_heads,
                    mlp_dim,
                    dropout_rate=dropout_rate,
                    activation_function=activation_function,
                )
                for _ in range(num_blocks)
            ]
        )

        # One raw score (logit) per vocabulary token, for every position.
        self.output_layer = nn.Linear(embedding_dim, vocabulary_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)


def create_model(
    vocabulary_size: int,
    max_length: int,
    embedding_dim: int = 256,
    mlp_dim: int = 256,
    num_heads: int = 2,
    num_blocks: int = 1,
    optimizer: Literal["adamw", "sgd"] = "adamw",
    learning_rate: float = 5e-4,
    dropout_rate: float = 0.0,
    activation_function: str = "relu",
    pad_token_id: int = 0,
) -> tuple[TransformerLanguageModel, torch.optim.Optimizer, CustomMaskPadLoss]:
    """Creates a transformer language model with its optimizer and loss.

    Example:
        model, optimizer, loss_fn = create_model(
            vocabulary_size=5000, max_length=100, embedding_dim=256,
            mlp_dim=512, num_heads=8, num_blocks=2)
        print(model)

    Args:
        vocabulary_size: Number of unique tokens.
        max_length: Maximum input sequence length.
        embedding_dim: Size of each token vector.
        mlp_dim: Hidden size of each feed-forward network.
        num_heads: Number of attention heads.
        num_blocks: Number of transformer blocks to stack.
        optimizer: 'adamw' (Adam with weight decay) or 'sgd'.
        learning_rate: Learning rate for the optimizer.
        dropout_rate: Dropout probability.
        activation_function: Activation in the feed-forward networks.
        pad_token_id: ID of the padding token, ignored by the loss.

    Returns:
        (model, optimizer, loss_fn).

    Raises:
        NotImplementedError: If an unsupported optimizer is specified.
    """
    model = TransformerLanguageModel(
        vocabulary_size,
        max_length,
        embedding_dim=embedding_dim,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
        activation_function=activation_function,
    )
    optimizer_instance = get_optimizer(model, optimizer, learning_rate)
    loss_fn = CustomMaskPadLoss(pad_token_id=pad_token_id)
    return model, optimizer_instance, loss_fn


def get_optimizer(
    model: nn.Module,
    optimizer_name: Literal["adamw", "sgd"],
    learning_rate: float,
) -> torch.optim.Optimizer:
    """Returns the optimizer for the model's parameters.

    Unlike Keras, a PyTorch optimizer must be told which parameters to update,
    so it takes the model as well.

    Args:
        model: The model whose parameters will be trained.
        optimizer_name: The name of the optimizer.
        learning_rate: The learning rate for the optimizer.

    Returns:
        The optimizer instance.

    Raises:
        NotImplementedError: If an unsupported optimizer is specified.
    """
    if optimizer_name.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.005
        )
    else:
        raise NotImplementedError(f"Optimizer {optimizer_name} is not implemented.")
