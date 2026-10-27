"""Core architectural layers for a custom transformer model.

This module contains the custom layers that form the building blocks of a
decoder-only transformer model, including embedding, self-attention, and
feed-forward network layers.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

# Standard choice from "Attention Is All You Need".
ANGLE_RATE_MULTIPLIER = 10000
ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}

class TokenAndPositionEmbedding(nn.Module):
    """Combines token embeddings with positional embeddings.

    Attributes:
        max_length: Maximum sequence length; number of position vectors.
        vocabulary_size: Number of tokens; rows in the token embedding table.
        embedding_dim: Size of each embedding vector (d_model).
        positional_embedding_type: 'simple' (learned) or 'sinusoidal' (fixed).

    Forward args:
        x: LongTensor of token IDs, shape (batch_size, sequence_length).

    Returns:
        Tensor of shape (batch_size, sequence_length, embedding_dim).
    """

    def __init__(
      self,
      max_length: int,
      vocabulary_size: int,
      embedding_dim: int,
      positional_embedding_type: str = "sinusoidal",
  ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.positional_embedding_type = positional_embedding_type

        # Row 0 (<PAD>) stays zero and is never trained
        self.token_embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            padding_idx=0,)


        if self.positional_embedding_type == "simple":
            self.pos_emb = nn.Embedding(
                num_embeddings=max_length, embedding_dim=embedding_dim
            )

        elif self.positional_embedding_type == "sinusoidal":
            self.register_buffer(
                "pos_encoding",
                self.positional_encoding(length=max_length, depth=embedding_dim),
            )

        else:
            raise NotImplementedError(
                "Positional embedding type"
                f" {self.positional_embedding_type}"
                " not implemented."
            )


    def positional_encoding(self, length: int, depth: int) -> torch.Tensor:

        """Creates a fixed sinusoidal positional encoding table.

        Args:
            length: Number of positions (rows).
            depth: Size of each position vector (columns). Must be even.

        Returns:
            Tensor of shape (length, depth).
        """
        if depth % 2 != 0:
            raise ValueError(f"depth must be even, got {depth}")

        depth = depth // 2

        positions = torch.arange(length, dtype=torch.float32)[:, None]      # (seq, 1)
        depths = torch.arange(depth, dtype=torch.float32)[None, :] / depth  # (1, depth)

        angle_rates = 1 / (ANGLE_RATE_MULTIPLIER**depths)                   # (1, depth)
        angle_rads = positions * angle_rates                                # (seq, depth)

        pos_encoding = torch.cat(
            [torch.sin(angle_rads), torch.cos(angle_rads)], dim=-1
        )                                                                   # (seq, 2*depth)

        return pos_encoding


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Combines token embeddings with positional embeddings.

        Args:
            x: LongTensor of token IDs, shape (batch_size, sequence_length).

        Returns:
            Tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        seq_len = x.shape[1]
        if seq_len > self.max_length:
            raise ValueError(f"sequence length {seq_len} > max_length {self.max_length}")

        token_embeddings = self.token_embedding(x)                     # (batch, seq, d)

        if self.positional_embedding_type == "sinusoidal":
            token_embeddings = token_embeddings * math.sqrt(self.embedding_dim)
            position_embeddings = self.pos_encoding[:seq_len]          # (seq, d)
        else:
            positions = torch.arange(seq_len, device=x.device)         # [0, 1, ..., seq-1]
            position_embeddings = self.pos_emb(positions)              # (seq, d)

        return token_embeddings + position_embeddings                  # (batch, seq, d)

class TransformerBlock(nn.Module):
    """One transformer block: self-attention, then a feed-forward network.

    Each sub-layer applies dropout, a residual connection and layer norm.

    Args:
        embedding_dim: Size of each token vector (d_model).
        num_heads: Number of attention heads.
        mlp_dim: Hidden size of the feed-forward network.
        dropout_rate: Dropout probability, between 0 and 1.
        activation_function: Activation in the feed-forward network.

    Forward args:
        x: Tensor of shape (batch_size, sequence_length, embedding_dim).

    Returns:
        Tensor of the same shape.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout_rate: float = 0.0,
        activation_function: str = "relu",
    ):
        super().__init__()

        self.self_attention = MultiHeadSelfAttention(
            embedding_dim, num_heads, dropout_rate
        )
        self.feed_forward = FeedForwardNetwork(
            embedding_dim, mlp_dim, dropout_rate, activation_function
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.self_attention(x)
        x = self.feed_forward(x)
        return x



class FeedForwardNetwork(nn.Module):
    """Two-layer feed-forward network with residual connection and layer norm.

    Args:
        embedding_dim: Size of each token vector.
        mlp_dim: Hidden size (usually 4 * embedding_dim).
        dropout_rate: Dropout probability on the FFN output.
        activation: Name of the activation in the hidden layer.

    Forward args:
        x: Tensor of shape (batch_size, sequence_length, embedding_dim).

    Returns:
        Tensor of the same shape.
    """

    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        dropout_rate: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}")

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),      # expand
            ACTIVATIONS[activation](),
            nn.Linear(mlp_dim, embedding_dim),      # project back
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        # Residual connection, then layer norm (post-norm).
        return self.layernorm(x + ffn_output)


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with residual connection and layer norm.

    Args:
        embedding_dim: Size of each token vector (d_model).
        num_heads: Number of attention heads; must divide embedding_dim.
        dropout_rate: Dropout probability on the attention output.

    Forward args:
        x: Tensor of shape (batch_size, sequence_length, embedding_dim).

    Returns:
        Tensor of the same shape.
    """

    def __init__(self, embedding_dim: int, num_heads: int, dropout_rate: float = 0.0):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim {embedding_dim} not divisible by num_heads {num_heads}")

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)   # mixes heads back together

        self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(embedding_dim)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        # (batch, seq, d) -> (batch, heads, seq, head_dim)
        batch, seq, _ = t.shape
        return t.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, d = x.shape

        # 1. Project to queries, keys and values, then split into heads.
        q = self._split_heads(self.query(x))             # (b, h, s, hd)
        k = self._split_heads(self.key(x))
        v = self._split_heads(self.value(x))

        # 2. How well each query matches each key, scaled.
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)   # (b, h, s, s)

        # 3. Causal mask: hide future positions.
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

        # 4. Scores -> percentages, then mix the values.
        weights = torch.softmax(scores, dim=-1)          # (b, h, s, s), rows sum to 1
        attn = weights @ v                               # (b, h, s, hd)

        # 5. Glue the heads back together and project.
        attn = attn.transpose(1, 2).reshape(batch, seq, d)   # (b, s, d)
        attn_output = self.out_proj(attn)

        # 6. Dropout, residual, layer norm (post-norm, as in DeepMind's code).
        attn_output = self.dropout(attn_output)
        return self.layernorm(x + attn_output)





# if __name__ == "__main__":
#     emb = TokenAndPositionEmbedding(max_length=3, vocabulary_size=4, embedding_dim=4)
#     print(emb.pos_encoding)
#     print(emb.pos_encoding.shape)




# emb = TokenAndPositionEmbedding(max_length=64, vocabulary_size=79, embedding_dim=32)
# print(emb.token_embedding.weight.shape)   # torch.Size([79, 32])
# print(emb.token_embedding.weight[0])      # all zeros