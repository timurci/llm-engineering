"""Embedding model for simple bigram next word prediction."""

from typing import Protocol, Self

import torch
from torch import nn


class EmbeddingModel(Protocol):
    """Embedding model protocol."""

    @property
    def encoder(self) -> nn.Module:
        """Encoder module."""
        ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the embedding model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size,).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, vocab_size).
        """
        ...

    def train(self, mode: bool = True) -> Self:  # noqa: FBT001, FBT002
        """Set model to training mode."""
        ...

    def eval(self) -> Self:
        """Set model to evaluation mode."""
        ...


class BigramEmbeddingModel(nn.Module):
    """Embedding model for simple bigram next word prediction."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        sparse: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the embedding model.

        Args:
            vocab_size: Size of the vocabulary.
            embedding_dim: Dimension of the embedding.
            sparse: Whether to use sparse embeddings (default: True).
        """
        super().__init__()
        self.encoder = nn.Embedding(vocab_size, embedding_dim, sparse=sparse)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the embedding model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size,).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, vocab_size).
        """
        return self.output(self.encoder(x))
