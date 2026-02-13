"""Dataset for retrieving bigram tokens."""

from typing import TYPE_CHECKING, NamedTuple

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Sequence


class TokenIndexBigram(NamedTuple):
    """Bigram representation of token indices."""

    left: torch.Tensor
    right: torch.Tensor


class BigramEmbeddingDataset(Dataset[TokenIndexBigram]):
    """Dataset for retrieving adjacent token bigrams from text chunks."""

    def __init__(self, chunks: Sequence[Sequence[int]]) -> None:
        """Initialize the dataset with precomputed token chunks.

        Args:
            chunks: Sequence of token index sequences (chunks).
                Empty or single-token chunks are ignored.
        """
        bigrams = [
            (chunk[i], chunk[i + 1]) for chunk in chunks for i in range(len(chunk) - 1)
        ]
        self._bigrams = torch.tensor(bigrams, dtype=torch.int)

    def __len__(self) -> int:
        """Return the total number of bigrams."""
        return len(self._bigrams)

    def __getitem__(self, index: int) -> TokenIndexBigram:
        """Return the bigram at the given index.

        Args:
            index: Index of the bigram to retrieve.

        Returns:
            TokenIndexBigram with left and right token tensors.
        """
        return TokenIndexBigram(
            self._bigrams[index, 0],
            self._bigrams[index, 1],
        )
