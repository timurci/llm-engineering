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
    """Dataset for retrieving adjacent token bigrams."""

    def __init__(self, tokens: Sequence[int]) -> None:
        """Initialize the dataset with a sequence of tokens.

        Args:
            tokens: Sequence of token indices.
        """
        self.tokens = tokens

    def __len__(self) -> int:
        """Return the number of bigram tokens in the dataset."""
        length = len(self.tokens) - 1
        return max(0, length)

    def __getitem__(self, index: int) -> TokenIndexBigram:
        """Return the token at the given index and the next token.

        Args:
            index: Index of the bigram to retrieve.

        Returns:
            Token indices as a named tuple.
        """
        left = torch.tensor(self.tokens[index], dtype=torch.int)
        right = torch.tensor(self.tokens[index + 1], dtype=torch.int)
        return TokenIndexBigram(left=left, right=right)
