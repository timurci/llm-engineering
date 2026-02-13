"""Tests for BigramEmbeddingDataset."""

import pytest
import torch
from torch.utils.data import DataLoader

from llm_engineering.embedding.dataset import BigramEmbeddingDataset, TokenIndexBigram


class TestBigramEmbeddingDataset:
    """Tests for BigramEmbeddingDataset."""

    def test_len_returns_correct_count(self) -> None:
        """Test __len__ returns number of bigrams (tokens - 1)."""
        tokens = [1, 2, 3, 4, 5]
        dataset = BigramEmbeddingDataset(tokens)

        assert len(dataset) == 4

    def test_len_returns_zero_for_single_token(self) -> None:
        """Test __len__ returns 0 when no bigrams can be formed."""
        tokens = [1]
        dataset = BigramEmbeddingDataset(tokens)

        assert len(dataset) == 0

    def test_len_returns_zero_for_empty_tokens(self) -> None:
        """Test __len__ returns 0 for empty token sequence."""
        tokens: list[int] = []
        dataset = BigramEmbeddingDataset(tokens)

        assert len(dataset) == 0

    def test_getitem_returns_correct_bigram(self) -> None:
        """Test __getitem__ returns correct left and right tokens."""
        tokens = [10, 20, 30, 40]
        dataset = BigramEmbeddingDataset(tokens)

        bigram = dataset[0]

        assert isinstance(bigram, TokenIndexBigram)
        assert bigram.left.item() == 10
        assert bigram.right.item() == 20

    def test_getitem_returns_correct_bigram_at_last_index(self) -> None:
        """Test __getitem__ returns correct bigram at last valid index."""
        tokens = [10, 20, 30, 40]
        dataset = BigramEmbeddingDataset(tokens)

        bigram = dataset[2]

        assert bigram.left.item() == 30
        assert bigram.right.item() == 40

    def test_getitem_returns_tensors_with_correct_shape(self) -> None:
        """Test __getitem__ returns scalar tensors."""
        tokens = [1, 2, 3]
        dataset = BigramEmbeddingDataset(tokens)

        bigram = dataset[0]

        assert bigram.left.shape == ()
        assert bigram.right.shape == ()

    def test_getitem_returns_long_tensors(self) -> None:
        """Test __getitem__ returns tensors with long dtype."""
        tokens = [1, 2, 3]
        dataset = BigramEmbeddingDataset(tokens)

        bigram = dataset[0]

        assert bigram.left.dtype == torch.long
        assert bigram.right.dtype == torch.long

    def test_dataloader_batch_shapes(self) -> None:
        """Test DataLoader produces batches with shape (batch_size,)."""
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        dataset = BigramEmbeddingDataset(tokens)
        batch_size = 3
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        for batch in dataloader:
            assert batch.left.shape == (batch_size,)
            assert batch.right.shape == (batch_size,)

    def test_dataloader_iteration_returns_correct_bigrams(self) -> None:
        """Test DataLoader yields correct bigram values in order."""
        tokens = [10, 20, 30, 40]
        dataset = BigramEmbeddingDataset(tokens)
        dataloader = DataLoader(dataset, batch_size=2)

        batches = list(dataloader)

        assert len(batches) == 2
        assert batches[0].left.tolist() == [10, 20]
        assert batches[0].right.tolist() == [20, 30]
        assert batches[1].left.tolist() == [30]
        assert batches[1].right.tolist() == [40]

    def test_getitem_index_out_of_range_raises(self) -> None:
        """Test __getitem__ raises IndexError for out of range index."""
        tokens = [1, 2, 3]
        dataset = BigramEmbeddingDataset(tokens)

        with pytest.raises(IndexError):
            _ = dataset[2]
