"""Tests for BigramEmbeddingDataset."""

import pytest
from torch.utils.data import DataLoader

from llm_engineering.embedding.dataset import BigramEmbeddingDataset, TokenIndexBigram


class TestBigramEmbeddingDataset:
    """Tests for BigramEmbeddingDataset."""

    def test_len_returns_correct_count_single_chunk(self) -> None:
        """Test __len__ returns number of bigrams for a single chunk."""
        chunks = [[1, 2, 3, 4, 5]]
        dataset = BigramEmbeddingDataset(chunks)

        assert len(dataset) == 4

    def test_len_returns_correct_count_multiple_chunks(self) -> None:
        """Test __len__ returns total bigrams across multiple chunks."""
        chunks = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        dataset = BigramEmbeddingDataset(chunks)

        assert len(dataset) == 6

    def test_len_returns_zero_for_empty_chunks(self) -> None:
        """Test __len__ returns 0 for empty chunk sequence."""
        chunks: list[list[int]] = []
        dataset = BigramEmbeddingDataset(chunks)

        assert len(dataset) == 0

    def test_empty_chunks_ignored(self) -> None:
        """Test empty chunks are filtered out."""
        chunks = [[1, 2], [], [3, 4]]
        dataset = BigramEmbeddingDataset(chunks)

        assert len(dataset) == 2

    def test_single_token_chunks_ignored(self) -> None:
        """Test single-token chunks are filtered out."""
        chunks = [[1, 2], [3], [4, 5, 6]]
        dataset = BigramEmbeddingDataset(chunks)

        assert len(dataset) == 3

    def test_getitem_returns_correct_bigram(self) -> None:
        """Test __getitem__ returns correct left and right tokens."""
        chunks = [[10, 20, 30, 40]]
        dataset = BigramEmbeddingDataset(chunks)

        bigram = dataset[0]

        assert isinstance(bigram, TokenIndexBigram)
        assert bigram.left.item() == 10
        assert bigram.right.item() == 20

    def test_getitem_returns_correct_bigram_at_last_index(self) -> None:
        """Test __getitem__ returns correct bigram at last valid index."""
        chunks = [[10, 20, 30, 40]]
        dataset = BigramEmbeddingDataset(chunks)

        bigram = dataset[2]

        assert bigram.left.item() == 30
        assert bigram.right.item() == 40

    def test_getitem_across_chunks(self) -> None:
        """Test __getitem__ correctly indexes across multiple chunks."""
        chunks = [[1, 2, 3], [4, 5], [6, 7, 8]]
        dataset = BigramEmbeddingDataset(chunks)

        assert dataset[0].left.item() == 1
        assert dataset[0].right.item() == 2
        assert dataset[1].left.item() == 2
        assert dataset[1].right.item() == 3
        assert dataset[2].left.item() == 4
        assert dataset[2].right.item() == 5
        assert dataset[3].left.item() == 6
        assert dataset[3].right.item() == 7
        assert dataset[4].left.item() == 7
        assert dataset[4].right.item() == 8

    def test_no_invalid_bigrams_at_boundaries(self) -> None:
        """Test no bigrams are created across chunk boundaries."""
        chunks = [[1, 2, 3], [4, 5, 6]]
        dataset = BigramEmbeddingDataset(chunks)

        all_bigrams = [
            (dataset[i].left.item(), dataset[i].right.item())
            for i in range(len(dataset))
        ]

        assert (3, 4) not in all_bigrams

    def test_getitem_returns_tensors_with_correct_shape(self) -> None:
        """Test __getitem__ returns scalar tensors."""
        chunks = [[1, 2, 3]]
        dataset = BigramEmbeddingDataset(chunks)

        bigram = dataset[0]

        assert bigram.left.shape == ()
        assert bigram.right.shape == ()

    def test_dataloader_batch_shapes(self) -> None:
        """Test DataLoader produces batches with shape (batch_size,)."""
        chunks = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
        dataset = BigramEmbeddingDataset(chunks)
        batch_size = 3
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        for batch in dataloader:
            assert batch.left.shape == (batch_size,)
            assert batch.right.shape == (batch_size,)

    def test_dataloader_iteration_returns_correct_bigrams(self) -> None:
        """Test DataLoader yields correct bigram values in order."""
        chunks = [[10, 20, 30, 40]]
        dataset = BigramEmbeddingDataset(chunks)
        dataloader = DataLoader(dataset, batch_size=2)

        batches = list(dataloader)

        assert len(batches) == 2
        assert batches[0].left.tolist() == [10, 20]
        assert batches[0].right.tolist() == [20, 30]
        assert batches[1].left.tolist() == [30]
        assert batches[1].right.tolist() == [40]

    def test_getitem_index_out_of_range_raises(self) -> None:
        """Test __getitem__ raises IndexError for out of range index."""
        chunks = [[1, 2, 3]]
        dataset = BigramEmbeddingDataset(chunks)

        with pytest.raises(IndexError):
            _ = dataset[2]
