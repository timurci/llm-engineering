"""Integration tests for BigramEmbeddingDataset and BigramEmbeddingModel."""

from torch.utils.data import DataLoader

from llm_engineering.embedding.dataset import BigramEmbeddingDataset
from llm_engineering.embedding.model import BigramEmbeddingModel


class TestDatasetModelIntegration:
    """Integration tests for dataset and model compatibility."""

    def test_forward_pass_output_shape(self) -> None:
        """Test model produces correct output shape from dataset batch."""
        vocab_size = 100
        embedding_dim = 16
        chunks = [[1, 2, 3, 4, 5, 6, 7, 8]]
        batch_size = 2

        dataset = BigramEmbeddingDataset(chunks)
        dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
        model = BigramEmbeddingModel(vocab_size=vocab_size, embedding_dim=embedding_dim)

        for batch in dataloader:
            output = model(batch.left)

            assert output.shape == (batch_size, vocab_size)

    def test_forward_pass_different_batch_sizes(self) -> None:
        """Test model handles different batch sizes correctly."""
        vocab_size = 100
        embedding_dim = 16
        chunks = [list(range(1, 21))]

        dataset = BigramEmbeddingDataset(chunks)
        model = BigramEmbeddingModel(vocab_size=vocab_size, embedding_dim=embedding_dim)

        for batch_size in [1, 3, 5, 7]:
            dataloader = DataLoader(dataset, batch_size=batch_size)
            for batch in dataloader:
                output = model(batch.left)
                assert output.shape[1] == vocab_size

    def test_embedding_layer_receives_correct_input(self) -> None:
        """Test embedding layer receives input within vocab range."""
        vocab_size = 100
        embedding_dim = 16
        chunks = [[0, 1, 2, 50, 99, 0, 99, 50]]
        batch_size = 2

        dataset = BigramEmbeddingDataset(chunks)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        model = BigramEmbeddingModel(vocab_size=vocab_size, embedding_dim=embedding_dim)

        for batch in dataloader:
            assert batch.left.max() < vocab_size
            assert batch.left.min() >= 0
            output = model(batch.left)
            assert output.shape[1] == vocab_size
