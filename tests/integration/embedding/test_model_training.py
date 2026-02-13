"""Integration tests for BigramEmbeddingModel training."""

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader

from llm_engineering.embedding.dataset import BigramEmbeddingDataset
from llm_engineering.embedding.model import BigramEmbeddingModel


class TestModelTraining:
    """Integration tests for model training."""

    def test_training_step_completes(self) -> None:
        """Test a single training step completes without error."""
        vocab_size = 50
        embedding_dim = 8
        chunks = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        batch_size = 2

        dataset = BigramEmbeddingDataset(chunks)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        model = BigramEmbeddingModel(vocab_size=vocab_size, embedding_dim=embedding_dim)
        optimizer = SGD(model.parameters(), lr=0.01)
        criterion = CrossEntropyLoss()

        model.train()
        initial_loss = None
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(batch.left)
            loss = criterion(output, batch.right.long())
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        assert initial_loss is not None

    def test_model_predicts_next_token(self) -> None:
        """Test model can be trained to predict next token."""
        vocab_size = 10
        embedding_dim = 8
        chunks = [[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]]
        batch_size = 4

        dataset = BigramEmbeddingDataset(chunks)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = BigramEmbeddingModel(vocab_size=vocab_size, embedding_dim=embedding_dim)
        optimizer = SGD(model.parameters(), lr=0.1)
        criterion = CrossEntropyLoss()

        model.train()
        for _ in range(10):
            for batch in dataloader:
                optimizer.zero_grad()
                output = model(batch.left)
                loss = criterion(output, batch.right.long())
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            input_token = torch.tensor([1])
            prediction = model(input_token)
            predicted_token = prediction.argmax(dim=1).item()

            assert predicted_token == 2
