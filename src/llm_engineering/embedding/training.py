"""Training module for embedding model."""

from typing import TYPE_CHECKING, NamedTuple, Protocol

import torch
from torch import nn

from llm_engineering.core.trackers import ExperimentTracker, Phase

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from llm_engineering.embedding.dataset import TokenIndexBigram
    from llm_engineering.embedding.model import EmbeddingModel


class LossFunction(Protocol):
    """Loss function protocol."""

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss between the output and target tensors.

        Args:
            output: The output tensor from the model.
            target: The target label tensor.

        Returns:
            The loss tensor.
        """
        ...


class ClassificationScore(NamedTuple):
    """Classification metrics for one epoch."""

    loss: float
    accuracy: float


class BigramEmbeddingTrainer:
    """Trainer class for embedding model."""

    def __init__(
        self,
        model: EmbeddingModel,
        optimizer: torch.optim.Optimizer,
        loss_fn: LossFunction | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: The embedding model to train.
            optimizer: The optimizer to use for training.
            loss_fn: The loss function to use for training.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

    def _train_epoch(
        self, loader: DataLoader[TokenIndexBigram], device: torch.device
    ) -> ClassificationScore:
        """Run training loop for one epoch.

        Args:
            loader: The data loader for the training data.
            device: The device to load the data.

        Returns:
            A classification score for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for batch in loader:
            self.optimizer.zero_grad()

            inputs = batch.left.to(device)
            targets = batch.right.to(device)
            outputs = self.model(inputs)

            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == targets).sum().item()
            total_samples += inputs.size(0)
        return ClassificationScore(
            loss=(total_loss / total_samples),
            accuracy=(total_correct / total_samples),
        )

    def _validate_epoch(
        self, loader: DataLoader[TokenIndexBigram], device: torch.device
    ) -> ClassificationScore:
        """Run validation loop for one epoch.

        Args:
            loader: The data loader for the validation data.
            device: The device to load the data.

        Returns:
            A classification score for the epoch.
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.inference_mode():
            for batch in loader:
                inputs = batch["left"].to(device)
                targets = batch["right"].to(device)
                outputs = self.model(inputs)

                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                total_correct += (outputs.argmax(dim=1) == targets).sum().item()
                total_samples += inputs.size(0)
        return ClassificationScore(
            loss=(total_loss / total_samples),
            accuracy=(total_correct / total_samples),
        )

    def train(
        self,
        train_loader: DataLoader[TokenIndexBigram],
        device: torch.device,
        epochs: int,
        val_loader: DataLoader[TokenIndexBigram] | None = None,
        experiment_trackers: list[ExperimentTracker] | None = None,
    ) -> None:
        """Train the model for a given number of epochs.

        Args:
            train_loader: The data loader for the training data.
            device: The device to load the data.
            epochs: The number of epochs to train for.
            val_loader: The data loader for the validation data.
            experiment_trackers: The experiment trackers to log metrics.
        """
        experiment_trackers = experiment_trackers or []
        for epoch in range(1, epochs + 1):
            train_metrics = self._train_epoch(train_loader, device)
            for tracker in experiment_trackers:
                tracker.log_metrics(Phase.TRAIN, epoch, train_metrics._asdict())
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader, device)
                for tracker in experiment_trackers:
                    tracker.log_metrics(Phase.VAL, epoch, val_metrics._asdict())
