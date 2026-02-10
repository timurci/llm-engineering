"""Training module for embedding model."""

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple, Protocol

import torch
from torch import nn

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


class TrainingPhase(StrEnum):
    """Training phase enum."""

    TRAINING = "training"
    VALIDATION = "validation"


class ClassificationScore(NamedTuple):
    """Classification metrics for one epoch."""

    loss: float
    accuracy: float


@dataclass(frozen=True)
class TrainingHistoryItem:
    """History item for one epoch."""

    phase: TrainingPhase
    epoch: int
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
        self.loss_fn = nn.CrossEntropyLoss() if loss_fn is None else loss_fn

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
        raise NotImplementedError

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
        raise NotImplementedError

    def train(
        self,
        train_loader: DataLoader[TokenIndexBigram],
        device: torch.device,
        epochs: int,
        val_loader: DataLoader[TokenIndexBigram] | None = None,
        logging_interval: int = 100,
    ) -> list[TrainingHistoryItem]:
        """Train the model for a given number of epochs.

        Args:
            train_loader: The data loader for the training data.
            device: The device to load the data.
            epochs: The number of epochs to train for.
            val_loader: The data loader for the validation data.
            logging_interval: The interval at which to log training progress.

        Returns:
            A list of training history items.
        """
        raise NotImplementedError
