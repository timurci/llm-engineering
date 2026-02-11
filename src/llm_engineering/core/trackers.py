"""A module for experiment tracking."""

from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, Self

if TYPE_CHECKING:
    from collections.abc import Mapping
    from logging import Logger
    from types import TracebackType


class Phase(StrEnum):
    """The phase of an experiment step."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MetricTracker(Protocol):
    """An interface for logging step metrics."""

    def log_metrics(
        self, phase: Phase, step: int, metrics: Mapping[str, float]
    ) -> None:
        """Logs the metrics of the current step (e.g., epoch, iteration).

        Args:
            phase: The phase of the experiment run.
            step: The step of the experiment run.
            metrics: The metrics to log.
        """
        ...


class ContextManager(Protocol):
    """An interface to support context management."""

    def __enter__(self) -> Self:
        """Enters the context."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exits the context."""
        ...


class ParamTracker(Protocol):
    """An interface for logging experiment parameters."""

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Logs the parameters of the current experiment run."""
        ...


class ExperimentTracker(ParamTracker, MetricTracker, Protocol):
    """An interface for experiment tracking."""

    ...  # noqa: PYI013


class StandardLoggingTracker(ExperimentTracker, ContextManager):
    """An experiment tracker for logging module in standard library."""

    def __init__(
        self, logger: Logger, logging_interval: int, experiment_steps: int | None = None
    ) -> None:
        """Initializes a new StandardLoggingTracker.

        Args:
            logger: The logger to use for logging.
            logging_interval: The interval at which to log metrics.
            experiment_steps: The total number of expected steps in the experiment.
        """
        self.logger = logger
        self.logging_interval = logging_interval
        self.experiment_steps = experiment_steps

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Logs the parameters of the current experiment run."""
        self.logger.info("Parameters: %s", params)

    def log_metrics(
        self, phase: Phase, step: int, metrics: Mapping[str, float]
    ) -> None:
        """Logs the metrics of the current experiment run."""
        if not (
            step in (1, self.experiment_steps) or step % self.logging_interval == 0
        ):
            return

        metrics_format = ", ".join(
            f"({key}, {value:.4f})" for key, value in metrics.items()
        )
        self.logger.info("(%s, %d): %s", phase, step, metrics_format)

    def __enter__(self) -> Self:
        """Enters the context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exits the context."""
