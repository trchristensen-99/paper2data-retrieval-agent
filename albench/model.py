"""Core model interface for sequence-to-function predictors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SequenceModel(ABC):
    """Abstract interface for oracle and student models."""

    @abstractmethod
    def predict(self, sequences: list[str]) -> np.ndarray:
        """Map input sequences to scalar predictions with shape ``(N,)``."""

    def uncertainty(self, sequences: list[str]) -> np.ndarray:
        """Return uncertainty values with shape ``(N,)``."""
        raise NotImplementedError

    def embed(self, sequences: list[str]) -> np.ndarray:
        """Return embeddings with shape ``(N, D)``."""
        raise NotImplementedError

    def fit(self, sequences: list[str], labels: np.ndarray) -> None:
        """Fit or update model parameters on labeled examples."""
        raise NotImplementedError
