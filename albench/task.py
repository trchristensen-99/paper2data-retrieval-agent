"""Task-level configuration objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskConfig:
    """Configuration for a sequence-to-function learning task."""

    name: str
    organism: str
    sequence_length: int
    data_root: str
    train_path: str | None = None
    val_path: str | None = None
    test_set: dict[str, str] = field(default_factory=dict)
    flanking_sequence: dict[str, str] = field(default_factory=dict)
    oracle_checkpoint: str | None = None
    sequence_filters: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
