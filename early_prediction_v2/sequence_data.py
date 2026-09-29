"""Compatibility types for the disabled sequence-model path.

The submitted Backblaze experiment is tabular.  Keeping the small container
avoids changing the dataset schema, while sequence construction fails clearly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SequencePeriod:
    features: Any = None
    labels: Any = None
    feature_dim: int | None = None
    window_size: int | None = None
    @property
    def shape(self) -> Any:
        return getattr(self.features, "shape", None)

    def __len__(self) -> int:
        if self.labels is not None:
            return len(self.labels)
        if self.features is not None:
            return len(self.features)
        return 0


def build_entity_sequence_periods(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(
        "Sequence models are not part of the submitted Backblaze experiment."
    )
