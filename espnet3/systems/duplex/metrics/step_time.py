"""Wall-clock time of a Moshi frame, against the 80 ms real-time budget."""

from __future__ import annotations

from typing import Sequence

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field

#: Duration of one frame, in milliseconds.
FRAME_MS = 80.0


class StepTime(BaseMetric):
    """Frame time over the conversations of a test set.

    Reads ``step_ms``, which the ``infer`` stage writes for each
    conversation (its median frame time), and averages it over the
    conversations, each counting once. Live serving keeps up with the user
    only when a frame takes less than 80 ms.

    Examples:
        >>> metric = StepTime()
        >>> metric.update(step_ms=["20.0", "40.0"])
        >>> metric.compute()
        {'step_ms': 30.0, 'real_time_factor': 0.375}
    """

    # The infer stage writes step_ms as text in its SCP file.
    inputs = (Field("step_ms", "text"),)
    outputs = (Field("step_ms", "number"), Field("real_time_factor", "number"))

    def __init__(self) -> None:
        """Register the per-conversation times."""
        super().__init__()
        self.add_state("times", [], "cat")

    def update(self, *, step_ms: Sequence[str]) -> None:
        """Add the mean frame time of each conversation."""
        self.times.extend(float(value) for value in step_ms)

    def compute(self) -> dict[str, float]:
        """Return the mean frame time and its ratio to the frame duration."""
        mean = sum(self.times) / len(self.times)
        return {"step_ms": mean, "real_time_factor": mean / FRAME_MS}
