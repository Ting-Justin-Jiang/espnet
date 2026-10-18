"""Full-Duplex-Bench v1.0 test sets: one user audio stream per sample."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

from egs3.full_duplex_bench.moshi.dataset.builder import (
    TASKS,
    benchmark_root,
    sample_dirs,
)
from espnet3.components.data.base_dataset import BaseDataset


class FullDuplexBenchDataset(BaseDataset):
    """Samples of one Full-Duplex-Bench v1.0 task, in sample-id order.

    The user's audio is resampled to ``sample_rate`` with the same
    ``torchaudio`` resampler the benchmark's Moshi client uses.

    Args:
        task: Task directory name, such as ``candor_turn_taking``.
        recipe_dir: Recipe root directory holding ``downloads/``.
        sample_rate: Sample rate of the returned ``speech``.

    Raises:
        ValueError: If ``task`` is unknown.
        FileNotFoundError: If the task has not been downloaded.

    Examples:
        >>> dataset = FullDuplexBenchDataset("candor_turn_taking", ".", 24000)
        >>> sorted(dataset[0])
        ['annotation', 'sample_id', 'speech', 'task', 'utt_id']
    """

    fields = {
        "utt_id": str,
        "speech": np.ndarray,
        "task": str,
        "sample_id": str,
        "annotation": str,
    }

    def __init__(self, task: str, recipe_dir: str | Path, sample_rate: int) -> None:
        """List the task's sample directories."""
        if task not in TASKS:
            raise ValueError(f"Unknown task {task!r}; expected one of {list(TASKS)}")
        self.task = task
        self.sample_rate = sample_rate
        self.annotation = TASKS[task]["annotation"]
        task_dir = benchmark_root(recipe_dir) / task
        if not task_dir.is_dir():
            raise FileNotFoundError(
                f"{task_dir} not found; run the create_dataset stage first."
            )
        self.sample_dirs = sample_dirs(task_dir)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return the user's audio and the annotation of sample ``idx``."""
        sample_dir = self.sample_dirs[int(idx)]
        audio, rate = sf.read(str(sample_dir / "input.wav"), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        speech = torchaudio.functional.resample(
            torch.from_numpy(audio), rate, self.sample_rate
        ).numpy()
        if self.annotation is None:
            annotation = ""
        else:
            annotation = str((sample_dir / self.annotation).resolve())
        return {
            "utt_id": f"{self.task}-{sample_dir.name}",
            "speech": speech,
            "task": self.task,
            "sample_id": sample_dir.name,
            "annotation": annotation,
        }
