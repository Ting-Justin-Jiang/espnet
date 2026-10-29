"""otoSpeech training windows: two-channel audio and the words of channel 0."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import soundfile as sf

from egs3.otospeech.duplex.dataset.builder import manifest_path
from espnet3.components.data.base_dataset import BaseDataset
from espnet3.contract import Field
from espnet3.contract.kinds.audio import Audio


class OtoSpeechDataset(BaseDataset):
    """Fixed-length windows of every session, once for each speaker.

    Both speakers are people, so each takes the model's side (channel 0) in
    turn: a session of ``n`` windows gives ``2 n`` items. Windows follow each
    other without overlap, as in kyutai's moshi-finetune; the last one of a
    session is shorter.

    Args:
        split: ``train`` or ``valid``.
        recipe_dir: Recipe root holding ``data/``.
        duration: Window length in seconds.

    Raises:
        FileNotFoundError: If the manifest is missing.
    """

    fields = (Field("speech", "audio", channels=2), Field("words", "segments"))

    def __init__(self, split: str, recipe_dir: str | Path, duration: float) -> None:
        """Read the manifest and list the windows."""
        path = manifest_path(recipe_dir, split)
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; run create_dataset first.")
        self.recipe_dir = Path(recipe_dir)
        self.duration = duration
        self.windows = []
        for line in path.read_text().splitlines():
            session = json.loads(line)
            for index in range(math.ceil(session["duration"] / duration)):
                for main in (0, 1):
                    self.windows.append((session, main, index * duration))

    def __len__(self) -> int:
        """Return the number of windows."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one window, its model speaker first, with that speaker's words."""
        session, main, start = self.windows[int(idx)]
        path = self.recipe_dir / session["audio"]
        rate = sf.info(str(path)).samplerate
        audio, _ = sf.read(
            str(path),
            start=round(start * rate),
            stop=round((start + self.duration) * rate),
            dtype="float32",
        )
        words = [
            {**word, "start": word["start"] - start, "end": word["end"] - start}
            for word in session["words"][main]
            if start <= word["start"] < start + self.duration
        ]
        return {"speech": Audio(audio[:, [main, 1 - main]], rate), "words": words}
