"""otoSpeech full-duplex dataset builder.

The release is a set of WebDataset shards; a sample is one session: a
two-channel FLAC (one speaker per channel, mapped by ``speakers[].channel``
in its JSON) and that JSON. ``build`` resamples each session to Mimi's rate,
transcribes each channel with word timestamps and writes one manifest line
per session:

    {"session_id": ..., "audio": "data/audio/<id>.flac", "duration": seconds,
     "words": [<channel 0 segments>, <channel 1 segments>]}

where a segment is ``{"text", "start", "end"}``, in seconds.
"""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
from importlib import resources
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import snapshot_download

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults


def _load_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)


_CFG = _load_config()["builder"]


def manifest_path(recipe_dir: str | Path, split: str) -> Path:
    """Return the manifest of ``split`` (``train`` or ``valid``)."""
    return Path(recipe_dir) / "data" / "manifest" / f"{split}.jsonl"


def _shards(recipe_dir: str | Path) -> list[Path]:
    root = Path(recipe_dir) / _CFG["download_dir"] / "data" / "train"
    return sorted(root.glob("shard-*.tar"))


def _sessions(shard: Path):
    """Yield ``(session_id, flac bytes, metadata)`` of every sample in ``shard``."""
    members: dict[str, dict[str, bytes]] = {}
    with tarfile.open(shard) as tar:
        for member in tar.getmembers():
            key, _, extension = member.name.partition(".")
            members.setdefault(key, {})[extension] = tar.extractfile(member).read()
    for key in sorted(members):
        yield key, members[key]["flac"], json.loads(members[key]["json"])


def _words(asr, audio: np.ndarray, rate: int) -> list[dict]:
    """Transcribe one channel; return its words as segments."""
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "channel.wav")
        sf.write(path, audio, rate)
        result = asr.transcribe([path], timestamps=True, verbose=False)[0]
    return [
        {"text": word["word"], "start": float(word["start"]), "end": float(word["end"])}
        for word in result.timestamp["word"]
    ]


class OtoSpeechBuilder(DatasetBuilder):
    """Download otoSpeech and build its train/valid manifests (GPU for ASR)."""

    def is_source_prepared(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check that every shard is downloaded."""
        return len(_shards(recipe_dir)) == _CFG["shards"]

    def prepare_source(self, recipe_dir: str | Path, **_kwargs) -> None:
        """Download the pinned shards into ``<recipe_dir>/<download_dir>``."""
        snapshot_download(
            _CFG["hf_repo"],
            repo_type="dataset",
            revision=_CFG["revision"],
            local_dir=Path(recipe_dir) / _CFG["download_dir"],
            allow_patterns=["data/train/*.tar"],
        )

    def is_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check that both manifests exist."""
        return all(manifest_path(recipe_dir, s).is_file() for s in ("train", "valid"))

    def build(self, recipe_dir: str | Path, **_kwargs) -> None:
        """Resample, transcribe and write each session, then the manifests."""
        from nemo.collections.asr.models import ASRModel

        recipe_dir = Path(recipe_dir)
        rate = _CFG["sample_rate"]
        asr = ASRModel.from_pretrained(model_name=_CFG["asr_model"]).to("cuda")
        asr.eval()
        audio_dir = recipe_dir / "data" / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for shard in _shards(recipe_dir):
            for session_id, flac, meta in _sessions(shard):
                audio, source_rate = sf.read(io.BytesIO(flac), dtype="float32")
                if audio.ndim != 2 or audio.shape[1] != 2:
                    raise ValueError(
                        f"{session_id}: expected 2 channels, got {audio.shape}"
                    )
                channels = sorted(speaker["channel"] for speaker in meta["speakers"])
                if channels != [0, 1]:
                    raise ValueError(f"{session_id}: speaker channels {channels}")
                audio = torchaudio.functional.resample(
                    torch.from_numpy(audio.T), source_rate, rate
                ).numpy()
                path = audio_dir / f"{session_id}.flac"
                sf.write(path, audio.T, rate)
                lines.append(
                    {
                        "session_id": session_id,
                        "audio": path.relative_to(recipe_dir).as_posix(),
                        "duration": audio.shape[1] / rate,
                        "words": [_words(asr, channel, rate) for channel in audio],
                    }
                )
        lines.sort(key=lambda line: line["session_id"])
        cut = len(lines) - _CFG["valid_sessions"]
        for split, part in (("train", lines[:cut]), ("valid", lines[cut:])):
            path = manifest_path(recipe_dir, split)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(line) + "\n" for line in part))
