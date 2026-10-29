"""Full-Duplex-Bench v1.0 turn-taking metrics (Lin et al., 2025).

A faithful port of the benchmark's evaluation code
(github.com/DanielLin94144/Full-Duplex-Bench, commit 3e799c4,
``v1_v1.5/get_transcript/asr.py`` and ``v1_v1.5/evaluation/eval_*.py``):

- The model's channel is written as 16-bit PCM and transcribed with word
  timestamps by ``nvidia/parakeet-tdt-0.6b-v2``. For user interruption, only
  the audio after the end of the interruption is transcribed.
- Pause handling, smooth turn-taking and user interruption: the model takes
  the turn (TOR = 1) when it says more than 3 words, or speaks for 1 s or
  more. Latency is the start of its first word minus the end of the user's
  turn (the start of the turn-taking marker) or of the interruption,
  clipped at 0, over the samples where it took the turn.
- Backchannel: Silero VAD segments of the model's channel, in time order. The
  first segment over 3 s takes the turn and ends the scan; until then each
  short segment counts as a backchannel and decides the turn by its word count
  and length, the last one winning. Frequency is the number of backchannels
  per second; JSD is the Jensen-Shannon distance between their 0.2 s histogram
  and the ICC reference.

The GPT-4o relevance score of user interruption is not computed: it needs the
OpenAI API. A score a task does not define is NaN.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch
import torchaudio
from nemo.collections.asr.models import ASRModel
from scipy.interpolate import interp1d
from scipy.spatial.distance import jensenshannon
from silero_vad import get_speech_timestamps, load_silero_vad

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field

#: How each task is scored.
TASK_RULES = {
    "candor_pause_handling": "pause_handling",
    "synthetic_pause_handling": "pause_handling",
    "candor_turn_taking": "turn_taking",
    "synthetic_user_interruption": "user_interruption",
    "icc_backchannel": "backchannel",
}

TURN_DURATION = 1.0
TURN_NUM_WORDS = 3
BACKCHANNEL_MAX_DURATION = 3.0
BACKCHANNEL_WINDOW = 0.2
EPSILON = 1e-10


def takes_turn(words: list[dict]) -> bool:
    """Return whether timestamped words amount to taking the turn."""
    if not words:
        return False
    duration = words[-1]["end"] - words[0]["start"]
    if duration < TURN_DURATION:
        return len(words) > TURN_NUM_WORDS
    return True


def _words_in_segment(words: list[dict], start: float, end: float) -> list[str]:
    """Return the words overlapping ``[start, end]``, as the benchmark selects them."""
    selected = []
    for word in words:
        inside = word["start"] >= start and word["end"] <= end
        crosses_end = word["start"] <= end and word["end"] > end
        crosses_start = word["start"] <= start and word["end"] > start
        if inside or crosses_end or crosses_start:
            selected.append(word["word"])
    return selected


class FullDuplexBench(BaseMetric):
    """Score each test set with the rule of its Full-Duplex-Bench task.

    Inputs come from the ``infer`` stage (``wav``) and from the test set
    (``task``, ``annotation``, ``sample_id``).

    Args:
        asr_model: NeMo name of the time-aligned ASR.
        backchannel_reference: ICC reference distribution, per sample id.
        device: Device of the ASR.
    """

    inputs = (
        Field("wav", "audio"),
        Field("task", "text"),
        Field("annotation", "text"),
        Field("sample_id", "text"),
    )
    outputs = (
        Field("TOR", "number"),
        Field("latency", "number"),
        Field("backchannel_freq", "number"),
        Field("backchannel_jsd", "number"),
    )

    def __init__(self, asr_model: str, backchannel_reference: str, device: str):
        """Load the ASR, the VAD and the backchannel reference."""
        super().__init__()
        self.asr = ASRModel.from_pretrained(model_name=asr_model).to(device).eval()
        self.vad = load_silero_vad()
        self.reference = json.loads(Path(backchannel_reference).read_text())
        self.add_state("turns", [], "cat")
        self.add_state("latencies", [], "cat")
        self.add_state("frequencies", [], "cat")
        self.add_state("divergences", [], "cat")
        self.add_state("records", [], "cat")

    def _transcribe(self, wav_path: str, offset: float) -> list[dict]:
        """Return the words of ``wav_path`` after ``offset`` seconds, timestamped."""
        waveform, rate = sf.read(wav_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        waveform = waveform[int(offset * rate) :]
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "output.wav")
            sf.write(path, waveform, rate)
            result = self.asr.transcribe([path], timestamps=True, verbose=False)[0]
        return [
            {
                "word": word["word"],
                "start": word["start"] + offset,
                "end": word["end"] + offset,
            }
            for word in result.timestamp["word"]
        ]

    def _backchannel(
        self, wav_path: str, words: list[dict], sample_id: str
    ) -> tuple[int, float, float, list[list[float]]]:
        """Return the turn, frequency, JSD and segments of a backchannel sample."""
        audio, rate = sf.read(wav_path, dtype="float32")
        wav = torch.from_numpy(audio)[None]
        wav = torchaudio.functional.resample(wav, rate, 16000)
        segments = get_speech_timestamps(wav, self.vad, return_seconds=True)
        turn = 0
        backchannels = []
        for segment in segments:
            start, end = segment["start"], segment["end"]
            duration = end - start
            if duration > BACKCHANNEL_MAX_DURATION:
                turn = 1
                break
            num_words = len(_words_in_segment(words, start, end))
            if num_words > 3:
                turn = 1
            elif duration < 1:
                turn = int(num_words > 2)
            else:
                turn = 1
            backchannels.append([start, end])

        total = wav.shape[-1] / 16000
        frequency = len(backchannels) / total
        if not backchannels:
            return turn, frequency, 1.0, backchannels
        histogram = np.zeros(int(total / BACKCHANNEL_WINDOW) + 1)
        for start, end in backchannels:
            first = int(start / BACKCHANNEL_WINDOW)
            last = min(int(end / BACKCHANNEL_WINDOW), len(histogram) - 1)
            histogram[first : last + 1] += 1
        histogram = histogram + EPSILON
        histogram = histogram / histogram.sum()
        reference = np.asarray(self.reference[sample_id])
        resized = interp1d(
            np.linspace(0, 1, len(reference)),
            reference,
            kind="linear",
            fill_value="extrapolate",
        )(np.linspace(0, 1, len(histogram)))
        return turn, frequency, float(jensenshannon(histogram, resized)), backchannels

    def update(
        self,
        *,
        wav: Sequence[str],
        task: Sequence[str],
        annotation: Sequence[str],
        sample_id: Sequence[str],
    ) -> None:
        """Score each sample with its task's rule."""
        for wav_path, task_name, annotation_path, sid in zip(
            wav, task, annotation, sample_id
        ):
            rule = TASK_RULES[task_name]
            latency = None
            if rule == "user_interruption":
                interrupt = json.loads(Path(annotation_path).read_text())[0]
                user_end = interrupt["timestamp"][1]
                words = self._transcribe(wav_path, offset=user_end)
            else:
                words = self._transcribe(wav_path, offset=0.0)

            if rule == "backchannel":
                turn, frequency, divergence, segments = self._backchannel(
                    wav_path, words, sid
                )
                self.frequencies.append(frequency)
                self.divergences.append(divergence)
            else:
                turn = int(takes_turn(words))
                segments = None
            if rule == "turn_taking" and turn == 1:
                marker = json.loads(Path(annotation_path).read_text())[0]
                latency = max(0.0, words[0]["start"] - marker["timestamp"][0])
            if rule == "user_interruption" and turn == 1:
                latency = max(0.0, words[0]["start"] - user_end)
            self.turns.append(turn)
            if latency is not None:
                self.latencies.append(latency)
            if self.collect_artifacts:
                self.records.append(
                    {
                        "task": task_name,
                        "sample_id": sid,
                        "TOR": turn,
                        "latency": latency,
                        "words": words,
                        "backchannels": segments,
                    }
                )

    def compute(self) -> dict[str, float]:
        """Average each score over the samples it is defined for (else NaN)."""
        return {
            "TOR": _mean(self.turns),
            "latency": _mean(self.latencies),
            "backchannel_freq": _mean(self.frequencies),
            "backchannel_jsd": _mean(self.divergences),
        }

    def write_artifacts(self, test_dir: Path) -> None:
        """Write each sample's words, decision and latency as JSON lines."""
        with (test_dir / "full_duplex_bench.jsonl").open("w") as handle:
            for record in self.records:
                handle.write(json.dumps(record) + "\n")


def _mean(values: list[float]) -> float:
    """Return the mean of ``values``, NaN when there is none."""
    if not values:
        return math.nan
    return float(np.mean(values))
