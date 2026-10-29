"""Fine-tuning a released full-duplex model with ESPnet3's ``train`` stage.

A training item is a window of a two-channel conversation and the words of
the model's channel in it:

- ``speech``: two-channel :class:`Audio` at Mimi's sample rate; channel 0
  is the speaker the model learns to be, channel 1 the user.
- ``words``: the words of channel 0 as ``segments`` (``text``, ``start``,
  ``end``), in seconds from the start of the window.

:class:`TextStream`, the dataset preprocessor, turns the words into the
model's text stream. :class:`DuplexLM`, the model, encodes both channels with
the frozen Mimi, as at inference, and trains the language model on every
stream. Both follow kyutai's moshi-finetune (MIT): the ``Interleaver`` with
``keep_main_only`` and the loss of its ``train.py``. The full 7B model is
trained with FSDP, under which :class:`FSDPClipGradNorm` clips the gradients.
"""

from __future__ import annotations

import math

import lightning
import numpy as np
import sentencepiece
import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from espnet3.contract.kinds.audio import Audio
from espnet3.systems.duplex.inference import resolve_file
from espnet3.systems.duplex.src.lm import (
    END_OF_TEXT_PADDING_ID,
    TEXT_PADDING_ID,
    ZERO_TOKEN_ID,
    LMModel,
    duplex_loss,
)
from espnet3.systems.duplex.src.mimi import MimiModel


class TextStream:
    """Dataset preprocessor: one text token per frame of the model's channel.

    A word's tokens are written one per frame from the frame its start falls
    in; if the frame before holds padding, it becomes
    ``END_OF_TEXT_PADDING_ID``. A word that starts before the previous one is
    fully written cuts it short. Every other frame holds ``TEXT_PADDING_ID``.
    Words of zero or negative duration are dropped.

    Args:
        hf_repo: Hugging Face repository of the released model, or ``None``
            for a local ``tokenizer_file``.
        revision: Repository revision.
        tokenizer_file: SentencePiece model of the text stream.
        sample_rate: Mimi's sample rate.
        frame_rate: Mimi's frame rate.
    """

    def __init__(
        self,
        hf_repo: str | None,
        revision: str | None,
        tokenizer_file: str,
        sample_rate: int,
        frame_rate: float,
    ) -> None:
        """Load the tokenizer."""
        self.tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=str(resolve_file(hf_repo, revision, tokenizer_file))
        )
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate / frame_rate)
        self.frame_rate = frame_rate

    def __call__(self, sample: dict) -> dict:
        """Return ``speech`` and its text stream ``text`` (int64, one per frame).

        ``speech`` comes back as a ``[N, 2]`` float32 array, time first, as
        the collate function pads it.

        Raises:
            ValueError: If ``speech`` is not two channels at ``sample_rate``.
        """
        speech: Audio = sample["speech"]
        if speech.rate != self.sample_rate or speech.channels != 2:
            raise ValueError(
                f"speech must be 2 channels at {self.sample_rate} Hz, got "
                f"{speech.channels} at {speech.rate} Hz."
            )
        num_frames = math.ceil(speech.array.shape[1] / self.frame_size)
        words = sorted(
            (word for word in sample["words"] if word["start"] < word["end"]),
            key=lambda word: word["start"],
        )
        text = [TEXT_PADDING_ID] * num_frames
        pending: list[int] = []
        index = 0
        for frame in range(num_frames):
            while (
                index < len(words)
                and words[index]["start"] * self.frame_rate < frame + 1
            ):
                pending = self.tokenizer.encode(words[index]["text"].strip())
                index += 1
            if pending:
                if frame > 0 and text[frame - 1] == TEXT_PADDING_ID:
                    text[frame - 1] = END_OF_TEXT_PADDING_ID
                text[frame] = pending.pop(0)
        return {
            "speech": np.ascontiguousarray(speech.array.T),
            "text": np.asarray(text, dtype=np.int64),
        }


class DuplexLM(nn.Module):
    """A released full-duplex model under the ESPnet3 training contract.

    The language model starts from the released weights, in float32, and is
    trained in full. Mimi is frozen and stays in eval mode; it encodes both
    channels in float32 inside :meth:`forward`, outside autocast.

    Args:
        lm: Keyword arguments of :class:`~.lm.LMModel`.
        mimi: Keyword arguments of :class:`~.mimi.MimiModel`.
        hf_repo: Hugging Face repository of the released model, or ``None``
            for local files.
        revision: Repository revision.
        lm_file: Released language model weights (safetensors).
        mimi_file: Released Mimi weights (safetensors).
        first_codebook_weight: Weight of the semantic audio codebook relative
            to the acoustic ones (kyutai: 100).
        text_padding_weight: Weight of the text padding targets (kyutai: 0.5).
    """

    def __init__(
        self,
        lm: dict,
        mimi: dict,
        hf_repo: str | None,
        revision: str | None,
        lm_file: str,
        mimi_file: str,
        first_codebook_weight: float,
        text_padding_weight: float,
    ) -> None:
        """Load the released weights and freeze Mimi."""
        super().__init__()
        self.mimi = MimiModel(**mimi)
        self.mimi.load_state_dict(load_file(resolve_file(hf_repo, revision, mimi_file)))
        self.mimi.requires_grad_(False)
        self.mimi.eval()
        # The 7B model is created without memory and takes the loaded tensors.
        with torch.device("meta"):
            self.lm = LMModel(**lm)
        state = load_file(resolve_file(hf_repo, revision, lm_file))
        self.lm.load_state_dict(
            {key: value.float() for key, value in state.items()}, assign=True
        )
        self.first_codebook_weight = first_codebook_weight
        self.text_padding_weight = text_padding_weight

    def train(self, mode: bool = True) -> DuplexLM:
        """Set the language model's mode; Mimi stays in eval mode."""
        super().train(mode)
        self.mimi.eval()
        return self

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, None]:
        """Return the loss of a batch, its stats and no weight.

        Args:
            speech: Both channels, ``[B, N, 2]``, zero-padded.
            speech_lengths: Samples of each item, ``[B]``.
            text: Text streams from :class:`TextStream`, ``[B, T]``,
                padded with ``ZERO_TOKEN_ID``.
            text_lengths: Frames of each item, ``[B]``.
        """
        frame_size = self.mimi.frame_size
        frames = torch.div(
            speech_lengths + frame_size - 1, frame_size, rounding_mode="floor"
        )
        if not torch.equal(frames, text_lengths):
            raise ValueError(
                f"Text streams of {text_lengths.tolist()} frames for audio of "
                f"{frames.tolist()} frames."
            )
        codes = torch.cat([text[:, None], self._encode(speech, frames)], dim=1)
        output = self.lm(codes)
        text_loss, audio_loss = duplex_loss(
            output, codes, self.first_codebook_weight, self.text_padding_weight
        )
        loss = text_loss + audio_loss
        stats = {
            "loss": loss.detach(),
            "text_loss": text_loss.detach(),
            "audio_loss": audio_loss.detach(),
        }
        return loss, stats, None

    @torch.no_grad()
    def _encode(self, speech: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        """Mimi codes of channel 0 then channel 1, ``[B, 2 * num_codebooks, T]``.

        Frames past an item's end are ``ZERO_TOKEN_ID``.
        """
        frame_size = self.mimi.frame_size
        num_frames = math.ceil(speech.shape[1] / frame_size)
        padded = F.pad(speech, (0, 0, 0, num_frames * frame_size - speech.shape[1]))
        with torch.autocast(speech.device.type, enabled=False):
            codes = torch.cat(
                [
                    self.mimi.encode(padded[:, None, :, channel].float())
                    for channel in range(2)
                ],
                dim=1,
            )
        valid = torch.arange(num_frames, device=speech.device) < frames[:, None]
        return torch.where(valid[:, None], codes, ZERO_TOKEN_ID)


class FSDPClipGradNorm(lightning.Callback):
    """Clip the gradient norm over all FSDP shards before each optimizer step.

    Lightning's FSDP precision plugin refuses ``gradient_clip_algorithm:
    norm``, since the norm must be taken over every rank's shard; the root
    FSDP module's ``clip_grad_norm_`` does that.

    Args:
        max_norm: Largest gradient norm (kyutai: 1).
    """

    def __init__(self, max_norm: float) -> None:
        """Keep the norm."""
        self.max_norm = max_norm

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        """Clip the gradients of the FSDP-wrapped model."""
        trainer.strategy.model.clip_grad_norm_(self.max_norm)
