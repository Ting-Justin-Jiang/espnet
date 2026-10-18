# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# Adapted from https://github.com/NVIDIA/personaplex (commit 3428dfd,
# moshi/moshi/models/lm.py and moshi/moshi/offline.py).

"""PersonaPlex's frame-by-frame generator, with its voice and role prompts.

PersonaPlex (Roy et al., 2026, arXiv:2602.06053) fine-tunes Moshi so that a
voice prompt and a text ("role") prompt, fed before the conversation, set
its voice and persona. Its language model is Moshi's with a depformer of 16
codebooks: it also predicts the user's 8 codebooks, which are teacher-forced
at inference.

Its generator differs from Moshi's :class:`~.lm.LMGen` in two ways:

- Any stream of a frame can be teacher-forced (``provided``): the prompts
  force all of them, a conversation forces the user's audio.
- The call that receives frame ``t`` runs the model on frame ``t - 1`` to
  predict frame ``t``. Moshi's generator runs on frame ``t`` when it
  receives it, so PersonaPlex answers one frame (80 ms) later.

Before the conversation, :meth:`PersonaPlexGen.run_prompts` steps through
the voice prompt, 0.5 s of silence, the role prompt as text tokens and 0.5 s
of silence, with the user's channel holding a 440 Hz tone.
"""

from __future__ import annotations

import torch

from espnet3.systems.duplex.src.cuda_graph import CUDAGraphed
from espnet3.systems.duplex.src.lm import LMModel, sample_token
from espnet3.systems.duplex.src.streaming import start_streaming, stop_streaming

#: Mimi codes of a silent frame, forced on PersonaPlex's channel in the prompts.
SILENCE_TOKENS = (948, 243, 1178, 546, 1736, 1030, 1978, 2008)
#: Mimi codes of a 440 Hz frame, forced on the user's channel in the prompts.
SINE_TOKENS = (430, 1268, 381, 1611, 1095, 1495, 56, 472)
#: Text padding token, forced on the text stream outside the role prompt.
TEXT_PADDING_ID = 3
#: Silence after the voice prompt and after the role prompt, in frames.
PROMPT_SILENCE_FRAMES = 6


def wrap_role_prompt(text: str) -> str:
    """Wrap a role prompt in the ``<system>`` tags PersonaPlex was trained with."""
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


class PersonaPlexGen:
    """Generate PersonaPlex's streams frame by frame, with teacher forcing.

    Args:
        lm: The language model (``dep_q == n_q``), in eval mode.
        temp: Audio sampling temperature.
        temp_text: Text sampling temperature.
        top_k: Audio top-k.
        top_k_text: Text top-k.
        cuda_graph: Whether to replay the temporal transformer and the depformer
            of each frame from CUDA graphs.
    """

    def __init__(
        self,
        lm: LMModel,
        temp: float,
        temp_text: float,
        top_k: int,
        top_k_text: int,
        cuda_graph: bool,
    ) -> None:
        """Keep the model and the sampling parameters."""
        if lm.training:
            raise RuntimeError("PersonaPlexGen needs the language model in eval mode.")
        if lm.dep_q != lm.n_q:
            raise ValueError("PersonaPlex's depformer predicts every audio stream.")
        self.lm = lm
        self.temp = temp
        self.temp_text = temp_text
        self.top_k = top_k
        self.top_k_text = top_k_text
        self.cuda_graph = cuda_graph
        self.max_delay = max(lm.delays)
        self.num_user = lm.n_q // 2
        self.cache: torch.Tensor | None = None
        self.provided: torch.Tensor | None = None
        self.initial: torch.Tensor | None = None
        self.delays: torch.Tensor | None = None
        self.offset = 0
        self._forward_text = None
        self._depformer = None

    @property
    def is_streaming(self) -> bool:
        """Whether a stream is open."""
        return self.cache is not None

    def start(self, batch_size: int, device: torch.device) -> None:
        """Open a stream: empty token cache and temporal transformer state.

        Raises:
            RuntimeError: If a stream is open.
            ValueError: If CUDA graphs are asked for on a device other than CUDA.
        """
        if self.cache is not None:
            raise RuntimeError("PersonaPlexGen is already streaming.")
        if self.cuda_graph and device.type != "cuda":
            raise ValueError(f"CUDA graphs need a CUDA device, got {device}.")
        shape = (batch_size, self.lm.num_codebooks, self.max_delay + 3)
        self.cache = torch.full(shape, -2, device=device, dtype=torch.long)
        self.provided = torch.zeros(shape, device=device, dtype=torch.bool)
        self.initial = self.lm.initial_tokens(device)
        self.delays = torch.tensor(self.lm.delays, device=device, dtype=torch.long)
        self.offset = 0
        start_streaming(self.lm.transformer, batch_size)
        if self.cuda_graph:
            self._forward_text = CUDAGraphed(self.lm.forward_text)
            self._depformer = CUDAGraphed(self._depformer_step)
        else:
            self._forward_text = self.lm.forward_text
            self._depformer = self._depformer_step

    def stop(self) -> None:
        """Close the stream and drop its graphs."""
        stop_streaming(self.lm)
        self.cache = None
        self.provided = None
        self.initial = None
        self.delays = None
        self.offset = 0
        self._forward_text = None
        self._depformer = None

    def _force(self, first: int, codes: torch.Tensor) -> None:
        """Write ``codes`` ``[B, K, 1]`` into streams ``first..first + K``, delayed."""
        capacity = self.cache.shape[2]
        for index in range(codes.shape[1]):
            stream = first + index
            position = (self.offset + self.lm.delays[stream]) % capacity
            self.cache[:, stream, position] = codes[:, index, 0]
            self.provided[:, stream, position] = True

    def _prepare(
        self,
        user_codes: torch.Tensor,
        agent_codes: torch.Tensor | None,
        text_token: int | None,
    ) -> tuple[int, int] | None:
        """Force the given tokens; return the input and target cache positions.

        Returns ``None`` for the first frame, which only fills the cache.
        """
        capacity = self.cache.shape[2]
        self._force(1 + self.lm.n_q - self.num_user, user_codes)
        if agent_codes is not None:
            self._force(1, agent_codes)
        if text_token is not None:
            position = (self.offset + self.lm.delays[0]) % capacity
            self.cache[:, 0, position] = text_token
            self.provided[:, 0, position] = True
        # Streams whose delay has not passed yet hold the initial token.
        for stream, delay in enumerate(self.lm.delays):
            if self.offset <= delay:
                self.cache[:, stream, self.offset % capacity] = self.initial[
                    :, stream, 0
                ]
                self.provided[:, stream, self.offset % capacity] = True
        if self.offset == 0:
            self.cache[:, :, 0] = self.initial[:, :, 0]
            self.offset += 1
            return None
        return (self.offset - 1) % capacity, self.offset % capacity

    def _finish(
        self,
        transformer_out: torch.Tensor,
        text_logits: torch.Tensor,
        input_position: int,
        target_position: int,
    ) -> torch.Tensor | None:
        """Sample the target frame where it is not forced; return a delayed frame."""
        cache = self.cache
        provided = self.provided
        target = cache[:, :, target_position]
        forced = provided[:, :, target_position]
        text = sample_token(text_logits.float(), self.temp_text, self.top_k_text)
        text = text[:, 0, 0]
        next_text = torch.where(forced[:, 0], target[:, 0], text)
        audio = self._depformer(
            next_text, transformer_out, target[:, 1:], forced[:, 1:]
        )

        provided[:, :, input_position] = False
        cache[:, 0, target_position] = torch.where(
            ~provided[:, 0, target_position], text, cache[:, 0, target_position]
        )
        cache[:, 1:, target_position] = torch.where(
            ~provided[:, 1:, target_position], audio, cache[:, 1:, target_position]
        )
        if self.offset <= self.max_delay:
            self.offset += 1
            return None
        read = (self.offset - self.max_delay + self.delays) % cache.shape[2]
        out = cache.gather(2, read.view(1, -1, 1).expand(cache.shape[0], -1, 1))
        self.offset += 1
        return out

    @torch.no_grad()
    def step(
        self,
        user_codes: torch.Tensor,
        agent_codes: torch.Tensor | None = None,
        text_token: int | None = None,
    ) -> torch.Tensor | None:
        """Consume one frame of the user, optionally forcing PersonaPlex's own.

        Args:
            user_codes: The user's Mimi codes of this frame, ``[B, n_q / 2, 1]``.
            agent_codes: PersonaPlex's codes to force, ``[B, n_q / 2, 1]``.
            text_token: Text token to force.

        Returns:
            ``None`` until the longest delay has passed, then every stream of
            frame ``offset - max_delay``, ``[B, 1 + n_q, 1]``.
        """
        if self.cache is None:
            raise RuntimeError("Call start() before step().")
        positions = self._prepare(user_codes, agent_codes, text_token)
        if positions is None:
            return None
        input_position, target_position = positions
        input_ = self.cache[:, :, input_position : input_position + 1]
        transformer_out, text_logits = self._forward_text(input_)
        return self._finish(
            transformer_out, text_logits, input_position, target_position
        )

    @torch.no_grad()
    def step_embeddings(self, embeddings: torch.Tensor) -> None:
        """Advance one frame on precomputed input embeddings ``[B, 1, dim]``.

        The released voice prompts are stored as these embeddings. The tokens
        written to the cache are placeholders that :meth:`run_prompts`
        overwrites with the prompt's own cache.
        """
        placeholder = self.initial[:, 1 : 1 + self.num_user]
        positions = self._prepare(placeholder, placeholder, TEXT_PADDING_ID)
        if positions is None:
            positions = self._prepare(placeholder, placeholder, TEXT_PADDING_ID)
        transformer_out, text_logits = self.lm.forward_embeddings(embeddings)
        self._finish(transformer_out, text_logits, *positions)

    @torch.no_grad()
    def run_prompts(self, voice_prompt: dict, role_tokens: list[int]) -> None:
        """Step through the voice prompt, silence, role prompt and silence.

        Args:
            voice_prompt: A released voice prompt: ``embeddings``
                ``[N, 1, 1, dim]`` and the token ``cache`` it ends with.
            role_tokens: Text tokens of the role prompt, ``<system>`` tags included.
        """
        if len(SILENCE_TOKENS) != self.num_user or len(SINE_TOKENS) != self.num_user:
            raise ValueError(
                f"The prompt frames have {len(SILENCE_TOKENS)} codes, the model "
                f"{self.num_user} codebooks per speaker."
            )
        device = self.cache.device
        silence = torch.tensor(SILENCE_TOKENS, device=device).view(1, -1, 1)
        sine = torch.tensor(SINE_TOKENS, device=device).view(1, -1, 1)
        for embeddings in voice_prompt["embeddings"].to(device):
            self.step_embeddings(embeddings)
        self.cache.copy_(voice_prompt["cache"].to(device))
        for _ in range(PROMPT_SILENCE_FRAMES):
            self.step(sine, silence, TEXT_PADDING_ID)
        for token in role_tokens:
            self.step(sine, silence, token)
        for _ in range(PROMPT_SILENCE_FRAMES):
            self.step(sine, silence, TEXT_PADDING_ID)

    def _depformer_step(
        self,
        text_token: torch.Tensor,
        transformer_out: torch.Tensor,
        audio_tokens: torch.Tensor,
        audio_forced: torch.Tensor,
    ) -> torch.Tensor:
        """Sample every audio codebook of one frame, ``[B, n_q]``.

        A forced codebook still draws its sample but feeds the forced token to
        the next one.
        """
        lm = self.lm
        tokens = []
        previous = text_token
        start_streaming(lm.depformer, text_token.shape[0])
        for index in range(lm.dep_q):
            logits = lm.forward_depformer(index, previous[:, None], transformer_out)
            sampled = sample_token(logits.float(), self.temp, self.top_k)[:, 0, 0]
            previous = torch.where(
                audio_forced[:, index], audio_tokens[:, index], sampled
            )
            tokens.append(sampled)
        stop_streaming(lm.depformer)
        return torch.stack(tokens, dim=1)
