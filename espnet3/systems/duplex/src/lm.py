# MIT License
#
# Copyright (c) Kyutai
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Adapted from https://github.com/kyutai-labs/moshi (moshi==0.2.13).

"""Moshi language model and its frame-by-frame generator (inference only).

Moshi models 17 token streams at 12.5 Hz: its inner-monologue text (stream 0),
its own 8 Mimi codebooks (streams 1-8) and the user's 8 Mimi codebooks
(streams 9-16). At each frame the temporal transformer reads the sum of the
embeddings of all streams of the previous frame; the text token is sampled
from its output, then the depth transformer ("depformer") samples Moshi's 8
audio tokens one after the other. Acoustic codebooks are delayed by
``delays`` frames relative to the text and semantic tokens.

Attribute names follow the released checkpoint.
"""

from __future__ import annotations

import torch
from torch import nn

from espnet3.systems.duplex.src.cuda_graph import CUDAGraphed
from espnet3.systems.duplex.src.streaming import start_streaming, stop_streaming
from espnet3.systems.duplex.src.transformer import RMSNorm, StreamingTransformer

#: Token id of a cache entry not generated yet.
UNGENERATED_TOKEN_ID = -2


class LMModel(nn.Module):
    """Temporal and depth transformers of Moshi.

    Args:
        dim: Temporal transformer dimension.
        num_heads: Temporal transformer heads.
        num_layers: Temporal transformer layers.
        hidden_scale: Feed-forward dimension of the temporal transformer, as a
            multiple of ``dim``.
        context: Attention window of the temporal transformer, in frames.
        max_period: RoPE maximum period of the temporal transformer.
        text_card: Text vocabulary size.
        card: Audio codebook size.
        n_q: Number of audio streams (Moshi's and the user's).
        dep_q: Number of audio streams Moshi generates.
        delays: Delay of each of the ``1 + n_q`` streams, in frames.
        depformer_dim: Depformer dimension.
        depformer_dim_feedforward: Depformer feed-forward dimension.
        depformer_num_heads: Depformer heads.
        depformer_num_layers: Depformer layers.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        hidden_scale: float,
        context: int,
        max_period: float,
        text_card: int,
        card: int,
        n_q: int,
        dep_q: int,
        delays: list[int],
        depformer_dim: int,
        depformer_dim_feedforward: int,
        depformer_num_heads: int,
        depformer_num_layers: int,
    ) -> None:
        """Create the embeddings, both transformers and the output heads."""
        super().__init__()
        if len(delays) != n_q + 1:
            raise ValueError(f"Expected {n_q + 1} delays, got {len(delays)}.")
        self.text_card = text_card
        self.card = card
        self.n_q = n_q
        self.dep_q = dep_q
        self.delays = list(delays)

        # The extra embedding row is the initial token (id ``card`` or ``text_card``).
        self.emb = nn.ModuleList([nn.Embedding(card + 1, dim) for _ in range(n_q)])
        self.text_emb = nn.Embedding(text_card + 1, dim)
        self.transformer = StreamingTransformer(
            d_model=dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=int(hidden_scale * dim),
            context=context,
            rope_max_period=max_period,
            norm="rms_norm_f32",
            layer_scale=None,
            gating="silu",
            weights_per_step=0,
        )
        self.out_norm = RMSNorm(dim)
        self.text_linear = nn.Linear(dim, text_card, bias=False)

        self.depformer_in = nn.ModuleList(
            [nn.Linear(dim, depformer_dim, bias=False) for _ in range(dep_q)]
        )
        self.depformer_text_emb = nn.Embedding(text_card + 1, depformer_dim)
        # Moshi's last audio token is never a depformer input.
        self.depformer_emb = nn.ModuleList(
            [nn.Embedding(card + 1, depformer_dim) for _ in range(dep_q - 1)]
        )
        self.depformer = StreamingTransformer(
            d_model=depformer_dim,
            num_heads=depformer_num_heads,
            num_layers=depformer_num_layers,
            dim_feedforward=depformer_dim_feedforward,
            context=None,
            rope_max_period=None,
            norm="rms_norm_f32",
            layer_scale=None,
            gating="silu",
            weights_per_step=dep_q,
        )
        self.linears = nn.ModuleList(
            [nn.Linear(depformer_dim, card, bias=False) for _ in range(dep_q)]
        )

    @property
    def num_codebooks(self) -> int:
        """Number of token streams, text included."""
        return self.n_q + 1

    def initial_tokens(self, device: torch.device) -> torch.Tensor:
        """Return the tokens fed before a stream starts, shape ``[1, 1 + n_q, 1]``."""
        tokens = torch.full(
            (1, self.num_codebooks, 1), self.card, device=device, dtype=torch.long
        )
        tokens[:, 0] = self.text_card
        return tokens

    def forward_text(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the temporal transformer.

        Args:
            sequence: Tokens of shape ``[B, 1 + n_q, S]``.

        Returns:
            Normalized hidden states ``[B, S, dim]`` and text logits
            ``[B, 1, S, text_card]``.
        """
        input_ = self.emb[0](sequence[:, 1])
        for index in range(1, self.n_q):
            input_ = input_ + self.emb[index](sequence[:, index + 1])
        input_ = input_ + self.text_emb(sequence[:, 0])
        transformer_out = self.out_norm(self.transformer(input_))
        text_logits = self.text_linear(transformer_out)[:, None]
        return transformer_out, text_logits

    def forward_depformer(
        self, index: int, token: torch.Tensor, transformer_out: torch.Tensor
    ) -> torch.Tensor:
        """Run one depformer step while the depformer is streaming.

        Args:
            index: Index of the audio codebook to predict.
            token: Previous token, shape ``[B, 1]``: the text token for
                ``index == 0``, else the audio token of codebook ``index - 1``.
            transformer_out: Temporal transformer output ``[B, 1, dim]``.

        Returns:
            Logits of codebook ``index``, shape ``[B, 1, 1, card]``.
        """
        if index == 0:
            token_emb = self.depformer_text_emb(token)
        else:
            token_emb = self.depformer_emb[index - 1](token)
        depformer_input = self.depformer_in[index](transformer_out) + token_emb
        output = self.depformer(depformer_input, weight_offset=index)
        logits = self.linears[index](output)
        return logits[:, None]


def sample_token(logits: torch.Tensor, temp: float, top_k: int) -> torch.Tensor:
    """Sample a token from the top-``top_k`` of ``softmax(logits / temp)``.

    A temperature of 0 takes the argmax. Sampling uses the exponential race
    of the reference implementation (``argmax(p / Exp(1))``), so that it draws
    the same random numbers.

    Args:
        logits: Logits of shape ``[*, card]``.
        temp: Temperature.
        top_k: Number of candidates kept.

    Returns:
        Tokens of shape ``[*]``.
    """
    if temp <= 0.0:
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits / temp, dim=-1)
    probs, indices = torch.topk(probs, min(top_k, probs.shape[-1]), dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    race = flat / torch.empty_like(flat).exponential_(1)
    choice = race.argmax(dim=-1, keepdim=True).reshape(*probs.shape[:-1], 1)
    return indices.gather(-1, choice)[..., 0]


class LMGen:
    """Generate Moshi's streams frame by frame while the user streams in.

    Args:
        lm: The language model, in eval mode.
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
            raise RuntimeError("LMGen needs the language model in eval mode.")
        self.lm = lm
        self.temp = temp
        self.temp_text = temp_text
        self.top_k = top_k
        self.top_k_text = top_k_text
        self.cuda_graph = cuda_graph
        self.max_delay = max(lm.delays)
        self.cache: torch.Tensor | None = None
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
            ValueError: If CUDA graphs are asked for on a device other than CUDA,
                where capture would record nothing and replays would do nothing.
        """
        if self.cache is not None:
            raise RuntimeError("LMGen is already streaming.")
        if self.cuda_graph and device.type != "cuda":
            raise ValueError(f"CUDA graphs need a CUDA device, got {device}.")
        # A ring buffer long enough to hold every token still waiting on its delay.
        self.cache = torch.full(
            (batch_size, self.lm.num_codebooks, self.max_delay + 2),
            UNGENERATED_TOKEN_ID,
            device=device,
            dtype=torch.long,
        )
        self.delays = torch.tensor(self.lm.delays, device=device, dtype=torch.long)
        self.offset = 0
        start_streaming(self.lm.transformer, batch_size)
        # Graphs capture the stream state just created, so each stream gets its own.
        if self.cuda_graph:
            self._forward_text = CUDAGraphed(self.lm.forward_text)
            self._depformer = CUDAGraphed(self._depformer_step)
        else:
            self._forward_text = self.lm.forward_text
            self._depformer = self._depformer_step

    def stop(self) -> None:
        """Close the stream and drop its graphs.

        Stops the depformer too, in case a frame was interrupted inside it.
        """
        stop_streaming(self.lm)
        self.cache = None
        self.delays = None
        self.offset = 0
        self._forward_text = None
        self._depformer = None

    @torch.no_grad()
    def step(self, user_codes: torch.Tensor) -> torch.Tensor | None:
        """Consume one frame of user codes and generate one frame of Moshi.

        Args:
            user_codes: The user's Mimi codes of this frame, ``[B, n_q - dep_q, 1]``.

        Returns:
            ``None`` during the first ``max_delay`` frames, then Moshi's text
            and audio tokens of frame ``offset - max_delay``, shape
            ``[B, 1 + dep_q, 1]``, with the delays undone.
        """
        lm = self.lm
        cache = self.cache
        if cache is None:
            raise RuntimeError("Call start() before step().")
        dep_q = lm.dep_q
        capacity = cache.shape[2]
        delays = self.delays

        # The user's codes of this frame are read once their delay has passed.
        write = (self.offset + delays[dep_q + 1 :]) % capacity
        cache[:, dep_q + 1 :].scatter_(
            -1, write.view(1, -1, 1).expand(cache.shape[0], -1, 1), user_codes
        )

        # Streams whose delay has not passed yet read the initial token.
        is_init = self.offset <= delays.view(-1, 1)
        position = self.offset % capacity
        input_ = cache[:, :, position : position + 1]
        input_ = torch.where(is_init, lm.initial_tokens(cache.device), input_)

        transformer_out, text_logits = self._forward_text(input_)
        text_token = sample_token(text_logits.float(), self.temp_text, self.top_k_text)
        text_token = text_token[:, 0, 0]
        audio_tokens = self._depformer(text_token, transformer_out)

        self.offset += 1
        position = self.offset % capacity
        cache[:, 0, position] = text_token
        cache[:, 1 : dep_q + 1, position] = audio_tokens
        if self.offset <= self.max_delay:
            return None
        read = (self.offset - self.max_delay + delays[: dep_q + 1]) % capacity
        return cache.gather(2, read.view(1, -1, 1).expand(cache.shape[0], -1, 1))

    def _depformer_step(
        self, text_token: torch.Tensor, transformer_out: torch.Tensor
    ) -> torch.Tensor:
        """Sample Moshi's ``dep_q`` audio tokens of one frame, ``[B, dep_q]``."""
        lm = self.lm
        tokens = []
        previous = text_token
        start_streaming(lm.depformer, text_token.shape[0])
        for index in range(lm.dep_q):
            logits = lm.forward_depformer(index, previous[:, None], transformer_out)
            previous = sample_token(logits.float(), self.temp, self.top_k)[:, 0, 0]
            tokens.append(previous)
        stop_streaming(lm.depformer)
        return torch.stack(tokens, dim=1)
