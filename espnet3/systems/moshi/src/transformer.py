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

"""Causal Transformer shared by Mimi and the Moshi language model.

Every attention in Moshi is causal. Mimi's transformers attend to a sliding
window of ``context`` steps with RoPE and LayerScale; the language model's
temporal transformer uses RoPE, a 3000-step window and SiLU gating; its depth
transformer (the "depformer") has no positional embedding and uses a separate
set of weights for each of its steps (``weights_per_step``).

While streaming, each :class:`StreamingTransformer` keeps one ring KV cache per
layer and the absolute position of its next input. Both live in device tensors
updated in place, so that a streaming step can be captured in a CUDA graph.
Outside streaming, the whole sequence is attended at once with the same mask,
which gives the same outputs as feeding it one step per call.

The ring cache holds ``context`` steps, as in the reference implementation. A
call of ``T > 1`` steps (Mimi takes 2 per frame) past the first ``context``
steps therefore lets its first ``T - 1`` queries see one to ``T - 1`` fewer
past steps than the full-sequence mask would.

Per-step weights are selected by ``weight_offset``, a Python integer the caller
passes: the index of the depformer step, which restarts at every frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from espnet3.systems.moshi.src.streaming import StreamingModule


class RMSNorm(nn.Module):
    """RMS normalization computed in float32 (``rms_norm_f32`` in Moshi)."""

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        """Create the learnt scale ``alpha`` of shape ``[1, 1, dim]``."""
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.full((1, 1, dim), 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` of shape ``[B, T, dim]`` and return it in its dtype."""
        x_dtype = x.dtype
        x = x.to(torch.float32)
        var = self.eps + torch.mean(x**2, dim=2, keepdim=True)
        return (x * (self.alpha.to(var) * torch.rsqrt(var))).to(x_dtype)


class LayerScale(nn.Module):
    """Learnt per-channel scale of a residual branch (Touvron et al., 2021)."""

    def __init__(self, channels: int, init: float) -> None:
        """Create the scale, initialized to ``init``."""
        super().__init__()
        self.scale = nn.Parameter(torch.full((channels,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale the last dimension of ``x``."""
        return self.scale * x


def _create_norm(norm: str, dim: int) -> nn.Module:
    """Return the normalization named ``norm``: Mimi's or the language model's."""
    if norm == "layer_norm":
        return nn.LayerNorm(dim, eps=1e-5)
    if norm == "rms_norm_f32":
        return RMSNorm(dim, eps=1e-8)
    raise ValueError(f"Unknown norm: {norm}")


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, offset: torch.Tensor, max_period: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate interleaved (real, imaginary) pairs of queries and keys.

    Args:
        q: Queries of shape ``[B, H, T, D]``.
        k: Keys of shape ``[B, H, T, D]``.
        offset: Absolute position of the first time step, a 0-dim long tensor.
        max_period: Longest rotation period.

    Returns:
        The rotated queries and keys, in their input dtype.
    """
    T, D = q.shape[2], q.shape[3]
    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))
    ts = offset.float() + torch.arange(T, device=q.device, dtype=torch.float32)
    ts = ts.view(1, 1, T, 1)
    rotr = torch.cos(freqs * ts)
    roti = torch.sin(freqs * ts)

    q = q.view(*q.shape[:-1], D // 2, 2)
    k = k.view(*k.shape[:-1], D // 2, 2)
    qr, qi = q[..., 0].float(), q[..., 1].float()
    kr, ki = k[..., 0].float(), k[..., 1].float()
    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr
    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr

    dtype = q.dtype
    qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1).view(*q.shape[:-2], D)
    ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1).view(*k.shape[:-2], D)
    return qo, ko


class RingKVCache:
    """Keys and values of the last ``capacity`` steps of a stream.

    Args:
        batch_size: Number of streams.
        num_heads: Number of attention heads.
        dim_per_head: Dimension of each head.
        capacity: Number of steps kept.
        device: Device of the cache.
        dtype: dtype of the cache.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        dim_per_head: int,
        capacity: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Allocate an empty cache."""
        self.capacity = capacity
        self.cache = torch.zeros(
            (2, batch_size, num_heads, capacity, dim_per_head),
            device=device,
            dtype=dtype,
        )
        self.end_offset = torch.zeros((), device=device, dtype=torch.long)

    def complete(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append ``T`` new steps and return every cached step.

        Args:
            k: New keys of shape ``[B, H, T, D]``.
            v: New values of shape ``[B, H, T, D]``.

        Returns:
            Keys and values of shape ``[B, H, capacity, D]`` and the absolute
            position of each cache slot, ``-1`` for a slot never written.
        """
        T = k.shape[2]
        device = self.cache.device
        indexes = (torch.arange(T, device=device) + self.end_offset) % self.capacity
        self.cache[0].index_copy_(2, indexes, k)
        self.cache[1].index_copy_(2, indexes, v)

        slots = torch.arange(self.capacity, device=device, dtype=torch.long)
        last_offset = self.end_offset + T - 1
        delta = slots - last_offset % self.capacity
        positions = torch.where(
            delta <= 0,
            last_offset + delta,
            last_offset + delta - self.capacity,
        )
        self.end_offset.add_(T)
        positions = torch.where(
            slots >= self.end_offset, torch.full_like(positions, -1), positions
        )
        return self.cache[0], self.cache[1], positions


def _per_step_linear(
    x: torch.Tensor, weight: torch.Tensor, num_weights: int, weight_offset: int
) -> torch.Tensor:
    """Apply a linear map; with many weights, step ``t`` uses ``weight_offset + t``.

    Args:
        x: Input of shape ``[B, T, C]``.
        weight: ``num_weights`` stacked weights, shape ``[num_weights * O, C]``.
        num_weights: Number of stacked weights; 1 shares one weight over time.
        weight_offset: Weight index of ``x[:, 0]``.

    Returns:
        Output of shape ``[B, T, O]``.
    """
    if num_weights == 1:
        return F.linear(x, weight)
    weights = weight.view(num_weights, -1, weight.shape[-1])
    ys = [
        F.linear(x[:, t : t + 1], weights[weight_offset + t]) for t in range(x.shape[1])
    ]
    return torch.cat(ys, dim=1)


class _OutProjection(nn.Module):
    """Container giving the attention output weight its checkpoint name."""

    def __init__(self, num_weights: int, dim: int) -> None:
        """Create ``weight`` of shape ``[num_weights * dim, dim]``."""
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_weights * dim, dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


class MultiheadAttention(nn.Module):
    """Causal multi-head self-attention.

    Args:
        embed_dim: Model dimension.
        num_heads: Number of heads.
        context: Number of past steps attended to, or ``None`` for all.
        rope_max_period: RoPE maximum period, or ``None`` for no RoPE.
        num_weights: Number of per-step projection weights (1 shares them).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        context: int | None,
        rope_max_period: float | None,
        num_weights: int,
    ) -> None:
        """Create the stacked input and output projections."""
        super().__init__()
        self.num_heads = num_heads
        self.context = context
        self.rope_max_period = rope_max_period
        self.num_weights = num_weights
        self.in_proj_weight = nn.Parameter(
            torch.empty(num_weights * 3 * embed_dim, embed_dim)
        )
        nn.init.kaiming_uniform_(self.in_proj_weight, a=math.sqrt(5))
        self.out_proj = _OutProjection(num_weights, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        offset: torch.Tensor,
        kv_cache: RingKVCache | None,
        weight_offset: int,
    ) -> torch.Tensor:
        """Attend from ``x`` of shape ``[B, T, C]`` starting at position ``offset``.

        Args:
            x: Normalized input of shape ``[B, T, C]``.
            offset: Absolute position of ``x[:, 0]``, a 0-dim long tensor.
            kv_cache: Cache of the past steps while streaming, else ``None``.
            weight_offset: Index of the per-step weights of ``x[:, 0]``.

        Returns:
            Attention output of shape ``[B, T, C]``.
        """
        T = x.shape[1]
        projected = _per_step_linear(
            x, self.in_proj_weight, self.num_weights, weight_offset
        )
        q, k, v = rearrange(
            projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads
        )
        if self.rope_max_period is not None:
            q, k = apply_rope(q, k, offset, self.rope_max_period)

        if kv_cache is None:
            keys, values = k, v
            pos_k = torch.arange(T, device=x.device, dtype=torch.long)
        else:
            keys, values, pos_k = kv_cache.complete(k, v)
        pos_q = offset + torch.arange(T, device=x.device, dtype=torch.long).view(-1, 1)
        delta = pos_q - pos_k
        attn_mask = (pos_k >= 0) & (delta >= 0)
        if self.context is not None:
            attn_mask = attn_mask & (delta < self.context)
        x = F.scaled_dot_product_attention(
            q, keys, values, attn_mask[None, None], dropout_p=0.0
        )
        x = rearrange(x, "b h t d -> b t (h d)")
        return _per_step_linear(
            x, self.out_proj.weight, self.num_weights, weight_offset
        )


class ActivationGating(nn.Module):
    """Gated feed-forward block, ``linear_out(silu(a) * b)``."""

    def __init__(self, dim: int, dim_feedforward: int) -> None:
        """Size the hidden layer as Moshi does: two thirds of ``dim_feedforward``."""
        super().__init__()
        if dim_feedforward == 4 * dim:
            hidden = (21 * dim) // 8
        else:
            hidden = (2 * dim_feedforward) // 3
        self.linear_in = nn.Linear(dim, 2 * hidden, bias=False)
        self.linear_out = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to ``x`` of shape ``[B, T, dim]``."""
        x = self.linear_in(x)
        B, T, _ = x.shape
        x = x.view(B, T, 2, -1)
        x = F.silu(x[..., 0, :]) * x[..., 1, :]
        return self.linear_out(x)


class TransformerLayer(nn.Module):
    """Pre-norm Transformer layer.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        dim_feedforward: Feed-forward dimension.
        context: Attention window in steps, or ``None`` for unbounded.
        rope_max_period: RoPE maximum period, or ``None`` for no RoPE.
        norm: ``"layer_norm"`` (Mimi) or ``"rms_norm_f32"`` (language model).
        layer_scale: Initial LayerScale value, or ``None`` for no LayerScale.
        gating: ``"none"`` for a GELU MLP (Mimi) or ``"silu"`` for gating.
        weights_per_step: Number of per-step weight sets, 0 to share them.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        context: int | None,
        rope_max_period: float | None,
        norm: str,
        layer_scale: float | None,
        gating: str,
        weights_per_step: int,
    ) -> None:
        """Create the attention, feed-forward block, norms and scales."""
        super().__init__()
        if weights_per_step > 0:
            num_weights = weights_per_step
        else:
            num_weights = 1
        self.num_weights = num_weights
        self.gating_type = gating
        self.self_attn = MultiheadAttention(
            d_model, num_heads, context, rope_max_period, num_weights
        )
        self.norm1 = _create_norm(norm, d_model)
        self.norm2 = _create_norm(norm, d_model)

        if gating == "none":
            if weights_per_step > 0:
                raise ValueError("weights_per_step requires gating.")
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
        elif gating == "silu":
            if weights_per_step > 0:
                self.gating = nn.ModuleList(
                    [
                        ActivationGating(d_model, dim_feedforward)
                        for _ in range(weights_per_step)
                    ]
                )
            else:
                self.gating = ActivationGating(d_model, dim_feedforward)
        else:
            raise ValueError(f"Unknown gating: {gating}")

        if layer_scale is None:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
        else:
            self.layer_scale_1 = LayerScale(d_model, layer_scale)
            self.layer_scale_2 = LayerScale(d_model, layer_scale)

    def _feed_forward(self, x: torch.Tensor, weight_offset: int) -> torch.Tensor:
        """Apply the feed-forward block; step ``t`` uses ``weight_offset + t``."""
        if self.gating_type == "none":
            return self.linear2(F.gelu(self.linear1(x)))
        if self.num_weights == 1:
            return self.gating(x)
        ys = [
            self.gating[weight_offset + t](x[:, t : t + 1]) for t in range(x.shape[1])
        ]
        return torch.cat(ys, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        offset: torch.Tensor,
        kv_cache: RingKVCache | None,
        weight_offset: int,
    ) -> torch.Tensor:
        """Run the layer on ``x`` ``[B, T, d_model]`` at position ``offset``."""
        update = self.self_attn(self.norm1(x), offset, kv_cache, weight_offset)
        x = x + self.layer_scale_1(update)
        x = x + self.layer_scale_2(self._feed_forward(self.norm2(x), weight_offset))
        return x


@dataclass
class _TransformerStream:
    """Absolute position of the next input step and one KV cache per layer."""

    offset: torch.Tensor
    kv_caches: list[RingKVCache]


class StreamingTransformer(StreamingModule):
    """Stack of causal :class:`TransformerLayer`.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        num_layers: Number of layers.
        dim_feedforward: Feed-forward dimension.
        context: Attention window in steps, or ``None`` for unbounded. It is
            also the KV cache capacity while streaming.
        rope_max_period: RoPE maximum period, or ``None`` for no RoPE.
        norm: ``"layer_norm"`` or ``"rms_norm_f32"``.
        layer_scale: Initial LayerScale value, or ``None``.
        gating: ``"none"`` or ``"silu"``.
        weights_per_step: Number of per-step weight sets, 0 to share them. With
            ``context`` unset it is the KV cache capacity while streaming.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        context: int | None,
        rope_max_period: float | None,
        norm: str,
        layer_scale: float | None,
        gating: str,
        weights_per_step: int,
    ) -> None:
        """Create the layers."""
        super().__init__()
        self.num_heads = num_heads
        self.dim_per_head = d_model // num_heads
        self.context = context
        self.weights_per_step = weights_per_step
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model,
                    num_heads,
                    dim_feedforward,
                    context,
                    rope_max_period,
                    norm,
                    layer_scale,
                    gating,
                    weights_per_step,
                )
                for _ in range(num_layers)
            ]
        )

    def init_stream(self, batch_size: int) -> _TransformerStream:
        """Return position 0 and an empty KV cache for every layer."""
        if self.context is not None:
            capacity = self.context
        elif self.weights_per_step > 0:
            capacity = self.weights_per_step
        else:
            raise RuntimeError("Streaming needs a context or weights_per_step.")
        weight = self.layers[0].self_attn.in_proj_weight
        kv_caches = [
            RingKVCache(
                batch_size,
                self.num_heads,
                self.dim_per_head,
                capacity,
                device=weight.device,
                dtype=weight.dtype,
            )
            for _ in self.layers
        ]
        offset = torch.zeros((), device=weight.device, dtype=torch.long)
        return _TransformerStream(offset=offset, kv_caches=kv_caches)

    def forward(self, x: torch.Tensor, weight_offset: int = 0) -> torch.Tensor:
        """Run all layers on ``x`` of shape ``[B, T, d_model]``.

        Args:
            x: Input sequence.
            weight_offset: Index of the per-step weights of ``x[:, 0]``.

        Returns:
            Output sequence of shape ``[B, T, d_model]``.
        """
        if self.stream is None:
            offset = torch.zeros((), device=x.device, dtype=torch.long)
            kv_caches = [None] * len(self.layers)
        else:
            offset = self.stream.offset
            kv_caches = self.stream.kv_caches
        for layer, kv_cache in zip(self.layers, kv_caches):
            x = layer(x, offset, kv_cache, weight_offset)
        if self.stream is not None:
            self.stream.offset.add_(x.shape[1])
        return x


class ProjectedTransformer(nn.Module):
    """Mimi's transformer, applied to latents of shape ``[B, C, T]``.

    The attribute name ``transformer`` follows the released checkpoint. Mimi
    uses equal input, model and output dimensions, so no projection is needed.
    """

    def __init__(self, **kwargs) -> None:
        """Create the inner :class:`StreamingTransformer` from ``kwargs``."""
        super().__init__()
        self.transformer = StreamingTransformer(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the transformer over the time axis of ``x`` of shape ``[B, C, T]``."""
        return self.transformer(x.transpose(1, 2)).transpose(1, 2)
