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

"""Mimi, the streaming neural audio codec of Moshi (inference only).

24 kHz audio is encoded to 8 codebooks at 12.5 Hz: a causal SEANet encoder
(hop 960, 25 Hz), a causal Transformer, a learnt stride-2 downsampling to
12.5 Hz and a split residual vector quantizer. Decoding mirrors these steps.
Every convolution is causal, so encoding a signal frame by frame (1920
samples, 80 ms) while streaming computes the same function as encoding it at
once, up to the 10 s attention window of the transformers (see
:mod:`espnet3.systems.moshi.src.transformer`). Streaming states are device
tensors updated in place, so that a frame can be captured in a CUDA graph.
Attribute names follow the released checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from espnet3.systems.moshi.src.quantizer import SplitResidualVectorQuantizer
from espnet3.systems.moshi.src.streaming import StreamingModule
from espnet3.systems.moshi.src.transformer import ProjectedTransformer


class _NormConv1d(nn.Module):
    """Container giving a ``Conv1d`` its checkpoint name (``conv.conv``)."""

    def __init__(self, *args, **kwargs) -> None:
        """Create the convolution."""
        super().__init__()
        self.conv = nn.Conv1d(*args, **kwargs)


class _NormConvTranspose1d(nn.Module):
    """Container giving a ``ConvTranspose1d`` its checkpoint name."""

    def __init__(self, *args, **kwargs) -> None:
        """Create the transposed convolution."""
        super().__init__()
        self.convtr = nn.ConvTranspose1d(*args, **kwargs)


@dataclass
class _ConvStream:
    """Input samples kept for the next call, and whether none was seen yet."""

    previous: torch.Tensor
    first: torch.Tensor


class StreamingConv1d(StreamingModule):
    """Causal 1D convolution, left-padded with ``kernel - stride`` samples.

    The padding is zeros (``pad_mode="constant"``) or copies of the first
    input sample (``pad_mode="replicate"``). While streaming, the padding of
    a call is the end of the previous call's input.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Kernel size.
        stride: Stride; the input length must be a multiple of it.
        dilation: Dilation.
        groups: Groups.
        bias: Whether the convolution has a bias.
        pad_mode: ``"constant"`` or ``"replicate"``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        pad_mode: str = "constant",
    ) -> None:
        """Create the convolution."""
        super().__init__()
        if pad_mode not in ("constant", "replicate"):
            raise ValueError(f"Unknown pad_mode: {pad_mode}")
        self.pad_mode = pad_mode
        self.conv = _NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.stride = stride
        self.padding = (kernel_size - 1) * dilation + 1 - stride

    def init_stream(self, batch_size: int) -> _ConvStream:
        """Return zero padding and ``first=True``."""
        weight = self.conv.conv.weight
        previous = torch.zeros(
            batch_size,
            self.conv.conv.in_channels,
            self.padding,
            device=weight.device,
            dtype=weight.dtype,
        )
        first = torch.ones((), device=weight.device, dtype=torch.bool)
        return _ConvStream(previous=previous, first=first)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve ``x`` of shape ``[B, C, T]``, ``T`` a multiple of the stride."""
        if x.shape[-1] % self.stride != 0:
            raise ValueError(
                f"Input length {x.shape[-1]} is not a multiple of {self.stride}."
            )
        if self.stream is None:
            state = self.init_stream(x.shape[0])
        else:
            state = self.stream
        if self.padding > 0:
            if self.pad_mode == "replicate":
                state.previous.copy_(
                    torch.where(state.first, x[..., :1], state.previous)
                )
            x = torch.cat([state.previous, x], dim=-1)
            state.previous.copy_(x[..., -self.padding :])
            state.first.fill_(False)
        return self.conv.conv(x)


@dataclass
class _ConvTransposeStream:
    """Output overlap of the previous call, without bias, to add to the next."""

    partial: torch.Tensor


class StreamingConvTranspose1d(StreamingModule):
    """Causal 1D transposed convolution.

    The last ``kernel - stride`` output samples overlap with the next input
    frame. Outside streaming they are dropped; while streaming they are added
    to the start of the next call's output.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Kernel size.
        stride: Stride.
        groups: Groups.
        bias: Whether the convolution has a bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """Create the transposed convolution."""
        super().__init__()
        self.convtr = _NormConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, groups=groups, bias=bias
        )
        self.overlap = kernel_size - stride

    def init_stream(self, batch_size: int) -> _ConvTransposeStream:
        """Return a zero overlap."""
        weight = self.convtr.convtr.weight
        partial = torch.zeros(
            batch_size,
            self.convtr.convtr.out_channels,
            self.overlap,
            device=weight.device,
            dtype=weight.dtype,
        )
        return _ConvTransposeStream(partial=partial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample ``x`` of shape ``[B, C, T]`` to ``[B, C', T * stride]``."""
        y = self.convtr.convtr(x)
        if self.overlap == 0:
            return y
        if self.stream is None:
            return y[..., : y.shape[-1] - self.overlap]
        y[..., : self.overlap] += self.stream.partial
        for_partial = y[..., -self.overlap :]
        bias = self.convtr.convtr.bias
        if bias is not None:
            for_partial = for_partial - bias[:, None]
        self.stream.partial.copy_(for_partial)
        return y[..., : -self.overlap]


class SEANetResnetBlock(nn.Module):
    """Residual block: ``x + conv1x1(elu(conv(elu(x))))``, halving the hidden width.

    Args:
        dim: Channels.
        kernel_size: Kernel size of the first convolution.
        dilation: Dilation of the first convolution.
        compress: Hidden width reduction.
    """

    def __init__(
        self, dim: int, kernel_size: int, dilation: int, compress: int
    ) -> None:
        """Create the two convolutions."""
        super().__init__()
        hidden = dim // compress
        self.block = nn.Sequential(
            nn.ELU(alpha=1.0),
            StreamingConv1d(dim, hidden, kernel_size, dilation=dilation),
            nn.ELU(alpha=1.0),
            StreamingConv1d(hidden, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block."""
        return x + self.block(x)


class SEANetEncoder(nn.Module):
    """Causal SEANet encoder.

    Args:
        channels: Audio channels.
        dimension: Latent dimension.
        n_filters: Width of the first block.
        n_residual_layers: Residual blocks per resolution.
        ratios: Upsampling ratios, in decoder order; the encoder uses them reversed.
        kernel_size: Kernel size of the first convolution.
        last_kernel_size: Kernel size of the last convolution.
        residual_kernel_size: Kernel size of the residual blocks.
        dilation_base: Dilation growth across the residual blocks.
        compress: Hidden width reduction of the residual blocks.
    """

    def __init__(
        self,
        channels: int,
        dimension: int,
        n_filters: int,
        n_residual_layers: int,
        ratios: list[int],
        kernel_size: int,
        last_kernel_size: int,
        residual_kernel_size: int,
        dilation_base: int,
        compress: int,
    ) -> None:
        """Create the convolution stack."""
        super().__init__()
        self.hop_length = math.prod(ratios)
        mult = 1
        model: list[nn.Module] = [
            StreamingConv1d(channels, mult * n_filters, kernel_size)
        ]
        for ratio in reversed(ratios):
            for j in range(n_residual_layers):
                model.append(
                    SEANetResnetBlock(
                        mult * n_filters,
                        residual_kernel_size,
                        dilation_base**j,
                        compress,
                    )
                )
            model.append(nn.ELU(alpha=1.0))
            model.append(
                StreamingConv1d(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                )
            )
            mult *= 2
        model.append(nn.ELU(alpha=1.0))
        model.append(StreamingConv1d(mult * n_filters, dimension, last_kernel_size))
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode audio ``[B, channels, T]`` to latents ``[B, dimension, T / hop]``."""
        return self.model(x)


class SEANetDecoder(nn.Module):
    """Causal SEANet decoder (arguments as in :class:`SEANetEncoder`)."""

    def __init__(
        self,
        channels: int,
        dimension: int,
        n_filters: int,
        n_residual_layers: int,
        ratios: list[int],
        kernel_size: int,
        last_kernel_size: int,
        residual_kernel_size: int,
        dilation_base: int,
        compress: int,
    ) -> None:
        """Create the convolution stack."""
        super().__init__()
        mult = 2 ** len(ratios)
        model: list[nn.Module] = [
            StreamingConv1d(dimension, mult * n_filters, kernel_size)
        ]
        for ratio in ratios:
            model.append(nn.ELU(alpha=1.0))
            model.append(
                StreamingConvTranspose1d(
                    mult * n_filters,
                    mult * n_filters // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                )
            )
            for j in range(n_residual_layers):
                model.append(
                    SEANetResnetBlock(
                        mult * n_filters // 2,
                        residual_kernel_size,
                        dilation_base**j,
                        compress,
                    )
                )
            mult //= 2
        model.append(nn.ELU(alpha=1.0))
        model.append(StreamingConv1d(n_filters, channels, last_kernel_size))
        self.model = nn.Sequential(*model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents ``[B, dimension, T]`` to audio ``[B, channels, T * hop]``."""
        return self.model(z)


class _ConvDownsample1d(nn.Module):
    """Learnt downsampling by ``stride``, kernel ``2 * stride``, replicate padding."""

    def __init__(self, stride: int, dimension: int) -> None:
        """Create the convolution."""
        super().__init__()
        self.conv = StreamingConv1d(
            dimension,
            dimension,
            kernel_size=2 * stride,
            stride=stride,
            bias=False,
            pad_mode="replicate",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample ``x`` of shape ``[B, C, T]``."""
        return self.conv(x)


class _ConvTrUpsample1d(nn.Module):
    """Learnt channel-wise upsampling by ``stride``, kernel ``2 * stride``."""

    def __init__(self, stride: int, dimension: int) -> None:
        """Create the transposed convolution."""
        super().__init__()
        self.convtr = StreamingConvTranspose1d(
            dimension,
            dimension,
            kernel_size=2 * stride,
            stride=stride,
            groups=dimension,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample ``x`` of shape ``[B, C, T]``."""
        return self.convtr(x)


class MimiModel(nn.Module):
    """Mimi codec.

    Args:
        seanet: Keyword arguments of :class:`SEANetEncoder` and :class:`SEANetDecoder`.
        transformer: Keyword arguments of the encoder and decoder
            :class:`~espnet3.systems.moshi.src.transformer.StreamingTransformer`.
        quantizer: Keyword arguments of
            :class:`~espnet3.systems.moshi.src.quantizer.SplitResidualVectorQuantizer`.
        sample_rate: Audio sample rate.
        frame_rate: Code frame rate.
        num_codebooks: Number of codebooks produced by :meth:`encode`.
    """

    def __init__(
        self,
        seanet: dict,
        transformer: dict,
        quantizer: dict,
        sample_rate: int,
        frame_rate: float,
        num_codebooks: int,
    ) -> None:
        """Create the encoder, decoder, transformers, resamplers and quantizer."""
        super().__init__()
        self.encoder = SEANetEncoder(**seanet)
        self.decoder = SEANetDecoder(**seanet)
        self.encoder_transformer = ProjectedTransformer(**transformer)
        self.decoder_transformer = ProjectedTransformer(**transformer)
        self.quantizer = SplitResidualVectorQuantizer(**quantizer)
        stride = sample_rate / self.encoder.hop_length / frame_rate
        if stride != int(stride):
            raise ValueError(f"Encoder rate / frame rate must be an integer: {stride}")
        dimension = seanet["dimension"]
        self.downsample = _ConvDownsample1d(int(stride), dimension)
        self.upsample = _ConvTrUpsample1d(int(stride), dimension)
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate / frame_rate)
        self.num_codebooks = num_codebooks

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode audio ``[B, 1, T]`` to codes ``[B, num_codebooks, ceil(T / frame)]``.

        The audio is right-padded with zeros to a whole number of frames.
        While streaming, pass exactly one or more whole frames per call.
        """
        length = x.shape[-1]
        extra = math.ceil(length / self.frame_size) * self.frame_size - length
        x = F.pad(x, (0, extra))
        emb = self.encoder(x)
        emb = self.encoder_transformer(emb)
        emb = self.downsample(emb)
        return self.quantizer.encode(emb, self.num_codebooks)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode codes ``[B, K, T]`` to audio ``[B, 1, T * frame_size]``."""
        emb = self.quantizer.decode(codes)
        emb = self.upsample(emb)
        emb = self.decoder_transformer(emb)
        return self.decoder(emb)
