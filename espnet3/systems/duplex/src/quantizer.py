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

"""Split residual vector quantizer of Mimi (inference only).

The first codebook is the "semantic" one (``rvq_first``), the next ones are
acoustic (``rvq_rest``). Each codebook holds the exponential moving averages
it was trained with; its centroids are ``embedding_sum / cluster_usage``.
Attribute names follow the released checkpoint.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _Codebook(nn.Module):
    """Euclidean codebook stored as EMA statistics."""

    def __init__(self, dim: int, codebook_size: int, epsilon: float = 1e-5) -> None:
        """Register the checkpoint buffers."""
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("_initialized", torch.tensor([False], dtype=torch.float))
        self.register_buffer("cluster_usage", torch.ones(codebook_size))
        self.register_buffer("embedding_sum", torch.zeros(codebook_size, dim))

    def embedding(self) -> torch.Tensor:
        """Return the centroids, shape ``[codebook_size, dim]``."""
        return self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the index of the centroid nearest to each row of ``x`` ``[N, D]``."""
        dists = torch.cdist(x[None], self.embedding()[None], p=2)[0]
        return dists.argmin(dim=-1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Return the centroids of ``codes``, shape ``[*codes.shape, dim]``."""
        return F.embedding(codes, self.embedding())


class _VectorQuantization(nn.Module):
    """One codebook applied to latents of shape ``[B, dim, T]``."""

    def __init__(self, dim: int, codebook_size: int) -> None:
        """Create the codebook."""
        super().__init__()
        self._codebook = _Codebook(dim, codebook_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return codes of shape ``[B, T]`` for ``x`` of shape ``[B, dim, T]``."""
        B, D, T = x.shape
        flat = x.transpose(1, 2).reshape(B * T, D)
        return self._codebook.encode(flat).view(B, T)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Return latents of shape ``[B, dim, T]`` for codes of shape ``[B, T]``."""
        return self._codebook.decode(codes).transpose(1, 2)


class _ResidualVectorQuantization(nn.Module):
    """Stack of codebooks, each quantizing the residual of the previous ones."""

    def __init__(self, dim: int, codebook_size: int, num_quantizers: int) -> None:
        """Create ``num_quantizers`` codebooks."""
        super().__init__()
        self.layers = nn.ModuleList(
            [_VectorQuantization(dim, codebook_size) for _ in range(num_quantizers)]
        )


class ResidualVectorQuantizer(nn.Module):
    """Residual vector quantizer with 1x1 input and output projections.

    Args:
        dimension: Codebook dimension.
        input_dimension: Latent dimension before quantization.
        output_dimension: Latent dimension after quantization.
        n_q: Number of codebooks.
        bins: Codebook size.
    """

    def __init__(
        self,
        dimension: int,
        input_dimension: int,
        output_dimension: int,
        n_q: int,
        bins: int,
    ) -> None:
        """Create the projections and the codebooks."""
        super().__init__()
        self.input_proj = nn.Conv1d(input_dimension, dimension, 1, bias=False)
        self.output_proj = nn.Conv1d(dimension, output_dimension, 1, bias=False)
        self.vq = _ResidualVectorQuantization(dimension, bins, n_q)

    def encode(self, x: torch.Tensor, n_q: int) -> torch.Tensor:
        """Return codes of shape ``[B, n_q, T]`` for ``x`` of shape ``[B, C, T]``."""
        residual = self.input_proj(x)
        codes = []
        for layer in self.vq.layers[:n_q]:
            indices = layer.encode(residual)
            residual = residual - layer.decode(indices)
            codes.append(indices)
        return torch.stack(codes, dim=1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Return latents of shape ``[B, C, T]`` for codes of shape ``[B, K, T]``."""
        quantized = self.vq.layers[0].decode(codes[:, 0])
        for index in range(1, codes.shape[1]):
            quantized = quantized + self.vq.layers[index].decode(codes[:, index])
        return self.output_proj(quantized)


class SplitResidualVectorQuantizer(nn.Module):
    """Semantic codebook(s) followed by acoustic residual codebooks.

    Args:
        dimension: Codebook dimension.
        input_dimension: Latent dimension before quantization.
        output_dimension: Latent dimension after quantization.
        n_q: Total number of codebooks in the checkpoint.
        n_q_semantic: Number of semantic codebooks.
        bins: Codebook size.
    """

    def __init__(
        self,
        dimension: int,
        input_dimension: int,
        output_dimension: int,
        n_q: int,
        n_q_semantic: int,
        bins: int,
    ) -> None:
        """Create the semantic and the acoustic quantizers."""
        super().__init__()
        self.n_q_semantic = n_q_semantic
        self.rvq_first = ResidualVectorQuantizer(
            dimension, input_dimension, output_dimension, n_q_semantic, bins
        )
        self.rvq_rest = ResidualVectorQuantizer(
            dimension, input_dimension, output_dimension, n_q - n_q_semantic, bins
        )

    def encode(self, x: torch.Tensor, n_q: int) -> torch.Tensor:
        """Return the first ``n_q`` codes, shape ``[B, n_q, T]``.

        The semantic and the acoustic quantizers both read ``x``: the acoustic
        codebooks do not quantize the semantic residual.
        """
        codes = self.rvq_first.encode(x, self.n_q_semantic)
        acoustic = self.rvq_rest.encode(x, n_q - self.n_q_semantic)
        return torch.cat([codes, acoustic], dim=1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Return latents of shape ``[B, C, T]`` for codes of shape ``[B, K, T]``."""
        quantized = self.rvq_first.decode(codes[:, : self.n_q_semantic])
        quantized = quantized + self.rvq_rest.decode(codes[:, self.n_q_semantic :])
        return quantized
