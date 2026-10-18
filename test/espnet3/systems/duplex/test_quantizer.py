"""Tests for Mimi's split residual vector quantizer."""

import torch

from espnet3.systems.duplex.src.quantizer import SplitResidualVectorQuantizer


def _quantizer():
    torch.manual_seed(0)
    quantizer = SplitResidualVectorQuantizer(
        dimension=4,
        input_dimension=6,
        output_dimension=6,
        n_q=4,
        n_q_semantic=1,
        bins=8,
    )
    for name, buffer in quantizer.named_buffers():
        if name.endswith("embedding_sum"):
            buffer.copy_(torch.randn_like(buffer))
        if name.endswith("cluster_usage"):
            buffer.copy_(torch.rand_like(buffer) + 0.5)
    return quantizer


def test_encode_picks_nearest_centroids():
    """Every code is the nearest centroid of the residual it quantizes."""
    quantizer = _quantizer()
    x = torch.randn(2, 6, 5)
    codes = quantizer.encode(x, 3)
    assert codes.shape == (2, 3, 5)

    first = quantizer.rvq_first
    latent = first.input_proj(x).transpose(1, 2)
    centroids = first.vq.layers[0]._codebook.embedding()
    expected = torch.cdist(latent, centroids[None]).argmin(-1)
    assert torch.equal(codes[:, 0], expected)

    rest = quantizer.rvq_rest
    residual = rest.input_proj(x).transpose(1, 2)
    for index in range(2):
        centroids = rest.vq.layers[index]._codebook.embedding()
        expected = torch.cdist(residual, centroids[None]).argmin(-1)
        assert torch.equal(codes[:, 1 + index], expected)
        residual = residual - centroids[expected]
