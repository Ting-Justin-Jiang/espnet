"""Tests for Mimi's split residual vector quantizer."""

import torch

from espnet3.systems.moshi.src.quantizer import SplitResidualVectorQuantizer


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


def test_decode_sums_both_quantizers():
    """Decoding adds the projected semantic and acoustic reconstructions."""
    quantizer = _quantizer()
    codes = torch.randint(0, 8, (2, 3, 5))
    expected = quantizer.rvq_first.decode(codes[:, :1]) + quantizer.rvq_rest.decode(
        codes[:, 1:]
    )
    torch.testing.assert_close(quantizer.decode(codes), expected)
    assert quantizer.decode(codes).shape == (2, 6, 5)


def test_centroids_are_ema_ratios():
    """Centroids are the EMA sums divided by the EMA usage."""
    codebook = _quantizer().rvq_first.vq.layers[0]._codebook
    expected = codebook.embedding_sum / codebook.cluster_usage[:, None]
    torch.testing.assert_close(codebook.embedding(), expected)
