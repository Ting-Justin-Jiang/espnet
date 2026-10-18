"""Tests for the Mimi codec of the Moshi port."""

import torch

from espnet3.systems.moshi.src.mimi import MimiModel, StreamingConv1d
from espnet3.systems.moshi.src.streaming import start_streaming, stop_streaming


def _mimi(config):
    torch.manual_seed(0)
    mimi = MimiModel(**config).eval()
    for name, buffer in mimi.named_buffers():
        if name.endswith("embedding_sum"):
            buffer.copy_(torch.randn_like(buffer))
    return mimi


def test_frame_size_and_code_shape(tiny_mimi_config):
    """Codes come at ``frame_rate``, one per ``frame_size`` samples."""
    mimi = _mimi(tiny_mimi_config)
    assert mimi.frame_size == 64
    with torch.no_grad():
        codes = mimi.encode(torch.randn(2, 1, 64 * 5))
        audio = mimi.decode(codes)
    assert codes.shape == (2, 3, 5)
    assert audio.shape == (2, 1, 64 * 5)


def test_encode_pads_to_whole_frames(tiny_mimi_config):
    """A partial last frame is padded with zeros."""
    mimi = _mimi(tiny_mimi_config)
    x = torch.randn(1, 1, 64 * 3 + 10)
    padded = torch.cat([x, torch.zeros(1, 1, 54)], dim=-1)
    with torch.no_grad():
        assert torch.equal(mimi.encode(x), mimi.encode(padded))


def test_streaming_matches_full_signal(tiny_mimi_config):
    """Frame-by-frame encoding and decoding compute the one-pass function."""
    mimi = _mimi(tiny_mimi_config)
    x = torch.randn(1, 1, 64 * 12)
    with torch.no_grad():
        full_latent = mimi.downsample(mimi.encoder_transformer(mimi.encoder(x)))
        full_codes = mimi.quantizer.encode(full_latent, 3)
        full_audio = mimi.decode(full_codes)
        start_streaming(mimi, 1)
        latents, audio = [], []
        for t in range(12):
            frame = x[..., t * 64 : (t + 1) * 64]
            latents.append(
                mimi.downsample(mimi.encoder_transformer(mimi.encoder(frame)))
            )
            audio.append(mimi.decode(full_codes[..., t : t + 1]))
        stop_streaming(mimi)
    torch.testing.assert_close(torch.cat(latents, -1), full_latent)
    torch.testing.assert_close(torch.cat(audio, -1), full_audio)


def test_replicate_padding_copies_first_sample():
    """A replicate-padded conv sees copies of the first sample, then the past."""
    conv = StreamingConv1d(1, 1, kernel_size=4, stride=2, bias=False)
    conv.pad_mode = "replicate"
    torch.nn.init.ones_(conv.conv.conv.weight)
    x = torch.arange(1.0, 7.0).view(1, 1, 6)
    with torch.no_grad():
        full = conv(x)
        start_streaming(conv, 1)
        streamed = torch.cat(
            [conv(x[..., :2]), conv(x[..., 2:4]), conv(x[..., 4:])], -1
        )
    # padding [1, 1] then windows [1, 1, 1, 2], [1, 2, 3, 4], [3, 4, 5, 6]
    assert full.flatten().tolist() == [5.0, 10.0, 18.0]
    assert torch.equal(streamed, full)
