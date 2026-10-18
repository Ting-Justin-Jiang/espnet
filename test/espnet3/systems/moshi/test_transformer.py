"""Tests for the streaming Transformer of the Moshi port."""

import pytest
import torch

from espnet3.systems.moshi.src.streaming import start_streaming, stop_streaming
from espnet3.systems.moshi.src.transformer import (
    RingKVCache,
    StreamingTransformer,
    apply_rope,
)


def _transformer(**overrides):
    kwargs = {
        "d_model": 16,
        "num_heads": 2,
        "num_layers": 2,
        "dim_feedforward": 32,
        "context": 4,
        "rope_max_period": 100,
        "norm": "rms_norm_f32",
        "layer_scale": None,
        "gating": "silu",
        "weights_per_step": 0,
    }
    kwargs.update(overrides)
    torch.manual_seed(0)
    return StreamingTransformer(**kwargs).eval()


MIMI_LAYOUT = {"norm": "layer_norm", "layer_scale": 0.1, "gating": "none"}


@pytest.mark.parametrize("overrides", [{}, MIMI_LAYOUT], ids=["lm", "mimi"])
def test_step_by_step_streaming_matches_full_sequence(overrides):
    """One step per call, far past the window, equals one pass."""
    model = _transformer(**overrides)
    x = torch.randn(2, 12, 16)
    with torch.no_grad():
        full = model(x)
        start_streaming(model, 2)
        streamed = torch.cat([model(x[:, t : t + 1]) for t in range(12)], dim=1)
        stop_streaming(model)
    torch.testing.assert_close(streamed, full, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("chunk", [2, 3])
@pytest.mark.parametrize("overrides", [{}, MIMI_LAYOUT], ids=["lm", "mimi"])
def test_chunked_streaming_matches_full_sequence_within_window(chunk, overrides):
    """Several steps per call equal one pass while the stream fits the window.

    Past the window the ring cache, like the reference implementation, keeps
    ``context`` steps, so the first queries of a call lose their oldest keys.
    """
    model = _transformer(context=12, **overrides)
    x = torch.randn(2, 12, 16)
    with torch.no_grad():
        full = model(x)
        start_streaming(model, 2)
        streamed = torch.cat(
            [model(x[:, t : t + chunk]) for t in range(0, 12, chunk)], dim=1
        )
        stop_streaming(model)
    torch.testing.assert_close(streamed, full, rtol=1e-5, atol=1e-5)


def test_context_limits_attention():
    """Each layer sees ``context`` steps: two layers of 3 see 4 steps back."""
    model = _transformer(context=3)
    x = torch.randn(1, 8, 16)
    y = x.clone()
    y[:, 0] = torch.randn(16)
    with torch.no_grad():
        out_x, out_y = model(x), model(y)
    assert not torch.allclose(out_x[:, 4], out_y[:, 4])
    torch.testing.assert_close(out_x[:, 5:], out_y[:, 5:])


def test_weights_per_step_streaming_matches_full_sequence():
    """The depformer layout: step ``t`` uses weight ``t``, streaming or not."""
    model = _transformer(context=None, rope_max_period=None, weights_per_step=3)
    x = torch.randn(2, 3, 16)
    with torch.no_grad():
        full = model(x)
        start_streaming(model, 2)
        streamed = torch.cat(
            [model(x[:, t : t + 1], weight_offset=t) for t in range(3)], dim=1
        )
        stop_streaming(model)
    torch.testing.assert_close(streamed, full)


def test_start_streaming_twice_raises():
    """A second start without a stop is an error, not a silent reset."""
    model = _transformer()
    start_streaming(model, 1)
    with pytest.raises(RuntimeError, match="already streaming"):
        start_streaming(model, 1)


def test_ring_kv_cache_positions():
    """Slots hold the absolute position last written there, -1 if never written."""
    cache = RingKVCache(1, 1, 2, capacity=4, device="cpu", dtype=torch.float32)
    _, _, positions = cache.complete(torch.zeros(1, 1, 3, 2), torch.zeros(1, 1, 3, 2))
    assert positions.tolist() == [0, 1, 2, -1]
    _, _, positions = cache.complete(torch.zeros(1, 1, 3, 2), torch.zeros(1, 1, 3, 2))
    assert positions.tolist() == [4, 5, 2, 3]


def test_apply_rope_offset_matches_full_range():
    """Rotating a slice at its offset equals slicing the full rotation."""
    q, k = torch.randn(2, 1, 2, 6, 8)
    full_q, full_k = apply_rope(q, k, torch.tensor(0), 100)
    part_q, part_k = apply_rope(q[:, :, 2:], k[:, :, 2:], torch.tensor(2), 100)
    torch.testing.assert_close(part_q, full_q[:, :, 2:])
    torch.testing.assert_close(part_k, full_k[:, :, 2:])
