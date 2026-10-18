"""Tests for CUDA graph replay of streaming steps."""

import pytest
import torch

from espnet3.systems.moshi.src.cuda_graph import CUDAGraphed

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graphs need a GPU"
)


def _accumulate(state):
    def fn(x):
        state.add_(x)
        return state * 2

    return fn


def test_replay_matches_eager_calls():
    """Warm-up, capture and replays all update the state like eager calls."""
    state = torch.zeros(3, device="cuda")
    graphed = CUDAGraphed(_accumulate(state))
    outs = [
        graphed(torch.full((3,), float(i), device="cuda")).clone() for i in range(1, 5)
    ]
    assert [out[0].item() for out in outs] == [2.0, 6.0, 12.0, 20.0]


def test_shape_change_raises():
    """A replay with another input shape is an error, not a broadcast."""
    graphed = CUDAGraphed(_accumulate(torch.zeros(3, device="cuda")))
    graphed(torch.ones(3, device="cuda"))
    graphed(torch.ones(3, device="cuda"))
    with pytest.raises(ValueError, match="Captured with shape"):
        graphed(torch.ones(1, device="cuda"))
