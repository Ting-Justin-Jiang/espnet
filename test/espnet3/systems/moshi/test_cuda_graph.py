"""Tests for CUDA graph replay of streaming steps."""

from test.espnet3.systems.moshi.test_lm import _generate

import pytest
import torch

from espnet3.systems.moshi.src.cuda_graph import CUDAGraphed
from espnet3.systems.moshi.src.lm import LMModel

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graphs need a GPU"
)


class _Accumulator:
    """A stateful step: adds its input to a device tensor in place."""

    def __init__(self):
        self.state = torch.zeros(3, device="cuda")

    def __call__(self, x):
        self.state.add_(x)
        return self.state * 2


def test_replay_matches_eager_calls():
    """Warm-up, capture and replays all update the state like eager calls."""
    graphed = CUDAGraphed(_Accumulator())
    outs = [
        graphed(torch.full((3,), float(i), device="cuda")).clone() for i in range(1, 5)
    ]
    assert [out[0].item() for out in outs] == [2.0, 6.0, 12.0, 20.0]


def test_shape_change_raises():
    """A replay with another input shape is an error, not a broadcast."""
    graphed = CUDAGraphed(_Accumulator())
    graphed(torch.ones(3, device="cuda"))
    graphed(torch.ones(3, device="cuda"))
    with pytest.raises(ValueError, match="Captured with shape"):
        graphed(torch.ones(1, device="cuda"))


def test_graphed_generation_matches_eager(tiny_lm_config):
    """Graphed frames, depformer state included, sample the eager tokens."""
    torch.manual_seed(0)
    lm = LMModel(**tiny_lm_config).eval().cuda()
    device = torch.device("cuda")
    assert torch.equal(_generate(lm, device, True), _generate(lm, device, False))
