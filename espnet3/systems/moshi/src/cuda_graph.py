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

"""Replay a streaming step from a CUDA graph.

A Moshi frame runs a few thousand small kernels. Launched one by one from
Python, they take longer than the 80 ms a frame lasts; replayed from a CUDA
graph, they take a fraction of it (see the system README).
"""

from __future__ import annotations

from typing import Any, Callable

import torch


class CUDAGraphed:
    """Call ``fn`` eagerly once, capture it on the second call, then replay it.

    ``fn`` must take tensors of fixed shapes and keep its state in tensors it
    updates in place, as the streaming modules do. The arguments of a call are
    copied into the captured inputs before the replay. The returned tensors
    are the graph's outputs: the next call overwrites them.

    Args:
        fn: The function to capture.
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        """Wrap ``fn``; nothing is captured yet."""
        self.fn = fn
        self.warmed_up = False
        self.graph: torch.cuda.CUDAGraph | None = None
        self.inputs: tuple[torch.Tensor, ...] = ()
        self.outputs: Any = None

    def __call__(self, *args: torch.Tensor) -> Any:
        """Run ``fn`` on ``args``."""
        if not self.warmed_up:
            self.warmed_up = True
            return self.fn(*args)
        if self.graph is None:
            self.inputs = tuple(arg.clone() for arg in args)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.outputs = self.fn(*self.inputs)
            self.graph.replay()
            return self.outputs
        for captured, arg in zip(self.inputs, args):
            if captured.shape != arg.shape:
                raise ValueError(
                    f"Captured with shape {tuple(captured.shape)}, "
                    f"called with {tuple(arg.shape)}."
                )
            captured.copy_(arg)
        self.graph.replay()
        return self.outputs
