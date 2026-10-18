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

"""Streaming state shared by the Moshi modules.

A module that keeps state across calls while audio or tokens arrive one frame
at a time subclasses :class:`StreamingModule`. Its ``stream`` attribute is
``None`` outside streaming. :func:`start_streaming` creates the state of every
such module under a root module, and :func:`stop_streaming` drops it.

Unlike the reference implementation, a stream has no per-item execution mask:
every item of a batch advances together.
"""

from __future__ import annotations

from typing import Any

from torch import nn


class StreamingModule(nn.Module):
    """Module whose forward reads and updates ``self.stream`` when it is set."""

    def __init__(self) -> None:
        """Start outside streaming."""
        super().__init__()
        self.stream: Any = None

    def init_stream(self, batch_size: int) -> Any:
        """Return the empty state of a stream of ``batch_size`` items."""
        raise NotImplementedError


def start_streaming(root: nn.Module, batch_size: int) -> None:
    """Create the stream state of every :class:`StreamingModule` under ``root``.

    Args:
        root: Module whose streaming submodules (``root`` included) start.
        batch_size: Number of streams processed together.

    Raises:
        RuntimeError: If one of the modules is already streaming.
    """
    for module in root.modules():
        if isinstance(module, StreamingModule):
            if module.stream is not None:
                raise RuntimeError(f"{type(module).__name__} is already streaming.")
            module.stream = module.init_stream(batch_size)


def stop_streaming(root: nn.Module) -> None:
    """Drop the stream state of every :class:`StreamingModule` under ``root``."""
    for module in root.modules():
        if isinstance(module, StreamingModule):
            module.stream = None
