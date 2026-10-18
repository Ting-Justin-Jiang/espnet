"""Tests for the Moshi duplex demo app, driven by a simulated microphone."""

import importlib.util
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from egs3.TEMPLATE.duplex.src.app import WARMUP_FRAMES, DuplexConversation
from espnet3.publication.demo.packing import pack_demo
from espnet3.utils.config_utils import load_and_merge_config


class _HalfEcho:
    """Stands in for MoshiInference: replies to each frame with half of it."""

    sample_rate = 24000
    frame_size = 1920

    def __init__(self):
        self.is_streaming = False
        self.frames = []

    def start(self):
        self.is_streaming = True

    def stop(self):
        self.is_streaming = False

    def step(self, frame):
        assert self.is_streaming and frame.shape == (1920,)
        self.frames.append(frame.copy())
        return 0.5 * frame, "a"


def _microphone(seconds, rate=48000):
    """Return int16 speech-like samples, as a browser microphone sends them."""
    t = np.arange(int(seconds * rate)) / rate
    return (8000 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)


def test_chunks_become_frames_and_replies():
    """Uneven microphone chunks are resampled and fed as whole 80 ms frames."""
    model = _HalfEcho()
    conversation = DuplexConversation(model)
    conversation.start()
    samples = _microphone(1.0)
    replies = []
    for start in range(0, len(samples), 3000):
        out = conversation(speech=(48000, samples[start : start + 3000]))
        if out["wav"] is not None:
            rate, reply = out["wav"]
            assert rate == 24000 and reply.dtype == np.int16
            replies.append(reply)
    conversation.stop()
    assert not model.is_streaming
    # Two warm-up frames of silence, then 1 s at 24 kHz: 12 whole frames.
    assert len(model.frames) == WARMUP_FRAMES + 12
    assert not np.any(np.concatenate(model.frames[:WARMUP_FRAMES]))
    expected = (0.5 * np.concatenate(model.frames) * 32767).astype(np.int16)
    np.testing.assert_array_equal(np.concatenate(replies), expected)
    assert out["text"] == "a" * (WARMUP_FRAMES + 12)


def test_no_reply_until_a_frame_completes():
    """A chunk that completes no frame returns no audio."""
    model = _HalfEcho()
    conversation = DuplexConversation(model)
    conversation.start()
    conversation(speech=(48000, _microphone(0.2)))
    out = conversation(speech=(48000, _microphone(0.01)))
    assert out["wav"] is None


def test_chunking_does_not_change_what_moshi_hears():
    """The streaming resampler gives the same frames for any chunking."""
    samples = _microphone(0.5)
    heard = []
    for chunk in (len(samples), 1111):
        model = _HalfEcho()
        conversation = DuplexConversation(model)
        conversation.start()
        for start in range(0, len(samples), chunk):
            conversation(speech=(48000, samples[start : start + chunk]))
        heard.append(np.concatenate(model.frames[WARMUP_FRAMES:]))
    np.testing.assert_allclose(heard[0], heard[1], atol=1e-6)
    expected = np.sin(2 * np.pi * 220 * np.arange(len(heard[0])) / 24000)
    # After the resampler's warm-up the signal is the original sine.
    np.testing.assert_allclose(
        heard[0][2000:4000], 8000 / 32768 * expected[2000:4000], atol=5e-3
    )


def test_restart_closes_the_open_conversation():
    """Starting again (a reloaded page) closes the conversation left open."""
    model = _HalfEcho()
    conversation = DuplexConversation(model)
    conversation.start()
    conversation(speech=(48000, _microphone(0.2)))
    conversation.start()
    assert model.is_streaming
    # Only the new conversation's warm-up frames are left.
    assert conversation.text == "a" * WARMUP_FRAMES


def test_packed_demo_builds(tmp_path, monkeypatch):
    """pack_demo writes a runnable app whose UI follows demo.yaml."""
    user_config = tmp_path / "demo.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "upload_demo": {"hf_repo": "owner/demo"},
                "ui": {"description": None, "title": "Talk to Moshi"},
                "pack": {"out_dir": str(tmp_path / "demo")},
            }
        ),
        user_config,
    )
    config = load_and_merge_config(
        user_config, "demo.yaml", default_package="egs3.TEMPLATE.duplex", resolve=False
    )
    out = pack_demo(SimpleNamespace(demo_config=config, exp_dir=None))
    packed = OmegaConf.load(out / "demo.yaml")
    assert packed.model.hf_repo == "kyutai/moshiko-pytorch-bf16"
    assert packed.model.dir_or_tag == "kyutai/moshiko-pytorch-bf16"

    spec = importlib.util.spec_from_file_location("packed_moshi_app", out / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seen = {}

    def fake_build_model(model_config):
        seen["target"] = model_config["_target_"]
        return _HalfEcho()

    monkeypatch.setattr(module, "build_model", fake_build_model)
    ui = module.build_demo(out)
    assert seen["target"] == "espnet3.systems.duplex.inference.MoshiInference"
    labels = {getattr(block, "label", None) for block in ui.blocks.values()}
    assert {"You (microphone)", "Moshi", "Moshi's inner monologue"} <= labels
