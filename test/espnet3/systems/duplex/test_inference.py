"""Tests for the full-duplex Moshi inference engine."""

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from espnet3.contract.inference import load_inference_contract
from espnet3.contract.kinds.audio import Audio
from espnet3.systems.base.inference_runner import InferenceRunner
from espnet3.systems.duplex.inference import DuplexInference
from espnet3.systems.duplex.src.lm import LMModel
from espnet3.systems.duplex.src.mimi import MimiModel


@pytest.fixture
def tiny_model(tmp_path, tiny_mimi_config, tiny_lm_config, tiny_tokenizer):
    """Build a tiny DuplexInference from random checkpoints saved to disk."""
    torch.manual_seed(0)
    mimi = MimiModel(**tiny_mimi_config)
    for name, buffer in mimi.named_buffers():
        if name.endswith("embedding_sum"):
            buffer.copy_(torch.randn_like(buffer))
    save_file(mimi.state_dict(), tmp_path / "mimi.safetensors")
    save_file(LMModel(**tiny_lm_config).state_dict(), tmp_path / "lm.safetensors")
    return DuplexInference(
        lm=tiny_lm_config,
        mimi=tiny_mimi_config,
        hf_repo=None,
        revision=None,
        lm_file=str(tmp_path / "lm.safetensors"),
        mimi_file=str(tmp_path / "mimi.safetensors"),
        tokenizer_file=str(tiny_tokenizer),
        generator="moshi",
        temp=0.8,
        temp_text=0.7,
        top_k=5,
        top_k_text=5,
        seed=7,
        cuda_graph=False,
        device="cpu",
    )


def test_output_is_frame_aligned_with_input(tiny_model):
    """The output is as long as the input; the delayed first frame is silent."""
    speech = 0.1 * np.random.default_rng(0).standard_normal(64 * 5 + 10)
    out = tiny_model(Audio(speech.astype(np.float32), 800))
    assert out["wav"].shape == (64 * 5 + 10,)
    assert out["wav"].dtype == np.float32
    assert np.all(out["wav"][:64] == 0)
    assert np.any(out["wav"][64:] != 0)
    assert isinstance(out["text"], str)
    assert out["step_ms"] > 0


def test_conversations_are_independent_and_reproducible(tiny_model):
    """Every conversation restarts from the seed."""
    speech = 0.1 * np.random.default_rng(1).standard_normal(64 * 6)
    first = tiny_model(Audio(speech.astype(np.float32), 800))
    second = tiny_model(Audio(speech.astype(np.float32), 800))
    np.testing.assert_array_equal(first["wav"], second["wav"])
    assert first["text"] == second["text"]


def test_live_steps_match_offline_call(tiny_model):
    """Stepping frames live gives what the offline simulation gives."""
    speech = (0.1 * np.random.default_rng(2).standard_normal(64 * 4)).astype(np.float32)
    offline = tiny_model(Audio(speech, 800))
    tiny_model.start()
    assert tiny_model.is_streaming
    live = [tiny_model.step(speech[i * 64 : (i + 1) * 64])[0] for i in range(4)]
    tiny_model.stop()
    assert not tiny_model.is_streaming
    np.testing.assert_array_equal(np.concatenate(live), offline["wav"])


def test_cuda_graph_needs_a_cuda_device(tmp_path, tiny_mimi_config, tiny_lm_config):
    """cuda_graph on CPU would silently replay nothing, so it is refused."""
    with pytest.raises(ValueError, match="cuda_graph needs a CUDA device"):
        DuplexInference(
            lm=tiny_lm_config,
            mimi=tiny_mimi_config,
            hf_repo=None,
            revision=None,
            lm_file=str(tmp_path / "unused"),
            mimi_file=str(tmp_path / "unused"),
            tokenizer_file=str(tmp_path / "unused"),
            generator="moshi",
            temp=0.8,
            temp_text=0.7,
            top_k=5,
            top_k_text=5,
            seed=7,
            cuda_graph=True,
            device="cpu",
        )


def test_step_needs_an_open_conversation(tiny_model):
    """Stepping outside a conversation is an error."""
    with pytest.raises(RuntimeError, match="Call start"):
        tiny_model.step(np.zeros(64, dtype=np.float32))


def test_step_rejects_partial_frames(tiny_model):
    """A frame of the wrong size is an error."""
    tiny_model.start()
    with pytest.raises(ValueError, match="Expected 64 samples"):
        tiny_model.step(np.zeros(10, dtype=np.float32))
    tiny_model.stop()


def test_moshi_refuses_a_role_prompt(tiny_model):
    """Only PersonaPlex takes a role prompt."""
    with pytest.raises(ValueError, match="Moshi takes no role prompt"):
        tiny_model.start(role_prompt="You enjoy having a good conversation.")


def test_contracted_inference_writes_the_declared_outputs(tiny_model):
    """The recipe's contract accepts the model's outputs and refuses a wrong rate.

    The contract is conf/inference.yaml's, at the tiny Mimi's 800 Hz.
    """
    contract = load_inference_contract(
        {
            "inputs": {
                "speech": {
                    "kind": "audio",
                    "sample_rate": 800,
                    "rate_policy": "reject",
                    "channels": 1,
                }
            },
            "outputs": {
                "wav": {"kind": "audio"},
                "text": {"kind": "text"},
                "step_ms": {"kind": "number"},
            },
        }
    )
    speech = np.zeros(64 * 3, dtype=np.float32)
    dataset = [
        {"utt_id": "a", "speech": Audio(speech, 800)},
        {"utt_id": "b", "speech": Audio(np.zeros(320, dtype=np.float32), 1600)},
    ]
    out = InferenceRunner.forward(
        0,
        dataset=dataset,
        model=tiny_model,
        input_key="speech",
        inference_contract=contract,
    )
    assert set(out) == {"utt_id", "wav", "text", "step_ms"}
    assert out["wav"].rate == 800 and out["wav"].array.shape == speech.shape
    with pytest.raises(ValueError, match="has sample rate 1600; expected 800"):
        InferenceRunner.forward(
            1,
            dataset=dataset,
            model=tiny_model,
            input_key="speech",
            inference_contract=contract,
        )


def test_call_refuses_audio_at_another_rate(tiny_model):
    """A recording at another rate would be read as the wrong speed."""
    with pytest.raises(ValueError, match="Expected mono audio at 800 Hz"):
        tiny_model(Audio(np.zeros(640, dtype=np.float32), 16000))
