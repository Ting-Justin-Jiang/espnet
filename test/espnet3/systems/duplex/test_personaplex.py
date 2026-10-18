"""Tests for PersonaPlex's prompted, teacher-forced generator."""

import tarfile

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from espnet3.systems.duplex.inference import DuplexInference
from espnet3.systems.duplex.src.lm import LMModel
from espnet3.systems.duplex.src.mimi import MimiModel
from espnet3.systems.duplex.src.personaplex import (
    PROMPT_SILENCE_FRAMES,
    SILENCE_TOKENS,
    SINE_TOKENS,
    TEXT_PADDING_ID,
    PersonaPlexGen,
    wrap_role_prompt,
)

CPU = torch.device("cpu")


@pytest.fixture
def plex_lm_config(tiny_lm_config):
    """TINY_LM with a depformer over every audio stream, as in PersonaPlex."""
    tiny_lm_config["dep_q"] = tiny_lm_config["n_q"]
    return tiny_lm_config


def _lm(config):
    torch.manual_seed(0)
    return LMModel(**config).eval()


def _voice_prompt(config, dtype, frames=3):
    """A voice prompt in the released format: embeddings and the final cache."""
    torch.manual_seed(1)
    cache = torch.randint(0, config["card"], (1, config["n_q"] + 1, 4))
    cache[:, 0] = TEXT_PADDING_ID
    embeddings = torch.randn(frames, 1, 1, config["dim"], dtype=dtype)
    return {"embeddings": embeddings, "cache": cache}


class _TinyCodebooks:
    """The prompt constants cut to the tiny model's 3 codebooks and card 16."""

    silence = torch.tensor(SILENCE_TOKENS[:3]) % 16
    sine = torch.tensor(SINE_TOKENS[:3]) % 16


def test_role_prompt_is_wrapped_once():
    """The <system> tags are added unless already there."""
    assert wrap_role_prompt(" Be nice. ") == "<system> Be nice. <system>"
    assert (
        wrap_role_prompt("<system> Be nice. <system>") == "<system> Be nice. <system>"
    )


def test_generator_needs_a_full_depformer(tiny_lm_config):
    """PersonaPlex's depformer predicts the user's streams too."""
    with pytest.raises(ValueError, match="every audio stream"):
        PersonaPlexGen(_lm(tiny_lm_config), 0.8, 0.7, 5, 5, cuda_graph=False)


def test_forced_streams_come_out_as_forced(plex_lm_config):
    """Forced text and audio, and the user's audio, are what the frame holds.

    The first call only fills the cache, as in the reference, so forcing
    starts with the second.
    """
    gen = PersonaPlexGen(_lm(plex_lm_config), 0.8, 0.7, 5, 5, cuda_graph=False)
    gen.start(1, CPU)
    torch.manual_seed(0)
    user = [torch.randint(0, 16, (1, 3, 1)) for _ in range(6)]
    agent = torch.randint(0, 16, (1, 3, 1))
    outs = [gen.step(user[0]), gen.step(user[1], agent, 7)]
    outs += [gen.step(codes) for codes in user[2:]]
    gen.stop()
    assert outs[0] is None and outs[1] is None
    # The frame received at call t comes out at call t + 1.
    assert outs[2][0, 0, 0] == 7
    assert torch.equal(outs[2][0, 1:4], agent[0])
    for index in range(2, 6):
        assert torch.equal(outs[index][0, 4:], user[index - 1][0])


def test_prompts_force_tone_silence_and_role(plex_lm_config, monkeypatch):
    """After the prompts the cache holds the role prompt's last frames."""
    import espnet3.systems.duplex.src.personaplex as module

    monkeypatch.setattr(
        module, "SILENCE_TOKENS", tuple(_TinyCodebooks.silence.tolist())
    )
    monkeypatch.setattr(module, "SINE_TOKENS", tuple(_TinyCodebooks.sine.tolist()))
    gen = PersonaPlexGen(_lm(plex_lm_config), 0.8, 0.7, 5, 5, cuda_graph=False)
    gen.start(1, CPU)
    gen.run_prompts(_voice_prompt(plex_lm_config, torch.float32), [11, 12])
    frames = 3 + PROMPT_SILENCE_FRAMES + 2 + PROMPT_SILENCE_FRAMES
    # The replay of the voice prompt's first frame only fills the cache.
    assert gen.offset == frames + 1
    position = (gen.offset - 1) % gen.cache.shape[2]
    assert gen.cache[0, 0, position] == TEXT_PADDING_ID
    assert torch.equal(
        gen.cache[0, 1, position : position + 1], _TinyCodebooks.silence[:1]
    )
    assert gen.cache[0, 4, position] == _TinyCodebooks.sine[0]
    gen.stop()


@pytest.fixture
def tiny_personaplex(tmp_path, tiny_mimi_config, plex_lm_config, tiny_tokenizer):
    """A tiny DuplexInference with the PersonaPlex generator and a voice archive."""
    torch.manual_seed(0)
    mimi = MimiModel(**tiny_mimi_config)
    for name, buffer in mimi.named_buffers():
        if name.endswith("embedding_sum"):
            buffer.copy_(torch.randn_like(buffer))
    save_file(mimi.state_dict(), tmp_path / "mimi.safetensors")
    save_file(LMModel(**plex_lm_config).state_dict(), tmp_path / "lm.safetensors")
    (tmp_path / "voices").mkdir()
    # DuplexInference runs the language model in bfloat16, as the voices are.
    voice = _voice_prompt(plex_lm_config, torch.bfloat16, frames=1)
    torch.save(voice, tmp_path / "voices" / "TINY.pt")
    with tarfile.open(tmp_path / "voices.tgz", "w:gz") as tar:
        tar.add(tmp_path / "voices" / "TINY.pt", arcname="voices/TINY.pt")
    (tmp_path / "voices" / "TINY.pt").unlink()
    (tmp_path / "voices").rmdir()
    return DuplexInference(
        lm=plex_lm_config,
        mimi=tiny_mimi_config,
        hf_repo=None,
        revision=None,
        lm_file=str(tmp_path / "lm.safetensors"),
        mimi_file=str(tmp_path / "mimi.safetensors"),
        tokenizer_file=str(tiny_tokenizer),
        generator="personaplex",
        temp=0.8,
        temp_text=0.7,
        top_k=5,
        top_k_text=5,
        seed=7,
        cuda_graph=False,
        voices_file=str(tmp_path / "voices.tgz"),
        voice_prompt="TINY.pt",
        device="cpu",
    )


def test_prompt_frames_must_match_the_codebooks(plex_lm_config):
    """The released prompt frames have 8 codes; a model with 3 is refused."""
    gen = PersonaPlexGen(_lm(plex_lm_config), 0.8, 0.7, 5, 5, cuda_graph=False)
    gen.start(1, CPU)
    with pytest.raises(ValueError, match="8 codes, the model 3 codebooks"):
        gen.run_prompts(_voice_prompt(plex_lm_config, torch.float32), [11])
    gen.stop()


def test_personaplex_conversation(tiny_personaplex, monkeypatch):
    """A role prompt is required; the output is frame-aligned.

    Seeding is DuplexInference's for both generators; test_inference.py
    checks that conversations are reproducible.
    """
    import espnet3.systems.duplex.src.personaplex as module

    monkeypatch.setattr(
        module, "SILENCE_TOKENS", tuple(_TinyCodebooks.silence.tolist())
    )
    monkeypatch.setattr(module, "SINE_TOKENS", tuple(_TinyCodebooks.sine.tolist()))
    # bfloat16 is slow on CPU: shorten the prompts' silences.
    monkeypatch.setattr(module, "PROMPT_SILENCE_FRAMES", 1)
    speech = (0.1 * np.random.default_rng(0).standard_normal(64 * 3)).astype(np.float32)
    with pytest.raises(ValueError, match="needs a role prompt"):
        tiny_personaplex(speech)
    out = tiny_personaplex(speech, role_prompt="abc")
    assert out["wav"].shape == speech.shape
    assert np.isfinite(out["wav"]).all()
