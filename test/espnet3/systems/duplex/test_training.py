"""Tests for fine-tuning: the text stream and the training model."""

from unittest import mock

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from espnet2.train.collate_fn import CommonCollateFn
from espnet3.contract.kinds.audio import Audio
from espnet3.systems.duplex.inference import DuplexInference
from espnet3.systems.duplex.src.lm import (
    END_OF_TEXT_PADDING_ID,
    TEXT_PADDING_ID,
    LMModel,
)
from espnet3.systems.duplex.src.mimi import MimiModel
from espnet3.systems.duplex.src.training import DuplexLM, FSDPClipGradNorm, TextStream

FRAME = 64  # samples per frame of the tiny Mimi (800 Hz, 12.5 Hz)


@pytest.fixture
def checkpoints(tmp_path, tiny_mimi_config, tiny_lm_config):
    """Save random tiny Mimi and LM weights as released checkpoints."""
    torch.manual_seed(0)
    mimi = MimiModel(**tiny_mimi_config)
    for name, buffer in mimi.named_buffers():
        if name.endswith("embedding_sum"):
            buffer.copy_(torch.randn_like(buffer))
    save_file(mimi.state_dict(), tmp_path / "mimi.safetensors")
    save_file(LMModel(**tiny_lm_config).state_dict(), tmp_path / "lm.safetensors")
    return tmp_path


@pytest.fixture
def model(checkpoints, tiny_mimi_config, tiny_lm_config):
    """A tiny DuplexLM loaded from ``checkpoints``."""
    return DuplexLM(
        lm=tiny_lm_config,
        mimi=tiny_mimi_config,
        hf_repo=None,
        revision=None,
        lm_file=str(checkpoints / "lm.safetensors"),
        mimi_file=str(checkpoints / "mimi.safetensors"),
        first_codebook_weight=100.0,
        text_padding_weight=0.5,
    )


@pytest.fixture
def text_stream(tiny_tokenizer):
    """The tiny tokenizer's text stream at the tiny Mimi's rates."""
    return TextStream(None, None, str(tiny_tokenizer), sample_rate=800, frame_rate=12.5)


def _word(text, start, end):
    """One word as a ``segments`` entry."""
    return {"text": text, "start": start, "end": end}


def _batch(text_stream, frames):
    """Collate one random conversation window per entry of ``frames``."""
    rng = np.random.default_rng(0)
    items = []
    for index, num_frames in enumerate(frames):
        sample = {
            "speech": Audio(rng.uniform(-0.5, 0.5, (num_frames * FRAME, 2)), 800),
            "words": [_word("ghi", 0.1, 0.3), _word("de", 0.4, 0.6)],
        }
        items.append((str(index), text_stream(sample)))
    _, batch = CommonCollateFn(int_pad_value=-1)(items)
    return batch


def test_text_stream_places_words_on_frames(text_stream):
    """Tokens start at a word's frame, after an end-of-padding frame.

    "de" starts before "ghi" is fully written and cuts it short; "f" has no
    duration and is dropped.
    """
    encode = text_stream.tokenizer.encode
    ghi, de, jkl = encode("ghi"), encode("de"), encode("jkl")
    assert TEXT_PADDING_ID not in ghi + de + jkl
    sample = {
        "speech": Audio(np.zeros((12 * FRAME, 2), dtype=np.float32), 800),
        "words": [
            _word("de", 0.35, 0.6),
            _word("ghi", 0.1, 0.4),
            _word("f", 0.7, 0.7),
            _word("jkl", 0.7, 0.9),
        ],
    }
    out = text_stream(sample)
    epad = END_OF_TEXT_PADDING_ID
    assert out["text"].tolist() == [epad, *ghi[:3], *de, epad, *jkl]
    assert out["speech"].dtype == np.float32
    assert out["speech"].shape == (12 * FRAME, 2)


def test_text_stream_refuses_other_rates_and_mono(text_stream):
    """A recording at another rate would be read at the wrong speed."""
    stereo = np.zeros((12 * FRAME, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="2 channels at 800 Hz, got 2 at 1600"):
        text_stream({"speech": Audio(stereo, 1600), "words": []})
    with pytest.raises(ValueError, match="got 1 at 800"):
        text_stream({"speech": Audio(stereo[:, 0], 800), "words": []})


def test_training_step_trains_the_language_model_only(model, text_stream):
    """Gradients reach every LM parameter and no Mimi parameter."""
    model.train()
    assert model.lm.training
    assert not model.mimi.training
    loss, stats, weight = model(**_batch(text_stream, [12, 9]))
    loss.backward()
    assert torch.isfinite(loss)
    assert set(stats) == {"loss", "text_loss", "audio_loss"}
    assert weight is None
    assert all(p.grad is None for p in model.mimi.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.lm.parameters())


def test_text_and_audio_lengths_must_agree(model, text_stream):
    """A text stream shorter than its audio would shift every target."""
    batch = _batch(text_stream, [12])
    batch["text"] = batch["text"][:, :-1]
    batch["text_lengths"] = batch["text_lengths"] - 1
    with pytest.raises(ValueError, match="Text streams of"):
        model(**batch)


def test_trained_language_model_loads_for_inference(
    model, checkpoints, tiny_tokenizer, tiny_lm_config, tiny_mimi_config
):
    """``lm_checkpoint`` takes the LM of a DuplexLM state dict, in bfloat16."""
    with torch.no_grad():
        model.lm.text_linear.weight.add_(1.0)
    torch.save(model.state_dict(), checkpoints / "trained.pth")
    inference = DuplexInference(
        lm=tiny_lm_config,
        mimi=tiny_mimi_config,
        hf_repo=None,
        revision=None,
        lm_file="unused",
        mimi_file=str(checkpoints / "mimi.safetensors"),
        tokenizer_file=str(tiny_tokenizer),
        generator="moshi",
        temp=0.8,
        temp_text=0.7,
        top_k=5,
        top_k_text=5,
        seed=0,
        cuda_graph=False,
        lm_checkpoint=str(checkpoints / "trained.pth"),
        device="cpu",
    )
    expected = model.lm.text_linear.weight.to(torch.bfloat16)
    assert torch.equal(inference.lm.text_linear.weight, expected)


def test_clipping_goes_through_the_root_fsdp_module():
    """Only the root FSDP module can take the norm over every rank's shard."""
    trainer = mock.Mock()
    FSDPClipGradNorm(max_norm=1.0).on_before_optimizer_step(trainer, None, None)
    trainer.strategy.model.clip_grad_norm_.assert_called_once_with(1.0)
