"""Tests for the Moshi language model and its frame-by-frame generator."""

import pytest
import torch

from espnet3.systems.duplex.src.lm import (
    TEXT_PADDING_ID,
    ZERO_TOKEN_ID,
    LMGen,
    LMModel,
    LMOutput,
    _delay,
    duplex_loss,
    sample_token,
)
from espnet3.systems.duplex.src.streaming import start_streaming, stop_streaming

CPU = torch.device("cpu")


def _lm(config):
    torch.manual_seed(0)
    return LMModel(**config).eval()


class _Recorder:
    """Wrap a callable and keep a copy of its first argument or of its output."""

    def __init__(self, fn, record_input):
        self.fn = fn
        self.record_input = record_input
        self.calls = []

    def __call__(self, *args):
        out = self.fn(*args)
        if self.record_input:
            self.calls.append(args[0].clone())
        else:
            self.calls.append(out.clone())
        return out


def test_sample_token_greedy_and_top1():
    """Temperature 0 and top-1 both take the argmax."""
    logits = torch.randn(3, 5, 10)
    assert torch.equal(sample_token(logits, 0.0, 4), logits.argmax(-1))
    assert torch.equal(sample_token(logits, 0.8, 1), logits.argmax(-1))


def test_sample_token_stays_in_top_k():
    """Sampled tokens are among the top-k logits."""
    torch.manual_seed(0)
    logits = torch.randn(500, 10)
    tokens = sample_token(logits, 1.0, 3)
    top = logits.topk(3, dim=-1).indices
    assert (tokens[:, None] == top).any(-1).all()


def test_first_frames_are_delayed(tiny_lm_config):
    """No frame comes out until the longest delay has passed."""
    gen = LMGen(_lm(tiny_lm_config), 0.8, 0.7, 5, 5, cuda_graph=False)
    gen.start(2, CPU)
    user = torch.randint(0, 16, (2, 3, 1))
    assert gen.step(user) is None
    out = gen.step(user)
    gen.stop()
    assert out.shape == (2, 4, 1)
    assert (out[:, 0] < 20).all() and (out[:, 1:] < 16).all() and (out >= 0).all()


def test_inputs_and_outputs_follow_delays(tiny_lm_config):
    """Each stream reads the token of its own delay; outputs undo the delays.

    TINY_LM delays: text 0, Moshi audio [0, 1, 1], user audio [0, 1, 1].
    """
    lm = _lm(tiny_lm_config)
    gen = LMGen(lm, 0.8, 0.7, 5, 5, cuda_graph=False)
    gen.start(1, CPU)
    inputs = _Recorder(gen._forward_text, record_input=True)
    audio = _Recorder(gen._depformer, record_input=False)
    gen._forward_text = inputs
    gen._depformer = audio
    torch.manual_seed(0)
    user = [torch.randint(0, 16, (1, 3, 1)) for _ in range(6)]
    outs = [gen.step(codes) for codes in user]
    gen.stop()

    initial = lm.initial_tokens(CPU)[0, :, 0]
    assert torch.equal(inputs.calls[0][0, :, 0], initial)
    delayed = torch.tensor([False, False, True, True, False, True, True])
    for s in range(1, 6):
        seen = inputs.calls[s][0, :, 0]
        # Moshi's streams read what it generated at the previous step ...
        expected = seen.clone()
        expected[1:4] = audio.calls[s - 1][0]
        # ... the user's semantic stream this frame, the acoustic ones the last.
        expected[4] = user[s][0, 0, 0]
        expected[5:] = user[s - 1][0, 1:, 0]
        # A delayed stream reads the initial token until its delay has passed.
        if s == 1:
            expected[delayed] = initial[delayed]
        assert torch.equal(seen, expected)
        # Frame s - 1 comes out at step s, with its delayed tokens from step s.
        frame = outs[s][0, :, 0]
        assert frame[0] == seen[0]
        assert frame[1] == audio.calls[s - 1][0, 0]
        assert torch.equal(frame[2:], audio.calls[s][0, 1:])


def _generate(lm, device, cuda_graph):
    """Run six seeded frames of constant user codes and return Moshi's tokens."""
    gen = LMGen(lm, 0.8, 0.7, 5, 5, cuda_graph=cuda_graph)
    gen.start(1, device)
    torch.manual_seed(3)
    outs = [gen.step(torch.full((1, 3, 1), i, device=device)) for i in range(6)]
    gen.stop()
    return torch.cat(outs[1:], dim=-1)


def test_generation_is_reproducible(tiny_lm_config):
    """The same seed gives the same conversation."""
    lm = _lm(tiny_lm_config)
    assert torch.equal(_generate(lm, CPU, False), _generate(lm, CPU, False))


def test_cuda_graph_needs_a_cuda_device(tiny_lm_config):
    """Capturing on CPU would record nothing, so it is refused."""
    gen = LMGen(_lm(tiny_lm_config), 0.8, 0.7, 5, 5, cuda_graph=True)
    with pytest.raises(ValueError, match="CUDA graphs need a CUDA device"):
        gen.start(1, CPU)


def test_stop_also_stops_an_interrupted_depformer(tiny_lm_config):
    """A frame interrupted inside the depformer does not block the next stream."""
    lm = _lm(tiny_lm_config)
    gen = LMGen(lm, 0.8, 0.7, 5, 5, cuda_graph=False)
    gen.start(1, CPU)
    start_streaming(lm.depformer, 1)
    gen.stop()
    gen.start(1, CPU)
    assert gen.step(torch.zeros(1, 3, 1, dtype=torch.long)) is None
    gen.stop()


def _codes(lm, num_frames, seed=1):
    """Random text and audio codes ``[1, 1 + n_q, num_frames]``."""
    generator = torch.Generator().manual_seed(seed)
    text = torch.randint(0, lm.text_card, (1, 1, num_frames), generator=generator)
    audio = torch.randint(0, lm.card, (1, lm.n_q, num_frames), generator=generator)
    return torch.cat([text, audio], dim=1)


def test_training_logits_are_the_streaming_logits(tiny_lm_config):
    """Training predicts each token as frame-by-frame generation would.

    Streaming reads frame ``s`` of the delayed streams, predicts the text of
    frame ``s`` and, fed the delayed tokens of step ``s``, each codebook
    ``k`` of frame ``s - delays[1 + k]``.
    """
    lm = _lm(tiny_lm_config)
    codes = _codes(lm, 8)
    with torch.no_grad():
        output = lm(codes)
        initial = lm.initial_tokens(CPU)
        delayed = torch.cat([initial, _delay(codes, lm.delays, initial)], dim=2)
        start_streaming(lm.transformer, 1)
        for step in range(8):
            out, text_logits = lm.forward_text(delayed[:, :, step : step + 1])
            torch.testing.assert_close(
                output.text_logits[0, 0, step], text_logits[0, 0, 0]
            )
            start_streaming(lm.depformer, 1)
            previous = delayed[:, 0, step + 1 : step + 2]
            for index in range(lm.dep_q):
                logits = lm.forward_depformer(index, previous, out)
                frame = step - lm.delays[1 + index]
                if frame >= 0:
                    torch.testing.assert_close(
                        output.logits[0, index, frame], logits[0, 0, 0]
                    )
                previous = delayed[:, 1 + index, step + 1 : step + 2]
            stop_streaming(lm.depformer)
        stop_streaming(lm.transformer)


def test_training_predictions_never_see_their_target(tiny_lm_config):
    """Changing a target token leaves its own prediction unchanged."""
    lm = _lm(tiny_lm_config)
    codes = _codes(lm, 6)
    with torch.no_grad():
        reference = lm(codes)
        for frame in range(6):
            changed = codes.clone()
            changed[0, 0, frame] = (codes[0, 0, frame] + 1) % lm.text_card
            text_logits = lm(changed).text_logits[0, 0, frame]
            assert torch.equal(text_logits, reference.text_logits[0, 0, frame])
            for index in range(lm.dep_q):
                if not reference.mask[0, index, frame]:
                    continue
                changed = codes.clone()
                changed[0, 1 + index, frame] = (
                    codes[0, 1 + index, frame] + 1
                ) % lm.card
                logits = lm(changed).logits[0, index, frame]
                assert torch.equal(logits, reference.logits[0, index, frame])


def test_padding_frames_are_masked_out(tiny_lm_config):
    """ZERO_TOKEN_ID frames are no target; the loss and gradients stay finite."""
    lm = _lm(tiny_lm_config).train()
    codes = torch.cat([_codes(lm, 7, seed=1), _codes(lm, 7, seed=2)])
    codes[1, :, 4:] = ZERO_TOKEN_ID
    output = lm(codes)
    assert not output.text_mask[1, 0, 4:].any()
    assert not output.mask[1, :, 4:].any()
    assert output.mask[0, 0].all()
    zero = torch.full((1,), ZERO_TOKEN_ID)
    assert not lm.text_emb(zero).any()
    text_loss, audio_loss = duplex_loss(output, codes, 100.0, 0.5)
    (text_loss + audio_loss).backward()
    assert torch.isfinite(text_loss) and torch.isfinite(audio_loss)
    assert all(torch.isfinite(p.grad).all() for p in lm.parameters())


def test_loss_weights_semantic_codebook_and_text_padding():
    """Each loss is the weighted mean of its per-target cross-entropies.

    Here every semantic and every padding target is predicted with certainty
    (cross-entropy 0) and every other target uniformly (log of the vocabulary
    size), so the weights decide the means.
    """
    card, text_card, dep_q, frames = 16, 20, 3, 4
    codes = torch.zeros(1, 1 + 2 * dep_q, frames, dtype=torch.long)
    codes[0, 0] = torch.tensor([TEXT_PADDING_ID, 7, 8, TEXT_PADDING_ID])
    logits = torch.zeros(1, dep_q, frames, card)
    logits[0, 0, :, 0] = 1e4
    text_logits = torch.zeros(1, 1, frames, text_card)
    text_logits[0, 0, [0, 3], TEXT_PADDING_ID] = 1e4
    output = LMOutput(
        logits=logits,
        mask=torch.ones(1, dep_q, frames, dtype=torch.bool),
        text_logits=text_logits,
        text_mask=torch.ones(1, 1, frames, dtype=torch.bool),
    )
    text_loss, audio_loss = duplex_loss(output, codes, 100.0, 0.5)
    log = torch.log(torch.tensor(float(card)))
    expected_audio = (2 * frames * log) / (100.0 * frames + 2 * frames)
    torch.testing.assert_close(audio_loss, expected_audio)
    text_log = torch.log(torch.tensor(float(text_card)))
    torch.testing.assert_close(text_loss, 2 * text_log / (2 + 2 * 0.5))
