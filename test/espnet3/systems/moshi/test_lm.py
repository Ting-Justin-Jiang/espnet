"""Tests for the Moshi language model and its frame-by-frame generator."""

import torch

from espnet3.systems.moshi.src.lm import LMGen, LMModel, sample_token

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


def test_generation_is_reproducible(tiny_lm_config):
    """The same seed gives the same conversation."""
    lm = _lm(tiny_lm_config)

    def run():
        gen = LMGen(lm, 0.8, 0.7, 5, 5, cuda_graph=False)
        gen.start(1, CPU)
        torch.manual_seed(3)
        outs = [gen.step(torch.full((1, 3, 1), i)) for i in range(6)]
        gen.stop()
        return torch.cat(outs[1:], dim=-1)

    assert torch.equal(run(), run())
