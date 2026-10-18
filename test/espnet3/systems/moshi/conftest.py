"""Tiny Moshi configurations shared by the Moshi tests.

The released model is far too large for unit tests, so these configs keep
every structural feature of it (a language model window shorter than the test
conversations, per-step depformer weights, acoustic delays, a learnt stride-4
resampler with replicate padding, split quantization) at toy sizes.
"""

import copy

import pytest
import sentencepiece as spm

TINY_MIMI = {
    "sample_rate": 800,
    "frame_rate": 12.5,
    "num_codebooks": 3,
    "seanet": {
        "channels": 1,
        "dimension": 16,
        "n_filters": 4,
        "n_residual_layers": 1,
        "ratios": [4, 4],
        "kernel_size": 7,
        "last_kernel_size": 3,
        "residual_kernel_size": 3,
        "dilation_base": 2,
        "compress": 2,
    },
    "transformer": {
        "d_model": 16,
        "num_heads": 2,
        "num_layers": 2,
        "dim_feedforward": 32,
        "context": 64,
        "rope_max_period": 100,
        "norm": "layer_norm",
        "layer_scale": 0.01,
        "gating": "none",
        "weights_per_step": 0,
    },
    "quantizer": {
        "dimension": 8,
        "input_dimension": 16,
        "output_dimension": 16,
        "n_q": 4,
        "n_q_semantic": 1,
        "bins": 16,
    },
}

TINY_LM = {
    "dim": 32,
    "num_heads": 4,
    "num_layers": 2,
    "hidden_scale": 2.0,
    "context": 5,
    "max_period": 100,
    "text_card": 20,
    "card": 16,
    "n_q": 6,
    "dep_q": 3,
    "delays": [0, 0, 1, 1, 0, 1, 1],
    "depformer_dim": 16,
    "depformer_dim_feedforward": 32,
    "depformer_num_heads": 2,
    "depformer_num_layers": 2,
}


@pytest.fixture(scope="session")
def tiny_tokenizer(tmp_path_factory):
    """Train a SentencePiece model with ``TINY_LM['text_card']`` pieces.

    Trained once per session, in fixture setup, outside the per-test budget.
    """
    directory = tmp_path_factory.mktemp("spm")
    corpus = directory / "corpus.txt"
    # 18 letters, the word boundary and <unk>: 20 pieces.
    corpus.write_text("abc def ghi jkl mno pqr\n" * 20)
    spm.SentencePieceTrainer.Train(
        input=str(corpus),
        model_prefix=str(directory / "tiny"),
        vocab_size=TINY_LM["text_card"],
        model_type="char",
        character_coverage=1.0,
        bos_id=-1,
        eos_id=-1,
    )
    model = directory / "tiny.model"
    assert spm.SentencePieceProcessor(model_file=str(model)).get_piece_size() == 20
    return model


@pytest.fixture
def tiny_mimi_config():
    """Return a fresh copy of ``TINY_MIMI``."""
    return copy.deepcopy(TINY_MIMI)


@pytest.fixture
def tiny_lm_config():
    """Return a fresh copy of ``TINY_LM``."""
    return copy.deepcopy(TINY_LM)
