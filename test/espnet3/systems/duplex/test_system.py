"""Tests for the duplex systems' stage declarations."""

from espnet3.contract.stages import roles
from espnet3.systems.duplex.system import DuplexEvaluationSystem, DuplexSystem


def test_training_system_prepares_data_from_the_training_config():
    """Fine-tuning recipes create and train on the training config's data."""
    assert [s.name for s in DuplexSystem.stages] == [
        "create_dataset",
        "train",
        "infer",
        "measure",
        "pack_demo",
        "upload_demo",
    ]
    assert roles(DuplexSystem) == ("training", "inference", "metrics", "demo")


def test_evaluation_system_has_no_training():
    """Benchmark recipes prepare their test sets from the inference config."""
    assert [s.name for s in DuplexEvaluationSystem.stages] == [
        "create_dataset",
        "infer",
        "measure",
        "pack_demo",
        "upload_demo",
    ]
    assert roles(DuplexEvaluationSystem) == ("inference", "metrics", "demo")
