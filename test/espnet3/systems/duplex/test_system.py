"""Tests for the duplex systems' stage declarations."""

from espnet3.contract.stages import roles
from espnet3.systems.duplex.system import DuplexEvaluationSystem, DuplexSystem


def test_training_system_publishes_its_model_and_demo():
    """Fine-tuning recipes train on the training config's data and publish."""
    assert [s.name for s in DuplexSystem.stages] == [
        "create_dataset",
        "train",
        "infer",
        "measure",
        "pack_model",
        "upload_model",
        "pack_demo",
        "upload_demo",
    ]
    assert roles(DuplexSystem) == (
        "training",
        "inference",
        "metrics",
        "publication",
        "demo",
    )


def test_evaluation_system_only_scores_released_models():
    """Benchmark recipes prepare their test sets from the inference config."""
    assert [s.name for s in DuplexEvaluationSystem.stages] == [
        "create_dataset",
        "infer",
        "measure",
    ]
    assert roles(DuplexEvaluationSystem) == ("inference", "metrics")
