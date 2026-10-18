"""Tests for the Moshi system's stage declaration."""

from espnet3.contract.stages import roles
from espnet3.systems.duplex.system import DuplexSystem


def test_stages_have_no_training():
    """The system runs the released model: data, inference, metrics, demo."""
    assert [s.name for s in DuplexSystem.stages] == [
        "create_dataset",
        "infer",
        "measure",
        "pack_demo",
        "upload_demo",
    ]
    assert roles(DuplexSystem) == ("inference", "metrics", "demo")
