"""Tests for the Moshi system's stage declaration."""

from espnet3.contract.stages import roles
from espnet3.systems.moshi.system import MoshiSystem


def test_stages_have_no_training():
    """The system runs the released model: data, inference, metrics, demo."""
    assert [s.name for s in MoshiSystem.stages] == [
        "create_dataset",
        "infer",
        "measure",
        "pack_demo",
        "upload_demo",
    ]
    assert roles(MoshiSystem) == ("inference", "metrics", "demo")
