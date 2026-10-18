"""Tests for the Moshi system's stage declaration."""

from omegaconf import OmegaConf

import espnet3.systems.base.system as base_module
from espnet3.contract.stages import roles, stage_log_dir
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


def test_create_dataset_prepares_the_inference_test_sets(tmp_path, monkeypatch):
    """create_dataset reads the inference config's dataset block."""
    config = OmegaConf.create(
        {
            "data_dir": str(tmp_path / "data"),
            "recipe_dir": str(tmp_path),
            "create_dataset": {"recipe_dir": str(tmp_path)},
            "dataset": {"test": [{"name": "a", "data_src_args": {"task": "a"}}]},
        }
    )
    calls = []

    class Builder:
        def is_source_prepared(self, **kwargs):
            calls.append(("is_source_prepared", kwargs))
            return False

        def prepare_source(self, **kwargs):
            calls.append(("prepare_source", kwargs))

        def is_built(self, **kwargs):
            calls.append(("is_built", kwargs))
            return True

    class Module:
        DatasetBuilder = Builder

    monkeypatch.setattr(
        base_module, "load_dataset_module", lambda data_src, recipe_dir: Module()
    )
    system = MoshiSystem(inference_config=config)
    system.create_dataset()
    kwargs = {"recipe_dir": str(tmp_path)}
    assert calls == [
        ("is_source_prepared", kwargs),
        ("prepare_source", kwargs),
        ("is_built", kwargs),
    ]
    assert stage_log_dir(system, "create_dataset") == tmp_path / "data"
