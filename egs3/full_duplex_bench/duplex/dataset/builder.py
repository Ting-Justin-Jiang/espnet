"""Full-Duplex-Bench v1.0 dataset builder."""

from __future__ import annotations

import urllib.request
import zipfile
from importlib import resources
from pathlib import Path

from huggingface_hub import hf_hub_download

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults


def _load_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)


_CONFIG = _load_config()
_CFG = _CONFIG["builder"]
TASKS = _CONFIG["tasks"]


def benchmark_root(recipe_dir: str | Path) -> Path:
    """Return the directory holding one sub-directory per task."""
    return Path(recipe_dir) / _CFG["download_dir"] / _CFG["version"]


def backchannel_reference(recipe_dir: str | Path) -> Path:
    """Return the ICC backchannel timing reference used by the backchannel JSD."""
    return Path(recipe_dir) / _CFG["download_dir"] / _CFG["backchannel_reference"]


def sample_dirs(task_dir: Path) -> list[Path]:
    """Return the sample directories of a task, in sample-id order."""
    return sorted(
        (path for path in task_dir.iterdir() if (path / "input.wav").is_file()),
        key=lambda path: int(path.name),
    )


class FullDuplexBenchBuilder(DatasetBuilder):
    """Download Full-Duplex-Bench v1.0 into ``<recipe_dir>/downloads``.

    Each task is a zip archive of one directory per sample, holding the user's
    audio (``input.wav``) and the task annotation. The recipe reads the
    extracted directories directly, so there is nothing to build after the
    download.
    """

    def is_source_prepared(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check that every task has all its samples and the reference exists."""
        root = benchmark_root(recipe_dir)
        for task, spec in TASKS.items():
            if not (root / task).is_dir():
                return False
            if len(sample_dirs(root / task)) != spec["samples"]:
                return False
        return backchannel_reference(recipe_dir).is_file()

    def prepare_source(self, recipe_dir: str | Path, **_kwargs) -> None:
        """Download and extract every task, and download the reference.

        Args:
            recipe_dir: Recipe root directory.
            **_kwargs: Unused extra options for API compatibility.
        """
        root = benchmark_root(recipe_dir)
        root.mkdir(parents=True, exist_ok=True)
        for task in TASKS:
            archive = hf_hub_download(
                _CFG["hf_repo"],
                f"{_CFG['version']}/{task}.zip",
                repo_type="dataset",
                revision=_CFG["revision"],
            )
            with zipfile.ZipFile(archive) as zip_file:
                members = [
                    name
                    for name in zip_file.namelist()
                    if name.startswith(f"{task}/") and "/." not in name
                ]
                zip_file.extractall(root, members=members)
        urllib.request.urlretrieve(
            _CFG["backchannel_reference_url"], backchannel_reference(recipe_dir)
        )

    def is_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Return source readiness because this recipe has no build artifacts."""
        return self.is_source_prepared(recipe_dir=recipe_dir)

    def build(self, recipe_dir: str | Path, **_kwargs) -> None:
        """Check the download, the only artifact this recipe uses.

        Raises:
            FileNotFoundError: If a task or the reference is missing.
        """
        if not self.is_source_prepared(recipe_dir=recipe_dir):
            raise FileNotFoundError(
                f"Full-Duplex-Bench is incomplete under {benchmark_root(recipe_dir)}."
            )
