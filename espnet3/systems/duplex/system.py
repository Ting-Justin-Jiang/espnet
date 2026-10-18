"""Moshi system: the staged pipeline of a full-duplex evaluation recipe."""

from typing import ClassVar

from espnet3.contract.stages import StageSpec
from espnet3.systems.base.system import BaseSystem


class DuplexSystem(BaseSystem):
    """System for recipes that run the released Moshi in full duplex.

    The stage order is ``create_dataset -> infer -> measure -> pack_demo ->
    upload_demo``, all inherited from ``BaseSystem``. The system runs the
    released model and has no training stage, so ``create_dataset``
    prepares the test sets of the inference config.

    Examples:
        >>> [s.name for s in DuplexSystem.stages]
        ['create_dataset', 'infer', 'measure', 'pack_demo', 'upload_demo']
    """

    stages: ClassVar[tuple[StageSpec, ...]] = (
        StageSpec(name="create_dataset", config="inference", log_dir="data_dir"),
        StageSpec(name="infer", config="inference", log_dir="inference_dir"),
        StageSpec(name="measure", config="metrics", log_dir="inference_dir"),
        StageSpec(name="pack_demo", config="demo", log_dir="pack.out_dir"),
        StageSpec(name="upload_demo", config="demo", log_dir="pack.out_dir"),
    )
