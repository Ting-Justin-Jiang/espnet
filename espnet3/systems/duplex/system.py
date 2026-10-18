"""Duplex systems: the staged pipelines of full-duplex recipes."""

from typing import ClassVar

from espnet3.contract.stages import StageSpec
from espnet3.systems.base.system import BaseSystem


class DuplexSystem(BaseSystem):
    """System for recipes that fine-tune a released full-duplex model.

    The stage order is ``create_dataset -> train -> infer -> measure ->
    pack_demo -> upload_demo``, all inherited from ``BaseSystem``.
    ``create_dataset`` prepares the datasets of the training config; the
    model, :class:`~espnet3.systems.duplex.src.training.DuplexLM`, is
    instantiated directly from ``training_config.model``. The model needs no
    feature statistics, so there is no ``collect_stats``.

    Examples:
        >>> [s.name for s in DuplexSystem.stages][:3]
        ['create_dataset', 'train', 'infer']
    """

    stages: ClassVar[tuple[StageSpec, ...]] = (
        StageSpec(name="create_dataset", config="training", log_dir="data_dir"),
        StageSpec(name="train", config="training", log_dir="exp_dir"),
        StageSpec(name="infer", config="inference", log_dir="inference_dir"),
        StageSpec(name="measure", config="metrics", log_dir="inference_dir"),
        StageSpec(name="pack_demo", config="demo", log_dir="pack.out_dir"),
        StageSpec(name="upload_demo", config="demo", log_dir="pack.out_dir"),
    )


class DuplexEvaluationSystem(BaseSystem):
    """System for benchmark recipes that run released full-duplex models.

    The stage order is ``create_dataset -> infer -> measure -> pack_demo ->
    upload_demo``. Nothing is trained, so ``create_dataset`` prepares the
    test sets of the inference config.

    Examples:
        >>> [s.name for s in DuplexEvaluationSystem.stages]
        ['create_dataset', 'infer', 'measure', 'pack_demo', 'upload_demo']
    """

    stages: ClassVar[tuple[StageSpec, ...]] = (
        StageSpec(name="create_dataset", config="inference", log_dir="data_dir"),
        StageSpec(name="infer", config="inference", log_dir="inference_dir"),
        StageSpec(name="measure", config="metrics", log_dir="inference_dir"),
        StageSpec(name="pack_demo", config="demo", log_dir="pack.out_dir"),
        StageSpec(name="upload_demo", config="demo", log_dir="pack.out_dir"),
    )
