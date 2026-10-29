"""Duplex systems: the staged pipelines of full-duplex recipes."""

from typing import ClassVar

from espnet3.contract.stages import StageSpec
from espnet3.systems.base.system import BaseSystem


class DuplexSystem(BaseSystem):
    """System for recipes that fine-tune and publish a full-duplex model.

    The stage order is ``create_dataset -> train -> infer -> measure ->
    pack_model -> upload_model -> pack_demo -> upload_demo``, all inherited
    from ``BaseSystem``. ``create_dataset`` prepares the datasets of the
    training config; the model,
    :class:`~espnet3.systems.duplex.src.training.DuplexLM`, is instantiated
    directly from ``training_config.model``. The model needs no feature
    statistics, so there is no ``collect_stats``. ``pack_model`` bundles the
    inference config with the trained model; the demo loads that bundle.

    Examples:
        >>> [s.name for s in DuplexSystem.stages][:3]
        ['create_dataset', 'train', 'infer']
    """

    stages: ClassVar[tuple[StageSpec, ...]] = (
        StageSpec(name="create_dataset", config="training", log_dir="data_dir"),
        StageSpec(name="train", config="training", log_dir="exp_dir"),
        StageSpec(name="infer", config="inference", log_dir="inference_dir"),
        StageSpec(name="measure", config="metrics", log_dir="inference_dir"),
        StageSpec(name="pack_model", config=("training", "publication")),
        StageSpec(name="upload_model", config="publication"),
        StageSpec(name="pack_demo", config="demo", log_dir="pack.out_dir"),
        StageSpec(name="upload_demo", config="demo", log_dir="pack.out_dir"),
    )


class DuplexEvaluationSystem(BaseSystem):
    """System for benchmark recipes that evaluate released full-duplex models.

    The stage order is ``create_dataset -> infer -> measure``. Nothing is
    trained, so ``create_dataset`` prepares the test sets of the inference
    config, which names the released model (as ``egs2`` benchmark recipes
    decode with ``--skip_train true --download_model``).

    Examples:
        >>> [s.name for s in DuplexEvaluationSystem.stages]
        ['create_dataset', 'infer', 'measure']
    """

    stages: ClassVar[tuple[StageSpec, ...]] = (
        StageSpec(name="create_dataset", config="inference", log_dir="data_dir"),
        StageSpec(name="infer", config="inference", log_dir="inference_dir"),
        StageSpec(name="measure", config="metrics", log_dir="inference_dir"),
    )
