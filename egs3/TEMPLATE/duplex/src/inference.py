"""Inference output formatting shared by Moshi recipes."""


def build_output(data, model_output, idx):
    """Turn one conversation into the record the infer stage writes.

    ``conf/inference.yaml`` points ``output_fn`` here. ``wav`` (Moshi's
    channel, frame-aligned with the user's) is written as a wav file;
    ``text`` (Moshi's inner monologue) and ``step_ms`` go to SCP files.

    Args:
        data: Dataset item with ``utt_id``.
        model_output: Output of ``MoshiInference.__call__``.
        idx: Index of the item within its test set (unused).

    Returns:
        Dict with ``utt_id``, ``wav``, ``text`` and ``step_ms``.
    """
    return {
        "utt_id": data["utt_id"],
        "wav": model_output["wav"],
        "text": model_output["text"],
        "step_ms": model_output["step_ms"],
    }
