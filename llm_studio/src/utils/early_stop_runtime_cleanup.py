"""Runtime cleanup for harmless Sliding Window and Early Stop side effects."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)

_INSTALLED = False
_ORIGINAL_PAD_TOKENS: Callable[..., dict] | None = None
_ORIGINAL_SAVE_PREDICTION_OUTPUTS: Callable[..., Any] | None = None


def _quiet_auxiliary_answer_padding(
    input_ids,
    attention_mask,
    max_length,
    pad_token_id,
    direction="left",
    prefix="",
):
    """Preserve legacy answer-field truncation without emitting misleading logs.

    Sliding Window training already slices the real training ``input_ids`` and
    ``labels`` to the configured window. The auxiliary ``answer_*`` field can still
    contain the full raw article, however. The base padding helper then logs that
    it truncates the sample even though only this non-loss helper field is being
    shortened. Pre-slice it exactly the same way so the base helper sees an input
    that is already within bounds.
    """
    if prefix == "answer_" and max_length < len(input_ids):
        input_ids = input_ids[-max_length:]
        attention_mask = attention_mask[-max_length:]

    if _ORIGINAL_PAD_TOKENS is None:
        raise RuntimeError("Original causal-LM pad_tokens helper is unavailable.")
    return _ORIGINAL_PAD_TOKENS(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        pad_token_id=pad_token_id,
        direction=direction,
        prefix=prefix,
    )


def _save_prediction_outputs_if_available(
    experiment_name: str,
    experiment_path: str,
):
    """Do not build a predictions ZIP when Early Stop happened before validation."""
    expected_files = (
        os.path.join(experiment_path, "validation_raw_predictions.pkl"),
        os.path.join(experiment_path, "validation_predictions.csv"),
    )
    if not any(os.path.isfile(path) for path in expected_files):
        logger.info(
            "No validation prediction files were produced before training stopped; "
            "skipping the predictions ZIP. The saved Early Stop model checkpoint "
            "and trainer state are unaffected."
        )
        return None

    if _ORIGINAL_SAVE_PREDICTION_OUTPUTS is None:
        raise RuntimeError("Original prediction export helper is unavailable.")
    return _ORIGINAL_SAVE_PREDICTION_OUTPUTS(experiment_name, experiment_path)


def install_early_stop_runtime_cleanup() -> None:
    """Install targeted compatibility wrappers before ``train.py`` is imported."""
    global _INSTALLED
    global _ORIGINAL_PAD_TOKENS
    global _ORIGINAL_SAVE_PREDICTION_OUTPUTS

    if _INSTALLED:
        return

    from llm_studio.src.datasets import text_causal_language_modeling_ds as causal_ds
    from llm_studio.src.utils import export_utils

    dataset_class = causal_ds.CustomDataset
    _ORIGINAL_PAD_TOKENS = dataset_class.pad_tokens
    _ORIGINAL_SAVE_PREDICTION_OUTPUTS = export_utils.save_prediction_outputs

    dataset_class.pad_tokens = staticmethod(_quiet_auxiliary_answer_padding)
    export_utils.save_prediction_outputs = _save_prediction_outputs_if_available

    _INSTALLED = True
