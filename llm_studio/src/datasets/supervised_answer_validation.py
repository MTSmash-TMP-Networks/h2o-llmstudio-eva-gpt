"""Validation guard for malformed supervised causal-LM chat rows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from llm_studio.src.datasets.conversation_chain_handler import get_plain_text_mask
from llm_studio.src.datasets.text_utils import clean_missing_text_values


def _format_row_reference(df: pd.DataFrame, cfg: Any, index: Any) -> str:
    """Return a compact, useful reference for one malformed dataframe row."""
    parts = [f"row={index}"]

    id_column = getattr(cfg.dataset, "id_column", None)
    if id_column not in (None, "None") and id_column in df.columns:
        parts.append(f"id={df.at[index, id_column]!r}")

    parent_id_column = getattr(cfg.dataset, "parent_id_column", None)
    if parent_id_column not in (None, "None") and parent_id_column in df.columns:
        parts.append(f"parent_id={df.at[index, parent_id_column]!r}")

    return " ".join(parts)


def _validate_supervised_answers(df: pd.DataFrame, cfg: Any, mode: str) -> None:
    """Reject chat rows that have no assistant answer.

    Raw ``Text`` rows used for continued pretraining are intentionally exempt.
    They are identified by the same plain-text detection logic used by the
    conversation-chain handler, so this guard cannot accidentally reject valid
    continued-pretraining samples.
    """
    answer_column = getattr(cfg.dataset, "answer_column", None)
    if not isinstance(answer_column, str) or answer_column not in df.columns:
        return

    answers = clean_missing_text_values(df[answer_column])
    plain_text_mask = get_plain_text_mask(df, cfg)
    invalid_mask = (~plain_text_mask) & answers.str.strip().eq("")

    if not invalid_mask.any():
        return

    invalid_indices = df.index[invalid_mask].tolist()
    examples = [_format_row_reference(df, cfg, index) for index in invalid_indices[:10]]
    omitted = len(invalid_indices) - len(examples)

    details = "\n".join(f"  - {example}" for example in examples)
    if omitted > 0:
        details += f"\n  - ... and {omitted} more"

    mode_prefix = f"{mode} " if mode else ""
    raise AssertionError(
        f"Invalid {mode_prefix}supervised training data: found "
        f"{len(invalid_indices)} row(s) with an empty assistant answer in "
        f"column '{answer_column}'.\n"
        "A supervised chat turn must contain both the user prompt and the "
        "assistant answer. Otherwise parent_id chaining can create malformed "
        "sequences such as '<|Assistentin|><|Benutzer|>'.\n"
        "Pure continued-pretraining rows in the 'Text' column are allowed and "
        "are not affected by this check.\n"
        f"Examples:\n{details}\n"
        "Fix or remove these chat rows before starting training."
    )


def install_supervised_answer_validation() -> None:
    """Install the validation guard on the causal-LM dataset class once."""
    from llm_studio.src.datasets import text_causal_language_modeling_ds

    dataset_cls = text_causal_language_modeling_ds.CustomDataset
    if getattr(dataset_cls, "_supervised_answer_validation_installed", False):
        return

    # The exported dataset may be a Fast/Structure-Aware subclass. ``sanity_check``
    # is a classmethod inherited from the base causal-LM dataset, so reading only
    # ``dataset_cls.__dict__`` can fail even though the method is available.
    original = dataset_cls.sanity_check.__func__

    def sanity_check(cls, df: pd.DataFrame, cfg: Any, mode: str = "train"):
        _validate_supervised_answers(df, cfg, mode)
        return original(cls, df, cfg, mode)

    dataset_cls.sanity_check = classmethod(sanity_check)
    dataset_cls._supervised_answer_validation_installed = True
