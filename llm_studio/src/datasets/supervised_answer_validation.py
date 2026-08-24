"""Validation guard for malformed supervised causal-LM chat rows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from llm_studio.src.datasets.context_only_chain_turns import (
    get_supervised_answer_mask,
    get_valid_context_only_mask,
)
from llm_studio.src.datasets.conversation_chain_handler import get_plain_text_mask


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
    """Reject empty assistant answers unless they are chained context-only turns.

    A chat row without an assistant answer is valid when it is an ancestor of a
    later row that does contain an assistant answer. Such rows represent multiple
    consecutive user/context messages before the assistant eventually responds.
    They stay in the prompt history but never become standalone loss targets.

    Raw ``Text`` rows used for continued pretraining remain explicitly exempt.
    """
    answer_column = getattr(cfg.dataset, "answer_column", None)
    if not isinstance(answer_column, (str, list, tuple)):
        return

    answer_mask = get_supervised_answer_mask(df, cfg)
    plain_text_mask = get_plain_text_mask(df, cfg)
    context_only_mask = get_valid_context_only_mask(df, cfg)
    invalid_mask = (~plain_text_mask) & (~answer_mask) & (~context_only_mask)

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
        f"{len(invalid_indices)} row(s) with an empty assistant answer that are "
        "not used as context for a later answered turn.\n"
        "Empty assistant answers are allowed for chained user/context-only turns "
        "when a descendant row later contains the assistant response. Those turns "
        "remain prompt context and are not trained as standalone targets.\n"
        "Terminal or orphan chat rows without an assistant answer are invalid.\n"
        "Pure continued-pretraining rows in the 'Text' column are allowed and are "
        "not affected by this check.\n"
        f"Examples:\n{details}\n"
        "Add the later answered child turn, add the missing assistant answer, or "
        "remove the unfinished row before starting training."
    )


def install_supervised_answer_validation() -> None:
    """Install the validation guard on the causal-LM dataset class once."""
    from llm_studio.src.datasets import text_causal_language_modeling_ds

    dataset_cls = text_causal_language_modeling_ds.CustomDataset
    if getattr(dataset_cls, "_supervised_answer_validation_installed", False):
        return

    original = dataset_cls.sanity_check.__func__

    def sanity_check(cls, df: pd.DataFrame, cfg: Any, mode: str = "train"):
        _validate_supervised_answers(df, cfg, mode)
        return original(cls, df, cfg, mode)

    dataset_cls.sanity_check = classmethod(sanity_check)
    dataset_cls._supervised_answer_validation_installed = True
