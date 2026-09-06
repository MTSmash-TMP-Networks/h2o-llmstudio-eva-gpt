"""Localize Text-only experiment GUI handling to the Experiment page.

PR #125 initially patched the generic recursive config renderer globally. The
renderer calls itself for every nested dataclass, so an Experiment-specific UI
extension should not live on that global recursion path. This module restores
the original renderer for recursive config construction and decorates only the
single top-level call made by the Experiment page.
"""

from __future__ import annotations

from typing import Any

from h2o_wave import Q


def _insert_after_dataset_separator(items: list[Any], controls: list[Any]) -> list[Any]:
    """Insert custom controls directly below the Dataset settings separator."""
    for index, item in enumerate(items):
        if getattr(item, "name", None) == "dataset_expander":
            return items[: index + 1] + controls + items[index + 1 :]
    return controls + items


def _localized_experiment_elements(
    cfg: Any,
    q: Q,
    limit: list[str] | None = None,
    pre: str = "experiment/start",
) -> list[Any]:
    from llm_studio.app_utils import text_only_training as text_mode

    original = text_mode._ORIGINAL_GET_UI_ELEMENTS_FOR_CFG
    if original is None:
        raise RuntimeError("Text-only training extension is not installed.")

    # Build the complete Experiment form exactly once through the original
    # renderer. Its recursive calls now also resolve to the original renderer.
    items = original(cfg=cfg, q=q, limit=limit, pre=pre)

    if pre != "experiment/start":
        return items

    dataset_cfg = getattr(cfg, "dataset", None)
    if dataset_cfg is None or not hasattr(dataset_cfg, "train_text_column"):
        return items

    text_mode._reset_experiment_mode_for_dataset(q)
    mode = text_mode._infer_experiment_mode(q)
    columns = text_mode._experiment_dataframe_columns(q)

    if mode == text_mode._TEXT_MODE:
        text_column = q.client[text_mode._EXPERIMENT_TEXT_COLUMN_KEY]
        persisted_prompt = text_mode._single_column(
            q.client["experiment/start/cfg/prompt_column"]
        )
        if text_column not in columns:
            if persisted_prompt in columns:
                text_column = persisted_prompt
            else:
                text_column = text_mode._preferred_text_column(columns)

        text_mode._apply_experiment_text_only_values(q, text_column)
        items = [
            item
            for item in items
            if getattr(item, "name", None)
            not in text_mode._EXPERIMENT_TEXT_ONLY_HIDDEN_FIELDS
        ]
    else:
        text_mode._apply_experiment_chat_values(
            q, mixed=mode == text_mode._MIXED_MODE
        )
        items = [
            item
            for item in items
            if getattr(item, "name", None)
            != "experiment/start/cfg/train_text_column"
        ]

    controls = text_mode._experiment_mode_controls(q, mode, columns)
    return _insert_after_dataset_separator(items, controls)


def install_experiment_training_mode_fix() -> None:
    """Keep Experiment Text-only UI out of the recursive global renderer."""
    from llm_studio.app_utils import text_only_training as text_mode
    from llm_studio.app_utils import utils as app_utils
    from llm_studio.app_utils.sections import experiment as experiment_section

    original = text_mode._ORIGINAL_GET_UI_ELEMENTS_FOR_CFG
    if original is None:
        raise RuntimeError("Install Text-only training mode before this fix.")

    # Critical part of the fix: restore the global function used recursively by
    # app_utils.get_ui_elements_for_cfg itself. Only experiment.py keeps the
    # localized decorator.
    app_utils.get_ui_elements_for_cfg = original
    experiment_section.get_ui_elements_for_cfg = _localized_experiment_elements
