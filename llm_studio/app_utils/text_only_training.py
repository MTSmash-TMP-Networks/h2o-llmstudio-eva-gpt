"""GUI and runtime support for pure-text continued pretraining datasets."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd
from h2o_wave import Q, ui

logger = logging.getLogger(__name__)

_TEXT_MODE = "text_only"
_CHAT_MODE = "chat"
_MODE_KEY = "dataset/import/training_mode"
_TEXT_COLUMN_KEY = "dataset/import/text_column"

_ORIGINAL_GET_DATASET_ELEMENTS = None
_ORIGINAL_GET_PLAIN_TEXT_MASK = None
_ORIGINAL_CONFIGURED_TEXT_COLUMNS = None
_ORIGINAL_APPLY_PLAIN_TEXT_ROWS = None
_INSTALLED = False

_CHAT_DATASET_FIELDS = {
    "dataset/import/cfg/system_column",
    "dataset/import/cfg/prompt_column",
    "dataset/import/cfg/answer_column",
    "dataset/import/cfg/parent_id_column",
    "dataset/import/cfg/id_column",
    "dataset/import/cfg/train_text_column",
}


def _single_column(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 1:
        return str(value[0])
    return None


def is_text_only_config(cfg: Any) -> bool:
    """Detect the persisted text-only representation without a new YAML field."""
    dataset = getattr(cfg, "dataset", cfg)
    if not bool(getattr(dataset, "train_text_column", False)):
        return False

    prompt_column = _single_column(getattr(dataset, "prompt_column", None))
    answer_column = _single_column(getattr(dataset, "answer_column", None))
    if prompt_column is None or answer_column is None:
        return False

    return prompt_column == answer_column


def get_text_only_column(cfg: Any) -> str | None:
    if not is_text_only_config(cfg):
        return None
    dataset = getattr(cfg, "dataset", cfg)
    return _single_column(getattr(dataset, "prompt_column", None))


def _dataframe_columns(q: Q) -> list[str]:
    dataframe = q.client["dataset/import/cfg/dataframe"]
    if isinstance(dataframe, pd.DataFrame):
        return [str(column) for column in dataframe.columns]
    return []


def _preferred_text_column(columns: list[str]) -> str:
    for candidate in (
        "Text",
        "text",
        "content",
        "Content",
        "document",
        "Document",
        "body",
        "Body",
    ):
        if candidate in columns:
            return candidate
    return columns[0] if columns else "Text"


def _infer_import_mode(q: Q, cfg: Any, columns: list[str]) -> str:
    current = q.client[_MODE_KEY]
    if current in (_CHAT_MODE, _TEXT_MODE):
        return current

    if is_text_only_config(cfg):
        return _TEXT_MODE

    # A single obvious corpus-text column is almost always continued pretraining.
    # This also makes wikimedia/wikipedia (mapped to `Text`) default correctly.
    if len(columns) == 1 and columns[0] in {
        "Text",
        "text",
        "content",
        "Content",
        "document",
        "Document",
        "body",
        "Body",
    }:
        return _TEXT_MODE

    return _CHAT_MODE


def _apply_text_only_values(q: Q, text_column: str) -> None:
    q.client[_MODE_KEY] = _TEXT_MODE
    q.client[_TEXT_COLUMN_KEY] = text_column

    # Persist text-only mode entirely with existing dataset config fields.
    # prompt == answer plus train_text_column=True is intentionally used as the
    # durable marker so old/new dataset YAML files stay compatible.
    q.client["dataset/import/cfg/train_text_column"] = True
    q.client["dataset/import/cfg/system_column"] = "None"
    q.client["dataset/import/cfg/prompt_column"] = (text_column,)
    q.client["dataset/import/cfg/answer_column"] = text_column
    q.client["dataset/import/cfg/parent_id_column"] = "None"
    q.client["dataset/import/cfg/id_column"] = "None"


def _apply_chat_values(q: Q) -> None:
    q.client[_MODE_KEY] = _CHAT_MODE
    q.client["dataset/import/cfg/train_text_column"] = False

    # When switching away from text-only, clear fields that were force-set to the
    # text column so LLM Studio can choose its normal preferred chat columns again.
    text_column = q.client[_TEXT_COLUMN_KEY]
    prompt = q.client["dataset/import/cfg/prompt_column"]
    answer = q.client["dataset/import/cfg/answer_column"]
    if text_column and _single_column(prompt) == text_column:
        q.client["dataset/import/cfg/prompt_column"] = None
    if text_column and _single_column(answer) == text_column:
        q.client["dataset/import/cfg/answer_column"] = None
    if q.client["dataset/import/cfg/system_column"] == "None":
        q.client["dataset/import/cfg/system_column"] = None
    if q.client["dataset/import/cfg/parent_id_column"] == "None":
        q.client["dataset/import/cfg/parent_id_column"] = None
    if q.client["dataset/import/cfg/id_column"] == "None":
        q.client["dataset/import/cfg/id_column"] = None


def _mode_controls(q: Q, mode: str, columns: list[str]) -> list[Any]:
    controls: list[Any] = [
        ui.dropdown(
            name=_MODE_KEY,
            label="Training mode",
            value=mode,
            required=True,
            trigger=True,
            choices=[
                ui.choice(_CHAT_MODE, "Chat / instruction training"),
                ui.choice(_TEXT_MODE, "Text only / continued pretraining"),
            ],
            tooltip=(
                "Choose Text only for Wikipedia, books, documentation and other "
                "plain-text corpora that should be learned without chat formatting."
            ),
        )
    ]

    if mode == _TEXT_MODE:
        text_column = q.client[_TEXT_COLUMN_KEY]
        if text_column not in columns:
            text_column = _preferred_text_column(columns)
            q.client[_TEXT_COLUMN_KEY] = text_column

        controls.extend(
            [
                ui.dropdown(
                    name=_TEXT_COLUMN_KEY,
                    label="Text column",
                    value=text_column,
                    required=True,
                    trigger=True,
                    choices=[ui.choice(column, column) for column in columns],
                    tooltip="Every non-empty row in this column is trained as raw text.",
                ),
                ui.message_bar(
                    type="info",
                    text=(
                        "Text-only mode: the selected column is trained directly as "
                        "continued-pretraining text. No System, Prompt, Assistant, "
                        "Parent ID or chat template is used."
                    ),
                ),
            ]
        )

    return controls


def _get_dataset_elements_with_training_mode(cfg: Any, q: Q) -> list[Any]:
    """Add the training-mode selector and hide irrelevant chat fields."""
    if _ORIGINAL_GET_DATASET_ELEMENTS is None:
        raise RuntimeError("Text-only training extension is not installed.")

    items = _ORIGINAL_GET_DATASET_ELEMENTS(cfg=cfg, q=q)

    # Only causal-LM datasets expose this legacy raw-text switch. Other problem
    # types keep their original dataset UI untouched.
    dataset_cfg = getattr(cfg, "dataset", None)
    if dataset_cfg is None or not hasattr(dataset_cfg, "train_text_column"):
        return items

    columns = _dataframe_columns(q)
    mode = _infer_import_mode(q, dataset_cfg, columns)

    if mode == _TEXT_MODE:
        text_column = q.client[_TEXT_COLUMN_KEY]
        if text_column not in columns:
            text_column = _preferred_text_column(columns)
        _apply_text_only_values(q, text_column)
        items = [
            item
            for item in items
            if getattr(item, "name", None) not in _CHAT_DATASET_FIELDS
        ]
    else:
        _apply_chat_values(q)
        # Hide the legacy switch; the new Training mode control replaces it.
        items = [
            item
            for item in items
            if getattr(item, "name", None)
            != "dataset/import/cfg/train_text_column"
        ]

    return _mode_controls(q, mode, columns) + items


def _configured_text_columns_with_text_only(cfg: Any) -> list[str]:
    if _ORIGINAL_CONFIGURED_TEXT_COLUMNS is None:
        return []

    columns = _ORIGINAL_CONFIGURED_TEXT_COLUMNS(cfg)
    text_column = get_text_only_column(cfg)
    if text_column is not None:
        columns = [column for column in columns if column != text_column]
    return columns


def _plain_text_mask_with_text_only(df: pd.DataFrame, cfg: Any) -> pd.Series:
    """In text-only mode every non-empty row in the selected column is raw text."""
    text_column = get_text_only_column(cfg)
    if text_column is None:
        if _ORIGINAL_GET_PLAIN_TEXT_MASK is None:
            return pd.Series(False, index=df.index)
        return _ORIGINAL_GET_PLAIN_TEXT_MASK(df, cfg)

    if text_column not in df.columns:
        return pd.Series(False, index=df.index)

    from llm_studio.src.datasets.text_utils import clean_missing_text_values

    return clean_missing_text_values(df[text_column]).str.strip() != ""


def _apply_plain_text_rows_with_selected_column(self: Any, df: pd.DataFrame) -> None:
    text_column = get_text_only_column_from_handler(self, df)
    if text_column is None:
        if _ORIGINAL_APPLY_PLAIN_TEXT_ROWS is not None:
            _ORIGINAL_APPLY_PLAIN_TEXT_ROWS(self, df)
        return

    if not self.plain_text_mask.any():
        return

    from llm_studio.src.datasets import conversation_chain_handler as chain_module
    from llm_studio.src.datasets.text_utils import clean_missing_text_values

    chain_module._patch_plain_text_custom_dataset()
    plain_texts = clean_missing_text_values(df[text_column]).tolist()
    plain_text_flags = self.plain_text_mask.tolist()
    self.prompts = [
        chain_module.PLAIN_TEXT_PROMPT if is_plain_text else prompt
        for prompt, is_plain_text in zip(self.prompts, plain_text_flags, strict=False)
    ]
    self.answers = [
        plain_texts[idx] if is_plain_text else answer
        for idx, (answer, is_plain_text) in enumerate(
            zip(self.answers, plain_text_flags, strict=False)
        )
    ]
    self.systems = [
        "" if is_plain_text else system
        for system, is_plain_text in zip(self.systems, plain_text_flags, strict=False)
    ]


def get_text_only_column_from_handler(handler: Any, df: pd.DataFrame) -> str | None:
    """Recover selected text column from the handler's prompt/answer setup."""
    # The handler does not retain cfg by design. In text-only mode get_texts() was
    # configured from the same single column as get_answers(), so find the matching
    # dataframe column by comparing the prepared values.
    if not getattr(handler, "plain_text_mask", pd.Series(dtype=bool)).any():
        return None

    for column in df.columns:
        values = df[column].fillna("").astype(str).tolist()
        prompts = [str(value) for value in handler.prompts]
        answers = [str(value) for value in handler.answers]
        if values == prompts and values == answers:
            return str(column)
    return None


def install_text_only_training_mode(handle: Callable[..., Any]) -> Callable[..., Any]:
    """Install GUI/runtime patches and return a wrapped Wave request handler."""
    global _INSTALLED
    global _ORIGINAL_GET_DATASET_ELEMENTS
    global _ORIGINAL_GET_PLAIN_TEXT_MASK
    global _ORIGINAL_CONFIGURED_TEXT_COLUMNS
    global _ORIGINAL_APPLY_PLAIN_TEXT_ROWS

    if _INSTALLED:
        return handle

    from llm_studio.app_utils import utils as app_utils
    from llm_studio.app_utils.sections import dataset as dataset_section
    from llm_studio.src.datasets import conversation_chain_handler as chain_module

    _ORIGINAL_GET_DATASET_ELEMENTS = app_utils.get_dataset_elements
    _ORIGINAL_GET_PLAIN_TEXT_MASK = chain_module.get_plain_text_mask
    _ORIGINAL_CONFIGURED_TEXT_COLUMNS = chain_module._configured_text_columns
    _ORIGINAL_APPLY_PLAIN_TEXT_ROWS = chain_module.ConversationChainHandler._apply_plain_text_rows

    app_utils.get_dataset_elements = _get_dataset_elements_with_training_mode
    dataset_section.get_dataset_elements = _get_dataset_elements_with_training_mode
    chain_module.get_plain_text_mask = _plain_text_mask_with_text_only
    chain_module._configured_text_columns = _configured_text_columns_with_text_only
    chain_module.ConversationChainHandler._apply_plain_text_rows = (
        _apply_plain_text_rows_with_selected_column
    )

    async def handle_with_text_training_mode(q: Q) -> None:
        submission = q.args.__wave_submission_name__

        if submission == _MODE_KEY:
            selected_mode = q.args[_MODE_KEY] or _CHAT_MODE
            q.client[_MODE_KEY] = selected_mode
            if selected_mode == _TEXT_MODE:
                columns = _dataframe_columns(q)
                text_column = q.client[_TEXT_COLUMN_KEY]
                if text_column not in columns:
                    text_column = _preferred_text_column(columns)
                _apply_text_only_values(q, text_column)
            else:
                _apply_chat_values(q)

            await dataset_section.dataset_import(
                q,
                step=3,
                edit=bool(q.client["dataset/import/edit"]),
            )
            return

        if submission == _TEXT_COLUMN_KEY:
            text_column = str(q.args[_TEXT_COLUMN_KEY] or "").strip()
            if text_column:
                _apply_text_only_values(q, text_column)
            await dataset_section.dataset_import(
                q,
                step=3,
                edit=bool(q.client["dataset/import/edit"]),
            )
            return

        await handle(q)

    _INSTALLED = True
    logger.info("Text-only continued-pretraining GUI mode enabled")
    return handle_with_text_training_mode
