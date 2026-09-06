"""GUI and runtime support for pure-text continued pretraining datasets."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd
from h2o_wave import Q, ui

logger = logging.getLogger(__name__)

_TEXT_MODE = "text_only"
_MIXED_MODE = "mixed"
_CHAT_MODE = "chat"
_MODE_KEY = "dataset/import/training_mode"
_TEXT_COLUMN_KEY = "dataset/import/text_column"
_EXPERIMENT_MODE_KEY = "experiment/start/training_mode"
_EXPERIMENT_TEXT_COLUMN_KEY = "experiment/start/text_column"
_EXPERIMENT_DATASET_KEY = "experiment/start/training_mode_dataset"

_ORIGINAL_GET_DATASET_ELEMENTS = None
_ORIGINAL_GET_UI_ELEMENTS_FOR_CFG = None
_ORIGINAL_GET_PLAIN_TEXT_MASK = None
_ORIGINAL_CONFIGURED_TEXT_COLUMNS = None
_ORIGINAL_HANDLER_INIT = None
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

_EXPERIMENT_TEXT_ONLY_HIDDEN_FIELDS = {
    "experiment/start/cfg/train_text_column",
    "experiment/start/cfg/system_column",
    "experiment/start/cfg/prompt_column",
    "experiment/start/cfg/prompt_column_separator",
    "experiment/start/cfg/answer_column",
    "experiment/start/cfg/parent_id_column",
    "experiment/start/cfg/id_column",
    "experiment/start/cfg/text_system_start",
    "experiment/start/cfg/text_prompt_start",
    "experiment/start/cfg/text_answer_separator",
    "experiment/start/cfg/add_eos_token_to_system",
    "experiment/start/cfg/add_eos_token_to_prompt",
    "experiment/start/cfg/add_eos_token_to_answer",
    "experiment/start/cfg/limit_chained_samples",
    "experiment/start/cfg/mask_prompt_labels",
    "experiment/start/cfg/mask_prompt_user_text_only",
    "experiment/start/cfg/only_last_answer",
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


def _training_mode_from_values(
    train_text_column: Any,
    prompt_column: Any,
    answer_column: Any,
) -> str:
    """Translate persisted dataset fields into the explicit GUI training mode."""
    if not bool(train_text_column):
        return _CHAT_MODE

    prompt = _single_column(prompt_column)
    answer = _single_column(answer_column)
    if prompt is not None and answer is not None and prompt == answer:
        return _TEXT_MODE

    return _MIXED_MODE


def _dataframe_columns(q: Q) -> list[str]:
    dataframe = q.client["dataset/import/cfg/dataframe"]
    if isinstance(dataframe, pd.DataFrame):
        return [str(column) for column in dataframe.columns]
    return []


def _experiment_dataframe_columns(q: Q) -> list[str]:
    dataframe = q.client["experiment/start/cfg/dataframe"]
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
    if current in (_CHAT_MODE, _MIXED_MODE, _TEXT_MODE):
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

    # Preserve the fork's existing mixed-chat-plus-Text behavior by default.
    if bool(getattr(cfg, "train_text_column", False)):
        return _MIXED_MODE

    return _CHAT_MODE


def _restore_chat_columns(q: Q) -> None:
    """Clear values that were force-set while Text-only mode was active."""
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


def _apply_chat_values(q: Q, *, mixed: bool) -> None:
    q.client[_MODE_KEY] = _MIXED_MODE if mixed else _CHAT_MODE
    q.client["dataset/import/cfg/train_text_column"] = mixed
    _restore_chat_columns(q)


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
                ui.choice(_MIXED_MODE, "Mixed chat + Text training"),
                ui.choice(_TEXT_MODE, "Text only / continued pretraining"),
            ],
            tooltip=(
                "Choose Text only for Wikipedia, books, documentation and other "
                "plain-text corpora. Mixed keeps the existing chat plus Text mode."
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
                    tooltip=(
                        "Every non-empty row in this column is trained as raw text."
                    ),
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
        _apply_chat_values(q, mixed=mode == _MIXED_MODE)
        # Hide the legacy switch; the new Training mode control replaces it.
        items = [
            item
            for item in items
            if getattr(item, "name", None) != "dataset/import/cfg/train_text_column"
        ]

    return _mode_controls(q, mode, columns) + items


def _reset_experiment_mode_for_dataset(q: Q) -> None:
    dataset_id = str(q.client["experiment/start/dataset"] or "")
    if q.client[_EXPERIMENT_DATASET_KEY] == dataset_id:
        return

    q.client[_EXPERIMENT_DATASET_KEY] = dataset_id
    q.client[_EXPERIMENT_MODE_KEY] = None
    q.client[_EXPERIMENT_TEXT_COLUMN_KEY] = None


def _infer_experiment_mode(q: Q) -> str:
    current = q.client[_EXPERIMENT_MODE_KEY]
    if current in (_CHAT_MODE, _MIXED_MODE, _TEXT_MODE):
        return current

    mode = _training_mode_from_values(
        q.client["experiment/start/cfg/train_text_column"],
        q.client["experiment/start/cfg/prompt_column"],
        q.client["experiment/start/cfg/answer_column"],
    )
    q.client[_EXPERIMENT_MODE_KEY] = mode
    return mode


def _apply_experiment_text_only_values(q: Q, text_column: str) -> None:
    q.client[_EXPERIMENT_MODE_KEY] = _TEXT_MODE
    q.client[_EXPERIMENT_TEXT_COLUMN_KEY] = text_column
    q.client["experiment/start/cfg/train_text_column"] = True
    q.client["experiment/start/cfg/system_column"] = "None"
    q.client["experiment/start/cfg/prompt_column"] = (text_column,)
    q.client["experiment/start/cfg/answer_column"] = text_column
    q.client["experiment/start/cfg/parent_id_column"] = "None"
    q.client["experiment/start/cfg/id_column"] = "None"


def _apply_experiment_chat_values(q: Q, *, mixed: bool) -> None:
    q.client[_EXPERIMENT_MODE_KEY] = _MIXED_MODE if mixed else _CHAT_MODE
    q.client["experiment/start/cfg/train_text_column"] = mixed


def _experiment_mode_controls(q: Q, mode: str, columns: list[str]) -> list[Any]:
    controls: list[Any] = [
        ui.dropdown(
            name=_EXPERIMENT_MODE_KEY,
            label="Training mode",
            value=mode,
            required=True,
            trigger=True,
            choices=[
                ui.choice(_CHAT_MODE, "Chat / instruction training"),
                ui.choice(_MIXED_MODE, "Mixed chat + Text training"),
                ui.choice(_TEXT_MODE, "Text only / continued pretraining"),
            ],
            tooltip=(
                "Controls how the selected dataset is interpreted for this experiment. "
                "Text only trains corpus text without applying the chat template."
            ),
        )
    ]

    if mode != _TEXT_MODE:
        return controls

    text_column = q.client[_EXPERIMENT_TEXT_COLUMN_KEY]
    persisted_prompt = _single_column(q.client["experiment/start/cfg/prompt_column"])
    if text_column not in columns:
        if persisted_prompt in columns:
            text_column = persisted_prompt
        else:
            text_column = _preferred_text_column(columns)
        q.client[_EXPERIMENT_TEXT_COLUMN_KEY] = text_column

    text_choices = list(columns)
    if not text_choices:
        text_choices = [text_column or "Text"]
        text_column = text_choices[0]
        q.client[_EXPERIMENT_TEXT_COLUMN_KEY] = text_column

    controls.extend(
        [
            ui.dropdown(
                name=_EXPERIMENT_TEXT_COLUMN_KEY,
                label="Text column",
                value=text_column,
                required=True,
                trigger=True,
                choices=[ui.choice(column, column) for column in text_choices],
                tooltip="Column whose non-empty rows are trained as raw text.",
            ),
            ui.message_bar(
                type="info",
                text=(
                    "This experiment uses continued pretraining. The selected text "
                    "column is trained directly; System, Prompt, Assistant, Parent "
                    "ID and chat-template settings are not used."
                ),
            ),
        ]
    )
    return controls


def _get_ui_elements_for_cfg_with_training_mode(
    cfg: Any,
    q: Q,
    limit: list[str] | None = None,
    pre: str = "experiment/start",
) -> list[Any]:
    """Expose the explicit training mode inside Experiment > Dataset settings."""
    if _ORIGINAL_GET_UI_ELEMENTS_FOR_CFG is None:
        raise RuntimeError("Experiment training-mode extension is not installed.")

    items = _ORIGINAL_GET_UI_ELEMENTS_FOR_CFG(cfg=cfg, q=q, limit=limit, pre=pre)

    # The wrapper is also used recursively. Only intercept the causal-LM dataset
    # dataclass; all other experiment config groups keep their original UI.
    if pre != "experiment/start" or not hasattr(cfg, "train_text_column"):
        return items

    _reset_experiment_mode_for_dataset(q)
    mode = _infer_experiment_mode(q)
    columns = _experiment_dataframe_columns(q)

    if mode == _TEXT_MODE:
        text_column = q.client[_EXPERIMENT_TEXT_COLUMN_KEY]
        persisted_prompt = _single_column(
            q.client["experiment/start/cfg/prompt_column"]
        )
        if text_column not in columns:
            if persisted_prompt in columns:
                text_column = persisted_prompt
            else:
                text_column = _preferred_text_column(columns)
        _apply_experiment_text_only_values(q, text_column)
        items = [
            item
            for item in items
            if getattr(item, "name", None) not in _EXPERIMENT_TEXT_ONLY_HIDDEN_FIELDS
        ]
    else:
        _apply_experiment_chat_values(q, mixed=mode == _MIXED_MODE)
        items = [
            item
            for item in items
            if getattr(item, "name", None) != "experiment/start/cfg/train_text_column"
        ]

    return _experiment_mode_controls(q, mode, columns) + items


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


def _handler_init_with_text_only(self: Any, df: pd.DataFrame, cfg: Any) -> None:
    """Keep the selected raw-text column available while the handler is built."""
    if _ORIGINAL_HANDLER_INIT is None:
        raise RuntimeError("Text-only handler patch is not installed.")

    self._text_only_column = get_text_only_column(cfg)
    _ORIGINAL_HANDLER_INIT(self, df, cfg)


def _apply_plain_text_rows_with_selected_column(self: Any, df: pd.DataFrame) -> None:
    text_column = getattr(self, "_text_only_column", None)
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


def install_text_only_training_mode(handle: Callable[..., Any]) -> Callable[..., Any]:
    """Install GUI/runtime patches and return a wrapped Wave request handler."""
    global _INSTALLED
    global _ORIGINAL_GET_DATASET_ELEMENTS
    global _ORIGINAL_GET_UI_ELEMENTS_FOR_CFG
    global _ORIGINAL_GET_PLAIN_TEXT_MASK
    global _ORIGINAL_CONFIGURED_TEXT_COLUMNS
    global _ORIGINAL_HANDLER_INIT
    global _ORIGINAL_APPLY_PLAIN_TEXT_ROWS

    if _INSTALLED:
        return handle

    from llm_studio.app_utils import utils as app_utils
    from llm_studio.app_utils.sections import dataset as dataset_section
    from llm_studio.app_utils.sections import experiment as experiment_section
    from llm_studio.src.datasets import conversation_chain_handler as chain_module

    _ORIGINAL_GET_DATASET_ELEMENTS = app_utils.get_dataset_elements
    _ORIGINAL_GET_UI_ELEMENTS_FOR_CFG = app_utils.get_ui_elements_for_cfg
    _ORIGINAL_GET_PLAIN_TEXT_MASK = chain_module.get_plain_text_mask
    _ORIGINAL_CONFIGURED_TEXT_COLUMNS = chain_module._configured_text_columns
    _ORIGINAL_HANDLER_INIT = chain_module.ConversationChainHandler.__init__
    _ORIGINAL_APPLY_PLAIN_TEXT_ROWS = (
        chain_module.ConversationChainHandler._apply_plain_text_rows
    )

    app_utils.get_dataset_elements = _get_dataset_elements_with_training_mode
    dataset_section.get_dataset_elements = _get_dataset_elements_with_training_mode
    app_utils.get_ui_elements_for_cfg = _get_ui_elements_for_cfg_with_training_mode
    experiment_section.get_ui_elements_for_cfg = (
        _get_ui_elements_for_cfg_with_training_mode
    )
    chain_module.get_plain_text_mask = _plain_text_mask_with_text_only
    chain_module._configured_text_columns = _configured_text_columns_with_text_only
    chain_module.ConversationChainHandler.__init__ = _handler_init_with_text_only
    chain_module.ConversationChainHandler._apply_plain_text_rows = (
        _apply_plain_text_rows_with_selected_column
    )

    async def handle_with_text_training_mode(q: Q) -> None:
        submission = q.args.__wave_submission_name__

        if submission == _MODE_KEY:
            selected_mode = q.args[_MODE_KEY] or _MIXED_MODE
            q.client[_MODE_KEY] = selected_mode
            if selected_mode == _TEXT_MODE:
                columns = _dataframe_columns(q)
                text_column = q.client[_TEXT_COLUMN_KEY]
                if text_column not in columns:
                    text_column = _preferred_text_column(columns)
                _apply_text_only_values(q, text_column)
            else:
                _apply_chat_values(q, mixed=selected_mode == _MIXED_MODE)

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
