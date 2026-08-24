"""Support chained user-context turns that intentionally have no assistant answer."""

from __future__ import annotations

import codecs
import logging
from typing import Any

import pandas as pd
import torch

from llm_studio.src.datasets.conversation_chain_handler import (
    ConversationChainHandler,
    get_plain_text_mask,
)
from llm_studio.src.datasets.text_utils import clean_missing_text_values

logger = logging.getLogger(__name__)

_CONTEXT_ONLY_PROMPT_PREFIX = "\x00__LLM_STUDIO_CONTEXT_ONLY_USER_TURN__\x00"
_MISSING_TEXT_MARKERS = {"", "none", "null", "nan", "na"}
_CACHE_SEMANTICS_VERSION = "context-only-user-turn-v1"


def _text_has_content(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return text.lower() not in _MISSING_TEXT_MARKERS


def get_supervised_answer_mask(df: pd.DataFrame, cfg: Any) -> pd.Series:
    """Return rows that contain an actual assistant target."""
    answer_column = getattr(cfg.dataset, "answer_column", None)
    mask = pd.Series(False, index=df.index)

    if isinstance(answer_column, (list, tuple)):
        columns = list(answer_column)
    elif isinstance(answer_column, str):
        columns = [answer_column]
    else:
        columns = []

    for column in columns:
        if column in df.columns:
            mask |= clean_missing_text_values(df[column]).str.strip().ne("")
    return mask


def _normalized_chain_ids(df: pd.DataFrame, cfg: Any) -> tuple[list[Any], list[Any]]:
    """Normalize IDs exactly like the conversation-chain handler where possible."""
    id_column = getattr(cfg.dataset, "id_column", None)
    parent_id_column = getattr(cfg.dataset, "parent_id_column", None)
    if (
        id_column in (None, "None")
        or parent_id_column in (None, "None")
        or id_column not in df.columns
        or parent_id_column not in df.columns
    ):
        return [], []

    try:
        sample_ids = df[id_column].astype(df[parent_id_column].dtype).tolist()
    except (TypeError, ValueError):
        sample_ids = df[id_column].tolist()
    parent_ids = df[parent_id_column].tolist()
    return sample_ids, parent_ids


def get_valid_context_only_mask(df: pd.DataFrame, cfg: Any) -> pd.Series:
    """Return empty-answer rows that feed a later answered descendant.

    A context-only row is legitimate only when parent chaining is configured and
    the row is an ancestor of at least one row with an actual assistant answer.
    This supports chains such as user -> user -> user -> assistant without turning
    the intermediate user turns into independent loss samples.
    """
    result = pd.Series(False, index=df.index)
    sample_ids, parent_ids = _normalized_chain_ids(df, cfg)
    if not sample_ids:
        return result

    answer_mask = get_supervised_answer_mask(df, cfg)
    sample_ids_set = set(sample_ids)
    normalized_parents = [
        parent_id
        if parent_id in sample_ids_set
        and not ConversationChainHandler._is_missing_parent_id(parent_id)
        else None
        for parent_id in parent_ids
    ]
    parent_map = {
        sample_id: parent_id
        for sample_id, parent_id in zip(sample_ids, normalized_parents, strict=False)
        if parent_id is not None
    }

    valid_context_ids: set[Any] = set()
    for sample_id, has_answer in zip(sample_ids, answer_mask.tolist(), strict=False):
        if not has_answer:
            continue
        current_id = sample_id
        visited = {current_id}
        while current_id in parent_map:
            parent_id = parent_map[current_id]
            if parent_id in visited:
                break
            valid_context_ids.add(parent_id)
            visited.add(parent_id)
            current_id = parent_id

    result_values = [
        (not has_answer) and sample_id in valid_context_ids
        for sample_id, has_answer in zip(sample_ids, answer_mask.tolist(), strict=False)
    ]
    return pd.Series(result_values, index=df.index)


def _mark_context_only_prompt(prompt: str) -> str:
    if prompt.startswith(_CONTEXT_ONLY_PROMPT_PREFIX):
        return prompt
    return f"{_CONTEXT_ONLY_PROMPT_PREFIX}{prompt}"


def _strip_context_only_prompt(prompt: str) -> tuple[str, bool]:
    if prompt.startswith(_CONTEXT_ONLY_PROMPT_PREFIX):
        return prompt[len(_CONTEXT_ONLY_PROMPT_PREFIX) :], True
    return prompt, False


def _install_chain_endpoint_filter() -> None:
    if getattr(
        ConversationChainHandler, "_context_only_endpoint_filter_installed", False
    ):
        return

    original = ConversationChainHandler.get_conversation_chain_ids

    def get_conversation_chain_ids(self, cfg, df):
        chains = original(self, cfg, df)
        parent_id_column = getattr(cfg.dataset, "parent_id_column", None)
        if (
            bool(getattr(cfg.dataset, "limit_chained_samples", False))
            or parent_id_column in (None, "None")
            or parent_id_column not in df.columns
        ):
            return chains

        endpoint_mask = get_supervised_answer_mask(df, cfg) | get_plain_text_mask(
            df, cfg
        )
        filtered = [
            chain for chain in chains if chain and bool(endpoint_mask.iloc[chain[-1]])
        ]
        removed = len(chains) - len(filtered)
        if removed:
            logger.info(
                "Limit Chained Samples=False: retained %s answer-bearing endpoints "
                "and treated %s empty-answer IDs as context-only turns.",
                len(filtered),
                removed,
            )
        return filtered

    ConversationChainHandler.get_conversation_chain_ids = get_conversation_chain_ids
    ConversationChainHandler._context_only_endpoint_filter_installed = True


def _encode_context_only_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a user-only turn without adding an assistant-start separator."""
    prompt, _ = _strip_context_only_prompt(prompt)
    prompt_parts = [
        (
            codecs.decode(self.cfg.dataset.text_prompt_start, "unicode_escape"),
            False,
        ),
        (prompt, True),
    ]
    if self.cfg.dataset.add_eos_token_to_prompt:
        prompt_parts.append((self.cfg.tokenizer._tokenizer_eos_token, False))

    encodings = []
    masks = []
    for text, mask_user_text in prompt_parts:
        if not text:
            continue
        input_ids = self.encode(
            self.tokenizer,
            text,
            self.cfg.tokenizer.max_length,
            "left",
        )["input_ids"]
        encodings.append(input_ids)
        masks.append(torch.full_like(input_ids, mask_user_text, dtype=torch.bool))

    if not encodings:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.bool)
    return (
        torch.cat(encodings)[-self.cfg.tokenizer.max_length :],
        torch.cat(masks)[-self.cfg.tokenizer.max_length :],
    )


def _install_dataset_encoding_support() -> None:
    from llm_studio.src.datasets import text_causal_language_modeling_ds

    dataset_cls = text_causal_language_modeling_ds.CustomDataset
    if getattr(dataset_cls, "_context_only_encoding_installed", False):
        return

    original_parse_answer = dataset_cls.parse_answer
    original_prepare = dataset_cls._prepare_input_text_dict
    original_get_sample_encoding = dataset_cls._get_sample_encoding
    original_get_prompt_encoding_and_mask = dataset_cls._get_prompt_encoding_and_mask
    original_postprocess_output = dataset_cls.postprocess_output

    def parse_answer(cfg: Any, answer: str):
        if not _text_has_content(answer):
            return ""
        return original_parse_answer(cfg, answer)

    def _prepare_input_text_dict(self, idx: int):
        input_text_dict = original_prepare(self, idx)
        input_text_dict["prompts"] = [
            _mark_context_only_prompt(prompt)
            if not _text_has_content(answer)
            else prompt
            for prompt, answer in zip(
                input_text_dict["prompts"],
                input_text_dict["answers"],
                strict=False,
            )
        ]
        return input_text_dict

    def _get_sample_encoding(self, system: str, prompt: str, answer: str):
        if not _text_has_content(answer):
            prompt = _mark_context_only_prompt(prompt)
        return original_get_sample_encoding(self, system, prompt, answer)

    def _get_prompt_encoding_and_mask(self, prompt: str):
        _, is_context_only = _strip_context_only_prompt(prompt)
        if is_context_only:
            return _encode_context_only_prompt(self, prompt)
        return original_get_prompt_encoding_and_mask(self, prompt)

    def postprocess_output(self, cfg, df: pd.DataFrame, output: dict):
        handler = self.conversation_chain_handler
        use_context_endpoint_targets = not bool(
            getattr(cfg.dataset, "limit_chained_samples", False)
        ) and len(handler) != len(handler.answers)
        if not use_context_endpoint_targets:
            return original_postprocess_output(self, cfg, df, output)

        full_answers = handler.answers
        endpoint_answers = [
            full_answers[idx] for idx in handler.get_conversation_end_ids()
        ]
        handler.answers = endpoint_answers
        try:
            return original_postprocess_output(self, cfg, df, output)
        finally:
            handler.answers = full_answers

    dataset_cls.parse_answer = staticmethod(parse_answer)
    dataset_cls._prepare_input_text_dict = _prepare_input_text_dict
    dataset_cls._get_sample_encoding = _get_sample_encoding
    dataset_cls._get_prompt_encoding_and_mask = _get_prompt_encoding_and_mask
    dataset_cls.postprocess_output = postprocess_output
    dataset_cls._context_only_encoding_installed = True


def _install_fast_layout_support() -> None:
    from llm_studio.src.datasets import (
        fast_sliding_window,
        structure_aware_sliding_window,
    )

    fast_cls = fast_sliding_window.FastSlidingWindowDataset
    if not getattr(fast_cls, "_context_only_prompt_parts_installed", False):
        original_prompt_parts = fast_cls._prompt_parts_with_masks

        def _prompt_parts_with_masks(self, prompt: str):
            prompt, is_context_only = _strip_context_only_prompt(prompt)
            parts = original_prompt_parts(self, prompt)
            if not is_context_only:
                return parts

            separator = codecs.decode(
                self.cfg.dataset.text_answer_separator, "unicode_escape"
            )
            if separator and parts and parts[-1] == (separator, False):
                parts = parts[:-1]
            return parts

        fast_cls._prompt_parts_with_masks = _prompt_parts_with_masks
        fast_cls._context_only_prompt_parts_installed = True

    if not getattr(fast_sliding_window, "_context_only_cache_version_installed", False):
        original_fast_cache_path = fast_sliding_window.get_cache_path

        def fast_cache_path(dataset, strategy):
            return original_fast_cache_path(
                dataset, f"{strategy}:{_CACHE_SEMANTICS_VERSION}"
            )

        fast_sliding_window.get_cache_path = fast_cache_path
        fast_sliding_window._context_only_cache_version_installed = True

    if not getattr(
        structure_aware_sliding_window,
        "_context_only_cache_version_installed",
        False,
    ):
        original_structure_cache_path = structure_aware_sliding_window.get_cache_path

        def structure_cache_path(dataset, strategy):
            return original_structure_cache_path(
                dataset, f"{strategy}:{_CACHE_SEMANTICS_VERSION}"
            )

        structure_aware_sliding_window.get_cache_path = structure_cache_path
        structure_aware_sliding_window._context_only_cache_version_installed = True


def install_context_only_chain_turns() -> None:
    """Install context-only chained-turn behavior once."""
    _install_chain_endpoint_filter()
    _install_dataset_encoding_support()
    _install_fast_layout_support()
