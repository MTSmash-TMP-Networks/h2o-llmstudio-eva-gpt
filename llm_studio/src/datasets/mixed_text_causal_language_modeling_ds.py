from typing import Any

import pandas as pd
import torch

from llm_studio.src.datasets.conversation_chain_handler import (
    PLAIN_TEXT_COLUMN,
    PLAIN_TEXT_PROMPT,
    ConversationChainHandler,
    get_plain_text_mask,
    prepare_plain_text_rows_for_chaining,
)
from llm_studio.src.datasets.text_causal_language_modeling_ds import (
    CustomDataset as BaseCustomDataset,
)
from llm_studio.src.datasets.text_utils import clean_missing_text_values, get_tokenizer


class MixedConversationChainHandler(ConversationChainHandler):
    """Conversation handler that can mix chat turns with raw Text rows."""

    def __init__(self, df, cfg):
        self.plain_text_mask = get_plain_text_mask(df, cfg)
        super().__init__(df, cfg)

        if self.plain_text_mask.any():
            plain_texts = clean_missing_text_values(df[PLAIN_TEXT_COLUMN]).tolist()
            self.prompts = [
                PLAIN_TEXT_PROMPT if is_plain_text else prompt
                for prompt, is_plain_text in zip(
                    self.prompts, self.plain_text_mask.tolist(), strict=False
                )
            ]
            self.answers = [
                plain_texts[idx] if is_plain_text else answer
                for idx, (answer, is_plain_text) in enumerate(
                    zip(self.answers, self.plain_text_mask.tolist(), strict=False)
                )
            ]
            self.systems = [
                "" if is_plain_text else system
                for system, is_plain_text in zip(
                    self.systems, self.plain_text_mask.tolist(), strict=False
                )
            ]

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["plain_texts"] = [
            bool(self.plain_text_mask.iloc[i]) for i in self.conversation_chain_ids[idx]
        ]
        return item


class CustomDataset(BaseCustomDataset):
    """Causal LM dataset with automatic raw Text-row detection.

    A row is handled as raw text when it contains a non-empty `Text` value while
    all configured prompt, answer, and system columns are empty. In that case the
    model sees only the text tokens, and labels are left unmasked for those tokens.
    """

    def __init__(self, df: pd.DataFrame, cfg: Any, mode: str = "train"):
        self.cfg = cfg
        self.mode = mode
        self.df = df.copy()
        self.tokenizer = get_tokenizer(self.cfg)
        self.conversation_chain_handler = MixedConversationChainHandler(self.df, cfg)
        self.sample_index = self._build_sample_index()

    def _prepare_input_text_dict(self, idx: int) -> dict[str, list[str]]:
        input_text_dict = self.conversation_chain_handler[idx]
        plain_texts = input_text_dict.get(
            "plain_texts", [False] * len(input_text_dict["prompts"])
        )

        input_text_dict["systems"] = [
            "" if is_plain_text else self.parse_system(self.cfg, system)
            for system, is_plain_text in zip(
                input_text_dict["systems"], plain_texts, strict=False
            )
        ]
        input_text_dict["prompts"] = [
            PLAIN_TEXT_PROMPT
            if is_plain_text
            else self.parse_prompt_body(self.cfg, prompt)
            for prompt, is_plain_text in zip(
                input_text_dict["prompts"], plain_texts, strict=False
            )
        ]
        input_text_dict["answers"] = [
            self.parse_answer(self.cfg, answer)
            for answer in input_text_dict["answers"]
        ]
        return input_text_dict

    def _get_input_ids_labels_and_encodings(
        self, idx: int, augment: bool = True, trim_to_max_length: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        input_text_dict = self._prepare_input_text_dict(idx)
        # Raw text samples should stay raw text; do not prepend random parents.
        if input_text_dict.get("plain_texts", [False])[-1]:
            augment = False

        _, prompt_encodings, answer_encodings, prompt_label_masks = self.get_encodings(
            input_text_dict=input_text_dict,
            augment=augment,
            trim_to_max_length=trim_to_max_length,
        )
        input_ids = torch.cat(
            [
                torch.cat([prompt_encoding, answer_encoding])
                for prompt_encoding, answer_encoding in zip(
                    prompt_encodings, answer_encodings, strict=False
                )
            ]
        )
        labels = self._get_raw_labels(
            prompt_encodings, answer_encodings, prompt_label_masks
        )
        return input_ids, labels, prompt_encodings, answer_encodings

    def _get_prompt_encoding_and_mask(
        self, prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt == PLAIN_TEXT_PROMPT:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.bool)
        return super()._get_prompt_encoding_and_mask(prompt)

    @classmethod
    def sanity_check(cls, df: pd.DataFrame, cfg: Any, mode: str = "train"):
        df_for_check = prepare_plain_text_rows_for_chaining(df, cfg).copy()
        plain_text_mask = get_plain_text_mask(df, cfg)

        if plain_text_mask.any():
            answer_column = cfg.dataset.answer_column
            if isinstance(answer_column, str):
                if answer_column not in df_for_check.columns:
                    df_for_check[answer_column] = ""
                df_for_check.loc[plain_text_mask, answer_column] = clean_missing_text_values(
                    df[PLAIN_TEXT_COLUMN]
                ).loc[plain_text_mask]

        return super().sanity_check(df_for_check, cfg, mode)
