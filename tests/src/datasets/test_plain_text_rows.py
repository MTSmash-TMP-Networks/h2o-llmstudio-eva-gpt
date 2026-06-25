from unittest.mock import patch

import pandas as pd
import torch

from llm_studio.python_configs.text_causal_language_modeling_config import (
    ConfigNLPCausalLMDataset,
    ConfigNLPCausalLMTokenizer,
    ConfigProblemBase,
)
from llm_studio.src.datasets.text_causal_language_modeling_ds import CustomDataset


class OrdinalCharacterTokenizer:
    pad_token_id = 0

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        input_ids = torch.tensor([ord(char) for char in text], dtype=torch.long)
        return {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(token_id)) for token_id in ids if int(token_id) != 0)


@patch("llm_studio.src.datasets.text_causal_language_modeling_ds.get_tokenizer")
def test_plain_text_rows_are_trained_raw_and_unmasked(mock_get_tokenizer):
    mock_get_tokenizer.return_value = OrdinalCharacterTokenizer()
    df = pd.DataFrame(
        {
            "id": ["chat-root", "", "instruction"],
            "Benutzer": ["Hallo", "", "Fasse zusammen"],
            "Assistentin": ["Hi", "", "Zusammenfassung"],
            "parent_id": [None, "", None],
            "system": ["", "", ""],
            "Kontext": ["", "", "Kontext"],
            "Text": ["", "Plain text only", ""],
        }
    )
    cfg = ConfigProblemBase(
        llm_backbone="unit-test",
        dataset=ConfigNLPCausalLMDataset(
            prompt_column=("Benutzer", "Kontext"),
            prompt_column_separator="\n",
            answer_column="Assistentin",
            parent_id_column="parent_id",
            id_column="id",
            system_column="system",
            text_prompt_start="<|prompt|>",
            text_answer_separator="<|answer|>",
            add_eos_token_to_prompt=False,
            add_eos_token_to_answer=False,
            mask_prompt_labels=True,
        ),
        tokenizer=ConfigNLPCausalLMTokenizer(max_length=128),
    )

    dataset = CustomDataset(df, cfg)
    assert dataset.conversation_chain_handler.conversation_chain_ids == [[0], [1], [2]]

    plain_sample = dataset[1]
    plain_tokens = plain_sample["input_ids"][plain_sample["attention_mask"].bool()]
    plain_labels = plain_sample["labels"][plain_sample["attention_mask"].bool()]

    assert dataset.tokenizer.decode(plain_tokens) == "Plain text only"
    assert torch.equal(plain_labels, plain_tokens)
    assert (plain_labels != -100).all()


@patch("llm_studio.src.datasets.text_causal_language_modeling_ds.get_tokenizer")
def test_regular_chat_rows_keep_prompt_masking(mock_get_tokenizer):
    mock_get_tokenizer.return_value = OrdinalCharacterTokenizer()
    df = pd.DataFrame(
        {
            "id": ["chat-root", "plain"],
            "Benutzer": ["Hallo", ""],
            "Assistentin": ["Hi", ""],
            "parent_id": [None, None],
            "system": ["", ""],
            "Text": ["", "Plain text only"],
        }
    )
    cfg = ConfigProblemBase(
        llm_backbone="unit-test",
        dataset=ConfigNLPCausalLMDataset(
            prompt_column=("Benutzer",),
            answer_column="Assistentin",
            parent_id_column="parent_id",
            id_column="id",
            system_column="system",
            text_prompt_start="<|prompt|>",
            text_answer_separator="<|answer|>",
            add_eos_token_to_prompt=False,
            add_eos_token_to_answer=False,
            mask_prompt_labels=True,
        ),
        tokenizer=ConfigNLPCausalLMTokenizer(max_length=128),
    )

    dataset = CustomDataset(df, cfg)
    chat_input_ids, chat_labels, prompt_encodings, _ = (
        dataset._get_input_ids_labels_and_encodings(0)
    )
    prompt_len = len(prompt_encodings[0])

    assert dataset.tokenizer.decode(chat_input_ids) == "<|prompt|>Hallo<|answer|>Hi"
    assert chat_labels[:prompt_len].tolist() == [-100] * prompt_len
    assert torch.equal(chat_labels[prompt_len:], chat_input_ids[prompt_len:])


@patch("llm_studio.src.datasets.text_causal_language_modeling_ds.get_tokenizer")
def test_plain_text_rows_are_ignored_when_toggle_is_disabled(mock_get_tokenizer):
    mock_get_tokenizer.return_value = OrdinalCharacterTokenizer()
    df = pd.DataFrame(
        {
            "id": ["plain"],
            "Benutzer": [""],
            "Assistentin": [""],
            "parent_id": [None],
            "system": [""],
            "Text": ["Plain text only"],
        }
    )
    cfg = ConfigProblemBase(
        llm_backbone="unit-test",
        dataset=ConfigNLPCausalLMDataset(
            prompt_column=("Benutzer",),
            answer_column="Assistentin",
            parent_id_column="parent_id",
            id_column="id",
            system_column="system",
            text_prompt_start="<|prompt|>",
            text_answer_separator="<|answer|>",
            add_eos_token_to_prompt=False,
            add_eos_token_to_answer=False,
            mask_prompt_labels=True,
        ),
        tokenizer=ConfigNLPCausalLMTokenizer(max_length=128),
    )
    cfg.dataset.train_text_column = False

    dataset = CustomDataset(df, cfg)
    sample = dataset[0]
    tokens = sample["input_ids"][sample["attention_mask"].bool()]

    assert dataset.tokenizer.decode(tokens) == "<|prompt|><|answer|>"
