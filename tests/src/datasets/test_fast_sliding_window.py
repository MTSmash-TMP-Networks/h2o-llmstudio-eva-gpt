from unittest.mock import patch

import pandas as pd
import torch

from llm_studio.python_configs.text_causal_language_modeling_config import (
    ConfigNLPCausalLMDataset,
    ConfigNLPCausalLMTokenizer,
    ConfigProblemBase,
)
from llm_studio.src.datasets.conversation_chain_handler import PLAIN_TEXT_PROMPT
from llm_studio.src.datasets.fast_sliding_window import FastSlidingWindowDataset
from llm_studio.src.datasets.text_causal_language_modeling_ds import CustomDataset


class BatchCharacterTokenizer:
    pad_token_id = 0
    name_or_path = "batch-character-tokenizer"
    vocab_size = 256

    def __init__(self):
        self.batch_calls = 0
        self.scalar_calls = 0

    @staticmethod
    def _ids(text):
        return list(range(1, len(text) + 1))

    def __call__(
        self,
        text,
        return_tensors=None,
        add_special_tokens=False,
        truncation=False,
        max_length=None,
        padding=False,
        return_attention_mask=True,
        return_token_type_ids=False,
    ):
        if isinstance(text, list):
            self.batch_calls += 1
            input_ids = [self._ids(value) for value in text]
            if truncation and max_length is not None:
                input_ids = [value[:max_length] for value in input_ids]
            return {"input_ids": input_ids}

        self.scalar_calls += 1
        input_ids = self._ids(text)
        if truncation and max_length is not None:
            input_ids = input_ids[:max_length]
        tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        return {
            "input_ids": tensor,
            "attention_mask": torch.ones_like(tensor),
        }


class ScalarOnlyCharacterTokenizer(BatchCharacterTokenizer):
    def __call__(self, text, **kwargs):
        if isinstance(text, list):
            raise TypeError("batch tokenization is not supported")
        return super().__call__(text, **kwargs)


def make_cfg(
    strategy="Sliding Window",
    overlap=2,
    max_length=10,
    backbone="unit-test",
):
    cfg = ConfigProblemBase(
        llm_backbone=backbone,
        dataset=ConfigNLPCausalLMDataset(
            train_dataframe="/path/to/train.csv",
            system_column="None",
            prompt_column=("prompt",),
            answer_column="answer",
            parent_id_column="None",
            text_prompt_start="",
            text_answer_separator="",
            add_eos_token_to_prompt=False,
            add_eos_token_to_answer=False,
        ),
        tokenizer=ConfigNLPCausalLMTokenizer(
            max_length=max_length,
            long_sample_strategy=strategy,
            sliding_window_overlap=overlap,
        ),
    )
    cfg.tokenizer._tokenizer_eos_token = "<EOS>"
    return cfg


def test_fast_dataset_is_installed():
    assert issubclass(CustomDataset, FastSlidingWindowDataset)


def test_fast_sliding_window_keeps_exact_window_semantics():
    tokenizer = BatchCharacterTokenizer()
    df = pd.DataFrame({"prompt": ["12345"], "answer": ["abcdefghi"]})

    with patch(
        "llm_studio.src.datasets.text_causal_language_modeling_ds.get_tokenizer",
        return_value=tokenizer,
    ):
        dataset = CustomDataset(df, make_cfg(), mode="train")

    assert dataset.sample_index == [(0, 0, 0), (0, 4, 6)]
    assert tokenizer.batch_calls == 1
    assert tokenizer.scalar_calls == 0


def test_batched_lengths_match_full_training_encodings():
    tokenizer = BatchCharacterTokenizer()
    df = pd.DataFrame(
        {
            "prompt": ["root prompt", "child prompt"],
            "answer": ["root answer", "child answer that is longer"],
            "system": ["system", ""],
            "id": [0, 1],
            "parent_id": [None, 0],
        }
    )
    cfg = make_cfg(max_length=16)
    cfg.dataset.system_column = "system"
    cfg.dataset.parent_id_column = "parent_id"
    cfg.dataset.id_column = "id"
    cfg.dataset.limit_chained_samples = True
    cfg.dataset.text_system_start = "System:"
    cfg.dataset.text_prompt_start = "Prompt:"
    cfg.dataset.text_answer_separator = "Answer:"
    cfg.dataset.add_eos_token_to_prompt = True
    cfg.dataset.add_eos_token_to_answer = True

    with patch(
        "llm_studio.src.datasets.text_causal_language_modeling_ds.get_tokenizer",
        return_value=tokenizer,
    ):
        dataset = CustomDataset(df, cfg, mode="train")

    batched_lengths = dataset._compute_sample_lengths_batched()
    serial_lengths = [
        len(
            dataset._get_input_ids_labels_and_encodings(
                idx,
                augment=False,
                trim_to_max_length=False,
            )[0]
        )
        for idx in range(len(dataset.conversation_chain_handler))
    ]
    assert batched_lengths == serial_lengths


def test_scalar_only_tokenizer_falls_back_without_changing_index():
    tokenizer = ScalarOnlyCharacterTokenizer()
    df = pd.DataFrame({"prompt": ["12345"], "answer": ["abcdefghi"]})

    with patch(
        "llm_studio.src.datasets.text_causal_language_modeling_ds.get_tokenizer",
        return_value=tokenizer,
    ):
        dataset = CustomDataset(df, make_cfg(), mode="train")

    assert dataset.sample_index == [(0, 0, 0), (0, 4, 6)]
    assert tokenizer.scalar_calls > 0


def test_sample_index_cache_skips_tokenization_on_next_start(tmp_path, monkeypatch):
    monkeypatch.setenv("H2O_LLM_STUDIO_CACHE_DIR", str(tmp_path))
    df = pd.DataFrame(
        {
            "prompt": ["p" * 15, "short"],
            "answer": ["a" * 15, "ok"],
        }
    )
    cfg = make_cfg(backbone="cache-test")
    first_tokenizer = BatchCharacterTokenizer()
    second_tokenizer = BatchCharacterTokenizer()

    with patch(
        "llm_studio.src.datasets.text_causal_language_modeling_ds.get_tokenizer",
        side_effect=[first_tokenizer, second_tokenizer],
    ):
        first_dataset = CustomDataset(df, cfg, mode="train")
        second_dataset = CustomDataset(df, cfg, mode="train")

    assert first_tokenizer.batch_calls > 0
    assert second_tokenizer.batch_calls == 0
    assert second_tokenizer.scalar_calls == 0
    assert second_dataset.sample_index == first_dataset.sample_index
    assert list((tmp_path / "sample_indices").glob("*.npy"))


def test_cache_key_changes_with_overlap(tmp_path, monkeypatch):
    monkeypatch.setenv("H2O_LLM_STUDIO_CACHE_DIR", str(tmp_path))
    df = pd.DataFrame({"prompt": ["p" * 15], "answer": ["a" * 15]})
    first_tokenizer = BatchCharacterTokenizer()
    second_tokenizer = BatchCharacterTokenizer()

    with patch(
        "llm_studio.src.datasets.text_causal_language_modeling_ds.get_tokenizer",
        side_effect=[first_tokenizer, second_tokenizer],
    ):
        CustomDataset(
            df,
            make_cfg(overlap=1, backbone="cache-test"),
            mode="train",
        )
        CustomDataset(
            df,
            make_cfg(overlap=4, backbone="cache-test"),
            mode="train",
        )

    assert first_tokenizer.batch_calls > 0
    assert second_tokenizer.batch_calls > 0
    assert len(list((tmp_path / "sample_indices").glob("*.npy"))) == 2


def test_plain_text_prompt_adds_no_chat_format_tokens():
    dataset = object.__new__(CustomDataset)
    dataset.cfg = make_cfg()
    assert dataset._prompt_parts(PLAIN_TEXT_PROMPT) == []
