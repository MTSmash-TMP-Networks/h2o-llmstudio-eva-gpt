from unittest.mock import patch

import pandas as pd
import torch

from llm_studio.python_configs.text_causal_language_modeling_config import (
    ConfigNLPCausalLMDataset,
    ConfigNLPCausalLMTokenizer,
    ConfigProblemBase,
)
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


def make_cfg(
    max_length=10,
    overlap=2,
    backbone="unit-test",
    only_last_answer=False,
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
            mask_prompt_labels=True,
            mask_prompt_user_text_only=True,
            only_last_answer=only_last_answer,
        ),
        tokenizer=ConfigNLPCausalLMTokenizer(
            max_length=max_length,
            long_sample_strategy="Sliding Window",
            sliding_window_overlap=overlap,
        ),
    )
    cfg.tokenizer._tokenizer_eos_token = "<EOS>"
    return cfg


def label_spans(labels):
    spans = []
    start = None
    for idx, trainable in enumerate((labels != -100).tolist()):
        if trainable and start is None:
            start = idx
        elif not trainable and start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, len(labels)))
    return tuple(spans)


def test_prompt_only_windows_are_removed_but_context_is_kept():
    tokenizer = BatchCharacterTokenizer()
    df = pd.DataFrame({"prompt": ["p" * 15], "answer": ["abc"]})

    with patch(
        "llm_studio.src.datasets.text_causal_language_modeling_ds.get_tokenizer",
        return_value=tokenizer,
    ):
        dataset = CustomDataset(df, make_cfg(), mode="train")

    # The prompt is capped to max_length before the answer is appended. The only
    # useful window starts at token 3 and therefore keeps seven prompt tokens as
    # visible context for the three answer targets.
    assert dataset.sample_index == [(0, 3, 0)]
    sample = dataset[0]
    assert torch.all(sample["labels"][:7] == -100)
    assert torch.all(sample["labels"][7:] != -100)
    assert torch.any(sample["labels"][1:] != -100)


def test_every_indexed_window_has_a_target_after_causal_shift():
    tokenizer = BatchCharacterTokenizer()
    df = pd.DataFrame(
        {
            "prompt": ["12345", "p" * 20, "short"],
            "answer": ["abcdefghi", "answer", "ok"],
        }
    )

    with patch(
        "llm_studio.src.datasets.text_causal_language_modeling_ds.get_tokenizer",
        return_value=tokenizer,
    ):
        dataset = CustomDataset(df, make_cfg(), mode="train")

    assert len(dataset) > 0
    for idx in range(len(dataset)):
        assert torch.any(dataset[idx]["labels"][1:] != -100)


def test_fully_masked_sample_is_removed_from_sliding_window_index():
    tokenizer = BatchCharacterTokenizer()
    df = pd.DataFrame({"prompt": ["p" * 20], "answer": [""]})

    with patch(
        "llm_studio.src.datasets.text_causal_language_modeling_ds.get_tokenizer",
        return_value=tokenizer,
    ):
        dataset = CustomDataset(df, make_cfg(), mode="train")

    assert dataset.sample_index == []


def test_batched_trainable_spans_match_serial_labels():
    tokenizer = BatchCharacterTokenizer()
    df = pd.DataFrame(
        {
            "prompt": ["root prompt", "child prompt"],
            "answer": ["root answer", "child answer"],
            "system": ["system", ""],
            "id": [0, 1],
            "parent_id": [None, 0],
        }
    )
    cfg = make_cfg(max_length=16, only_last_answer=True)
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

    layouts = dataset._compute_sample_layouts_batched()
    for idx, layout in enumerate(layouts):
        input_ids, labels, _, _ = dataset._get_input_ids_labels_and_encodings(
            idx,
            augment=False,
            trim_to_max_length=False,
        )
        assert layout.length == len(input_ids)
        assert layout.trainable_spans == label_spans(labels)


def test_cache_key_changes_with_label_masking(tmp_path, monkeypatch):
    monkeypatch.setenv("H2O_LLM_STUDIO_CACHE_DIR", str(tmp_path))
    df = pd.DataFrame(
        {
            "prompt": ["root", "child"],
            "answer": ["first answer", "second answer"],
            "id": [0, 1],
            "parent_id": [None, 0],
        }
    )
    first_cfg = make_cfg(backbone="cache-test", only_last_answer=False)
    second_cfg = make_cfg(backbone="cache-test", only_last_answer=True)
    for cfg in (first_cfg, second_cfg):
        cfg.dataset.parent_id_column = "parent_id"
        cfg.dataset.id_column = "id"

    with patch(
        "llm_studio.src.datasets.text_causal_language_modeling_ds.get_tokenizer",
        side_effect=[BatchCharacterTokenizer(), BatchCharacterTokenizer()],
    ):
        CustomDataset(df, first_cfg, mode="train")
        CustomDataset(df, second_cfg, mode="train")

    assert len(list((tmp_path / "sample_indices").glob("*.npy"))) == 2
