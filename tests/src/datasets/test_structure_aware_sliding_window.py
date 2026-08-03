from unittest.mock import patch

import pandas as pd
import torch

from llm_studio.python_configs.text_causal_language_modeling_config import (
    ConfigNLPCausalLMDataset,
    ConfigNLPCausalLMTokenizer,
    ConfigProblemBase,
)
from llm_studio.src.datasets import structure_aware_sliding_window as structure_module
from llm_studio.src.datasets.fast_sliding_window import (
    FastSlidingWindowDataset,
    _SampleLayout,
)
from llm_studio.src.datasets.structure_aware_sliding_window import (
    StructureAwareSlidingWindowDataset,
)
from llm_studio.src.datasets.text_causal_language_modeling_ds import CustomDataset


class CharacterTokenizer:
    pad_token_id = 0
    name_or_path = "structure-aware-character-tokenizer"
    vocab_size = 256

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
        del add_special_tokens, padding, return_token_type_ids
        if isinstance(text, list):
            input_ids = [[ord(char) for char in value] for value in text]
            if truncation and max_length is not None:
                input_ids = [value[:max_length] for value in input_ids]
            return {"input_ids": input_ids}

        input_ids = [ord(char) for char in text]
        if truncation and max_length is not None:
            input_ids = input_ids[:max_length]
        tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        return {
            "input_ids": tensor,
            "attention_mask": torch.ones_like(tensor),
        }


def make_cfg(max_length=24, overlap=12):
    cfg = ConfigProblemBase(
        llm_backbone="unit-test",
        dataset=ConfigNLPCausalLMDataset(
            train_dataframe="/path/to/train.csv",
            system_column="system",
            prompt_column=("prompt",),
            answer_column="answer",
            parent_id_column="None",
            text_system_start="<S>",
            text_prompt_start="<U>",
            text_answer_separator="<A>",
            add_eos_token_to_system=False,
            add_eos_token_to_prompt=False,
            add_eos_token_to_answer=False,
            mask_prompt_labels=True,
        ),
        tokenizer=ConfigNLPCausalLMTokenizer(
            max_length=max_length,
            long_sample_strategy="Sliding Window",
            sliding_window_overlap=overlap,
        ),
    )
    cfg.tokenizer._tokenizer_eos_token = ""
    return cfg


def test_structure_aware_dataset_wraps_fast_indexer():
    assert issubclass(CustomDataset, FastSlidingWindowDataset)
    assert CustomDataset is StructureAwareSlidingWindowDataset


def test_structure_aware_starts_prefer_answer_boundaries(monkeypatch):
    monkeypatch.setattr(structure_module, "_MIN_LOCAL_OVERLAP_TOKENS", 4)
    dataset = object.__new__(StructureAwareSlidingWindowDataset)
    layout = _SampleLayout(
        length=50,
        trainable_spans=((14, 25), (32, 50)),
    )

    starts = dataset._get_structure_aware_starts(
        layout=layout,
        max_length=20,
        overlap=6,
    )

    assert starts[0] == 0
    assert starts[1] == 14
    assert starts[-1] == 30
    assert all(
        current <= previous + 20
        for previous, current in zip(starts, starts[1:], strict=False)
    )


def test_later_window_repeats_masked_system_and_prompt_context(monkeypatch):
    monkeypatch.setattr(structure_module, "_MIN_LOCAL_OVERLAP_TOKENS", 4)
    monkeypatch.setattr(structure_module, "_ANCHOR_MAX_TOKENS", 8)
    tokenizer = CharacterTokenizer()
    cfg = make_cfg()
    df = pd.DataFrame(
        {
            "system": ["SYS"],
            "prompt": ["QUESTION"],
            "answer": ["abcdefghijklmnopqrstuvwxyz0123456789"],
        }
    )

    with patch(
        "llm_studio.src.datasets.text_causal_language_modeling_ds.get_tokenizer",
        return_value=tokenizer,
    ):
        dataset = CustomDataset(df, cfg, mode="train")

    original_idx, window_start, prefix_mask = next(
        entry
        for entry in dataset.sample_index
        if entry[1] is not None and entry[1] > 0 and entry[2] > 4
    )
    input_ids, labels, prompt_encodings, answer_encodings = (
        dataset._get_input_ids_labels_and_encodings(
            original_idx,
            augment=False,
            trim_to_max_length=False,
        )
    )
    anchor = dataset._get_structure_anchor(
        original_idx=original_idx,
        window_start=window_start,
        prefix_label_mask_len=prefix_mask,
        labels=labels,
        prompt_encodings=prompt_encodings,
        answer_encodings=answer_encodings,
    )
    window_ids, window_labels = dataset._compose_structure_aware_window(
        original_idx=original_idx,
        window_start=window_start,
        prefix_label_mask_len=prefix_mask,
        input_ids=input_ids,
        labels=labels,
        prompt_encodings=prompt_encodings,
        answer_encodings=answer_encodings,
    )

    assert 0 < len(anchor) <= 8
    assert "".join(chr(token) for token in anchor.tolist()).startswith("<S>SYS")
    assert "".join(chr(token) for token in anchor.tolist()).endswith("A>")
    assert torch.equal(window_ids[: len(anchor)], anchor)
    assert torch.equal(
        window_ids[prefix_mask:],
        input_ids[window_start + prefix_mask : window_start + cfg.tokenizer.max_length],
    )
    assert torch.all(window_labels[:prefix_mask] == -100)
    assert torch.equal(
        window_labels[prefix_mask:],
        labels[window_start + prefix_mask : window_start + cfg.tokenizer.max_length],
    )
    assert torch.any(window_labels[prefix_mask:] != -100)


def test_small_overlap_keeps_original_window_unchanged(monkeypatch):
    monkeypatch.setattr(structure_module, "_MIN_LOCAL_OVERLAP_TOKENS", 64)
    dataset = object.__new__(StructureAwareSlidingWindowDataset)
    dataset.cfg = make_cfg(max_length=10, overlap=2)

    input_ids = torch.arange(20)
    labels = input_ids.clone()
    prompt_encodings = [torch.tensor([1, 2, 3])]
    answer_encodings = [torch.arange(3, 20)]
    dataset._get_system_anchor_ids = lambda original_idx: torch.empty(
        0, dtype=torch.long
    )

    window_ids, window_labels = dataset._compose_structure_aware_window(
        original_idx=0,
        window_start=8,
        prefix_label_mask_len=2,
        input_ids=input_ids,
        labels=labels,
        prompt_encodings=prompt_encodings,
        answer_encodings=answer_encodings,
    )

    assert torch.equal(window_ids, input_ids[8:18])
    assert window_labels[:2].tolist() == [-100, -100]
    assert torch.equal(window_labels[2:], labels[10:18])
