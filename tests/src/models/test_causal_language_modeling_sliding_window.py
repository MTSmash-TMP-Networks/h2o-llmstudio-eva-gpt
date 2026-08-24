from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

from llm_studio.python_configs.text_causal_language_modeling_config import (
    ConfigNLPCausalLMDataset,
    ConfigNLPCausalLMTokenizer,
)
from llm_studio.src.datasets.text_causal_language_modeling_ds import CustomDataset


class CharacterTokenizer:
    pad_token_id = 0

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        input_ids = torch.tensor([ord(char) for char in text], dtype=torch.long)
        return {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": torch.ones_like(input_ids).unsqueeze(0),
        }


def _cfg(long_sample_strategy="Sliding Window"):
    return SimpleNamespace(
        dataset=ConfigNLPCausalLMDataset(
            system_column="None",
            prompt_column=("prompt",),
            answer_column="answer",
            parent_id_column="parent_id",
            id_column="id",
            text_system_start="",
            text_prompt_start="",
            text_answer_separator=" ",
            add_eos_token_to_system=False,
            add_eos_token_to_prompt=False,
            add_eos_token_to_answer=False,
            mask_prompt_labels=True,
            only_last_answer=False,
        ),
        tokenizer=ConfigNLPCausalLMTokenizer(
            max_length=10,
            long_sample_strategy=long_sample_strategy,
            sliding_window_overlap=2,
            _tokenizer_eos_token="",
        ),
        augmentation=SimpleNamespace(
            skip_parent_probability=0.0,
            random_parent_probability=0.0,
        ),
    )


def test_sliding_window_indexes_untrimmed_conversation_turns():
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "parent_id": ["None", "1", "2"],
            "prompt": ["p1", "p2", "p3"],
            "answer": ["a1", "a2", "a3"],
        }
    )

    with patch(
        "llm_studio.src.datasets.text_causal_language_modeling_ds.get_tokenizer",
        return_value=CharacterTokenizer(),
    ):
        dataset = CustomDataset(df, _cfg(), mode="train")

    # Limit Chained Samples=False creates one endpoint sample for every ID.
    # Parent answers remain visible context but are masked from the endpoint loss.
    # The prompt-only first window of ID 3 is therefore removed from the supervised
    # sliding-window index; only the window containing ID 3's answer remains.
    assert dataset.sample_index == [(0, None, 0), (1, None, 0), (2, 5, 0)]

    endpoint_window = dataset[len(dataset) - 1]
    endpoint_tokens = endpoint_window["input_ids"][
        endpoint_window["attention_mask"].bool()
    ]
    assert "".join(chr(token) for token in endpoint_tokens.tolist()) == "p2 a2p3 a3"

    trained_texts = []
    for sample in dataset:
        trained_texts.append(
            "".join(
                chr(token) for token in sample["labels"].tolist() if token != -100
            )
        )

    assert "a1" in trained_texts[0]
    assert "a2" not in trained_texts[0]
    assert "a1" not in trained_texts[1]
    assert "a2" in trained_texts[1]
    assert "a1" not in trained_texts[2]
    assert "a2" not in trained_texts[2]
    assert "a3" in trained_texts[2]

    endpoint_labels = endpoint_window["labels"][:10]
    assert endpoint_labels[:7].tolist() == [-100] * 7
    assert endpoint_labels[7:].tolist() == [ord(" "), ord("a"), ord("3")]


def test_sliding_window_starts_mask_actual_duplicate_prefix_for_shifted_final_window():
    windows = CustomDataset._get_sliding_window_starts_and_prefix_masks(
        sample_length=25,
        max_length=10,
        overlap=2,
    )

    assert windows == [(0, 0), (8, 2), (15, 3)]


def test_sliding_window_prefix_masks_handle_edge_cases():
    assert CustomDataset._get_sliding_window_starts_and_prefix_masks(9, 10, 2) == [
        (0, 0)
    ]
    assert CustomDataset._get_sliding_window_starts_and_prefix_masks(25, 10, 0) == [
        (0, 0),
        (10, 0),
        (15, 5),
    ]
    assert CustomDataset._get_sliding_window_starts_and_prefix_masks(5, 1, 99) == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
    ]


def test_sliding_window_train_insight_prompt_matches_window_context():
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "parent_id": ["None", "1", "2"],
            "prompt": ["p1", "p2", "p3"],
            "answer": ["a1", "a2", "a3"],
        }
    )

    with patch(
        "llm_studio.src.datasets.text_causal_language_modeling_ds.get_tokenizer",
        return_value=CharacterTokenizer(),
    ):
        dataset = CustomDataset(df, _cfg(), mode="train")

    second_window = dataset[len(dataset) - 1]
    prompt_tokens = second_window["prompt_input_ids"][
        second_window["prompt_attention_mask"].bool()
    ]

    assert "".join(chr(token) for token in prompt_tokens.tolist()) == "p2 a2p3"
