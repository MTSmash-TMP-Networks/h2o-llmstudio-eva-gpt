from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

from llm_studio.src.datasets.text_causal_language_modeling_ds import CustomDataset


class CharacterTokenizer:
    pad_token_id = 0

    def __call__(self, text, return_tensors="pt", add_special_tokens=False, **kwargs):
        if isinstance(text, list):
            return {"input_ids": [[ord(char) for char in value] for value in text]}
        input_ids = torch.tensor([ord(char) for char in text], dtype=torch.long)
        return {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": torch.ones_like(input_ids).unsqueeze(0),
        }


def _cfg(*, strategy="Truncate", max_length=256):
    return SimpleNamespace(
        dataset=SimpleNamespace(
            train_text_column=True,
            prompt_column=("prompt",),
            answer_column="answer",
            system_column="system",
            parent_id_column="parent_id",
            id_column="id",
            limit_chained_samples=False,
            text_system_start="<S>",
            text_prompt_start="<U>",
            text_answer_separator="<A>",
            add_eos_token_to_system=True,
            add_eos_token_to_prompt=True,
            add_eos_token_to_answer=True,
            mask_prompt_labels=False,
            mask_prompt_user_text_only=False,
            only_last_answer=False,
        ),
        tokenizer=SimpleNamespace(
            max_length=max_length,
            long_sample_strategy=strategy,
            sliding_window_overlap=4,
            _tokenizer_eos_token="<E>",
        ),
        augmentation=SimpleNamespace(
            skip_parent_probability=0.0,
            random_parent_probability=0.0,
        ),
    )


def _decode(tokens):
    return "".join(chr(token) for token in tokens.tolist())


def _trainable_text(sample):
    return "".join(
        chr(token) for token in sample["labels"].tolist() if token != -100
    )


def _label_spans(labels):
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


def test_multiple_user_only_turns_feed_one_later_assistant_target():
    df = pd.DataFrame(
        {
            "id": ["id_1", "id_2", "id_3"],
            "parent_id": [None, "id_1", "id_2"],
            "system": ["", "", ""],
            "prompt": ["first", "second", "third"],
            "answer": ["", "", "answer"],
        }
    )
    cfg = _cfg()

    with patch(
        "llm_studio.src.datasets.text_causal_language_modeling_ds.get_tokenizer",
        return_value=CharacterTokenizer(),
    ):
        dataset = CustomDataset(df, cfg, mode="train")

    assert dataset.conversation_chain_handler.conversation_chain_ids == [[0, 1, 2]]
    assert len(dataset) == 1

    sample = dataset[0]
    visible = sample["input_ids"][sample["attention_mask"].bool()]
    visible_text = _decode(visible)

    assert visible_text == "<U>first<E><U>second<E><U>third<E><A>answer<E>"
    assert visible_text.count("<A>") == 1
    assert _trainable_text(sample) == "answer<E>"
    assert cfg.dataset.mask_prompt_labels is True
    assert cfg.dataset.only_last_answer is True


def test_answered_ids_still_train_once_around_context_only_turns():
    df = pd.DataFrame(
        {
            "id": ["id_1", "id_2", "id_3"],
            "parent_id": [None, "id_1", "id_2"],
            "system": ["", "", ""],
            "prompt": ["question one", "more context", "question two"],
            "answer": ["answer one", "", "answer two"],
        }
    )

    with patch(
        "llm_studio.src.datasets.text_causal_language_modeling_ds.get_tokenizer",
        return_value=CharacterTokenizer(),
    ):
        dataset = CustomDataset(df, _cfg(), mode="train")

    assert dataset.conversation_chain_handler.conversation_chain_ids == [
        [0],
        [0, 1, 2],
    ]
    assert len(dataset) == 2
    assert _trainable_text(dataset[0]) == "answer one<E>"
    assert _trainable_text(dataset[1]) == "answer two<E>"

    second_visible = dataset[1]["input_ids"][dataset[1]["attention_mask"].bool()]
    second_visible_text = _decode(second_visible)
    assert "answer one<E>" in second_visible_text
    assert "<U>more context<E><U>question two<E><A>" in second_visible_text


def test_fast_sliding_layout_matches_serial_context_only_encoding():
    df = pd.DataFrame(
        {
            "id": ["id_1", "id_2", "id_3"],
            "parent_id": [None, "id_1", "id_2"],
            "system": ["", "", ""],
            "prompt": ["context-1111", "context-2222", "question-3333"],
            "answer": ["", "", "final-answer"],
        }
    )
    cfg = _cfg(strategy="Sliding Window", max_length=20)

    with patch(
        "llm_studio.src.datasets.text_causal_language_modeling_ds.get_tokenizer",
        return_value=CharacterTokenizer(),
    ):
        dataset = CustomDataset(df, cfg, mode="train")

    layouts = dataset._compute_sample_layouts_batched()
    assert len(layouts) == 1

    input_ids, labels, _, _ = dataset._get_input_ids_labels_and_encodings(
        0,
        augment=False,
        trim_to_max_length=False,
    )
    assert layouts[0].length == len(input_ids)
    assert layouts[0].trainable_spans == _label_spans(labels)
    assert len(dataset) >= 1
    for idx in range(len(dataset)):
        assert torch.any(dataset[idx]["labels"][1:] != -100)
