import re
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from llm_studio.app_utils.default_datasets import (
    prepare_default_dataset_causal_language_modeling,
)
from llm_studio.python_configs.text_causal_language_modeling_config import (
    ConfigNLPCausalLMDataset,
    ConfigNLPCausalLMTokenizer,
    ConfigProblemBase,
)
from llm_studio.src.datasets.text_causal_language_modeling_ds import CustomDataset


def test_prepare_default_dataset(tmp_path):
    df = prepare_default_dataset_causal_language_modeling(tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert set(df.keys()) == set(
        ["instruction", "output", "id", "parent_id", "lang", "rank"]
    )
    assert df.shape == (13026, 6)


def test_clean_output():
    output = {
        "predicted_text": np.array(
            [
                "This is a test",
                "This is a test <stop> This is a test",
                "This is a test <stop2> This is a test",
                "This is a test <stop3> <stop> This is a test",
                "<stop2> <stop> This is a test",
                "This is a test <stop>",
            ]
        )
    }

    cfg = mock.MagicMock()
    cfg.tokenizer._stop_words = ["<stop>", "<stop2>", "<stop3>"]

    predicted_text_clean = CustomDataset.clean_output(output=output, cfg=cfg)[
        "predicted_text"
    ]
    assert predicted_text_clean == [
        "This is a test",
        "This is a test",
        "This is a test",
        "This is a test",
        "",
        "This is a test",
    ]


def test_sanity_check_raises_error():
    mock_config = MagicMock()
    mock_config.dataset.parent_id_column = "parent_id"
    mock_config.dataset.id_column = "id"
    mock_config.dataset.answer_column = "answer"

    df_1 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "parent_id": [2, None, 4, 1],
            "answer": ["a", "b", "c", "d"],
            "other_data": ["a", "b", "c", "d"],
        }
    )
    CustomDataset.sanity_check(df_1, mock_config)

    df_2 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "parent_id": [None, None, None, None],
            "answer": ["a", "b", "c", "d"],
            "other_data": ["a", "b", "c", "d"],
        }
    )
    CustomDataset.sanity_check(df_2, mock_config)

    invalid_df_1 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "parent_id": [1, 2, 3, 4],
            "answer": ["a", "b", "c", "d"],
            "other_data": ["a", "b", "c", "d"],
        }
    )
    with pytest.raises(
        AssertionError,
        match=r"Parent id column:.* is the same as id column for some rows",
    ):
        CustomDataset.sanity_check(invalid_df_1, mock_config)

    invalid_df_2 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "parent_id": [2, 3, 4, 1],
            "other_data": ["a", "b", "c", "d"],
        }
    )
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Did not find any conversation chain. "
            "Please ensure that some parent ids are empty."
            "\n"
            "Conversations are chained using parent id, "
            "start conversation record should have empty parent id."
            "\n"
            f"Parent id column checked:{mock_config.dataset.parent_id_column}"
        ),
    ):
        CustomDataset.sanity_check(invalid_df_2, mock_config)


@pytest.fixture
def mock_auto_tokenizer():
    # from
    # https://github.com/deepset-ai/haystack/blob/b5aef24a7ebac55cb4ba492baf81a85598700b94/test/conftest.py#L908
    with patch(
        "transformers.AutoTokenizer.from_pretrained", autospec=True
    ) as mock_from_pretrained:
        yield mock_from_pretrained


def test_init(mock_auto_tokenizer):
    df = pd.DataFrame(
        {
            "col_A": [1, 2, 3],
            "col_B": [4, 5, 6],
        }
    )
    cfg = mock.MagicMock()
    cfg.dataset.prompt_column = "col_A"
    cfg.dataset.answer_column = "col_B"
    cfg.dataset.parent_id_column = "None"
    cfg.dataset.system_column = "None"

    cfg.dataset.text_system_start = ""
    cfg.dataset.text_prompt_start = ""
    cfg.dataset.text_answer_separator = ""

    cfg.tokenizer.tokenizer_kwargs = '{"use_fast": true, "add_prefix_space": false}'

    dataset = CustomDataset(df, cfg)

    assert dataset.df.equals(df)
    assert dataset.mode == "train"


def test_getitem():
    df = pd.DataFrame(
        {
            "prompt": ["prompt 1", "prompt 2", "prompt 3"],
            "answer": ["answer 1", "answer 2", "answer 3"],
            "parent_id": [None, 0, 1],
            "system": ["system 1", "system 2", "system 3"],
            "id": [0, 1, 2],
        }
    )

    cfg = ConfigProblemBase(
        dataset=ConfigNLPCausalLMDataset(
            prompt_column=("prompt",),
            answer_column="answer",
            parent_id_column="parent_id",
            system_column="system",
            text_system_start="System:",
            text_prompt_start="Prompt:",
            text_answer_separator="Answer:",
            add_eos_token_to_answer=True,
            limit_chained_samples=True,
        ),
        tokenizer=ConfigNLPCausalLMTokenizer(max_length=513),
    )

    cfg.llm_backbone = "EleutherAI/pythia-2.8b-deduped"

    dataset = CustomDataset(df, cfg)
    assert len(dataset) == 1

    result = dataset[0]
    assert isinstance(result, dict)
    assert set(result.keys()) == {
        "labels",
        "input_ids",
        "attention_mask",
        "prompt_input_ids",
        "prompt_attention_mask",
        "answer_input_ids",
        "answer_attention_mask",
    }

    assert (
        dataset.tokenizer.decode(result["input_ids"], skip_special_tokens=True)
        == "System:system 1"
        "Prompt:prompt 1"
        "Answer:answer 1"
        "Prompt:prompt 2"
        "Answer:answer 2"
        "Prompt:prompt 3"
        "Answer:answer 3"
    )

    assert (
        dataset.tokenizer.decode(result["prompt_input_ids"], skip_special_tokens=True)
        == "System:system 1"
        "Prompt:prompt 1"
        "Answer:answer 1"
        "Prompt:prompt 2"
        "Answer:answer 2"
        "Prompt:prompt 3"
        "Answer:"
    )

    assert (
        dataset.tokenizer.decode(result["input_ids"], skip_special_tokens=False)
        == "<|endoftext|>" * 475 + "System:system 1"
        "<|endoftext|>"
        "Prompt:prompt 1"
        "<|endoftext|>"
        "Answer:answer 1"
        "<|endoftext|>"
        "Prompt:prompt 2"
        "<|endoftext|>"
        "Answer:answer 2"
        "<|endoftext|>"
        "Prompt:prompt 3"
        "<|endoftext|>"
        "Answer:answer 3"
        "<|endoftext|>"
    )

    assert result["input_ids"].shape == (513,)
    assert result["prompt_input_ids"].shape == (513,)


def test_getitem_no_chaining():
    df = pd.DataFrame(
        {
            "prompt": ["prompt 1", "prompt 2", "prompt 3"],
            "answer": ["answer 1", "answer 2", "answer 3"],
            "parent_id": [None, 0, 1],
            "system": ["system 1", "system 2", "system 3"],
            "id": [0, 1, 2],
        }
    )

    cfg = ConfigProblemBase(
        dataset=ConfigNLPCausalLMDataset(
            prompt_column=("prompt",),
            answer_column="answer",
            parent_id_column="None",
            system_column="system",
            text_system_start="System:",
            text_prompt_start="Prompt:",
            text_answer_separator="Answer:",
            add_eos_token_to_answer=True,
        ),
        tokenizer=ConfigNLPCausalLMTokenizer(max_length=513),
    )

    cfg.llm_backbone = "EleutherAI/pythia-2.8b-deduped"

    dataset = CustomDataset(df, cfg)
    assert len(dataset) == 3

    for i in range(3):
        result = dataset[i]
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            "labels",
            "input_ids",
            "attention_mask",
            "prompt_input_ids",
            "prompt_attention_mask",
            "answer_input_ids",
            "answer_attention_mask",
        }

        assert (
            dataset.tokenizer.decode(result["input_ids"], skip_special_tokens=True)
            == f"System:system {i + 1}"
            f"Prompt:prompt {i + 1}"
            f"Answer:answer {i + 1}"
        )

        assert (
            dataset.tokenizer.decode(
                result["prompt_input_ids"], skip_special_tokens=True
            )
            == f"System:system {i + 1}"
            f"Prompt:prompt {i + 1}"
            "Answer:"
        )


def test_encode():
    df = pd.DataFrame(
        {
            "prompt": ["a", "a"],
            "answer": ["b", "b"],
            "parent_id": [None, 0],
            "id": [0, 1],
        }
    )

    cfg = ConfigProblemBase(
        dataset=ConfigNLPCausalLMDataset(
            prompt_column=("prompt",),
            answer_column="answer",
            parent_id_column="parent_id",
            text_prompt_start="<|prompt|>",
            text_answer_separator="<|answer|>",
            add_eos_token_to_answer=True,
            limit_chained_samples=True,
        ),
        tokenizer=ConfigNLPCausalLMTokenizer(
            max_length=64,
            tokenizer_kwargs='{"use_fast": true, "add_prefix_space": false}',
        ),
    )

    cfg.llm_backbone = "h2oai/h2o-danube2-1.8b-base"

    dataset = CustomDataset(df, cfg)
    assert len(dataset) == 1

    result = dataset[0]

    labels = result["labels"]
    assert (labels != -100).sum() == 28

    out = dataset.tokenizer.decode(result["input_ids"]).replace("<unk>", "")
    assert out == "<|prompt|>a</s><|answer|>b</s><|prompt|>a</s><|answer|>b</s>"


def test_encode_maxlength():
    df = pd.DataFrame(
        {
            "prompt": ["a", "a"],
            "answer": ["b", "a b"],
            "parent_id": [None, 0],
            "id": [0, 1],
        }
    )

    cfg = ConfigProblemBase(
        dataset=ConfigNLPCausalLMDataset(
            prompt_column=("prompt",),
            answer_column="answer",
            parent_id_column="parent_id",
            text_prompt_start="<|prompt|>",
            text_answer_separator="<|answer|>",
            add_eos_token_to_answer=True,
            limit_chained_samples=True,
        ),
        tokenizer=ConfigNLPCausalLMTokenizer(
            max_length=2,
            tokenizer_kwargs='{"use_fast": true, "add_prefix_space": false}',
        ),
    )

    cfg.llm_backbone = "h2oai/h2o-danube2-1.8b-base"

    dataset = CustomDataset(df, cfg)
    assert len(dataset) == 1

    result = dataset[0]
    out = dataset.tokenizer.decode(result["input_ids"]).replace("<unk>", "")
    assert out == "a b"


def test_preprocess_dataframe_personalize():
    df = pd.DataFrame(
        {
            "prompt": ["Open Assistant", "a"],
            "answer": ["b", "LAION b"],
            "parent_id": [None, 0],
            "id": [0, 1],
        }
    )

    cfg = ConfigProblemBase(
        dataset=ConfigNLPCausalLMDataset(
            prompt_column=("prompt",),
            answer_column="answer",
            parent_id_column="parent_id",
            chatbot_author="H2O.ai",
            chatbot_name="Danube",
            personalize=True,
        ),
    )

    cfg.llm_backbone = "h2oai/h2o-danube2-1.8b-base"

    assert df["prompt"].str.contains("Open Assistant").any()
    assert df["answer"].str.contains("LAION").any()

    dataset = CustomDataset(df, cfg)
    df = dataset.preprocess_dataframe(df, cfg)

    assert df["prompt"].str.contains("Danube").any()
    assert df["answer"].str.contains("H2O.ai").any()


def test_preprocess_dataframe_no_personalize():
    df = pd.DataFrame(
        {
            "prompt": ["Open Assistant", "a"],
            "answer": ["b", "LAION b"],
            "parent_id": [None, 0],
            "id": [0, 1],
        }
    )

    cfg = ConfigProblemBase(
        dataset=ConfigNLPCausalLMDataset(
            prompt_column=("prompt",),
            answer_column="answer",
            parent_id_column="parent_id",
            chatbot_author="H2O.ai",
            chatbot_name="Danube",
            personalize=False,
        ),
    )

    cfg.llm_backbone = "h2oai/h2o-danube2-1.8b-base"

    assert df["prompt"].str.contains("Open Assistant").any()
    assert df["answer"].str.contains("LAION").any()

    dataset = CustomDataset(df, cfg)
    df_processed = dataset.preprocess_dataframe(df.copy(), cfg)

    assert df_processed["prompt"].str.contains("Open Assistant").any()
    assert df_processed["answer"].str.contains("LAION").any()
    assert not df_processed["prompt"].str.contains("Danube").any()
    assert not df_processed["answer"].str.contains("H2O.ai").any()

    assert df_processed.equals(df)


class CharacterTokenizer:
    pad_token_id = 0

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        input_ids = torch.arange(1, len(text) + 1, dtype=torch.long)
        return {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": torch.ones(1, len(text), dtype=torch.long),
        }


def make_long_sample_cfg(long_sample_strategy="Truncate", sliding_window_overlap=0):
    return ConfigProblemBase(
        llm_backbone="unit-test",
        dataset=ConfigNLPCausalLMDataset(
            system_column="None",
            prompt_column=("prompt",),
            answer_column="answer",
            text_prompt_start="",
            text_answer_separator="",
            add_eos_token_to_prompt=False,
            add_eos_token_to_answer=False,
        ),
        tokenizer=ConfigNLPCausalLMTokenizer(
            max_length=10,
            long_sample_strategy=long_sample_strategy,
            sliding_window_overlap=sliding_window_overlap,
        ),
    )


@patch("llm_studio.src.datasets.text_causal_language_modeling_ds.get_tokenizer")
def test_skip_long_samples_in_training(mock_get_tokenizer):
    mock_get_tokenizer.return_value = CharacterTokenizer()
    df = pd.DataFrame(
        {
            "prompt": ["short", "this prompt is too long"],
            "answer": ["ok", "this answer is also too long"],
        }
    )
    dataset = CustomDataset(
        df, make_long_sample_cfg(long_sample_strategy="Skip"), mode="train"
    )

    assert len(dataset) == 1
    assert dataset.sample_index == [(0, None)]


@patch("llm_studio.src.datasets.text_causal_language_modeling_ds.get_tokenizer")
def test_sliding_window_expands_long_training_samples(mock_get_tokenizer):
    mock_get_tokenizer.return_value = CharacterTokenizer()
    df = pd.DataFrame({"prompt": ["12345"], "answer": ["abcdefghi"]})
    dataset = CustomDataset(
        df,
        make_long_sample_cfg(
            long_sample_strategy="Sliding Window", sliding_window_overlap=2
        ),
        mode="train",
    )

    assert dataset.sample_index == [(0, 0), (0, 4)]
    assert len(dataset) == 2
    first_sample = dataset[0]
    second_sample = dataset[1]
    assert first_sample["attention_mask"].sum() == 10
    assert second_sample["attention_mask"].sum() == 10


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
def test_mask_prompt_labels_keeps_prompt_format_tokens_trainable(mock_get_tokenizer):
    mock_get_tokenizer.return_value = OrdinalCharacterTokenizer()
    df = pd.DataFrame({"prompt": ["hi"], "answer": ["ok"]})
    cfg = ConfigProblemBase(
        llm_backbone="unit-test",
        dataset=ConfigNLPCausalLMDataset(
            system_column="None",
            prompt_column=("prompt",),
            answer_column="answer",
            text_prompt_start="<|prompt|>",
            text_answer_separator="<|answer|>",
            add_eos_token_to_prompt=True,
            add_eos_token_to_answer=False,
            mask_prompt_labels=True,
            mask_prompt_user_text_only=True,
        ),
        tokenizer=ConfigNLPCausalLMTokenizer(max_length=128),
    )
    cfg.tokenizer._tokenizer_eos_token = "<EOS>"

    dataset = CustomDataset(df, cfg)
    input_ids, labels, _, _ = dataset._get_input_ids_labels_and_encodings(0)
    text = dataset.tokenizer.decode(input_ids)

    assert text == "<|prompt|>hi<EOS><|answer|>ok"
    assert torch.equal(labels[: len("<|prompt|>")], input_ids[: len("<|prompt|>")])
    assert labels[len("<|prompt|>") : len("<|prompt|>hi")].tolist() == [-100, -100]
    assert torch.equal(labels[len("<|prompt|>hi") :], input_ids[len("<|prompt|>hi") :])


@patch("llm_studio.src.datasets.text_causal_language_modeling_ds.get_tokenizer")
def test_mask_prompt_user_text_only_restores_legacy_full_prompt_mask(
    mock_get_tokenizer,
):
    mock_get_tokenizer.return_value = OrdinalCharacterTokenizer()
    df = pd.DataFrame({"prompt": ["hi"], "answer": ["ok"]})
    cfg = ConfigProblemBase(
        llm_backbone="unit-test",
        dataset=ConfigNLPCausalLMDataset(
            system_column="None",
            prompt_column=("prompt",),
            answer_column="answer",
            text_prompt_start="<|prompt|>",
            text_answer_separator="<|answer|>",
            add_eos_token_to_prompt=True,
            add_eos_token_to_answer=False,
            mask_prompt_labels=True,
            mask_prompt_user_text_only=False,
        ),
        tokenizer=ConfigNLPCausalLMTokenizer(max_length=128),
    )
    cfg.tokenizer._tokenizer_eos_token = "<EOS>"

    dataset = CustomDataset(df, cfg)
    input_ids, labels, prompt_encodings, _ = (
        dataset._get_input_ids_labels_and_encodings(0)
    )
    prompt_len = len(prompt_encodings[0])

    assert labels[:prompt_len].tolist() == [-100] * prompt_len
    assert torch.equal(labels[prompt_len:], input_ids[prompt_len:])
