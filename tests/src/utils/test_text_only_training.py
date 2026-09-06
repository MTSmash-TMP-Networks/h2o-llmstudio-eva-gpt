from types import SimpleNamespace

import pandas as pd

from llm_studio.app_utils.huggingface_import import (
    _detect_huggingface_columns,
    _prepare_fallback_dataset_for_text_training,
    _preferred_text_column as _preferred_hf_text_column,
)
from llm_studio.app_utils.text_only_training import (
    _plain_text_mask_with_text_only,
    _preferred_text_column,
    get_text_only_column,
    is_text_only_config,
)


def _cfg(*, train_text_column=True, prompt_column=("Text",), answer_column="Text"):
    return SimpleNamespace(
        dataset=SimpleNamespace(
            train_text_column=train_text_column,
            prompt_column=prompt_column,
            answer_column=answer_column,
        )
    )


def test_text_only_config_is_persisted_with_existing_fields():
    cfg = _cfg()
    assert is_text_only_config(cfg) is True
    assert get_text_only_column(cfg) == "Text"


def test_text_only_supports_arbitrary_text_column_names():
    cfg = _cfg(prompt_column=("content",), answer_column="content")
    assert is_text_only_config(cfg) is True
    assert get_text_only_column(cfg) == "content"


def test_text_only_mask_uses_selected_column_and_skips_empty_rows():
    cfg = _cfg(prompt_column=("content",), answer_column="content")
    df = pd.DataFrame({"content": ["Alpha", "", None, "Beta"]})

    mask = _plain_text_mask_with_text_only(df, cfg)

    assert mask.tolist() == [True, False, False, True]


def test_chat_config_is_not_misdetected_as_text_only():
    cfg = _cfg(prompt_column=("instruction",), answer_column="output")
    assert is_text_only_config(cfg) is False
    assert get_text_only_column(cfg) is None


def test_disabled_raw_text_switch_remains_chat_mode():
    cfg = _cfg(train_text_column=False)
    assert is_text_only_config(cfg) is False


def test_preferred_text_column_prioritizes_common_corpus_names():
    assert _preferred_text_column(["id", "text", "title"]) == "text"
    assert _preferred_text_column(["id", "content"]) == "content"
    assert _preferred_text_column(["only_column"]) == "only_column"


def test_huggingface_text_column_prefers_common_corpus_names():
    assert _preferred_hf_text_column(["id", "url", "title", "text"]) == "text"
    assert _preferred_hf_text_column(["id", "document"]) == "document"
    assert _preferred_hf_text_column(["only_column"]) == "only_column"
    assert _preferred_hf_text_column(["id", "title"]) is None


def test_huggingface_column_detection_uses_streaming_metadata(monkeypatch):
    calls = {}

    def fake_load_dataset(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return SimpleNamespace(column_names=["id", "title", "text"])

    monkeypatch.setattr(
        "llm_studio.app_utils.huggingface_import.load_dataset", fake_load_dataset
    )

    columns = _detect_huggingface_columns(
        dataset_name="wikimedia/wikipedia",
        config="20231101.de",
        split="train",
        token=None,
    )

    assert columns == ["id", "title", "text"]
    assert calls["args"] == ("wikimedia/wikipedia", "20231101.de")
    assert calls["kwargs"]["split"] == "train"
    assert calls["kwargs"]["streaming"] is True


def test_fallback_huggingface_dataset_projects_selected_text_column():
    class FakeDataset:
        def __init__(self):
            self.column_names = ["id", "body", "title"]

        def select_columns(self, columns):
            self.column_names = list(columns)
            return self

        def rename_column(self, source, target):
            self.column_names = [target if value == source else value for value in self.column_names]
            return self

    dataset = _prepare_fallback_dataset_for_text_training(FakeDataset(), "body")

    assert dataset.column_names == ["Text"]
