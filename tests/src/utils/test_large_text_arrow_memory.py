from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
from sklearn.model_selection import train_test_split

from llm_studio.src.datasets import conversation_chain_handler as chains
from llm_studio.src.utils.large_text_arrow_memory import (
    _ArrowTextSequence,
    _clean_missing_text_values,
    _clean_text_rows_arrow,
    _handler_init_arrow,
    _is_arrow_backed_string,
    _promote_string_columns_to_large,
)


def _arrow_dataframe(values):
    return pa.table({"Text": values}).to_pandas(types_mapper=pd.ArrowDtype)


def _cfg():
    return SimpleNamespace(
        dataset=SimpleNamespace(
            train_text_column=True,
            prompt_column=("Text",),
            answer_column="Text",
            parent_id_column="None",
            system_column="None",
        )
    )


def test_clean_text_preserves_arrow_storage():
    df = _arrow_dataframe(["alpha", None, "nan", "beta"])
    assert _is_arrow_backed_string(df["Text"])

    cleaned = _clean_missing_text_values(df["Text"])

    assert _is_arrow_backed_string(cleaned)
    assert cleaned.tolist() == ["alpha", "", "", "beta"]


def test_clean_rows_drops_empty_without_python_object_conversion():
    df = _arrow_dataframe(["alpha", " ", "beta", None])

    cleaned = _clean_text_rows_arrow(df, "Text", rank=0, label="test")

    assert _is_arrow_backed_string(cleaned["Text"])
    assert cleaned["Text"].tolist() == ["alpha", "beta"]


def test_promote_string_columns_to_large_keeps_random_split_arrow_backed():
    table = pa.table(
        {
            "Text": ["eins", "zwei", "drei", "vier", "fuenf", "sechs"],
            "row_id": [1, 2, 3, 4, 5, 6],
        }
    )

    promoted, columns = _promote_string_columns_to_large(table)

    assert columns == ["Text"]
    assert pa.types.is_large_string(promoted.schema.field("Text").type)
    assert promoted.schema.field("row_id").type == table.schema.field("row_id").type

    df = promoted.to_pandas(types_mapper=pd.ArrowDtype)
    train_df, val_df = train_test_split(df, test_size=0.33, random_state=1337)

    assert _is_arrow_backed_string(train_df["Text"])
    assert _is_arrow_backed_string(val_df["Text"])
    assert "large_string" in str(train_df["Text"].dtype)
    assert sorted(train_df["Text"].tolist() + val_df["Text"].tolist()) == sorted(
        df["Text"].tolist()
    )


def test_pure_text_handler_keeps_arrow_answers_lazy(monkeypatch):
    df = _arrow_dataframe(["eins", "zwei", "drei"])
    monkeypatch.setattr(chains, "_patch_plain_text_custom_dataset", lambda: None)
    handler = chains.ConversationChainHandler.__new__(chains.ConversationChainHandler)

    _handler_init_arrow(handler, df, _cfg())

    assert isinstance(handler.answers, _ArrowTextSequence)
    assert handler.answers[0] == "eins"
    assert handler.answers[-1] == "drei"
    assert handler.prompts[1] == chains.PLAIN_TEXT_PROMPT
    assert handler.conversation_chain_ids[2] == [2]
