from types import SimpleNamespace

import pandas as pd
import pytest

from llm_studio.src.datasets.text_causal_language_modeling_ds import CustomDataset


def _cfg():
    return SimpleNamespace(
        dataset=SimpleNamespace(
            train_text_column=True,
            prompt_column=("prompt",),
            answer_column="answer",
            system_column="system",
            parent_id_column="parent_id",
            id_column="id",
        )
    )


def test_rejects_empty_answer_inside_conversation_chain():
    df = pd.DataFrame(
        {
            "id": ["id_1", "id_2"],
            "parent_id": [None, "id_1"],
            "system": ["", ""],
            "prompt": [
                "Ist es moeglich ein LLM von Scratch wachsen zu lassen?",
                "Naja der Ansatz waere meines Erachtens gar nicht so schwierig.",
            ],
            "answer": ["", "Eine vollstaendige Antwort."],
            "Text": ["", ""],
        }
    )

    with pytest.raises(AssertionError) as exc_info:
        CustomDataset.sanity_check(df, _cfg(), mode="train")

    message = str(exc_info.value)
    assert "empty assistant answer" in message
    assert "id='id_1'" in message
    assert "parent_id=None" in message
    assert "<|Assistentin|><|Benutzer|>" in message


@pytest.mark.parametrize("empty_value", ["   ", "null", "None", "NaN", "na"])
def test_rejects_textual_missing_answer_markers(empty_value):
    df = pd.DataFrame(
        {
            "id": ["id_1"],
            "parent_id": [None],
            "system": [""],
            "prompt": ["Frage"],
            "answer": [empty_value],
            "Text": [""],
        }
    )

    with pytest.raises(AssertionError, match="empty assistant answer"):
        CustomDataset.sanity_check(df, _cfg(), mode="train")


def test_allows_valid_supervised_conversation_chain():
    df = pd.DataFrame(
        {
            "id": ["id_1", "id_2"],
            "parent_id": [None, "id_1"],
            "system": ["", ""],
            "prompt": ["Frage 1", "Frage 2"],
            "answer": ["Antwort 1", "Antwort 2"],
            "Text": ["", ""],
        }
    )

    CustomDataset.sanity_check(df, _cfg(), mode="train")


def test_allows_pure_text_row_without_assistant_answer():
    df = pd.DataFrame(
        {
            "id": ["text_1"],
            "parent_id": [None],
            "system": [""],
            "prompt": [""],
            "answer": [""],
            "Text": ["Dies ist ein reines Continued-Pretraining-Sample."],
        }
    )

    CustomDataset.sanity_check(df, _cfg(), mode="train")
