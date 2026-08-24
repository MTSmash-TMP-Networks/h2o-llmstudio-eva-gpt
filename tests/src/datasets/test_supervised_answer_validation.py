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


def test_allows_empty_answer_as_context_for_later_answered_turn():
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

    CustomDataset.sanity_check(df, _cfg(), mode="train")


@pytest.mark.parametrize("empty_value", [None, float("nan"), "", "   ", "null"])
def test_allows_missing_answer_cells_as_chained_context(empty_value):
    df = pd.DataFrame(
        {
            "id": ["id_1", "id_2"],
            "parent_id": [None, "id_1"],
            "system": ["", ""],
            "prompt": ["Nur Kontext", "Jetzt beantworten"],
            "answer": [empty_value, "Antwort"],
            "Text": ["", ""],
        }
    )

    CustomDataset.sanity_check(df, _cfg(), mode="train")


def test_allows_multiple_context_only_turns_before_assistant_answer():
    df = pd.DataFrame(
        {
            "id": ["id_1", "id_2", "id_3", "id_4"],
            "parent_id": [None, "id_1", "id_2", "id_3"],
            "system": ["", "", "", ""],
            "prompt": [
                "Erste Information",
                "Zweite Information",
                "Dritte Information",
                "Bitte beantworte das jetzt.",
            ],
            "answer": ["", "", "", "Die gemeinsame Antwort."],
            "Text": ["", "", "", ""],
        }
    )

    CustomDataset.sanity_check(df, _cfg(), mode="train")


@pytest.mark.parametrize("empty_value", ["   ", "null", "None", "NaN", "na"])
def test_rejects_terminal_empty_answer_markers(empty_value):
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


def test_rejects_unfinished_leaf_after_valid_context_chain():
    df = pd.DataFrame(
        {
            "id": ["id_1", "id_2", "id_3"],
            "parent_id": [None, "id_1", "id_2"],
            "system": ["", "", ""],
            "prompt": ["Kontext", "Beantwortete Frage", "Unfertige Folgefrage"],
            "answer": ["", "Antwort", ""],
            "Text": ["", "", ""],
        }
    )

    with pytest.raises(AssertionError) as exc_info:
        CustomDataset.sanity_check(df, _cfg(), mode="train")

    message = str(exc_info.value)
    assert "not used as context for a later answered turn" in message
    assert "id='id_3'" in message


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
