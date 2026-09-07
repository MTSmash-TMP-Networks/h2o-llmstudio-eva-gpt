import logging

import torch

from llm_studio.src.utils import early_stop_runtime_cleanup as cleanup


def test_auxiliary_answer_padding_preslices_without_changing_legacy_result(monkeypatch):
    calls = []

    def fake_pad_tokens(**kwargs):
        calls.append(kwargs)
        return {
            "input_ids": kwargs["input_ids"],
            "attention_mask": kwargs["attention_mask"],
        }

    monkeypatch.setattr(cleanup, "_ORIGINAL_PAD_TOKENS", fake_pad_tokens)
    input_ids = torch.tensor([10, 11, 12, 13, 14])
    attention_mask = torch.ones_like(input_ids)

    cleanup._quiet_auxiliary_answer_padding(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=3,
        pad_token_id=0,
        direction="right",
        prefix="answer_",
    )

    assert calls[0]["input_ids"].tolist() == [12, 13, 14]
    assert calls[0]["attention_mask"].tolist() == [1, 1, 1]
    assert calls[0]["prefix"] == "answer_"


def test_real_training_input_is_not_presliced_by_cleanup(monkeypatch):
    calls = []

    def fake_pad_tokens(**kwargs):
        calls.append(kwargs)
        return kwargs

    monkeypatch.setattr(cleanup, "_ORIGINAL_PAD_TOKENS", fake_pad_tokens)
    input_ids = torch.tensor([10, 11, 12, 13, 14])

    cleanup._quiet_auxiliary_answer_padding(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_length=3,
        pad_token_id=0,
        prefix="",
    )

    assert calls[0]["input_ids"].tolist() == [10, 11, 12, 13, 14]


def test_prediction_zip_is_skipped_when_early_stop_has_no_validation_files(
    monkeypatch, tmp_path, caplog
):
    delegated = []

    def fake_save_predictions(experiment_name, experiment_path):
        delegated.append((experiment_name, experiment_path))
        return "predictions.zip"

    monkeypatch.setattr(
        cleanup, "_ORIGINAL_SAVE_PREDICTION_OUTPUTS", fake_save_predictions
    )
    with caplog.at_level(logging.INFO):
        result = cleanup._save_prediction_outputs_if_available("exp", str(tmp_path))

    assert result is None
    assert delegated == []
    assert "skipping the predictions ZIP" in caplog.text


def test_prediction_zip_delegates_when_validation_outputs_exist(monkeypatch, tmp_path):
    validation_file = tmp_path / "validation_predictions.csv"
    validation_file.write_text("prediction\nhello\n", encoding="utf-8")

    delegated = []

    def fake_save_predictions(experiment_name, experiment_path):
        delegated.append((experiment_name, experiment_path))
        return "predictions.zip"

    monkeypatch.setattr(
        cleanup, "_ORIGINAL_SAVE_PREDICTION_OUTPUTS", fake_save_predictions
    )
    result = cleanup._save_prediction_outputs_if_available("exp", str(tmp_path))

    assert result == "predictions.zip"
    assert delegated == [("exp", str(tmp_path))]
