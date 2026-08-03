from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

from llm_studio.src.models import text_causal_language_modeling_model as model_module
from llm_studio.src.plots import text_causal_language_modeling_plots as plots_module


class _DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)


class _DummyLoss(nn.Module):
    def forward(self, logits, labels):
        return torch.tensor(0.25)


class _DummyPerplexity(nn.Module):
    def forward(self, logits, labels):
        return torch.tensor([2.0, 3.0])


class _WhitespaceTokenizer:
    def __init__(self):
        self.vocabulary = {}

    def _encode(self, text):
        token_ids = []
        for token in text.split():
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary) + 1
            token_ids.append(self.vocabulary[token])
        return token_ids

    def __call__(
        self,
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    ):
        del add_special_tokens, padding, truncation
        if isinstance(texts, str):
            return {"input_ids": self._encode(texts)}
        return {"input_ids": [self._encode(text) for text in texts]}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return self._encode(text)


def test_perplexity_validation_also_generates_predictions(monkeypatch):
    cfg = SimpleNamespace(
        architecture=SimpleNamespace(gradient_checkpointing=False),
        prediction=SimpleNamespace(metric="Perplexity"),
        tokenizer=SimpleNamespace(_padding_side="left"),
    )

    model = model_module.Model.__new__(model_module.Model)
    nn.Module.__init__(model)
    model.cfg = cfg
    model.backbone = _DummyBackbone()
    model.loss_fn = _DummyLoss()
    model.perplexity = _DummyPerplexity()
    model.eval()

    logits = torch.randn(2, 4, 8)
    generated_ids = torch.tensor([[4, 5, 6], [7, 8, 9]])
    generation_calls = []

    monkeypatch.setattr(
        model_module,
        "forward",
        lambda backbone, input_ids, attention_mask: SimpleNamespace(logits=logits),
    )

    def generate(self, batch, cfg, streamer=None):
        generation_calls.append((batch, cfg, streamer))
        return generated_ids

    monkeypatch.setattr(model_module.Model, "generate", generate)

    batch = {
        "input_ids": torch.ones((2, 4), dtype=torch.long),
        "attention_mask": torch.ones((2, 4), dtype=torch.long),
        "labels": torch.ones((2, 4), dtype=torch.long),
    }
    output = model.forward(batch, padding=False)

    assert torch.equal(output["perplexity"], torch.tensor([2.0, 3.0]))
    assert torch.equal(output["predicted_answer_ids"], generated_ids)
    assert output["predicted_answer_ids"].device.type == "cpu"
    assert len(generation_calls) == 1


def test_autoregressive_match_statistics_penalize_wrong_and_extra_tokens():
    statistics = plots_module.get_autoregressive_match_statistics(
        target_texts=["alpha beta", "same text"],
        predicted_texts=["alpha gamma extra", "same text"],
        tokenizer=_WhitespaceTokenizer(),
    )

    assert statistics.loc[0, "Metric (Strict Exact Match %)"] == 0.0
    assert statistics.loc[0, "Metric (Token Accuracy %)"] == 100.0 / 3.0
    assert statistics.loc[0, "Metric (Length Match %)"] == 0.0
    assert statistics.loc[0, "Target Token Count"] == 2
    assert statistics.loc[0, "Predicted Token Count"] == 3
    assert statistics.loc[1, "Metric (Strict Exact Match %)"] == 100.0
    assert statistics.loc[1, "Metric (Token Accuracy %)"] == 100.0
    assert statistics.loc[1, "Metric (Length Match %)"] == 100.0


def test_perplexity_insights_include_exact_match_and_full_validation_summary(
    monkeypatch, tmp_path
):
    conversations = [
        {
            "systems": [""],
            "prompts": ["First prompt"],
            "answers": ["Reference answer"],
        },
        {
            "systems": [""],
            "prompts": ["Second prompt"],
            "answers": ["Second target"],
        },
    ]
    monkeypatch.setattr(
        plots_module,
        "get_conversation_chains",
        lambda val_df, cfg, limit_chained_samples: conversations,
    )
    monkeypatch.setattr(
        plots_module,
        "get_tokenizer",
        lambda cfg: _WhitespaceTokenizer(),
    )

    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            limit_chained_samples=False,
            prompt_column=("instruction",),
            answer_column="output",
        ),
        prediction=SimpleNamespace(metric="Perplexity"),
        output_directory=str(tmp_path),
    )
    val_outputs = {
        "predicted_text": np.array(["Reference answer", "Second wrong"]),
        "metrics": np.array([1.0004, 2.718281]),
    }

    plot_data = plots_module.plot_validation_predictions(
        val_outputs=val_outputs,
        cfg=cfg,
        val_df=pd.DataFrame(
            {
                "instruction": ["First prompt", "Second prompt"],
                "output": ["Reference answer", "Second target"],
            }
        ),
        mode="validation",
    )
    dataframe = pd.read_parquet(plot_data.data)

    summary = dataframe.iloc[0]
    first_prediction = dataframe.iloc[1]
    second_prediction = dataframe.iloc[2]

    input_text_column_name = (
        "Input Text (tokenization max length setting "
        "may truncate the input text during training/inference)"
    )
    assert summary["Sample"] == "Validation Summary"
    assert "1/2" in summary[input_text_column_name]
    assert summary["Metric (Strict Exact Match %)"] == 50.0
    assert summary["Metric (Token Accuracy %)"] == 75.0
    assert summary["Metric (Length Match %)"] == 100.0
    assert summary["Target Token Count"] == 2.0
    assert summary["Predicted Token Count"] == 2.0
    assert np.isclose(summary["Metric (Perplexity)"], 1.8593405, atol=1e-6)

    assert first_prediction["Target Text"] == "Reference answer"
    assert first_prediction["Predicted Text"] == "Reference answer"
    assert first_prediction["Metric (Strict Exact Match %)"] == 100.0
    assert first_prediction["Metric (Token Accuracy %)"] == 100.0
    assert first_prediction["Metric (Perplexity)"] == 1.0004

    assert second_prediction["Target Text"] == "Second target"
    assert second_prediction["Predicted Text"] == "Second wrong"
    assert second_prediction["Metric (Strict Exact Match %)"] == 0.0
    assert second_prediction["Metric (Token Accuracy %)"] == 50.0
    assert second_prediction["Metric (Length Match %)"] == 100.0
