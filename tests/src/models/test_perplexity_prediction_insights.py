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
        self.generation_config = SimpleNamespace(
            max_new_tokens=256,
            pad_token_id=0,
        )


class _DummyLoss(nn.Module):
    def forward(self, logits, labels):
        return torch.tensor(0.25)


class _DummyPerplexity(nn.Module):
    def forward(self, logits, labels):
        return torch.tensor([2.0, 3.0, 4.0, 5.0])


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


def test_perplexity_validation_generates_only_evenly_spaced_insight_samples(
    monkeypatch,
):
    cfg = SimpleNamespace(
        architecture=SimpleNamespace(gradient_checkpointing=False),
        training=SimpleNamespace(lora=False),
        environment=SimpleNamespace(
            use_deepspeed=False,
            _local_rank=0,
            _curr_val_step=4,
        ),
        prediction=SimpleNamespace(
            metric="Perplexity",
            perplexity_generation_samples=3,
            min_length_inference=2,
            max_length_inference=256,
        ),
        tokenizer=SimpleNamespace(_padding_side="left"),
    )

    model = model_module.Model.__new__(model_module.Model)
    nn.Module.__init__(model)
    model.cfg = cfg
    model.backbone = _DummyBackbone()
    model.backbone_config = SimpleNamespace(pad_token_id=0)
    model.loss_fn = _DummyLoss()
    model.perplexity = _DummyPerplexity()
    model.eval()

    logits = torch.randn(4, 4, 8)
    generated_ids = torch.tensor([[4, 5, 6], [7, 8, 9]])
    generation_calls = []

    monkeypatch.setattr(
        model_module,
        "forward",
        lambda backbone, input_ids, attention_mask: SimpleNamespace(logits=logits),
    )

    def generate(self, batch, cfg, streamer=None):
        generation_calls.append(
            {
                "batch_size": batch["input_ids"].shape[0],
                "max_new_tokens": self.backbone.generation_config.max_new_tokens,
            }
        )
        return generated_ids

    monkeypatch.setattr(model_module.Model, "generate", generate)

    batch = {
        "input_ids": torch.ones((4, 4), dtype=torch.long),
        "attention_mask": torch.ones((4, 4), dtype=torch.long),
        "labels": torch.ones((4, 4), dtype=torch.long),
        "answer_attention_mask": torch.tensor(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
            ]
        ),
        "validation_sample_index": torch.tensor([0, 3, 6, 9]),
        "validation_sample_count": torch.tensor([10, 10, 10, 10]),
    }
    output = model.forward(batch, padding=False)

    assert torch.equal(output["perplexity"], torch.tensor([2.0, 3.0, 4.0, 5.0]))
    assert torch.equal(
        output["prediction_generated"],
        torch.tensor([True, False, False, True]),
    )
    assert torch.equal(output["predicted_answer_ids"][0], generated_ids[0])
    assert torch.equal(output["predicted_answer_ids"][3], generated_ids[1])
    assert torch.all(output["predicted_answer_ids"][1:3] == 0)
    assert generation_calls == [{"batch_size": 2, "max_new_tokens": 10}]
    assert model.backbone.generation_config.max_new_tokens == 256


def test_perplexity_generation_indices_are_deterministic_and_evenly_spaced():
    assert model_module.get_perplexity_generation_indices(10, 3) == (0, 4, 9)
    assert model_module.get_perplexity_generation_indices(10, 1) == (5,)
    assert model_module.get_perplexity_generation_indices(3, 10) == (0, 1, 2)
    assert model_module.get_perplexity_generation_indices(10, 0) == ()


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


def test_perplexity_summary_uses_full_metric_set_and_generated_subset(
    monkeypatch, tmp_path
):
    conversations = [
        {
            "systems": [""],
            "prompts": [f"Prompt {index}"],
            "answers": [target],
        }
        for index, target in enumerate(
            ["alpha beta", "not generated", "third target", "same text"]
        )
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
        "predicted_text": np.array(["alpha beta", "", "", "same wrong"], dtype=object),
        "prediction_generated": np.array([True, False, False, True]),
        "metrics": np.array([1.0, 2.0, 3.0, 4.0]),
    }

    plot_data = plots_module.plot_validation_predictions(
        val_outputs=val_outputs,
        cfg=cfg,
        val_df=pd.DataFrame(
            {
                "instruction": [f"Prompt {index}" for index in range(4)],
                "output": [
                    "alpha beta",
                    "not generated",
                    "third target",
                    "same text",
                ],
            }
        ),
        mode="validation",
    )
    dataframe = pd.read_parquet(plot_data.data)

    summary = dataframe.iloc[0]
    sample_rows = dataframe.iloc[1:].set_index("Sample")
    input_text_column_name = (
        "Input Text (tokenization max length setting "
        "may truncate the input text during training/inference)"
    )

    assert "2/4" in summary[input_text_column_name]
    assert "1/2" in summary[input_text_column_name]
    assert summary["Metric (Strict Exact Match %)"] == 50.0
    assert summary["Metric (Perplexity)"] == 2.5
    assert sample_rows.loc["1", "Prediction Scope"] == "Generated insight sample"
    assert sample_rows.loc["2", "Prediction Scope"] == "Perplexity only"
    assert np.isnan(sample_rows.loc["2", "Metric (Strict Exact Match %)"])
    assert "Not generated" in sample_rows.loc["2", "Predicted Text"]
