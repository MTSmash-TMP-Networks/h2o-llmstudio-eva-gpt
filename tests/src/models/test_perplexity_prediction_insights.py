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


def test_perplexity_insights_table_contains_generated_text(monkeypatch, tmp_path):
    monkeypatch.setattr(
        plots_module,
        "get_conversation_chains",
        lambda val_df, cfg, limit_chained_samples: [
            {
                "systems": [""],
                "prompts": ["Explain perplexity."],
                "answers": ["Reference answer"],
            }
        ],
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
        "predicted_text": np.array(["Generated answer"]),
        "metrics": np.array([2.718]),
    }

    plot_data = plots_module.plot_validation_predictions(
        val_outputs=val_outputs,
        cfg=cfg,
        val_df=pd.DataFrame(
            {"instruction": ["Explain perplexity."], "output": ["Reference answer"]}
        ),
        mode="validation",
    )
    dataframe = pd.read_parquet(plot_data.data)

    assert dataframe.loc[0, "Target Text"] == "Reference answer"
    assert dataframe.loc[0, "Predicted Text"] == "Generated answer"
    assert dataframe.loc[0, "Metric (Perplexity)"] == 2.718
