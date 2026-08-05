"""Fast sampled autoregressive insights for full-set perplexity validation."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import torch

from llm_studio.src.datasets import text_causal_language_modeling_ds as base_ds
from llm_studio.src.plots import text_causal_language_modeling_plots as base_plots
from llm_studio.src.utils.plot_utils import PlotData, format_for_markdown_visualization

_NOT_GENERATED_TEXT = "Not generated for this sample (Perplexity was still calculated)."
_GENERATED_SCOPE = "Generated insight sample"
_PERPLEXITY_ONLY_SCOPE = "Perplexity only"


def _install_dataset_sampling_metadata() -> None:
    dataset_class = base_ds.CustomDataset
    if getattr(dataset_class, "_perplexity_sampling_installed", False):
        return

    original_getitem = dataset_class.__getitem__
    original_postprocess_output = dataset_class.postprocess_output

    def getitem_with_validation_position(self, idx: int) -> dict:
        sample = original_getitem(self, idx)
        if self.mode != "train":
            original_idx = idx
            if hasattr(self, "sample_index"):
                original_idx = int(self.sample_index[idx][0])
            sample["validation_sample_index"] = torch.tensor(
                original_idx, dtype=torch.long
            )
            sample["validation_sample_count"] = torch.tensor(
                len(self.conversation_chain_handler), dtype=torch.long
            )
        return sample

    def postprocess_output_with_sampled_predictions(
        self, cfg, df: pd.DataFrame, output: dict
    ) -> dict:
        if cfg.prediction.metric == "Perplexity" and "predicted_text" in output:
            output = self.clean_output(output, cfg)

        output = original_postprocess_output(self, cfg, df, output)
        if cfg.prediction.metric != "Perplexity":
            return output
        if "predicted_text" not in output or "prediction_generated" not in output:
            return output

        generated_mask = np.asarray(output["prediction_generated"], dtype=bool)
        predicted_texts = np.asarray(output["predicted_text"], dtype=object)
        predicted_texts = predicted_texts.copy()
        predicted_texts[~generated_mask] = _NOT_GENERATED_TEXT
        output["predicted_text"] = predicted_texts
        return output

    dataset_class.__getitem__ = getitem_with_validation_position
    dataset_class.postprocess_output = postprocess_output_with_sampled_predictions
    dataset_class._perplexity_sampling_installed = True


def _get_prediction_mask(val_outputs: dict, sample_count: int) -> np.ndarray:
    if "prediction_generated" in val_outputs:
        mask = np.asarray(val_outputs["prediction_generated"], dtype=bool).reshape(-1)
        if len(mask) >= sample_count:
            return mask[:sample_count]
        return np.pad(mask, (0, sample_count - len(mask)), constant_values=False)
    if "predicted_text" in val_outputs:
        return np.ones(sample_count, dtype=bool)
    return np.zeros(sample_count, dtype=bool)


def _align_predicted_texts(
    val_outputs: dict, generated_mask: np.ndarray, sample_count: int
) -> list[str]:
    values = [str(value) for value in val_outputs.get("predicted_text", [])]
    if len(values) < sample_count:
        values.extend([_NOT_GENERATED_TEXT] * (sample_count - len(values)))
    values = values[:sample_count]
    return [
        value if generated else _NOT_GENERATED_TEXT
        for value, generated in zip(values, generated_mask, strict=False)
    ]


def _sampled_validation_summary_row(
    df: pd.DataFrame,
    input_text_column_name: str,
    metric_column_name: str | None,
) -> dict[str, Any]:
    summary = {column: None for column in df.columns}
    generated_df = df[df["Prediction Scope"] == _GENERATED_SCOPE]
    generated_count = len(generated_df)
    total_count = len(df)
    exact_matches = 0
    if generated_count:
        exact_matches = int(
            (generated_df["Metric (Strict Exact Match %)"] == 100.0).sum()
        )

    summary["Sample"] = "Validation Summary"
    if generated_count:
        summary[input_text_column_name] = (
            f"Autoregressive comparison sample: {generated_count}/{total_count}. "
            f"Exact generated answers: {exact_matches}/{generated_count} "
            f"({generated_df['Metric (Strict Exact Match %)'].mean():.3f}%)."
        )
    else:
        summary[input_text_column_name] = (
            f"Perplexity calculated for all {total_count} samples; "
            "autoregressive insight generation was disabled."
        )
    summary["Target Text"] = (
        "Perplexity aggregates the complete validation set. Exact Match, Token "
        "Accuracy and Length Match aggregate only generated insight samples."
    )
    summary["Predicted Text"] = (
        "Generated samples are selected deterministically across the validation set."
    )
    summary["Prediction Scope"] = f"{generated_count} generated / {total_count} total"

    sampled_columns = [
        "Metric (Strict Exact Match %)",
        "Metric (Token Accuracy %)",
        "Metric (Length Match %)",
        "Target Token Count",
        "Predicted Token Count",
    ]
    if generated_count:
        for column in sampled_columns:
            summary[column] = generated_df[column].mean()
    if metric_column_name is not None:
        summary[metric_column_name] = df[metric_column_name].mean()
    return summary


def _limit_rows(
    df: pd.DataFrame, metric_column_name: str | None, maximum_rows: int = 900
) -> pd.DataFrame:
    if len(df) <= maximum_rows:
        return df

    generated_df = df[df["Prediction Scope"] == _GENERATED_SCOPE]
    other_df = df[df["Prediction Scope"] != _GENERATED_SCOPE]
    remaining = max(maximum_rows - len(generated_df), 0)
    if remaining == 0:
        return generated_df.iloc[:maximum_rows].reset_index(drop=True)

    if metric_column_name is None:
        selected_other = other_df.sample(
            n=min(remaining, len(other_df)), random_state=42
        )
    else:
        other_df = other_df.sort_values(by=metric_column_name)
        low_count = remaining // 3
        high_count = remaining // 3
        middle_count = remaining - low_count - high_count
        middle = other_df.iloc[low_count : len(other_df) - high_count]
        middle_count = min(middle_count, len(middle))
        selected_other = pd.concat(
            [
                other_df.iloc[:low_count],
                middle.sample(n=middle_count, random_state=42),
                other_df.iloc[len(other_df) - high_count :]
                if high_count
                else other_df.iloc[0:0],
            ]
        )

    return pd.concat([generated_df, selected_other]).reset_index(drop=True)


def plot_validation_predictions(
    val_outputs: dict, cfg: Any, val_df: pd.DataFrame, mode: str
) -> PlotData:
    conversations = base_plots.get_conversation_chains(
        val_df, cfg, limit_chained_samples=cfg.dataset.limit_chained_samples
    )
    prompt_column_name = (
        cfg.dataset.prompt_column
        if len(cfg.dataset.prompt_column) > 1
        else cfg.dataset.prompt_column[0]
    )

    target_texts = [str(conversation["answers"][-1]) for conversation in conversations]
    input_texts = []
    for conversation in conversations:
        input_text = conversation["systems"][0]
        prompts = conversation["prompts"]
        answers = list(conversation["answers"])
        answers[-1] = ""
        for prompt, answer in zip(prompts, answers, strict=False):
            input_text += (
                f" **{prompt_column_name}:** "
                f"{prompt}\n\n"
                f"**{cfg.dataset.answer_column}:** "
                f"{answer}\n\n"
            )
        input_texts.append(input_text)

    sample_count = len(target_texts)
    generated_mask = _get_prediction_mask(val_outputs, sample_count)
    predicted_texts = _align_predicted_texts(val_outputs, generated_mask, sample_count)
    input_text_column_name = (
        "Input Text (tokenization max length setting "
        "may truncate the input text during training/inference)"
    )
    df = pd.DataFrame(
        {
            "Sample": [str(index + 1) for index in range(sample_count)],
            input_text_column_name: input_texts,
            "Target Text": target_texts,
            "Predicted Text": predicted_texts,
            "Prediction Scope": np.where(
                generated_mask, _GENERATED_SCOPE, _PERPLEXITY_ONLY_SCOPE
            ),
        }
    )

    comparison_columns = [
        "Metric (Strict Exact Match %)",
        "Metric (Token Accuracy %)",
        "Metric (Length Match %)",
        "Target Token Count",
        "Predicted Token Count",
    ]
    for column in comparison_columns:
        df[column] = np.nan

    generated_positions = np.flatnonzero(generated_mask)
    if len(generated_positions):
        tokenizer = base_plots.get_tokenizer(cfg)
        comparison_df = base_plots.get_autoregressive_match_statistics(
            target_texts=[target_texts[index] for index in generated_positions],
            predicted_texts=[predicted_texts[index] for index in generated_positions],
            tokenizer=tokenizer,
        )
        for column in comparison_columns:
            df.loc[generated_positions, column] = comparison_df[column].to_numpy()

    metric_column_name = None
    metric_decimals = 3
    if val_outputs.get("metrics") is not None:
        metric_column_name = f"Metric ({cfg.prediction.metric})"
        df[metric_column_name] = np.asarray(val_outputs["metrics"])[:sample_count]
        metric_decimals = 6 if cfg.prediction.metric == "Perplexity" else 3

    if val_outputs.get("explanations") is not None:
        df["Explanation"] = np.asarray(val_outputs["explanations"])[:sample_count]

    summary_row = _sampled_validation_summary_row(
        df=df,
        input_text_column_name=input_text_column_name,
        metric_column_name=metric_column_name,
    )

    df = pd.concat(
        [
            df[df["Prediction Scope"] == _GENERATED_SCOPE],
            df[df["Prediction Scope"] != _GENERATED_SCOPE],
        ]
    ).reset_index(drop=True)
    df = _limit_rows(df, metric_column_name)
    df = pd.concat([pd.DataFrame([summary_row]), df], ignore_index=True)

    for column in [input_text_column_name, "Target Text", "Predicted Text"]:
        df[column] = df[column].apply(format_for_markdown_visualization)

    if metric_column_name is not None:
        df[metric_column_name] = df[metric_column_name].round(decimals=metric_decimals)

    for column in [
        "Metric (Strict Exact Match %)",
        "Metric (Token Accuracy %)",
        "Metric (Length Match %)",
        "Target Token Count",
        "Predicted Token Count",
    ]:
        df[column] = df[column].round(decimals=3)

    path = os.path.join(cfg.output_directory, f"{mode}_viz.parquet")
    df.to_parquet(path)
    return PlotData(data=path, encoding="df")


def install_perplexity_validation_sampling() -> None:
    """Install dataset metadata and sampled validation-insight rendering."""
    _install_dataset_sampling_metadata()
    base_plots.plot_validation_predictions = plot_validation_predictions
