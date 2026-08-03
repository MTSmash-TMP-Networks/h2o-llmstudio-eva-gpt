import hashlib
import os
from typing import Any

import pandas as pd

from llm_studio.src.datasets.conversation_chain_handler import get_conversation_chains
from llm_studio.src.datasets.text_utils import get_tokenizer
from llm_studio.src.utils.data_utils import read_dataframe_drop_missing_labels
from llm_studio.src.utils.plot_utils import (
    PlotData,
    format_for_markdown_visualization,
    list_to_markdown_representation,
)


class Plots:
    @classmethod
    def plot_batch(cls, batch, cfg) -> PlotData:
        tokenizer = get_tokenizer(cfg)
        df = create_batch_prediction_df(batch, tokenizer)
        path = os.path.join(cfg.output_directory, "batch_viz.parquet")
        df.to_parquet(path)
        return PlotData(path, encoding="df")

    @classmethod
    def plot_data(cls, cfg) -> PlotData:
        """
        Plots the data in a scrollable table.
        We limit the number of rows to max 600 to avoid rendering issues in Wave.
        As the data visualization is instantiated on every page load, we cache the
        data visualization in a parquet file.
        """
        config_id = (
            str(cfg.dataset.train_dataframe)
            + str(cfg.dataset.system_column)
            + str(cfg.dataset.prompt_column)
            + str(cfg.dataset.answer_column)
            + str(cfg.dataset.parent_id_column)
        )
        config_hash = hashlib.md5(config_id.encode(), usedforsecurity=False).hexdigest()
        path = os.path.join(
            os.path.dirname(cfg.dataset.train_dataframe),
            f"__meta_info__{config_hash}_data_viz.parquet",
        )
        if os.path.exists(path):
            return PlotData(path, encoding="df")

        df = read_dataframe_drop_missing_labels(cfg.dataset.train_dataframe, cfg)

        conversations = get_conversation_chains(df, cfg, limit_chained_samples=True)

        # Limit to max 15 prompt-conversation-answer rounds
        # This yields to max 5 * sum_{i=1}^{15} i = 600 rows in the DataFrame
        max_conversation_length = min(
            max([len(conversation["prompts"]) for conversation in conversations]), 15
        )

        conversations_to_display = []
        for conversation_length in range(1, max_conversation_length + 1):
            conversations_to_display += [
                conversation
                for conversation in conversations
                if len(conversation["prompts"]) == conversation_length
            ][:5]

        # Convert into a scrollable table by transposing the dataframe
        df_transposed = pd.DataFrame(columns=["Sample Number", "Field", "Content"])

        i = 0
        for sample_number, conversation in enumerate(conversations_to_display):
            if conversation["systems"][0] != "":
                df_transposed.loc[i] = [
                    sample_number,
                    "System",
                    conversation["systems"][0],
                ]
                i += 1
            for prompt, answer in zip(
                conversation["prompts"], conversation["answers"], strict=False
            ):
                df_transposed.loc[i] = [
                    sample_number,
                    "Prompt",
                    prompt,
                ]
                i += 1
                df_transposed.loc[i] = [
                    sample_number,
                    "Answer",
                    answer,
                ]
                i += 1

        df_transposed["Content"] = df_transposed["Content"].apply(
            format_for_markdown_visualization
        )

        df_transposed.to_parquet(path)

        return PlotData(path, encoding="df")

    @classmethod
    def plot_validation_predictions(
        cls, val_outputs: dict, cfg: Any, val_df: pd.DataFrame, mode: str
    ) -> PlotData:
        return plot_validation_predictions(val_outputs, cfg, val_df, mode)


def _batch_tokenize_for_comparison(tokenizer, texts: list[str]) -> list[list[int]]:
    """Tokenize texts without adding model-specific BOS/EOS tokens."""
    try:
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        if texts and encoded and isinstance(encoded[0], int):
            encoded = [encoded]
        return [list(token_ids) for token_ids in encoded]
    except (AttributeError, KeyError, TypeError):
        return [
            list(tokenizer.encode(text, add_special_tokens=False)) for text in texts
        ]


def get_autoregressive_match_statistics(
    target_texts: list[str], predicted_texts: list[str], tokenizer
) -> pd.DataFrame:
    """Compare freely generated answers with validation targets.

    Strict Exact Match compares the decoded strings without normalization. Token
    Accuracy compares tokens at the same position and uses the longer sequence as
    denominator, so missing and additional tokens are both penalized.
    """
    if len(target_texts) != len(predicted_texts):
        raise ValueError(
            "Target and predicted text counts must match: "
            f"{len(target_texts)} != {len(predicted_texts)}"
        )

    target_texts = [str(text) for text in target_texts]
    predicted_texts = [str(text) for text in predicted_texts]
    target_token_ids = _batch_tokenize_for_comparison(tokenizer, target_texts)
    predicted_token_ids = _batch_tokenize_for_comparison(tokenizer, predicted_texts)

    strict_exact_matches = []
    token_accuracies = []
    length_matches = []
    target_token_counts = []
    predicted_token_counts = []

    for target_text, predicted_text, target_ids, predicted_ids in zip(
        target_texts,
        predicted_texts,
        target_token_ids,
        predicted_token_ids,
        strict=False,
    ):
        target_len = len(target_ids)
        predicted_len = len(predicted_ids)
        denominator = max(target_len, predicted_len)
        matching_tokens = sum(
            target_id == predicted_id
            for target_id, predicted_id in zip(target_ids, predicted_ids, strict=False)
        )
        token_accuracy = (
            100.0 if denominator == 0 else 100.0 * matching_tokens / denominator
        )

        strict_exact_matches.append(100.0 if target_text == predicted_text else 0.0)
        token_accuracies.append(token_accuracy)
        length_matches.append(100.0 if target_len == predicted_len else 0.0)
        target_token_counts.append(target_len)
        predicted_token_counts.append(predicted_len)

    return pd.DataFrame(
        {
            "Metric (Strict Exact Match %)": strict_exact_matches,
            "Metric (Token Accuracy %)": token_accuracies,
            "Metric (Length Match %)": length_matches,
            "Target Token Count": target_token_counts,
            "Predicted Token Count": predicted_token_counts,
        }
    )


def _validation_summary_row(
    df: pd.DataFrame,
    input_text_column_name: str,
    metric_column_name: str | None,
) -> dict[str, Any]:
    total = len(df)
    exact_matches = int((df["Metric (Strict Exact Match %)"] == 100.0).sum())
    summary = {column: None for column in df.columns}
    summary["Sample"] = "Validation Summary"
    summary[input_text_column_name] = (
        f"Exact generated answers: {exact_matches}/{total} "
        f"({df['Metric (Strict Exact Match %)'].mean():.3f}%)."
    )
    summary["Target Text"] = (
        "Aggregate values across the complete validation set before table sampling."
    )
    summary["Predicted Text"] = (
        "Strict Exact Match uses the actual autoregressive generated text."
    )
    summary["Metric (Strict Exact Match %)"] = df[
        "Metric (Strict Exact Match %)"
    ].mean()
    summary["Metric (Token Accuracy %)"] = df["Metric (Token Accuracy %)"].mean()
    summary["Metric (Length Match %)"] = df["Metric (Length Match %)"].mean()
    summary["Target Token Count"] = df["Target Token Count"].mean()
    summary["Predicted Token Count"] = df["Predicted Token Count"].mean()
    if metric_column_name is not None:
        summary[metric_column_name] = df[metric_column_name].mean()
    return summary


def plot_validation_predictions(
    val_outputs: dict, cfg: Any, val_df: pd.DataFrame, mode: str
) -> PlotData:
    conversations = get_conversation_chains(
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
        answers = conversation["answers"]
        # exclude last answer
        answers[-1] = ""
        for prompt, answer in zip(prompts, answers, strict=False):
            input_text += (
                f" **{prompt_column_name}:** "
                f"{prompt}\n\n"
                f"**{cfg.dataset.answer_column}:** "
                f"{answer}\n\n"
            )
        input_texts += [input_text]

    has_predictions = "predicted_text" in val_outputs
    if has_predictions:
        predicted_texts = [str(text) for text in val_outputs["predicted_text"]]
    else:
        predicted_texts = [
            "No predictions are generated for the selected metric"
        ] * len(target_texts)

    input_text_column_name = (
        "Input Text (tokenization max length setting "
        "may truncate the input text during training/inference)"
    )
    df = pd.DataFrame(
        {
            "Sample": [str(index + 1) for index in range(len(target_texts))],
            input_text_column_name: input_texts,
            "Target Text": target_texts,
            "Predicted Text": predicted_texts,
        }
    )

    if has_predictions:
        tokenizer = get_tokenizer(cfg)
        comparison_df = get_autoregressive_match_statistics(
            target_texts=target_texts,
            predicted_texts=predicted_texts,
            tokenizer=tokenizer,
        )
        df = pd.concat([df, comparison_df], axis=1)

    metric_column_name = None
    metric_decimals = 3
    if val_outputs.get("metrics") is not None:
        metric_column_name = f"Metric ({cfg.prediction.metric})"
        df[metric_column_name] = val_outputs["metrics"]
        metric_decimals = 6 if cfg.prediction.metric == "Perplexity" else 3

    if val_outputs.get("explanations") is not None:
        df["Explanation"] = val_outputs["explanations"]

    summary_row = None
    if has_predictions and len(df) > 0:
        summary_row = _validation_summary_row(
            df=df,
            input_text_column_name=input_text_column_name,
            metric_column_name=metric_column_name,
        )

    if len(df) > 900:
        if metric_column_name is not None:
            df.sort_values(by=metric_column_name, inplace=True)
            df = pd.concat(
                [
                    df.iloc[:300],
                    df.iloc[300:-300].sample(n=300, random_state=42),
                    df.iloc[-300:],
                ]
            ).reset_index(drop=True)
        else:
            df = df.sample(n=900, random_state=42).reset_index(drop=True)

    if summary_row is not None:
        df = pd.concat([pd.DataFrame([summary_row]), df], ignore_index=True)

    for column in [input_text_column_name, "Target Text", "Predicted Text"]:
        df[column] = df[column].apply(format_for_markdown_visualization)

    if metric_column_name is not None:
        df[metric_column_name] = df[metric_column_name].round(decimals=metric_decimals)

    percentage_columns = [
        "Metric (Strict Exact Match %)",
        "Metric (Token Accuracy %)",
        "Metric (Length Match %)",
    ]
    for column in percentage_columns:
        if column in df.columns:
            df[column] = df[column].round(decimals=3)

    token_count_columns = ["Target Token Count", "Predicted Token Count"]
    for column in token_count_columns:
        if column in df.columns:
            df[column] = df[column].round(decimals=3)

    path = os.path.join(cfg.output_directory, f"{mode}_viz.parquet")
    df.to_parquet(path)
    return PlotData(data=path, encoding="df")


def create_batch_prediction_df(
    batch, tokenizer, ids_for_tokenized_text="input_ids", labels_column="labels"
):
    df = pd.DataFrame(
        {
            "Prompt Text": [
                tokenizer.decode(input_ids, skip_special_tokens=True)
                for input_ids in batch["prompt_input_ids"].detach().cpu().numpy()
            ]
        }
    )
    df["Prompt Text"] = df["Prompt Text"].apply(format_for_markdown_visualization)
    if labels_column in batch.keys():
        df["Answer Text"] = [
            tokenizer.decode(
                [label for label in labels if label != -100],
                skip_special_tokens=True,
            )
            for labels in batch.get(labels_column, batch[ids_for_tokenized_text])
            .detach()
            .cpu()
            .numpy()
        ]
    tokens_list = [
        tokenizer.convert_ids_to_tokens(input_ids)
        for input_ids in batch[ids_for_tokenized_text].detach().cpu().numpy()
    ]
    masks_list = [
        [label != -100 for label in labels]
        for labels in batch.get(labels_column, batch[ids_for_tokenized_text])
        .detach()
        .cpu()
        .numpy()
    ]
    df["Tokenized Text"] = [
        list_to_markdown_representation(
            tokens, masks, pad_token=tokenizer.pad_token, num_chars=100
        )
        for tokens, masks in zip(tokens_list, masks_list, strict=False)
    ]
    # limit to 2000 rows, still renders fast in wave
    df = df.iloc[:2000]
    # Convert into a scrollable table by transposing the dataframe
    df_transposed = pd.DataFrame(columns=["Sample Number", "Field", "Content"])
    has_answer = "Answer Text" in df.columns
    for i, row in df.iterrows():
        offset = 2 + int(has_answer)
        df_transposed.loc[i * offset] = [
            i,
            "Prompt Text",
            row["Prompt Text"],
        ]
        if has_answer:
            df_transposed.loc[i * offset + 1] = [
                i,
                "Answer Text",
                row["Answer Text"],
            ]
        df_transposed.loc[i * offset + 1 + int(has_answer)] = [
            i,
            "Tokenized Text",
            row["Tokenized Text"],
        ]
    return df_transposed
