"""Keep dataset statistics responsive for large sharded Parquet corpora."""

from __future__ import annotations

import functools
import logging

from llm_studio.app_utils.huggingface_parquet import (
    is_parquet_directory,
    parquet_directory_row_count,
)
from llm_studio.src.datasets.conversation_chain_handler import get_conversation_chains
from llm_studio.src.utils.config_utils import load_config_yaml

logger = logging.getLogger(__name__)

_STATISTICS_SAMPLE_ROWS = 10_000
_INSTALLED = False


def _compute_sharded_statistics(dataset_path: str, cfg_path: str) -> dict:
    """Compute representative statistics without materializing the full corpus."""
    from llm_studio.app_utils.sections import dataset as dataset_section

    total_rows = parquet_directory_row_count(dataset_path)
    sample_rows = min(total_rows, _STATISTICS_SAMPLE_ROWS)
    df_train = dataset_section.read_dataframe(dataset_path, n_rows=sample_rows)
    cfg = load_config_yaml(cfg_path)
    conversations = get_conversation_chains(
        df=df_train, cfg=cfg, limit_chained_samples=True
    )

    stats_dict: dict = {}
    for chat_type in ["prompts", "answers"]:
        text_lengths = [
            [len(text.split(" ")) for text in conversation[chat_type]]
            for conversation in conversations
        ]
        stats_dict[chat_type] = [item for sublist in text_lengths for item in sublist]

    input_texts = []
    for conversation in conversations:
        input_text = conversation["systems"][0]
        prompts = conversation["prompts"]
        answers = conversation["answers"]
        for prompt, answer in zip(prompts, answers, strict=False):
            input_text += prompt + answer
        input_texts.append(input_text)

    stats_dict["complete_conversations"] = [
        len(text.split(" ")) for text in input_texts
    ]
    stats_dict["number_of_prompts"] = [
        len(conversation["prompts"]) for conversation in conversations
    ]
    stats_dict["df_stats"] = dataset_section.get_frame_stats(df_train)
    stats_dict["_sampled_rows"] = len(df_train)
    stats_dict["_total_rows"] = total_rows

    logger.info(
        "Computed sharded dataset statistics from %s/%s rows instead of loading the "
        "complete Parquet corpus.",
        len(df_train),
        total_rows,
    )
    return stats_dict


def install_large_dataset_statistics() -> None:
    """Patch statistics computation without replacing the Wave UI renderer."""
    global _INSTALLED
    if _INSTALLED:
        return

    from llm_studio.app_utils.sections import dataset as dataset_section

    original_compute = dataset_section.compute_dataset_statistics

    @functools.lru_cache(maxsize=128)
    def compute_dataset_statistics(
        dataset_path: str, cfg_path: str, cfg_hash: str
    ) -> dict:
        if is_parquet_directory(dataset_path):
            return _compute_sharded_statistics(dataset_path, cfg_path)
        return original_compute(dataset_path, cfg_path, cfg_hash)

    # Keep H2O Wave's original show_statistics_tab implementation untouched. Page
    # card attributes are Wave Ref proxies after assignment to q.page; mutating
    # ``q.page[...].items`` as if it were a normal Python list raises
    # ``TypeError: 'Ref' object is not callable``. The original renderer already
    # calls the module-level compute_dataset_statistics symbol, so replacing only
    # that function is sufficient to keep large datasets bounded and responsive.
    dataset_section.compute_dataset_statistics = compute_dataset_statistics
    _INSTALLED = True
