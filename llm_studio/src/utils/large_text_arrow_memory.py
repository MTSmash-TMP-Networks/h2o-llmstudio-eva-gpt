"""Arrow-backed host-memory path for very large sharded text-only corpora."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Callable

import pandas as pd
import pyarrow.dataset as pa_dataset

logger = logging.getLogger(__name__)

_INSTALLED = False
_ORIGINAL_HANDLER_INIT: Callable[..., None] | None = None
_ORIGINAL_CLEAN_MISSING: Callable[..., pd.Series] | None = None


def _is_arrow_backed_string(values: pd.Series) -> bool:
    dtype = values.dtype
    arrow_dtype = getattr(pd, "ArrowDtype", None)
    if arrow_dtype is not None and isinstance(dtype, arrow_dtype):
        return True
    string_dtype = getattr(pd, "StringDtype", None)
    return bool(
        string_dtype is not None
        and isinstance(dtype, string_dtype)
        and getattr(dtype, "storage", None) == "pyarrow"
    )


def _clean_missing_text_values(values: pd.Series) -> pd.Series:
    """Clean text while preserving Arrow buffers instead of Python string objects."""
    if not _is_arrow_backed_string(values):
        if _ORIGINAL_CLEAN_MISSING is None:
            return values.fillna("").astype(str)
        return _ORIGINAL_CLEAN_MISSING(values)

    values = values.fillna("")
    normalized = values.str.strip().str.lower()
    return values.mask(normalized.isin(["nan", "none", "null", "na"]), "")


class _ArrowTextSequence(Sequence[str]):
    """Expose an Arrow extension array as strings without materializing all rows."""

    def __init__(self, values: Any):
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [str(value) for value in self.values[index]]
        value = self.values[int(index)]
        return "" if pd.isna(value) else str(value)


def _read_rank_partitioned_dataframe_arrow(
    path: str, rank: int, world_size: int
) -> pd.DataFrame:
    """Read one rank's Parquet shards into Arrow-backed pandas columns."""
    from llm_studio.app_utils.huggingface_parquet import (
        _metadata_projection,
        list_parquet_shards,
    )
    from llm_studio.src.utils import sharded_parquet_training as sharded

    all_shards = list_parquet_shards(path)
    local_shards = sharded._rank_partition_shards(path, rank, world_size)
    dataset = pa_dataset.dataset(local_shards, format="parquet")
    columns, aliases = _metadata_projection(path)
    available_columns = list(dataset.schema.names)

    if columns is not None:
        columns = [column for column in columns if column in available_columns]
        if not columns:
            raise ValueError(
                "Configured columns are not present in Parquet dataset directory "
                f"{path}."
            )

    logger.info(
        "Rank %s/%s loading %s of %s Parquet shards with Arrow-backed text storage.",
        rank,
        world_size,
        len(local_shards),
        len(all_shards),
    )
    table = dataset.to_table(columns=columns)
    try:
        df = table.to_pandas(types_mapper=pd.ArrowDtype).reset_index(drop=True)
    except (TypeError, AttributeError):
        # Compatibility fallback for older pandas/pyarrow combinations. The caller
        # still works, only without the memory optimization.
        df = table.to_pandas().reset_index(drop=True)
    if aliases:
        df = df.rename(columns=aliases)

    footprint_mb = float(df.memory_usage(index=True, deep=True).sum()) / (1024 * 1024)
    logger.info(
        "Rank %s/%s loaded %s text rows; pandas/Arrow logical footprint %.1f MB.",
        rank,
        world_size,
        len(df),
        footprint_mb,
    )
    return df


def _clean_text_rows_arrow(
    df: pd.DataFrame, text_column: str, *, rank: int, label: str
) -> pd.DataFrame:
    """Remove empty text without converting Arrow strings to Python objects."""
    df = df.copy()
    text_values = _clean_missing_text_values(df[text_column])
    df[text_column] = text_values
    non_empty = text_values.str.strip().ne("")
    dropped = int((~non_empty).sum())
    if dropped:
        logger.info(
            "Rank %s removed %s empty rows from its %s text partition.",
            rank,
            dropped,
            label,
        )
        df = df.loc[non_empty].copy()
    return df.reset_index(drop=True)


def _handler_init_arrow(self: Any, df: pd.DataFrame, cfg: Any) -> None:
    """Use an Arrow-backed answer sequence for pure continued-pretraining text."""
    from llm_studio.src.datasets import conversation_chain_handler as chains

    pure_text_column = chains.get_pure_text_training_column(df, cfg)
    if pure_text_column is not None and _is_arrow_backed_string(df[pure_text_column]):
        text_values = _clean_missing_text_values(df[pure_text_column])
        non_empty_mask = text_values.str.strip().ne("")
        if bool(non_empty_mask.all()):
            sample_count = len(df)
            self.plain_text_mask = pd.Series(True, index=df.index, dtype=bool)
            self.conversation_chain_ids = chains._SingletonConversationChains(sample_count)
            self.prompts = chains._ConstantTextSequence(
                chains.PLAIN_TEXT_PROMPT, sample_count
            )
            self.answers = _ArrowTextSequence(text_values.array)
            self.systems = chains._ConstantTextSequence("", sample_count)
            chains._patch_plain_text_custom_dataset()
            logger.info(
                "Prepared Arrow-backed optimized text-only conversation handler for "
                "%s rows without materializing Python string copies.",
                sample_count,
            )
            return

    if _ORIGINAL_HANDLER_INIT is None:
        raise RuntimeError("Original ConversationChainHandler.__init__ unavailable.")
    _ORIGINAL_HANDLER_INIT(self, df, cfg)


def install_large_text_arrow_memory() -> None:
    """Install Arrow-backed sharded-text readers before training data is prepared."""
    global _INSTALLED
    global _ORIGINAL_CLEAN_MISSING
    global _ORIGINAL_HANDLER_INIT

    if _INSTALLED:
        return

    from llm_studio.src.datasets import conversation_chain_handler, text_utils
    from llm_studio.src.utils import sharded_parquet_training

    _ORIGINAL_CLEAN_MISSING = text_utils.clean_missing_text_values
    _ORIGINAL_HANDLER_INIT = conversation_chain_handler.ConversationChainHandler.__init__

    # The sharded training module resolves these helpers through module globals at
    # runtime, so replacing them here keeps its existing rank balancing/splitting
    # behavior while changing only the in-memory representation.
    sharded_parquet_training._read_rank_partitioned_dataframe = (
        _read_rank_partitioned_dataframe_arrow
    )
    sharded_parquet_training._clean_text_rows = _clean_text_rows_arrow

    text_utils.clean_missing_text_values = _clean_missing_text_values
    conversation_chain_handler.clean_missing_text_values = _clean_missing_text_values
    conversation_chain_handler.ConversationChainHandler.__init__ = _handler_init_arrow

    _INSTALLED = True
