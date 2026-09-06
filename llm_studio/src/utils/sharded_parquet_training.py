"""Training-process support for sharded Hugging Face Parquet directories."""

from __future__ import annotations

import logging
from typing import Any, Callable

import pandas as pd
import pyarrow.dataset as pa_dataset
import torch
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

_PREPARTITIONED_MARKER = "_llm_studio_rank_partitioned_parquet"
_INSTALLED = False
_ORIGINAL_READ_DATAFRAME: Callable[..., pd.DataFrame] | None = None
_ORIGINAL_GET_DATA: Callable[..., tuple[pd.DataFrame, pd.DataFrame]] | None = None
_ORIGINAL_GET_TRAIN_DATASET: Callable[..., Any] | None = None
_ORIGINAL_GET_TRAIN_DATALOADER: Callable[..., Any] | None = None


def _read_training_dataframe(
    path: str,
    *,
    original_reader: Callable[..., pd.DataFrame],
    n_rows: int = -1,
    meta_only: bool = False,
    non_missing_columns: list[str] | None = None,
    verbose: bool = False,
    handling: str = "warn",
    fill_columns: list[str] | None = None,
    fill_value: Any = "",
    mode: str = "",
) -> pd.DataFrame:
    """Read sharded Parquet directories with their logical column aliases."""
    from llm_studio.app_utils.huggingface_parquet import (
        _read_parquet_directory,
        is_parquet_directory,
    )

    if is_parquet_directory(path):
        return _read_parquet_directory(
            path=path,
            n_rows=n_rows,
            meta_only=meta_only,
            non_missing_columns=non_missing_columns,
            verbose=verbose,
            handling=handling,
            fill_columns=fill_columns,
            fill_value=fill_value,
            mode=mode,
        )

    return original_reader(
        path=path,
        n_rows=n_rows,
        meta_only=meta_only,
        non_missing_columns=non_missing_columns,
        verbose=verbose,
        handling=handling,
        fill_columns=fill_columns,
        fill_value=fill_value,
        mode=mode,
    )


def _single_column(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return str(value[0])
    return None


def _is_text_only_sharded_training(cfg: Any) -> bool:
    """Return True for distributed continued-pretraining Parquet imports."""
    from llm_studio.app_utils.huggingface_parquet import (
        is_parquet_directory,
        read_parquet_directory_metadata,
    )

    path = getattr(cfg.dataset, "train_dataframe", "")
    if not is_parquet_directory(path):
        return False

    environment = getattr(cfg, "environment", None)
    if environment is None:
        return False
    if not bool(getattr(environment, "_distributed", False)):
        return False
    if int(getattr(environment, "_world_size", 1) or 1) <= 1:
        return False

    metadata = read_parquet_directory_metadata(path)
    if metadata.get("training_mode") == "text_only":
        return True

    if not bool(getattr(cfg.dataset, "train_text_column", False)):
        return False

    prompt = _single_column(getattr(cfg.dataset, "prompt_column", None))
    answer = _single_column(getattr(cfg.dataset, "answer_column", None))
    parent = getattr(cfg.dataset, "parent_id_column", "None")
    return prompt is not None and prompt == answer and parent in ("None", None)


def _rank_partition_shards(path: str, rank: int, world_size: int) -> list[str]:
    from llm_studio.app_utils.huggingface_parquet import list_parquet_shards

    shards = list_parquet_shards(path)
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"Invalid distributed rank {rank} for world size {world_size}."
        )
    if len(shards) < world_size:
        raise ValueError(
            "Sharded text-only training needs at least one Parquet shard per GPU. "
            f"Found {len(shards)} shards for {world_size} distributed ranks."
        )
    return shards[rank::world_size]


def _read_rank_partitioned_dataframe(
    path: str, rank: int, world_size: int
) -> pd.DataFrame:
    """Materialize only the Parquet shards assigned to one distributed rank."""
    from llm_studio.app_utils.huggingface_parquet import (
        _metadata_projection,
        list_parquet_shards,
    )

    all_shards = list_parquet_shards(path)
    local_shards = _rank_partition_shards(path, rank, world_size)
    dataset = pa_dataset.dataset(local_shards, format="parquet")
    columns, aliases = _metadata_projection(path)
    available_columns = list(dataset.schema.names)

    if columns is not None:
        columns = [column for column in columns if column in available_columns]
        if not columns:
            raise ValueError(
                f"Configured columns are not present in Parquet dataset directory {path}."
            )

    logger.info(
        "Rank %s/%s loading %s of %s Parquet shards for text-only training.",
        rank,
        world_size,
        len(local_shards),
        len(all_shards),
    )
    table = dataset.to_table(columns=columns)
    df = table.to_pandas().reset_index(drop=True)
    if aliases:
        df = df.rename(columns=aliases)

    logger.info(
        "Rank %s/%s loaded %s text rows from its local Parquet shard partition.",
        rank,
        world_size,
        len(df),
    )
    return df


def _distributed_min(value: int, cfg: Any) -> int:
    """Return the minimum integer value across all initialized ranks."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return int(value)

    backend = str(torch.distributed.get_backend()).lower()
    if "nccl" in backend:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL process group is active but CUDA is unavailable.")
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")

    tensor = torch.tensor([int(value)], dtype=torch.long, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
    return int(tensor.item())


def _sample_local_dataframe(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if fraction >= 1.0:
        return df
    n_rows = max(10, int(len(df) * fraction))
    return df.sample(n=min(n_rows, len(df)), random_state=7331, replace=False)


def _balance_dataframe_rows(df: pd.DataFrame, cfg: Any, label: str) -> pd.DataFrame:
    """Keep every rank on the same row count so distributed epochs stay aligned."""
    target_rows = _distributed_min(len(df), cfg)
    if target_rows <= 0:
        raise ValueError(f"Distributed {label} partition contains no usable rows.")
    if len(df) != target_rows:
        logger.info(
            "Rank %s trimming %s rows from %s to %s to match the shortest rank.",
            getattr(cfg.environment, "_local_rank", 0),
            label,
            len(df),
            target_rows,
        )
        df = df.iloc[:target_rows].copy()
    return df.reset_index(drop=True)


def _prepare_rank_partitioned_data(cfg: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare one rank's unique shard partition without loading the full corpus."""
    if _ORIGINAL_READ_DATAFRAME is None:
        raise RuntimeError("Sharded Parquet training support is not installed.")

    rank = int(getattr(cfg.environment, "_local_rank", 0) or 0)
    world_size = int(getattr(cfg.environment, "_world_size", 1) or 1)
    train_path = cfg.dataset.train_dataframe

    local_df = _read_rank_partitioned_dataframe(train_path, rank, world_size)
    text_column = _single_column(getattr(cfg.dataset, "prompt_column", None))
    if text_column is None:
        text_column = _single_column(getattr(cfg.dataset, "answer_column", None))
    if text_column is None or text_column not in local_df.columns:
        raise ValueError(
            "Text-only sharded training could not resolve the configured text column. "
            f"Configured={text_column!r}, available={list(local_df.columns)!r}."
        )
    local_df[text_column] = local_df[text_column].fillna("").astype(str)

    validation_strategy = getattr(cfg.dataset, "validation_strategy", "automatic")
    if validation_strategy == "automatic":
        logger.info(
            "Rank %s creating its automatic validation split from local shards.", rank
        )
        train_df, val_df = train_test_split(
            local_df,
            test_size=cfg.dataset.validation_size,
            random_state=1337,
        )
    elif validation_strategy == "custom":
        validation_path = getattr(cfg.dataset, "validation_dataframe", "")
        if validation_path in ("", "None", None):
            raise ValueError(
                "No validation dataframe provided for custom validation strategy."
            )

        from llm_studio.app_utils.huggingface_parquet import is_parquet_directory

        train_df = local_df
        if is_parquet_directory(validation_path):
            val_df = _read_rank_partitioned_dataframe(validation_path, rank, world_size)
        else:
            val_df = _ORIGINAL_READ_DATAFRAME(validation_path)
        if text_column in val_df.columns:
            val_df[text_column] = val_df[text_column].fillna("").astype(str)
    else:
        raise ValueError(
            f"Unsupported validation strategy for sharded text training: "
            f"{validation_strategy!r}."
        )

    data_sample = float(getattr(cfg.dataset, "data_sample", 1.0))
    data_sample_choice = getattr(
        cfg.dataset, "data_sample_choice", ("Train", "Validation")
    )
    if data_sample < 1.0:
        if "Train" in data_sample_choice:
            train_df = _sample_local_dataframe(train_df, data_sample)
        if "Validation" in data_sample_choice:
            val_df = _sample_local_dataframe(val_df, data_sample)

    if bool(getattr(cfg.training, "train_validation_data", False)):
        train_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

    train_df = cfg.dataset.dataset_class.preprocess_dataframe(train_df, cfg)
    val_df = cfg.dataset.dataset_class.preprocess_dataframe(val_df, cfg)

    train_df = _balance_dataframe_rows(train_df, cfg, "training")
    val_df = _balance_dataframe_rows(val_df, cfg, "validation")
    train_df.attrs[_PREPARTITIONED_MARKER] = True
    val_df.attrs[_PREPARTITIONED_MARKER] = True

    if hasattr(cfg.environment, "_distributed_inference"):
        cfg.environment._distributed_inference = False
        logger.info(
            "Distributed validation gather disabled for rank-partitioned text data; "
            "rank 0 evaluates its balanced local validation partition."
        )

    return train_df, val_df


def _get_data_with_rank_partitioning(cfg: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    if _is_text_only_sharded_training(cfg):
        return _prepare_rank_partitioned_data(cfg)
    if _ORIGINAL_GET_DATA is None:
        raise RuntimeError("Original get_data function is unavailable.")
    return _ORIGINAL_GET_DATA(cfg)


def _get_train_dataset_with_rank_partitioning(train_df: pd.DataFrame, cfg: Any):
    if _ORIGINAL_GET_TRAIN_DATASET is None:
        raise RuntimeError("Original get_train_dataset function is unavailable.")

    dataset = _ORIGINAL_GET_TRAIN_DATASET(train_df=train_df, cfg=cfg)
    if not bool(train_df.attrs.get(_PREPARTITIONED_MARKER, False)):
        return dataset

    setattr(dataset, _PREPARTITIONED_MARKER, True)
    target_samples = _distributed_min(len(dataset), cfg)
    if target_samples <= 0:
        raise ValueError("Distributed training partition contains no usable samples.")

    if len(dataset) != target_samples:
        sample_index = getattr(dataset, "sample_index", None)
        if not isinstance(sample_index, list):
            raise ValueError(
                "Cannot balance rank-partitioned dataset because it has no mutable "
                "sample_index."
            )
        logger.info(
            "Rank %s trimming encoded training samples from %s to %s so all ranks "
            "run the same number of DeepSpeed steps.",
            getattr(cfg.environment, "_local_rank", 0),
            len(dataset),
            target_samples,
        )
        dataset.sample_index = sample_index[:target_samples]

    return dataset


def _get_train_dataloader_with_rank_partitioning(train_ds: Any, cfg: Any):
    if _ORIGINAL_GET_TRAIN_DATALOADER is None:
        raise RuntimeError("Original get_train_dataloader function is unavailable.")
    if not bool(getattr(train_ds, _PREPARTITIONED_MARKER, False)):
        return _ORIGINAL_GET_TRAIN_DATALOADER(train_ds=train_ds, cfg=cfg)

    # Every distributed rank already owns a disjoint shard partition. Applying the
    # normal DistributedSampler here would divide each local partition a second time
    # and train on only 1/world_size of it again.
    previous_distributed = cfg.environment._distributed
    cfg.environment._distributed = False
    try:
        dataloader = _ORIGINAL_GET_TRAIN_DATALOADER(train_ds=train_ds, cfg=cfg)
    finally:
        cfg.environment._distributed = previous_distributed

    logger.info(
        "Rank %s uses its complete local shard partition; the extra "
        "DistributedSampler layer is disabled.",
        getattr(cfg.environment, "_local_rank", 0),
    )
    return dataloader


def install_sharded_parquet_training_support() -> None:
    """Patch core training helpers for large sharded text-only datasets."""
    global _INSTALLED
    global _ORIGINAL_GET_DATA
    global _ORIGINAL_GET_TRAIN_DATALOADER
    global _ORIGINAL_GET_TRAIN_DATASET
    global _ORIGINAL_READ_DATAFRAME

    if _INSTALLED:
        return

    from llm_studio.src.utils import data_utils

    _ORIGINAL_READ_DATAFRAME = data_utils.read_dataframe
    _ORIGINAL_GET_DATA = data_utils.get_data
    _ORIGINAL_GET_TRAIN_DATASET = data_utils.get_train_dataset
    _ORIGINAL_GET_TRAIN_DATALOADER = data_utils.get_train_dataloader
    original_reader = _ORIGINAL_READ_DATAFRAME

    def training_reader(
        path: str,
        n_rows: int = -1,
        meta_only: bool = False,
        non_missing_columns: list[str] | None = None,
        verbose: bool = False,
        handling: str = "warn",
        fill_columns: list[str] | None = None,
        fill_value: Any = "",
        mode: str = "",
    ) -> pd.DataFrame:
        return _read_training_dataframe(
            path,
            original_reader=original_reader,
            n_rows=n_rows,
            meta_only=meta_only,
            non_missing_columns=non_missing_columns,
            verbose=verbose,
            handling=handling,
            fill_columns=fill_columns,
            fill_value=fill_value,
            mode=mode,
        )

    data_utils.read_dataframe = training_reader
    data_utils.get_data = _get_data_with_rank_partitioning
    data_utils.get_train_dataset = _get_train_dataset_with_rank_partitioning
    data_utils.get_train_dataloader = _get_train_dataloader_with_rank_partitioning
    _INSTALLED = True
    logger.info("Distributed sharded Parquet text-training support enabled")
