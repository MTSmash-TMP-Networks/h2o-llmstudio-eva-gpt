"""Persistent cache helpers for causal-LM long-sample indices."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
_CACHE_VERSION = 3
_PREPARTITIONED_MARKER = "_llm_studio_rank_partitioned_parquet"


def get_cache_path(dataset: Any, strategy: str) -> Path | None:
    """Return a content-addressed cache path, or ``None`` when unavailable."""
    if os.getenv("H2O_LLM_STUDIO_DISABLE_SAMPLE_INDEX_CACHE", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return None

    configured_root = os.getenv("H2O_LLM_STUDIO_CACHE_DIR", "").strip()
    if configured_root:
        root = Path(configured_root).expanduser()
    else:
        if str(getattr(dataset.cfg, "llm_backbone", "")) == "unit-test":
            return None
        dataframe_path = Path(
            str(getattr(dataset.cfg.dataset, "train_dataframe", ""))
        ).expanduser()
        root = (
            dataframe_path.parent / ".h2o_llmstudio_cache"
            if dataframe_path.is_file()
            else Path.home() / ".cache" / "h2o_llmstudio"
        )

    root = root / "sample_indices"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exception:
        logger.warning("Sample-index cache is unavailable at %s: %s", root, exception)
        return None
    return root / f"long-sample-index-{_cache_key(dataset, strategy)}.npy"


def _file_signature(path: str | os.PathLike[str]) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        stat = resolved.stat()
    except OSError:
        return {"path": str(resolved), "size": None, "mtime_ns": None}
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _rank_source_signatures(path: Any, rank: int, world_size: int) -> list[dict[str, Any]]:
    """Describe one rank's source files without reading their text payload."""
    if path in (None, "", "None"):
        return []

    source = os.fspath(path)
    try:
        from llm_studio.app_utils.huggingface_parquet import (
            is_parquet_directory,
            list_parquet_shards,
        )

        if is_parquet_directory(source):
            shards = list_parquet_shards(source)
            if world_size > 1 and len(shards) >= world_size:
                shards = shards[rank::world_size]
            return [_file_signature(shard) for shard in shards]
    except (ImportError, OSError, ValueError):
        pass

    return [_file_signature(source)]


def _prepartitioned_source_fingerprint(dataset: Any) -> str | None:
    """Build a stable cache identity for rank-partitioned Parquet training.

    Hashing a DataFrame containing hundreds of thousands of long, mostly unique text
    values can temporarily consume several gigabytes per distributed rank. The
    sharded training path is deterministic, so source-file metadata plus the split
    settings identifies the rank-local row selection without touching every article.
    """
    if not bool(dataset.df.attrs.get(_PREPARTITIONED_MARKER, False)):
        return None

    cfg = dataset.cfg
    environment = getattr(cfg, "environment", None)
    rank = int(getattr(environment, "_local_rank", 0) or 0)
    world_size = max(int(getattr(environment, "_world_size", 1) or 1), 1)
    train_path = getattr(cfg.dataset, "train_dataframe", "")
    validation_path = getattr(cfg.dataset, "validation_dataframe", "")

    payload = {
        "rank": rank,
        "world_size": world_size,
        "train_sources": _rank_source_signatures(train_path, rank, world_size),
        "validation_sources": _rank_source_signatures(
            validation_path, rank, world_size
        ),
        "row_count": int(len(dataset.df)),
        "validation_strategy": getattr(
            cfg.dataset, "validation_strategy", "automatic"
        ),
        "validation_size": float(getattr(cfg.dataset, "validation_size", 0.0) or 0.0),
        "data_sample": float(getattr(cfg.dataset, "data_sample", 1.0) or 1.0),
        "data_sample_choice": list(
            getattr(cfg.dataset, "data_sample_choice", ("Train", "Validation"))
        ),
        "train_validation_data": bool(
            getattr(getattr(cfg, "training", None), "train_validation_data", False)
        ),
        "selection_version": 2,
    }
    return hashlib.blake2b(
        json.dumps(payload, sort_keys=True, default=str).encode(), digest_size=20
    ).hexdigest()


def _cache_key(dataset: Any, strategy: str) -> str:
    cfg = dataset.cfg
    settings = {
        "version": _CACHE_VERSION,
        "dataset_class": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
        "columns": list(dataset.df.columns),
        "dtypes": [str(dtype) for dtype in dataset.df.dtypes],
        "backbone": str(getattr(cfg, "llm_backbone", "")),
        "tokenizer_class": (
            f"{type(dataset.tokenizer).__module__}."
            f"{type(dataset.tokenizer).__qualname__}"
        ),
        "tokenizer_name": str(getattr(dataset.tokenizer, "name_or_path", "")),
        "tokenizer_vocab_size": getattr(dataset.tokenizer, "vocab_size", None),
        "tokenizer_kwargs": str(getattr(cfg.tokenizer, "tokenizer_kwargs", "")),
        "max_length": int(cfg.tokenizer.max_length),
        "strategy": strategy,
        "overlap": int(getattr(cfg.tokenizer, "sliding_window_overlap", 0)),
        "eos_token": str(getattr(cfg.tokenizer, "_tokenizer_eos_token", "")),
        "dataset_settings": {
            name: getattr(cfg.dataset, name, None)
            for name in (
                "system_column",
                "prompt_column",
                "answer_column",
                "parent_id_column",
                "id_column",
                "text_system_start",
                "text_prompt_start",
                "text_answer_separator",
                "add_eos_token_to_system",
                "add_eos_token_to_prompt",
                "add_eos_token_to_answer",
                "limit_chained_samples",
                "mask_prompt_labels",
                "mask_prompt_user_text_only",
                "only_last_answer",
                "train_text_column",
                "personalize",
            )
        },
    }
    digest = hashlib.blake2b(digest_size=20)
    digest.update(json.dumps(settings, sort_keys=True, default=str).encode())

    source_fingerprint = _prepartitioned_source_fingerprint(dataset)
    if source_fingerprint is not None:
        logger.info(
            "Using lightweight sharded source fingerprint for the long-sample "
            "cache (%s rows); full DataFrame text hashing is skipped.",
            len(dataset.df),
        )
        digest.update(b"rank-partitioned:")
        digest.update(source_fingerprint.encode())
        return digest.hexdigest()

    try:
        row_hash = pd.util.hash_pandas_object(
            dataset.df, index=True, categorize=True
        ).to_numpy(dtype=np.uint64, copy=False)
    except (TypeError, ValueError):
        row_hash = pd.util.hash_pandas_object(
            dataset.df.astype(str), index=True, categorize=True
        ).to_numpy(dtype=np.uint64, copy=False)
    digest.update(row_hash.tobytes())
    return digest.hexdigest()


def load_index(
    path: Path | None, sample_count: int
) -> list[tuple[int, int | None, int]] | None:
    """Load and validate a cached sample index."""
    if path is None or not path.is_file():
        return None
    try:
        values = np.load(path, allow_pickle=False)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("unexpected sample-index shape")
        if len(values) and (
            values[:, 0].min() < 0
            or values[:, 0].max() >= sample_count
            or values[:, 1].min() < -1
            or values[:, 2].min() < 0
        ):
            raise ValueError("sample-index values are outside valid bounds")
        return [
            (
                int(original_idx),
                None if int(window_start) < 0 else int(window_start),
                int(prefix_mask),
            )
            for original_idx, window_start, prefix_mask in values
        ]
    except (OSError, ValueError, EOFError) as exception:
        logger.warning("Ignoring invalid sample-index cache %s: %s", path, exception)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def save_index(
    path: Path | None, index: Sequence[tuple[int, int | None, int]]
) -> None:
    """Atomically save a sample index without duplicating it as a Python list."""
    if path is None:
        return

    values = np.empty((len(index), 3), dtype=np.int64)
    for row, (original_idx, window_start, prefix_mask) in enumerate(index):
        values[row, 0] = int(original_idx)
        values[row, 1] = -1 if window_start is None else int(window_start)
        values[row, 2] = int(prefix_mask)

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as cache_file:
            np.save(cache_file, values, allow_pickle=False)
        os.replace(temporary, path)
    except OSError as exception:
        logger.warning("Could not write sample-index cache %s: %s", path, exception)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
