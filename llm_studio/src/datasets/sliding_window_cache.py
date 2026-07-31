"""Persistent cache helpers for causal-LM long-sample indices."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
_CACHE_VERSION = 1


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
                "train_text_column",
                "personalize",
            )
        },
    }
    digest = hashlib.blake2b(digest_size=20)
    digest.update(json.dumps(settings, sort_keys=True, default=str).encode())
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


def save_index(path: Path | None, index: list[tuple[int, int | None, int]]) -> None:
    """Atomically save a sample index."""
    if path is None:
        return
    values = np.asarray(
        [
            (original_idx, -1 if window_start is None else window_start, prefix_mask)
            for original_idx, window_start, prefix_mask in index
        ],
        dtype=np.int64,
    ).reshape((-1, 3))
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
