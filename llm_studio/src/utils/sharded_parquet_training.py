"""Training-process support for sharded Hugging Face Parquet directories."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

_INSTALLED = False
_ORIGINAL_READ_DATAFRAME: Callable[..., pd.DataFrame] | None = None


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


def install_sharded_parquet_training_support() -> None:
    """Patch only the core dataframe reader used by the training subprocess."""
    global _INSTALLED
    global _ORIGINAL_READ_DATAFRAME

    if _INSTALLED:
        return

    from llm_studio.src.utils import data_utils

    _ORIGINAL_READ_DATAFRAME = data_utils.read_dataframe
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
    _INSTALLED = True
