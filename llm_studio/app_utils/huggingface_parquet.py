import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

import pandas as pd
import pyarrow.dataset as pa_dataset
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

_METADATA_FILENAME = ".llm_studio_hf_dataset.json"
_PARQUET_SUFFIXES = (".pq", ".parquet")
_ORIGINAL_SCAN_FILES = None
_ORIGINAL_READ_DATAFRAME = None
_ORIGINAL_IS_VALID_DATA_FRAME = None
_INSTALLED = False


def is_parquet_directory(path: object) -> bool:
    """Return True for a logical dataframe backed by multiple Parquet shards."""
    if path is None:
        return False

    path_str = os.fspath(path)
    if not os.path.isdir(path_str):
        return False
    if not path_str.lower().endswith(_PARQUET_SUFFIXES):
        return False

    return bool(list_parquet_shards(path_str))


def list_parquet_shards(path: str) -> list[str]:
    """List data shards while ignoring Hugging Face/local metadata files."""
    shards: list[str] = []
    if not os.path.isdir(path):
        return shards

    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [name for name in dirnames if name != ".cache"]
        for filename in filenames:
            if filename.startswith("__meta_info__"):
                continue
            if filename.lower().endswith(_PARQUET_SUFFIXES):
                shards.append(os.path.join(dirpath, filename))
    return sorted(shards)


def write_parquet_directory_metadata(path: str, metadata: dict[str, Any]) -> None:
    """Persist non-secret import metadata next to the downloaded shards."""
    os.makedirs(path, exist_ok=True)
    metadata_path = os.path.join(path, _METADATA_FILENAME)
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2, sort_keys=True)


def read_parquet_directory_metadata(path: str) -> dict[str, Any]:
    metadata_path = os.path.join(path, _METADATA_FILENAME)
    try:
        with open(metadata_path, encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def parquet_directory_row_count(path: str) -> int:
    """Count rows from Parquet metadata without materializing the dataset."""
    return sum(
        int(pq.ParquetFile(shard).metadata.num_rows)
        for shard in list_parquet_shards(path)
    )


def _metadata_projection(path: str) -> tuple[list[str] | None, dict[str, str]]:
    metadata = read_parquet_directory_metadata(path)
    columns = metadata.get("columns")
    aliases = metadata.get("column_aliases", {})

    if not isinstance(columns, list) or not all(isinstance(x, str) for x in columns):
        columns = None
    if not isinstance(aliases, dict):
        aliases = {}

    clean_aliases = {
        str(source): str(target)
        for source, target in aliases.items()
        if isinstance(source, str) and isinstance(target, str)
    }
    return columns, clean_aliases


def _logical_columns(path: str, schema_names: list[str]) -> list[str]:
    columns, aliases = _metadata_projection(path)
    selected = columns if columns is not None else schema_names
    return [aliases.get(column, column) for column in selected if column in schema_names]


def _read_parquet_directory(
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
    shards = list_parquet_shards(path)
    if not shards:
        raise ValueError(f"Parquet dataset directory contains no shards: {path}")

    dataset = pa_dataset.dataset(shards, format="parquet")
    columns, aliases = _metadata_projection(path)
    available_columns = list(dataset.schema.names)

    if columns is not None:
        columns = [column for column in columns if column in available_columns]
        if not columns:
            raise ValueError(
                f"Configured columns are not present in Parquet dataset directory {path}."
            )

    if meta_only:
        logical_columns = _logical_columns(path, available_columns)
        return pd.DataFrame(columns=logical_columns)

    if n_rows > -1:
        table = dataset.head(n_rows, columns=columns)
    else:
        table = dataset.to_table(columns=columns)

    df = table.to_pandas().reset_index(drop=True)
    if aliases:
        df = df.rename(columns=aliases)

    non_missing_columns = [] if non_missing_columns is None else non_missing_columns
    fill_columns = [] if fill_columns is None else fill_columns
    fill_columns = [column for column in fill_columns if column in df.columns]

    if fill_columns:
        df[fill_columns] = df[fill_columns].fillna(fill_value)

    non_missing_columns = [
        column for column in non_missing_columns if column in df.columns
    ]
    if non_missing_columns:
        original_size = df.shape[0]
        non_missing_index = df[non_missing_columns].dropna().index
        dropped_index = [idx for idx in df.index if idx not in non_missing_index]
        df = df.loc[non_missing_index].reset_index(drop=True)
        new_size = df.shape[0]

        if new_size < original_size and verbose:
            logger.warning(
                "Dropped %s rows when reading dataframe '%s' due to missing values "
                "in columns %s.",
                original_size - new_size,
                path,
                non_missing_columns,
            )

            if handling == "error":
                dropped_preview: list[Any] = dropped_index
                if len(dropped_preview) > 10:
                    dropped_preview = dropped_preview[:5] + ["..."] + dropped_preview[-5:]
                dropped_str = ", ".join(str(idx) for idx in dropped_preview)
                prefix = f"{mode} " if mode else ""
                raise ValueError(
                    f"{prefix}dataset contains {len(dropped_index)} rows with missing "
                    f"values in one of the following columns: {non_missing_columns} "
                    f"in the following rows: {dropped_str}"
                )

    return df


def read_dataframe(
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
    """Read regular dataframes unchanged and add support for Parquet directories."""
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

    if _ORIGINAL_READ_DATAFRAME is None:
        raise RuntimeError("Parquet directory support is not installed.")

    return _ORIGINAL_READ_DATAFRAME(
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


def read_dataframe_for_ui(
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
    """Use schema-only reads when the import UI only needs available columns."""
    if is_parquet_directory(path) and n_rows < 0 and not meta_only:
        meta_only = True

    return read_dataframe(
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


def is_valid_data_frame(path: str, csv_rows: int = 100) -> bool:
    if is_parquet_directory(path):
        try:
            pa_dataset.dataset(list_parquet_shards(path), format="parquet").schema
            return True
        except Exception as error:
            logger.error("Invalid Parquet dataset directory %s: %s", path, error)
            return False

    if _ORIGINAL_IS_VALID_DATA_FRAME is None:
        raise RuntimeError("Parquet directory support is not installed.")
    return _ORIGINAL_IS_VALID_DATA_FRAME(path, csv_rows=csv_rows)


def _path_is_inside(path: str, directory: str) -> bool:
    try:
        return os.path.commonpath(
            [os.path.abspath(path), os.path.abspath(directory)]
        ) == os.path.abspath(directory)
    except ValueError:
        return False


def _scan_files_with_parquet_directories(
    dirname: str,
    extensions: tuple[str, ...] = (
        ".csv",
        ".CSV",
        ".pq",
        ".PQ",
        ".parquet",
        ".PARQUET",
    ),
) -> list[str]:
    if _ORIGINAL_SCAN_FILES is None:
        raise RuntimeError("Parquet directory support is not installed.")

    regular_files = _ORIGINAL_SCAN_FILES(dirname, extensions)

    parquet_directories: list[str] = []
    for dirpath, dirnames, _ in os.walk(dirname):
        for directory_name in list(dirnames):
            path = os.path.join(dirpath, directory_name)
            if is_parquet_directory(path):
                parquet_directories.append(path)
                dirnames.remove(directory_name)

    if not parquet_directories:
        return regular_files

    regular_files = [
        path
        for path in regular_files
        if not any(_path_is_inside(path, directory) for directory in parquet_directories)
    ]
    return sorted(regular_files + parquet_directories)


@contextmanager
def limit_parquet_directory_reads(max_rows: int = 2000) -> Iterator[None]:
    """Limit only sharded dataframe reads while the import wizard builds previews."""
    from llm_studio.src.utils import data_utils

    previous_reader = data_utils.read_dataframe

    def limited_reader(
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
        if is_parquet_directory(path) and n_rows < 0 and not meta_only:
            n_rows = max_rows
        return previous_reader(
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

    data_utils.read_dataframe = limited_reader
    try:
        yield
    finally:
        data_utils.read_dataframe = previous_reader


def install_parquet_directory_support() -> None:
    """Patch LLM Studio's existing dataframe helpers for sharded Parquet datasets."""
    global _INSTALLED
    global _ORIGINAL_SCAN_FILES
    global _ORIGINAL_READ_DATAFRAME
    global _ORIGINAL_IS_VALID_DATA_FRAME

    if _INSTALLED:
        return

    from llm_studio.app_utils import utils as app_utils
    from llm_studio.app_utils.sections import dataset as dataset_section
    from llm_studio.src import possible_values
    from llm_studio.src.utils import data_utils

    _ORIGINAL_SCAN_FILES = possible_values._scan_files
    _ORIGINAL_READ_DATAFRAME = data_utils.read_dataframe
    _ORIGINAL_IS_VALID_DATA_FRAME = data_utils.is_valid_data_frame

    possible_values._scan_files = _scan_files_with_parquet_directories
    data_utils.read_dataframe = read_dataframe
    data_utils.is_valid_data_frame = is_valid_data_frame

    # These modules imported the helpers by name, so patch their references too.
    app_utils.read_dataframe = read_dataframe_for_ui
    app_utils.is_valid_data_frame = is_valid_data_frame
    dataset_section.read_dataframe = read_dataframe

    _INSTALLED = True
    logger.info("Sharded Parquet dataset directory support enabled")
