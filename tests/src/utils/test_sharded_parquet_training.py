from types import SimpleNamespace

import pandas as pd

from llm_studio.app_utils.huggingface_parquet import (
    limit_parquet_directory_reads,
    write_parquet_directory_metadata,
)
from llm_studio.python_configs import cfg_checks
from llm_studio.src.utils import data_utils
from llm_studio.src.utils.sharded_parquet_training import _read_training_dataframe


def _make_sharded_dataset(tmp_path, rows=2505):
    dataset_dir = tmp_path / "wikipedia.parquet"
    dataset_dir.mkdir()

    first = rows // 2
    pd.DataFrame(
        {
            "id": range(first),
            "text": [f"article-{index}" for index in range(first)],
        }
    ).to_parquet(dataset_dir / "part-000.parquet", index=False)
    pd.DataFrame(
        {
            "id": range(first, rows),
            "text": [f"article-{index}" for index in range(first, rows)],
        }
    ).to_parquet(dataset_dir / "part-001.parquet", index=False)

    write_parquet_directory_metadata(
        str(dataset_dir),
        {
            "columns": ["text"],
            "column_aliases": {"text": "Text"},
            "source_text_column": "text",
            "training_mode": "text_only",
        },
    )
    return dataset_dir


def test_training_reader_preserves_logical_text_alias(tmp_path):
    dataset_dir = _make_sharded_dataset(tmp_path, rows=5)
    delegated = False

    def regular_reader(**kwargs):
        nonlocal delegated
        delegated = True
        return pd.DataFrame()

    df = _read_training_dataframe(
        str(dataset_dir),
        original_reader=regular_reader,
        n_rows=3,
    )

    assert delegated is False
    assert list(df.columns) == ["Text"]
    assert df["Text"].tolist() == ["article-0", "article-1", "article-2"]


def test_run_config_sanity_check_caps_sharded_parquet_reads(tmp_path, monkeypatch):
    dataset_dir = _make_sharded_dataset(tmp_path)
    original_reader = data_utils.read_dataframe

    def sharded_reader(
        path,
        n_rows=-1,
        meta_only=False,
        non_missing_columns=None,
        verbose=False,
        handling="warn",
        fill_columns=None,
        fill_value="",
        mode="",
    ):
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

    monkeypatch.setattr(data_utils, "read_dataframe", sharded_reader)
    monkeypatch.setattr(
        cfg_checks,
        "check_for_common_errors",
        lambda cfg: {"title": [], "message": [], "type": []},
    )
    monkeypatch.setattr(
        cfg_checks,
        "check_for_logging_errors",
        lambda cfg: {"title": [], "message": [], "type": []},
    )

    observed_rows = []

    def check():
        observed_rows.append(len(data_utils.read_dataframe(str(dataset_dir))))
        return {"title": [], "message": [], "type": []}

    cfg = SimpleNamespace(check=check)
    errors = cfg_checks.check_config_for_errors(cfg)

    assert errors == {"title": [], "message": [], "type": []}
    assert observed_rows == [2000]


def test_limit_context_does_not_change_regular_dataframe_reads(tmp_path):
    csv_path = tmp_path / "small.csv"
    pd.DataFrame({"Text": ["a", "b", "c"]}).to_csv(csv_path, index=False)

    with limit_parquet_directory_reads(max_rows=1):
        df = data_utils.read_dataframe(str(csv_path))

    assert len(df) == 3
