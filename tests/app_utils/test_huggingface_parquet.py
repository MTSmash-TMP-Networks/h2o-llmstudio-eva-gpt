import pandas as pd

from llm_studio.app_utils.huggingface_parquet import (
    is_parquet_directory,
    parquet_directory_row_count,
    read_dataframe,
    write_parquet_directory_metadata,
)


def test_sharded_parquet_directory_projects_wikipedia_text(tmp_path):
    dataset_dir = tmp_path / "wikipedia_20231101.de_train.parquet"
    shard_dir = dataset_dir / "20231101.de"
    shard_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "id": ["1", "2"],
            "url": ["https://example/1", "https://example/2"],
            "title": ["One", "Two"],
            "text": ["Article one", "Article two"],
        }
    ).to_parquet(shard_dir / "train-00000-of-00002.parquet", index=False)
    pd.DataFrame(
        {
            "id": ["3"],
            "url": ["https://example/3"],
            "title": ["Three"],
            "text": ["Article three"],
        }
    ).to_parquet(shard_dir / "train-00001-of-00002.parquet", index=False)

    write_parquet_directory_metadata(
        str(dataset_dir),
        {
            "source": "huggingface",
            "dataset": "wikimedia/wikipedia",
            "config": "20231101.de",
            "split": "train",
            "columns": ["text"],
            "column_aliases": {"text": "Text"},
        },
    )

    assert is_parquet_directory(str(dataset_dir)) is True
    assert parquet_directory_row_count(str(dataset_dir)) == 3

    metadata_df = read_dataframe(str(dataset_dir), meta_only=True)
    assert list(metadata_df.columns) == ["Text"]
    assert metadata_df.empty

    preview_df = read_dataframe(str(dataset_dir), n_rows=2)
    assert list(preview_df.columns) == ["Text"]
    assert preview_df["Text"].tolist() == ["Article one", "Article two"]
