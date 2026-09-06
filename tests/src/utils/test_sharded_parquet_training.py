from types import SimpleNamespace

import pandas as pd

from llm_studio.app_utils import large_dataset_statistics as large_stats
from llm_studio.app_utils.huggingface_parquet import (
    limit_parquet_directory_reads,
    list_parquet_shards,
    write_parquet_directory_metadata,
)
from llm_studio.app_utils.sections import dataset as dataset_section
from llm_studio.python_configs import cfg_checks
from llm_studio.src.datasets import conversation_chain_handler as chain_handler
from llm_studio.src.utils import data_utils
from llm_studio.src.utils import sharded_parquet_training as sharded_training
from llm_studio.src.utils.sharded_parquet_training import (
    _is_text_only_sharded_training,
    _rank_partition_shards,
    _read_rank_partitioned_dataframe,
    _read_training_dataframe,
)


def _write_metadata(dataset_dir):
    write_parquet_directory_metadata(
        str(dataset_dir),
        {
            "columns": ["text"],
            "column_aliases": {"text": "Text"},
            "source_text_column": "text",
            "training_mode": "text_only",
        },
    )


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

    _write_metadata(dataset_dir)
    return dataset_dir


def _make_many_shards(tmp_path, shards=8, rows_per_shard=3):
    dataset_dir = tmp_path / "wikipedia-many.parquet"
    dataset_dir.mkdir()

    article = 0
    for shard_index in range(shards):
        rows = list(range(article, article + rows_per_shard))
        pd.DataFrame(
            {
                "id": rows,
                "text": [f"article-{index}" for index in rows],
            }
        ).to_parquet(
            dataset_dir / f"part-{shard_index:03d}.parquet",
            index=False,
        )
        article += rows_per_shard

    _write_metadata(dataset_dir)
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


def test_rank_partitions_are_disjoint_and_cover_all_shards(tmp_path):
    dataset_dir = _make_many_shards(tmp_path, shards=8, rows_per_shard=2)
    all_shards = list_parquet_shards(str(dataset_dir))
    partitions = [
        _rank_partition_shards(str(dataset_dir), rank, 4) for rank in range(4)
    ]

    assert all(len(partition) == 2 for partition in partitions)
    flattened = [shard for partition in partitions for shard in partition]
    assert sorted(flattened) == sorted(all_shards)
    assert len(set(flattened)) == len(all_shards)


def test_rank_partition_reader_loads_only_its_shards_and_keeps_alias(tmp_path):
    dataset_dir = _make_many_shards(tmp_path, shards=4, rows_per_shard=2)

    rank_zero = _read_rank_partitioned_dataframe(str(dataset_dir), 0, 2)
    rank_one = _read_rank_partitioned_dataframe(str(dataset_dir), 1, 2)

    assert list(rank_zero.columns) == ["Text"]
    assert list(rank_one.columns) == ["Text"]
    assert rank_zero["Text"].tolist() == [
        "article-0",
        "article-1",
        "article-4",
        "article-5",
    ]
    assert rank_one["Text"].tolist() == [
        "article-2",
        "article-3",
        "article-6",
        "article-7",
    ]


def test_text_only_sharded_mode_enables_rank_partitioning(tmp_path):
    dataset_dir = _make_many_shards(tmp_path, shards=4, rows_per_shard=2)
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            train_dataframe=str(dataset_dir),
            train_text_column=True,
            prompt_column=("Text",),
            answer_column="Text",
            parent_id_column="None",
        ),
        environment=SimpleNamespace(_distributed=True, _world_size=4),
    )

    assert _is_text_only_sharded_training(cfg) is True


def test_prepartitioned_dataloader_disables_second_distributed_sampler(monkeypatch):
    observed_distributed = []

    def fake_get_train_dataloader(*, train_ds, cfg):
        observed_distributed.append(cfg.environment._distributed)
        return "loader"

    monkeypatch.setattr(
        sharded_training,
        "_ORIGINAL_GET_TRAIN_DATALOADER",
        fake_get_train_dataloader,
    )
    train_ds = SimpleNamespace()
    setattr(train_ds, sharded_training._PREPARTITIONED_MARKER, True)
    cfg = SimpleNamespace(environment=SimpleNamespace(_distributed=True, _local_rank=2))

    result = sharded_training._get_train_dataloader_with_rank_partitioning(
        train_ds, cfg
    )

    assert result == "loader"
    assert observed_distributed == [False]
    assert cfg.environment._distributed is True


def test_prepartitioned_dataset_balances_encoded_sample_count(monkeypatch):
    class DummyDataset:
        def __init__(self):
            self.sample_index = list(range(7))

        def __len__(self):
            return len(self.sample_index)

    monkeypatch.setattr(
        sharded_training,
        "_ORIGINAL_GET_TRAIN_DATASET",
        lambda **kwargs: DummyDataset(),
    )
    monkeypatch.setattr(sharded_training, "_distributed_min", lambda value, cfg: 5)

    train_df = pd.DataFrame({"Text": ["a"]})
    train_df.attrs[sharded_training._PREPARTITIONED_MARKER] = True
    cfg = SimpleNamespace(environment=SimpleNamespace(_local_rank=0))

    dataset = sharded_training._get_train_dataset_with_rank_partitioning(train_df, cfg)

    assert len(dataset) == 5
    assert dataset.sample_index == [0, 1, 2, 3, 4]
    assert getattr(dataset, sharded_training._PREPARTITIONED_MARKER) is True


def test_pure_text_handler_skips_generic_chat_materialization(monkeypatch):
    rows = 2000
    df = pd.DataFrame({"Text": [f"article text {index}" for index in range(rows)]})
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            train_text_column=True,
            prompt_column=("Text",),
            answer_column="Text",
            parent_id_column="None",
            system_column="None",
        )
    )

    def fail_generic_get_texts(*args, **kwargs):
        raise AssertionError("generic chat text materialization must not run")

    monkeypatch.setattr(chain_handler, "get_texts", fail_generic_get_texts)
    monkeypatch.setattr(chain_handler, "_patch_plain_text_custom_dataset", lambda: None)

    handler = chain_handler.ConversationChainHandler(df, cfg)

    assert len(handler) == rows
    assert not isinstance(handler.conversation_chain_ids, list)
    assert handler.conversation_chain_ids[1999] == [1999]
    assert handler[1999] == {
        "prompts": [chain_handler.PLAIN_TEXT_PROMPT],
        "answers": ["article text 1999"],
        "systems": [""],
    }


def test_same_sharded_train_and_validation_path_is_loaded_once(tmp_path, monkeypatch):
    dataset_dir = _make_many_shards(tmp_path, shards=4, rows_per_shard=2)
    calls = []

    def fake_rank_reader(path, rank, world_size):
        calls.append((path, rank, world_size))
        return pd.DataFrame({"Text": [f"article-{index}" for index in range(20)]})

    monkeypatch.setattr(
        sharded_training, "_read_rank_partitioned_dataframe", fake_rank_reader
    )
    monkeypatch.setattr(sharded_training, "_distributed_min", lambda value, cfg: value)
    monkeypatch.setattr(sharded_training, "_ORIGINAL_READ_DATAFRAME", lambda path: None)

    dataset_class = SimpleNamespace(preprocess_dataframe=lambda df, cfg: df)
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            train_dataframe=str(dataset_dir),
            validation_dataframe=str(dataset_dir),
            validation_strategy="custom",
            validation_size=0.2,
            prompt_column=("Text",),
            answer_column="Text",
            data_sample=1.0,
            data_sample_choice=("Train", "Validation"),
            dataset_class=dataset_class,
        ),
        environment=SimpleNamespace(
            _local_rank=0,
            _world_size=2,
            _distributed_inference=True,
        ),
        training=SimpleNamespace(train_validation_data=False),
    )

    train_df, val_df = sharded_training._prepare_rank_partitioned_data(cfg)

    assert len(calls) == 1
    assert len(train_df) == 16
    assert len(val_df) == 4


def test_sharded_statistics_use_bounded_sample(monkeypatch):
    observed = []

    def fake_read_dataframe(path, n_rows=-1, **kwargs):
        observed.append((path, n_rows))
        return pd.DataFrame({"Text": ["eins zwei", "drei vier fünf"]})

    cfg = SimpleNamespace()
    monkeypatch.setattr(dataset_section, "read_dataframe", fake_read_dataframe)
    monkeypatch.setattr(large_stats, "parquet_directory_row_count", lambda path: 2_845_308)
    monkeypatch.setattr(large_stats, "load_config_yaml", lambda path: cfg)
    monkeypatch.setattr(
        large_stats,
        "get_conversation_chains",
        lambda df, cfg, limit_chained_samples: [
            {"systems": [""], "prompts": [""], "answers": [text]}
            for text in df["Text"].tolist()
        ],
    )

    stats = large_stats._compute_sharded_statistics("wikipedia.parquet", "cfg.yaml")

    assert observed == [("wikipedia.parquet", 10_000)]
    assert stats["_sampled_rows"] == 2
    assert stats["_total_rows"] == 2_845_308
