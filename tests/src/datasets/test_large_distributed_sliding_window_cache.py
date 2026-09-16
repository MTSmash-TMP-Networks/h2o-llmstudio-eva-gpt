from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from llm_studio.src.datasets import sliding_window_cache as cache


def _dataset(tmp_path: Path, *, rank: int = 0):
    train_path = tmp_path / "train.csv"
    train_path.write_text("id,prompt,answer\n1,p,a\n", encoding="utf-8")
    cfg = SimpleNamespace(
        llm_backbone="cache-test",
        environment=SimpleNamespace(
            _distributed=True,
            _local_rank=rank,
            _world_size=4,
        ),
        training=SimpleNamespace(train_validation_data=False),
        tokenizer=SimpleNamespace(
            max_length=4096,
            sliding_window_overlap=512,
            tokenizer_kwargs="{}",
            _tokenizer_eos_token="<eos>",
        ),
        dataset=SimpleNamespace(
            train_dataframe=str(train_path),
            validation_dataframe="",
            validation_strategy="automatic",
            validation_size=0.1,
            data_sample=1.0,
            data_sample_choice=("Train", "Validation"),
            system_column="system",
            prompt_column=("prompt",),
            answer_column="answer",
            parent_id_column="parent_id",
            id_column="id",
            text_system_start="",
            text_prompt_start="",
            text_answer_separator="",
            add_eos_token_to_system=False,
            add_eos_token_to_prompt=False,
            add_eos_token_to_answer=True,
            limit_chained_samples=True,
            mask_prompt_labels=True,
            mask_prompt_user_text_only=False,
            only_last_answer=False,
            train_text_column=False,
            personalize=False,
        ),
    )
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "prompt": ["one", "two", "three"],
            "answer": ["alpha", "beta", "gamma"],
            "parent_id": ["", "", ""],
            "system": ["eva", "eva", "eva"],
        }
    )
    tokenizer = SimpleNamespace(
        name_or_path="test-tokenizer",
        vocab_size=1234,
    )
    return SimpleNamespace(df=df, cfg=cfg, tokenizer=tokenizer, mode="train")


def test_large_distributed_cache_key_skips_full_dataframe_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_LARGE_DISTRIBUTED_ROWS", 2)
    dataset = _dataset(tmp_path, rank=0)

    def fail_full_hash(*args, **kwargs):
        raise AssertionError("large distributed chat cache must not hash full DataFrame")

    monkeypatch.setattr(cache.pd.util, "hash_pandas_object", fail_full_hash)

    key = cache._cache_key(dataset, "Sliding Window:structure-aware-v1")

    assert key


def test_large_distributed_cache_key_is_shared_across_ranks(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_LARGE_DISTRIBUTED_ROWS", 2)
    rank_zero = _dataset(tmp_path, rank=0)
    rank_three = _dataset(tmp_path, rank=3)

    key_zero = cache._cache_key(rank_zero, "Sliding Window:structure-aware-v1")
    key_three = cache._cache_key(rank_three, "Sliding Window:structure-aware-v1")

    assert key_zero == key_three


def test_nonzero_rank_waits_for_shared_cache_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_LARGE_DISTRIBUTED_ROWS", 2)
    monkeypatch.setattr(cache, "_SHARED_CACHE_POLL_SECONDS", 0.0)
    monkeypatch.setenv("H2O_LLM_STUDIO_SHARED_INDEX_WAIT_SECONDS", "0")
    dataset = _dataset(tmp_path, rank=2)
    cache_path = tmp_path / "index.npy"

    cache._wait_for_rank_zero_cache(dataset, cache_path)

    # With a zero-second timeout the helper must return without creating anything;
    # rank-local fallback remains available to avoid a permanent distributed hang.
    assert not cache_path.exists()


def test_small_dataset_keeps_content_hashing(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_LARGE_DISTRIBUTED_ROWS", 100)
    dataset = _dataset(tmp_path, rank=0)
    called = {"value": False}
    original = cache.pd.util.hash_pandas_object

    def tracked_hash(*args, **kwargs):
        called["value"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(cache.pd.util, "hash_pandas_object", tracked_hash)

    key = cache._cache_key(dataset, "Sliding Window:structure-aware-v1")

    assert key
    assert called["value"] is True
