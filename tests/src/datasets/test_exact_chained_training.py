from types import SimpleNamespace

from llm_studio.src.datasets.exact_chained_training import (
    _apply_exact_chain_settings,
    _pad_distributed_sample_index,
    _uses_unlimited_chained_samples,
)


def _cfg(
    *,
    limit_chained_samples=False,
    parent_id_column="parent_id",
    distributed=False,
    world_size=1,
):
    return SimpleNamespace(
        dataset=SimpleNamespace(
            parent_id_column=parent_id_column,
            limit_chained_samples=limit_chained_samples,
            mask_prompt_labels=False,
            only_last_answer=False,
        ),
        training=SimpleNamespace(drop_last_batch=True),
        environment=SimpleNamespace(
            _distributed=distributed,
            _world_size=world_size,
        ),
    )


def test_unlimited_chains_enable_exact_per_id_supervision():
    cfg = _cfg()

    assert _uses_unlimited_chained_samples(cfg)
    assert _apply_exact_chain_settings(cfg, mode="train")

    assert cfg.dataset.mask_prompt_labels is True
    assert cfg.dataset.only_last_answer is True
    assert cfg.training.drop_last_batch is False


def test_limited_chains_keep_explicit_training_settings():
    cfg = _cfg(limit_chained_samples=True)

    assert not _uses_unlimited_chained_samples(cfg)
    assert not _apply_exact_chain_settings(cfg, mode="train")

    assert cfg.dataset.mask_prompt_labels is False
    assert cfg.dataset.only_last_answer is False
    assert cfg.training.drop_last_batch is True


def test_no_parent_column_keeps_single_turn_training_settings():
    cfg = _cfg(parent_id_column="None")

    assert not _uses_unlimited_chained_samples(cfg)
    assert not _apply_exact_chain_settings(cfg, mode="train")

    assert cfg.dataset.mask_prompt_labels is False
    assert cfg.dataset.only_last_answer is False
    assert cfg.training.drop_last_batch is True


def test_validation_uses_same_exact_target_semantics_without_touching_batch_setting():
    cfg = _cfg()

    assert _apply_exact_chain_settings(cfg, mode="validation")

    assert cfg.dataset.mask_prompt_labels is True
    assert cfg.dataset.only_last_answer is True
    assert cfg.training.drop_last_batch is True


def test_distributed_index_padding_keeps_every_original_sample():
    cfg = _cfg(distributed=True, world_size=4)
    dataset = SimpleNamespace(
        sample_index=[
            (0, None, 0),
            (1, None, 0),
            (2, None, 0),
            (3, None, 0),
            (4, None, 0),
        ]
    )
    original = list(dataset.sample_index)

    padding = _pad_distributed_sample_index(dataset, cfg, mode="train")

    assert padding == 3
    assert len(dataset.sample_index) == 8
    assert dataset.sample_index[: len(original)] == original
    assert len(dataset.sample_index) % cfg.environment._world_size == 0
    assert dataset._distributed_sample_padding == 3


def test_distributed_index_padding_is_not_used_for_limited_chains():
    cfg = _cfg(limit_chained_samples=True, distributed=True, world_size=4)
    dataset = SimpleNamespace(
        sample_index=[
            (0, None, 0),
            (1, None, 0),
            (2, None, 0),
            (3, None, 0),
            (4, None, 0),
        ]
    )

    assert _pad_distributed_sample_index(dataset, cfg, mode="train") == 0
    assert len(dataset.sample_index) == 5
