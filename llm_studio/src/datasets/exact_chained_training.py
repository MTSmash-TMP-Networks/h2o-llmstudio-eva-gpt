"""Runtime semantics for per-ID training of unlimited conversation chains."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _uses_unlimited_chained_samples(cfg: Any) -> bool:
    """Return whether parent chaining is enabled without limiting to leaf samples."""
    parent_id_column = getattr(cfg.dataset, "parent_id_column", None)
    if parent_id_column in (None, "None"):
        return False
    return not bool(getattr(cfg.dataset, "limit_chained_samples", False))


def _validate_long_sample_strategy(cfg: Any, mode: str) -> None:
    """Reject settings that can silently remove endpoint IDs from training."""
    if mode != "train" or not _uses_unlimited_chained_samples(cfg):
        return

    tokenizer_cfg = getattr(cfg, "tokenizer", None)
    strategy = getattr(tokenizer_cfg, "long_sample_strategy", "Truncate")
    if strategy == "Skip":
        raise ValueError(
            "Limit Chained Samples=False requires every training ID to remain in "
            "the dataset, but Long Sample Strategy='Skip' can silently remove "
            "long endpoint samples. Use 'Truncate' or 'Sliding Window' instead."
        )


def _apply_exact_chain_settings(cfg: Any, mode: str) -> bool:
    """Configure unlimited chains so each endpoint ID is the sole supervised turn.

    With ``limit_chained_samples=False`` the chain handler deliberately creates one
    prefix sample ending at every row/ID. Earlier turns must therefore be context
    only; otherwise their answers are supervised again in every descendant prefix
    and early conversation turns become heavily over-weighted.
    """
    if not _uses_unlimited_chained_samples(cfg):
        return False

    _validate_long_sample_strategy(cfg, mode)

    changed = False
    if not bool(getattr(cfg.dataset, "mask_prompt_labels", True)):
        cfg.dataset.mask_prompt_labels = True
        changed = True
    if not bool(getattr(cfg.dataset, "only_last_answer", False)):
        cfg.dataset.only_last_answer = True
        changed = True

    if mode == "train" and hasattr(cfg, "training"):
        if bool(getattr(cfg.training, "drop_last_batch", False)):
            cfg.training.drop_last_batch = False
            changed = True

    if changed:
        logger.info(
            "Limit Chained Samples=False: using per-ID endpoint supervision "
            "(parent turns are context only, only the endpoint answer contributes "
            "loss, incomplete train batches are retained)."
        )
    return True


def _pad_distributed_sample_index(dataset: Any, cfg: Any, mode: str) -> int:
    """Pad minimally so the existing DDP sampler cannot drop real samples.

    The repository currently creates ``DistributedSampler(..., drop_last=True)``.
    Making the dataset length divisible by world size prevents that sampler from
    discarding tail samples. At most ``world_size - 1`` duplicate index entries are
    added to keep all ranks equally sized; every original endpoint remains present.
    """
    if mode != "train" or not _uses_unlimited_chained_samples(cfg):
        return 0

    environment = getattr(cfg, "environment", None)
    if environment is None or not bool(getattr(environment, "_distributed", False)):
        return 0

    try:
        world_size = int(getattr(environment, "_world_size", 1))
    except (TypeError, ValueError):
        world_size = 1
    if world_size <= 1:
        return 0

    sample_index = getattr(dataset, "sample_index", None)
    if not isinstance(sample_index, list) or not sample_index:
        return 0

    remainder = len(sample_index) % world_size
    if remainder == 0:
        return 0

    padding = world_size - remainder
    repeats = (padding + len(sample_index) - 1) // len(sample_index)
    sample_index.extend((sample_index * repeats)[:padding])
    dataset._distributed_sample_padding = padding
    logger.info(
        "Padded unlimited chained training index by %s sample(s) for %s DDP ranks "
        "so DistributedSampler(drop_last=True) does not discard real IDs.",
        padding,
        world_size,
    )
    return padding


def install_exact_chained_training() -> None:
    """Install per-ID unlimited-chain behavior on the exported causal-LM dataset."""
    from llm_studio.src.datasets import text_causal_language_modeling_ds

    dataset_cls = text_causal_language_modeling_ds.CustomDataset
    if getattr(dataset_cls, "_exact_chained_training_installed", False):
        return

    original_init = dataset_cls.__init__

    def __init__(self, df, cfg, mode: str = "train"):
        _apply_exact_chain_settings(cfg, mode)
        original_init(self, df=df, cfg=cfg, mode=mode)
        _pad_distributed_sample_index(self, cfg, mode)

    dataset_cls.__init__ = __init__
    dataset_cls._exact_chained_training_installed = True
