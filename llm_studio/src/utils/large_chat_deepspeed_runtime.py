"""Low-host-memory DeepSpeed loader path for large in-memory chat datasets."""

from __future__ import annotations

import logging
from typing import Any, Callable

import deepspeed
import torch

logger = logging.getLogger(__name__)

_PRESERVE_LOADER_MARKER = "_llm_studio_preserve_deepspeed_loader"
_RANK_PARTITIONED_MARKER = "_llm_studio_rank_partitioned_parquet"
_LARGE_DATASET_ROWS = 100_000
_LARGE_PROCESS_RSS_BYTES = 4 * 1024 * 1024 * 1024
_INSTALLED = False
_ORIGINAL_GET_TRAIN_DATALOADER: Callable[..., Any] | None = None
_ORIGINAL_GET_VAL_DATALOADER: Callable[..., Any] | None = None
_ORIGINAL_WRAP_MODEL_DISTRIBUTED: Callable[..., Any] | None = None


def _rank(cfg: Any) -> int:
    return int(getattr(getattr(cfg, "environment", None), "_local_rank", 0) or 0)


def _current_rss_bytes() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _is_target_deepspeed_chat(cfg: Any) -> bool:
    environment = getattr(cfg, "environment", None)
    training = getattr(cfg, "training", None)
    return bool(
        environment is not None
        and training is not None
        and getattr(environment, "use_deepspeed", False)
        and getattr(environment, "_distributed", False)
        and int(getattr(environment, "_world_size", 1) or 1) > 1
        and not getattr(training, "lora", False)
        and getattr(cfg, "problem_type", "") == "text_causal_language_modeling"
    )


def _is_rank_partitioned_dataset(value: Any) -> bool:
    dataset = getattr(value, "dataset", value)
    return bool(getattr(dataset, _RANK_PARTITIONED_MARKER, False))


def _dataset_length(dataset: Any) -> int:
    try:
        return int(len(dataset))
    except (TypeError, AttributeError):
        return 0


def _is_large_in_memory_dataset(dataset: Any, cfg: Any) -> bool:
    if not _is_target_deepspeed_chat(cfg):
        return False
    # The sharded Parquet runtime already owns this case and has stronger rank-local
    # guarantees. Do not overlap the two policies.
    if _is_rank_partitioned_dataset(dataset):
        return False
    return bool(
        _dataset_length(dataset) >= _LARGE_DATASET_ROWS
        or _current_rss_bytes() >= _LARGE_PROCESS_RSS_BYTES
    )


class _EpochAwareLoaderProxy:
    """Advance a preserved DistributedSampler once for every training epoch."""

    def __init__(self, loader: Any):
        self._loader = loader
        self._next_epoch = 0
        setattr(self, _PRESERVE_LOADER_MARKER, True)

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self):
        sampler = getattr(self._loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self._next_epoch)
            self._next_epoch += 1
        return iter(self._loader)

    def __getattr__(self, name: str):
        return getattr(self._loader, name)


def _build_with_zero_workers(
    original: Callable[..., Any], dataset: Any, cfg: Any, *, kind: str
):
    environment = getattr(cfg, "environment", None)
    if environment is None:
        if kind == "train":
            return original(train_ds=dataset, cfg=cfg)
        return original(val_ds=dataset, cfg=cfg)

    previous_workers = getattr(environment, "number_of_workers", 0)
    environment.number_of_workers = 0
    try:
        if kind == "train":
            loader = original(train_ds=dataset, cfg=cfg)
        else:
            loader = original(val_ds=dataset, cfg=cfg)
    finally:
        environment.number_of_workers = previous_workers

    rss_mb = _current_rss_bytes() / (1024 * 1024)
    logger.info(
        "Rank %s uses num_workers=0 for large in-memory %s data (%s samples; "
        "host RSS %.1f MB) to avoid forking the complete conversation dataset.",
        _rank(cfg),
        kind,
        _dataset_length(dataset),
        rss_mb,
    )
    return loader


def _get_train_dataloader_low_memory(train_ds: Any, cfg: Any):
    if _ORIGINAL_GET_TRAIN_DATALOADER is None:
        raise RuntimeError("Original training DataLoader builder is unavailable.")
    if not _is_large_in_memory_dataset(train_ds, cfg):
        return _ORIGINAL_GET_TRAIN_DATALOADER(train_ds=train_ds, cfg=cfg)

    loader = _build_with_zero_workers(
        _ORIGINAL_GET_TRAIN_DATALOADER,
        train_ds,
        cfg,
        kind="train",
    )
    return _EpochAwareLoaderProxy(loader)


def _get_val_dataloader_low_memory(val_ds: Any, cfg: Any):
    if _ORIGINAL_GET_VAL_DATALOADER is None:
        raise RuntimeError("Original validation DataLoader builder is unavailable.")
    if not _is_large_in_memory_dataset(val_ds, cfg):
        return _ORIGINAL_GET_VAL_DATALOADER(val_ds=val_ds, cfg=cfg)
    return _build_with_zero_workers(
        _ORIGINAL_GET_VAL_DATALOADER,
        val_ds,
        cfg,
        kind="validation",
    )


def _should_preserve_loader(train_dataloader: Any, cfg: Any) -> bool:
    return bool(
        _is_target_deepspeed_chat(cfg)
        and getattr(train_dataloader, _PRESERVE_LOADER_MARKER, False)
    )


def _wrap_model_distributed_low_memory(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    cfg: Any,
):
    if _ORIGINAL_WRAP_MODEL_DISTRIBUTED is None:
        raise RuntimeError("Original distributed model wrapper is unavailable.")

    if not _should_preserve_loader(train_dataloader, cfg):
        return _ORIGINAL_WRAP_MODEL_DISTRIBUTED(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            cfg=cfg,
        )

    from llm_studio.src.utils import modeling_utils

    ds_config = modeling_utils.get_ds_config(cfg)
    ds_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model.backbone,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        training_data=None,
        config_params=ds_config,
    )
    model.backbone = ds_engine
    model.init_deepspeed()  # type: ignore[attr-defined]

    logger.info(
        "Rank %s preserves the existing distributed train/validation DataLoaders "
        "through DeepSpeed initialization; no second DeepSpeed DataLoader or worker "
        "pool is created for the in-memory conversation dataset.",
        _rank(cfg),
    )
    return model, optimizer, train_dataloader, val_dataloader, lr_scheduler


def install_large_chat_deepspeed_runtime() -> None:
    """Install the large in-memory chat safeguards before train.py imports helpers."""
    global _INSTALLED
    global _ORIGINAL_GET_TRAIN_DATALOADER
    global _ORIGINAL_GET_VAL_DATALOADER
    global _ORIGINAL_WRAP_MODEL_DISTRIBUTED

    if _INSTALLED:
        return

    from llm_studio.src.utils import data_utils, modeling_utils

    _ORIGINAL_GET_TRAIN_DATALOADER = data_utils.get_train_dataloader
    _ORIGINAL_GET_VAL_DATALOADER = data_utils.get_val_dataloader
    _ORIGINAL_WRAP_MODEL_DISTRIBUTED = modeling_utils.wrap_model_distributed

    data_utils.get_train_dataloader = _get_train_dataloader_low_memory
    data_utils.get_val_dataloader = _get_val_dataloader_low_memory
    modeling_utils.wrap_model_distributed = _wrap_model_distributed_low_memory
    _INSTALLED = True
