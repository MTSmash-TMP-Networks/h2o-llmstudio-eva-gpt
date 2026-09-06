"""Low-host-memory DeepSpeed runtime for large rank-partitioned text corpora."""

from __future__ import annotations

import contextlib
import gc
import hashlib
import logging
import os
import tempfile
from collections.abc import Iterator
from typing import Any, Callable

import deepspeed
import torch

logger = logging.getLogger(__name__)

_RANK_PARTITIONED_MARKER = "_llm_studio_rank_partitioned_parquet"
_LARGE_CHECKPOINT_BYTES = 512 * 1024 * 1024
_INSTALLED = False
_ORIGINAL_GET_TRAIN_DATALOADER: Callable[..., Any] | None = None
_ORIGINAL_GET_VAL_DATASET: Callable[..., Any] | None = None
_ORIGINAL_GET_VAL_DATALOADER: Callable[..., Any] | None = None
_ORIGINAL_LOAD_CHECKPOINT: Callable[..., Any] | None = None
_ORIGINAL_WRAP_MODEL_DISTRIBUTED: Callable[..., Any] | None = None


def _is_rank_partitioned_dataset(value: Any) -> bool:
    """Return whether a Dataset/DataLoader owns one pre-sharded rank partition."""
    dataset = getattr(value, "dataset", value)
    return bool(getattr(dataset, _RANK_PARTITIONED_MARKER, False))


def _rank(cfg: Any) -> int:
    return int(getattr(getattr(cfg, "environment", None), "_local_rank", 0) or 0)


def _force_zero_workers(
    original: Callable[..., Any], dataset: Any, cfg: Any, *, kind: str
):
    """Build a DataLoader without forked workers for multi-GB in-memory text data."""
    if not _is_rank_partitioned_dataset(dataset):
        if kind == "train":
            return original(train_ds=dataset, cfg=cfg)
        return original(val_ds=dataset, cfg=cfg)

    environment = getattr(cfg, "environment", None)
    if environment is None:
        if kind == "train":
            return original(train_ds=dataset, cfg=cfg)
        return original(val_ds=dataset, cfg=cfg)

    previous_workers = getattr(environment, "number_of_workers", 0)
    environment.number_of_workers = 0
    try:
        if kind == "train":
            dataloader = original(train_ds=dataset, cfg=cfg)
        else:
            dataloader = original(val_ds=dataset, cfg=cfg)
    finally:
        environment.number_of_workers = previous_workers

    logger.info(
        "Rank %s uses num_workers=0 for the rank-partitioned %s DataLoader to "
        "avoid forking the multi-GB text corpus into worker processes.",
        _rank(cfg),
        kind,
    )
    return dataloader


def _get_train_dataloader_low_memory(train_ds: Any, cfg: Any):
    if _ORIGINAL_GET_TRAIN_DATALOADER is None:
        raise RuntimeError("Original training DataLoader builder is unavailable.")
    return _force_zero_workers(
        _ORIGINAL_GET_TRAIN_DATALOADER,
        train_ds,
        cfg,
        kind="train",
    )


def _get_val_dataset_rank_marker(val_df: Any, cfg: Any):
    if _ORIGINAL_GET_VAL_DATASET is None:
        raise RuntimeError("Original validation Dataset builder is unavailable.")
    dataset = _ORIGINAL_GET_VAL_DATASET(val_df=val_df, cfg=cfg)
    if bool(getattr(val_df, "attrs", {}).get(_RANK_PARTITIONED_MARKER, False)):
        setattr(dataset, _RANK_PARTITIONED_MARKER, True)
    return dataset


def _get_val_dataloader_low_memory(val_ds: Any, cfg: Any):
    if _ORIGINAL_GET_VAL_DATALOADER is None:
        raise RuntimeError("Original validation DataLoader builder is unavailable.")
    return _force_zero_workers(
        _ORIGINAL_GET_VAL_DATALOADER,
        val_ds,
        cfg,
        kind="validation",
    )


def _should_preserve_rank_dataloaders(train_dataloader: Any, cfg: Any) -> bool:
    environment = getattr(cfg, "environment", None)
    training = getattr(cfg, "training", None)
    return bool(
        environment is not None
        and training is not None
        and getattr(environment, "use_deepspeed", False)
        and not getattr(training, "lora", False)
        and _is_rank_partitioned_dataset(train_dataloader)
    )


def _wrap_model_distributed_low_memory(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    cfg: Any,
):
    """Keep rank-local loaders instead of letting DeepSpeed repartition them again."""
    if _ORIGINAL_WRAP_MODEL_DISTRIBUTED is None:
        raise RuntimeError("Original distributed model wrapper is unavailable.")

    if not _should_preserve_rank_dataloaders(train_dataloader, cfg):
        return _ORIGINAL_WRAP_MODEL_DISTRIBUTED(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            cfg=cfg,
        )

    # ``training_data=...`` makes DeepSpeed construct another DeepSpeedDataLoader.
    # That loader applies another DistributedSampler to data that is already unique
    # per rank and defaults to extra local IO workers.  Both are wrong for the
    # rank-partitioned multi-GB corpus, so initialize only the engine/optimizer and
    # keep the DataLoaders we already built above.
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
        "Rank %s preserves its pre-sharded train/validation DataLoaders through "
        "DeepSpeed initialization; no second DistributedSampler or DeepSpeed "
        "worker pool is created.",
        _rank(cfg),
    )
    return model, optimizer, train_dataloader, val_dataloader, lr_scheduler


def _normalize_path(value: Any) -> str | None:
    try:
        return os.path.realpath(os.fspath(value))
    except TypeError:
        return None


def _is_large_distributed_checkpoint(cfg: Any, weights_path: str | None) -> bool:
    if weights_path is None or not os.path.isfile(weights_path):
        return False
    environment = getattr(cfg, "environment", None)
    if environment is None or not getattr(environment, "use_deepspeed", False):
        return False
    if not getattr(environment, "_distributed", False):
        return False
    if int(getattr(environment, "_world_size", 1) or 1) <= 1:
        return False
    try:
        return os.path.getsize(weights_path) >= _LARGE_CHECKPOINT_BYTES
    except OSError:
        return False


@contextlib.contextmanager
def _serialized_checkpoint_slot(weights_path: str, cfg: Any) -> Iterator[None]:
    """Serialize large checkpoint loads per host so four ranks do not peak together."""
    try:
        import fcntl
    except ImportError:
        yield
        return

    digest = hashlib.sha256(os.path.realpath(weights_path).encode("utf-8")).hexdigest()
    lock_path = os.path.join(
        tempfile.gettempdir(),
        f"llmstudio-checkpoint-{digest[:20]}.lock",
    )
    rank = _rank(cfg)
    logger.info(
        "Rank %s waiting for the serialized large-checkpoint host-RAM load slot.",
        rank,
    )
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            logger.info(
                "Rank %s acquired the large-checkpoint load slot; other local ranks "
                "remain blocked until this checkpoint copy is released.",
                rank,
            )
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _memory_mapped_torch_load(weights_path: str) -> Iterator[None]:
    """Use torch.load(mmap=True) for one checkpoint, with a compatibility fallback."""
    original_torch_load = torch.load
    normalized_target = _normalize_path(weights_path)

    def mmap_torch_load(*args, **kwargs):
        source = args[0] if args else kwargs.get("f")
        normalized_source = _normalize_path(source)
        if normalized_source != normalized_target or "mmap" in kwargs:
            return original_torch_load(*args, **kwargs)

        mmap_kwargs = dict(kwargs)
        mmap_kwargs["mmap"] = True
        try:
            return original_torch_load(*args, **mmap_kwargs)
        except (RuntimeError, TypeError, ValueError) as exception:
            logger.warning(
                "Memory-mapped checkpoint loading is unavailable for %s (%s); "
                "falling back to the normal torch.load path.",
                weights_path,
                exception,
            )
            return original_torch_load(*args, **kwargs)

    torch.load = mmap_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def _load_checkpoint_low_memory(
    cfg: Any,
    model: torch.nn.Module,
    strict: bool = True,
    weights_path: str | None = None,
):
    """Load a large distributed checkpoint without a four-rank host-RAM spike."""
    if _ORIGINAL_LOAD_CHECKPOINT is None:
        raise RuntimeError("Original checkpoint loader is unavailable.")

    resolved_path = weights_path
    if resolved_path is None:
        resolved_path = getattr(
            getattr(cfg, "architecture", None), "pretrained_weights", None
        )
    normalized_path = _normalize_path(resolved_path)
    if not _is_large_distributed_checkpoint(cfg, normalized_path):
        return _ORIGINAL_LOAD_CHECKPOINT(
            cfg=cfg,
            model=model,
            strict=strict,
            weights_path=weights_path,
        )

    size_mb = os.path.getsize(normalized_path) / (1024 * 1024)
    logger.info(
        "Rank %s will load %.1f MB checkpoint %s with mmap and serialized local-rank "
        "access to avoid simultaneous host-RAM copies.",
        _rank(cfg),
        size_mb,
        normalized_path,
    )
    with _serialized_checkpoint_slot(normalized_path, cfg):
        with _memory_mapped_torch_load(normalized_path):
            result = _ORIGINAL_LOAD_CHECKPOINT(
                cfg=cfg,
                model=model,
                strict=strict,
                weights_path=weights_path,
            )
        gc.collect()
    return result


def install_large_text_deepspeed_runtime() -> None:
    """Install host-RAM safeguards before ``train.py`` imports runtime helpers."""
    global _INSTALLED
    global _ORIGINAL_GET_TRAIN_DATALOADER
    global _ORIGINAL_GET_VAL_DATASET
    global _ORIGINAL_GET_VAL_DATALOADER
    global _ORIGINAL_LOAD_CHECKPOINT
    global _ORIGINAL_WRAP_MODEL_DISTRIBUTED

    if _INSTALLED:
        return

    from llm_studio.src.utils import data_utils, modeling_utils

    _ORIGINAL_GET_TRAIN_DATALOADER = data_utils.get_train_dataloader
    _ORIGINAL_GET_VAL_DATASET = data_utils.get_val_dataset
    _ORIGINAL_GET_VAL_DATALOADER = data_utils.get_val_dataloader
    _ORIGINAL_LOAD_CHECKPOINT = modeling_utils.load_checkpoint
    _ORIGINAL_WRAP_MODEL_DISTRIBUTED = modeling_utils.wrap_model_distributed

    data_utils.get_train_dataloader = _get_train_dataloader_low_memory
    data_utils.get_val_dataset = _get_val_dataset_rank_marker
    data_utils.get_val_dataloader = _get_val_dataloader_low_memory
    modeling_utils.load_checkpoint = _load_checkpoint_low_memory
    modeling_utils.wrap_model_distributed = _wrap_model_distributed_low_memory
    _INSTALLED = True
