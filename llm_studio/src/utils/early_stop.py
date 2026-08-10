"""Graceful early-stop coordination and complete checkpoint persistence."""

from __future__ import annotations

import json
import logging
import math
import os
import random
from typing import Any

import numpy as np
import torch

from llm_studio.python_configs.base import DefaultConfigProblemBase
from llm_studio.src.utils.modeling_utils import save_checkpoint

logger = logging.getLogger(__name__)

TRAINER_STATE_FILENAME = "trainer_state.pth"
CHECKPOINT_MANIFEST_FILENAME = "checkpoint_manifest.json"
DEEPSPEED_RESUME_DIRNAME = "deepspeed_resume"
EARLY_STOP_CHECKPOINT_DIRNAME = "early_stop_checkpoint"
EARLY_STOP_POINTER_FILENAME = "early_stop_checkpoint_path.txt"


def _distributed_ready(cfg: DefaultConfigProblemBase) -> bool:
    return bool(
        getattr(cfg.environment, "_distributed", False)
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )


def _is_primary_rank(cfg: DefaultConfigProblemBase) -> bool:
    if _distributed_ready(cfg):
        return int(getattr(cfg.environment, "_rank", 0)) == 0
    return int(getattr(cfg.environment, "_local_rank", 0)) == 0


def _barrier(cfg: DefaultConfigProblemBase) -> None:
    if _distributed_ready(cfg):
        torch.distributed.barrier()


def is_early_stop_requested(
    stop_training_path: str, cfg: DefaultConfigProblemBase
) -> bool:
    """Return one synchronized stop decision for every distributed rank.

    The UI creates a shared marker file. In distributed training one process may
    observe that marker a little earlier than another. Reducing the local flags with
    MAX makes the decision collective, so every rank enters checkpoint saving and
    exits training together.
    """

    requested = os.path.exists(stop_training_path)
    if not _distributed_ready(cfg):
        return requested

    backend = torch.distributed.get_backend()
    if backend == "nccl":
        device = torch.device(getattr(cfg.environment, "_device", "cuda"))
    else:
        device = torch.device("cpu")
    requested_tensor = torch.tensor(
        [int(requested)], dtype=torch.int32, device=device
    )
    torch.distributed.all_reduce(
        requested_tensor, op=torch.distributed.ReduceOp.MAX
    )
    return bool(requested_tensor.item())


def is_clean_accumulation_boundary(itr: int, grad_accumulation: int) -> bool:
    """Return whether no unapplied micro-batch gradients are pending."""

    grad_accumulation = max(int(grad_accumulation), 1)
    return itr == 0 or itr % grad_accumulation == 0


def get_early_stop_checkpoint_path(cfg: DefaultConfigProblemBase) -> str:
    """Choose a stop checkpoint path without overwriting an existing best model."""

    output_directory = cfg.output_directory
    root_checkpoint = os.path.join(output_directory, "checkpoint.pth")
    if (
        getattr(cfg.training, "save_checkpoint", "last") == "best"
        and os.path.exists(root_checkpoint)
    ):
        return os.path.join(output_directory, EARLY_STOP_CHECKPOINT_DIRNAME)
    return output_directory


def _metadata(
    cfg: DefaultConfigProblemBase,
    epoch: int,
    iteration: int,
    best_val_metric: float | None,
) -> dict[str, Any]:
    best_metric = None
    if best_val_metric is not None:
        numeric_best = float(best_val_metric)
        if math.isfinite(numeric_best):
            best_metric = numeric_best

    return {
        "version": 1,
        "reason": "early_stop",
        "epoch": int(epoch),
        "iteration": int(iteration),
        "current_step": int(getattr(cfg.environment, "_curr_step", 0)),
        "best_val_metric": best_metric,
    }


def _save_non_deepspeed_training_state(
    path: str,
    cfg: DefaultConfigProblemBase,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    metadata: dict[str, Any],
) -> None:
    if not _is_primary_rank(cfg):
        return

    state: dict[str, Any] = dict(metadata)
    state["optimizer"] = optimizer.state_dict() if optimizer is not None else None
    state["scheduler"] = scheduler.state_dict() if scheduler is not None else None
    state["scaler"] = scaler.state_dict() if scaler is not None else None
    state["python_random_state"] = random.getstate()
    state["numpy_random_state"] = np.random.get_state()
    state["torch_random_state"] = torch.get_rng_state()
    if torch.cuda.is_available():
        state["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()

    final_path = os.path.join(path, TRAINER_STATE_FILENAME)
    temporary_path = final_path + ".tmp"
    torch.save(state, temporary_path)
    os.replace(temporary_path, final_path)


def _save_deepspeed_training_state(
    path: str,
    cfg: DefaultConfigProblemBase,
    model: torch.nn.Module,
    metadata: dict[str, Any],
) -> None:
    """Keep a native DeepSpeed checkpoint including optimizer/scheduler state."""

    resume_path = os.path.join(path, DEEPSPEED_RESUME_DIRNAME)
    model.save_checkpoint(resume_path, client_state=metadata)  # type: ignore[attr-defined]
    _barrier(cfg)


def _write_manifest(
    path: str,
    cfg: DefaultConfigProblemBase,
    metadata: dict[str, Any],
) -> None:
    if not _is_primary_rank(cfg):
        return

    manifest = dict(metadata)
    manifest.update(
        {
            "complete": True,
            "model_checkpoint": "checkpoint.pth",
            "training_state": (
                DEEPSPEED_RESUME_DIRNAME
                if cfg.environment.use_deepspeed
                else TRAINER_STATE_FILENAME
            ),
        }
    )
    final_path = os.path.join(path, CHECKPOINT_MANIFEST_FILENAME)
    temporary_path = final_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary_path, final_path)


def _update_early_stop_pointer(
    checkpoint_path: str, cfg: DefaultConfigProblemBase
) -> None:
    if not _is_primary_rank(cfg):
        return

    pointer_path = os.path.join(
        cfg.output_directory, EARLY_STOP_POINTER_FILENAME
    )
    if os.path.abspath(checkpoint_path) == os.path.abspath(cfg.output_directory):
        try:
            os.remove(pointer_path)
        except FileNotFoundError:
            pass
        return

    relative_path = os.path.relpath(checkpoint_path, cfg.output_directory)
    temporary_path = pointer_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        file.write(relative_path + "\n")
    os.replace(temporary_path, pointer_path)


def save_early_stop_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    cfg: DefaultConfigProblemBase,
    epoch: int,
    iteration: int,
    best_val_metric: float | None,
    stop_training_path: str,
) -> str:
    """Save a complete, reloadable early-stop checkpoint and clear the marker.

    ``checkpoint.pth`` remains the normal inference/model checkpoint. For regular
    PyTorch training, ``trainer_state.pth`` additionally contains optimizer,
    scheduler, AMP scaler, progress and RNG state. DeepSpeed keeps its native resume
    checkpoint so sharded optimizer/scheduler state is not lost.

    When ``save_checkpoint=best`` already produced a root checkpoint, the current
    stopped weights are written to ``early_stop_checkpoint/`` instead of destroying
    the validated best model.
    """

    checkpoint_path = get_early_stop_checkpoint_path(cfg)
    os.makedirs(checkpoint_path, exist_ok=True)
    metadata = _metadata(cfg, epoch, iteration, best_val_metric)

    logger.info("Saving early-stop model checkpoint to %s", checkpoint_path)
    save_checkpoint(model=model, path=checkpoint_path, cfg=cfg)

    if cfg.environment.use_deepspeed:
        _save_deepspeed_training_state(checkpoint_path, cfg, model, metadata)
    else:
        _save_non_deepspeed_training_state(
            checkpoint_path,
            cfg,
            optimizer,
            scheduler,
            scaler,
            metadata,
        )

    _barrier(cfg)
    _write_manifest(checkpoint_path, cfg, metadata)
    _update_early_stop_pointer(checkpoint_path, cfg)
    _barrier(cfg)

    # Remove the shared marker only after every rank completed the checkpoint.
    if _is_primary_rank(cfg):
        try:
            os.remove(stop_training_path)
        except FileNotFoundError:
            pass
    _barrier(cfg)

    logger.info("Early-stop checkpoint completed at %s", checkpoint_path)
    return checkpoint_path
