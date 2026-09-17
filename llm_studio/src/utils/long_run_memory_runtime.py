"""Memory safeguards for long-running DeepSpeed causal-LM experiments.

Large validation sets can temporarily allocate substantial CPU memory.  Two paths are
particularly expensive for long jobs:

* decoded prediction text was stored in fixed-width NumPy unicode arrays.  When many
  batches are concatenated, NumPy promotes every row to the width of the longest
  generated prediction;
* DeepSpeed's regular Causal-LM checkpoint path reloaded the just-written 16-bit
  checkpoint into CPU RAM only to inspect optional classification/regression heads.

This runtime keeps the existing training semantics while removing those avoidable
allocations and aggressively returns freed validation/checkpoint memory to the OS.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import shutil
from typing import Any, Callable

import numpy as np
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

logger = logging.getLogger(__name__)

_INSTALLED = False
_ORIGINAL_POSTPROCESS_BATCH: Callable[..., Any] | None = None
_ORIGINAL_RUN_EVAL: Callable[..., Any] | None = None
_ORIGINAL_SAVE_CHECKPOINT: Callable[..., Any] | None = None


def _rank(cfg: Any) -> int:
    return int(getattr(getattr(cfg, "environment", None), "_local_rank", 0) or 0)


def _current_rss_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return None


def _log_memory(cfg: Any, stage: str) -> None:
    parts = [f"Rank {_rank(cfg)} memory at {stage}"]
    rss_mb = _current_rss_mb()
    if rss_mb is not None:
        parts.append(f"host RSS={rss_mb:.1f} MB")

    environment = getattr(cfg, "environment", None)
    device = getattr(environment, "_device", None) if environment is not None else None
    if torch.cuda.is_available() and device is not None:
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            mib = 1024 * 1024
            parts.extend(
                [
                    f"CUDA allocated={allocated / mib:.1f} MB",
                    f"reserved={reserved / mib:.1f} MB",
                    f"free={free_bytes / mib:.1f} MB",
                    f"total={total_bytes / mib:.1f} MB",
                ]
            )
        except Exception:
            pass

    logger.info("; ".join(parts))


def _malloc_trim() -> None:
    """Return free glibc arenas to Linux when available."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        trim = getattr(libc, "malloc_trim", None)
        if trim is not None:
            trim(0)
    except (OSError, AttributeError):
        pass


def _release_memory(cfg: Any, stage: str) -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    _malloc_trim()
    _log_memory(cfg, stage)


def _is_deepspeed_causal_lm(cfg: Any) -> bool:
    environment = getattr(cfg, "environment", None)
    return bool(
        environment is not None
        and getattr(environment, "use_deepspeed", False)
        and getattr(cfg, "problem_type", "") == "text_causal_language_modeling"
    )


def _postprocess_batch_predictions_object_strings(self, output: dict) -> dict:
    """Keep decoded batch predictions as object strings, not fixed-width unicode."""
    if _ORIGINAL_POSTPROCESS_BATCH is None:
        raise RuntimeError("Original batch prediction postprocessor is unavailable.")

    result = _ORIGINAL_POSTPROCESS_BATCH(self, output)
    predicted_text = result.get("predicted_text")
    if isinstance(predicted_text, np.ndarray) and predicted_text.dtype.kind in {"U", "S"}:
        # Validation concatenates all batches.  Fixed-width unicode would promote
        # every one of hundreds of thousands of rows to the longest generated text.
        result["predicted_text"] = predicted_text.astype(object, copy=False)
    return result


def _save_checkpoint_low_memory(model, path: str, cfg: Any) -> None:
    """Save DeepSpeed Causal-LM weights without reloading them into host RAM."""
    if _ORIGINAL_SAVE_CHECKPOINT is None:
        raise RuntimeError("Original checkpoint saver is unavailable.")
    if not _is_deepspeed_causal_lm(cfg):
        return _ORIGINAL_SAVE_CHECKPOINT(model=model, path=path, cfg=cfg)
    if not path:
        raise ValueError(f"Path must be provided. Received {path}.")

    os.makedirs(path, exist_ok=True)
    _log_memory(cfg, "before Causal-LM checkpoint")

    checkpoint_path = os.path.join(path, "checkpoint.pth")
    status = model.save_16bit_model(path, "checkpoint.pth")
    if status:
        if _rank(cfg) == 0:
            logger.info(
                "Saved DeepSpeed Causal-LM checkpoint to %s without reloading the "
                "just-written weights into host RAM.",
                checkpoint_path,
            )
    else:
        # Preserve the legacy ZeRO fallback for configurations where a gathered
        # 16-bit checkpoint cannot be produced directly.
        ds_checkpoint = os.path.join(path, "ds_checkpoint")
        model.save_checkpoint(ds_checkpoint)
        if _rank(cfg) == 0:
            state_dict = get_fp32_state_dict_from_zero_checkpoint(ds_checkpoint)
            try:
                torch.save({"model": state_dict}, checkpoint_path)
            finally:
                del state_dict
                shutil.rmtree(ds_checkpoint, ignore_errors=True)

    _release_memory(cfg, "after Causal-LM checkpoint cleanup")


def _run_eval_with_memory_cleanup(*args, **kwargs):
    """Run validation and return released temporary memory to the OS afterwards."""
    if _ORIGINAL_RUN_EVAL is None:
        raise RuntimeError("Original validation loop is unavailable.")

    cfg = kwargs.get("cfg")
    if cfg is None and args:
        cfg = args[0]
    if cfg is not None:
        _log_memory(cfg, "before validation")

    try:
        return _ORIGINAL_RUN_EVAL(*args, **kwargs)
    finally:
        if cfg is not None:
            _release_memory(cfg, "after validation cleanup")


def install_long_run_memory_runtime() -> None:
    """Install long-run validation/checkpoint safeguards after train.py is imported."""
    global _INSTALLED
    global _ORIGINAL_POSTPROCESS_BATCH
    global _ORIGINAL_RUN_EVAL
    global _ORIGINAL_SAVE_CHECKPOINT

    if _INSTALLED:
        return

    import llm_studio.train as train_module
    from llm_studio.src.datasets import text_causal_language_modeling_ds as causal_ds
    from llm_studio.src.utils import modeling_utils

    dataset_class = causal_ds.CustomDataset
    _ORIGINAL_POSTPROCESS_BATCH = dataset_class.postprocess_batch_predictions
    _ORIGINAL_RUN_EVAL = train_module.run_eval
    _ORIGINAL_SAVE_CHECKPOINT = train_module.save_checkpoint

    dataset_class.postprocess_batch_predictions = _postprocess_batch_predictions_object_strings
    train_module.run_eval = _run_eval_with_memory_cleanup
    train_module.save_checkpoint = _save_checkpoint_low_memory
    modeling_utils.save_checkpoint = _save_checkpoint_low_memory

    _INSTALLED = True
    logger.info("Installed long-run validation/checkpoint memory safeguards.")
