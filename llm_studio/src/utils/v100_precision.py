"""Safe precision policy for full-weight training on FP16-only CUDA GPUs."""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _selected_cuda_indices(cfg: Any) -> list[int]:
    """Return valid logical CUDA indices selected by the experiment."""
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        return []

    configured_gpus = getattr(getattr(cfg, "environment", None), "gpus", ())
    indices: list[int] = []
    for gpu in configured_gpus:
        value = str(gpu).strip().lower()
        if value.startswith("cuda:"):
            value = value.split(":", maxsplit=1)[1]
        try:
            index = int(value)
        except ValueError:
            continue
        if 0 <= index < device_count and index not in indices:
            indices.append(index)

    return indices or list(range(device_count))


def selected_cuda_supports_bfloat16(cfg: Any) -> bool | None:
    """Return native BF16 support for selected CUDA devices when detectable."""
    if not torch.cuda.is_available():
        return None

    indices = _selected_cuda_indices(cfg)
    if not indices:
        return None

    try:
        # Native BF16 Tensor Core support starts with NVIDIA Ampere (SM 8.x).
        return all(torch.cuda.get_device_capability(index)[0] >= 8 for index in indices)
    except Exception:
        is_supported = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(is_supported):
            try:
                return bool(is_supported())
            except Exception:
                pass
    return None


def _selected_cuda_names(cfg: Any) -> str:
    names: list[str] = []
    for index in _selected_cuda_indices(cfg):
        try:
            name = torch.cuda.get_device_name(index)
        except Exception:
            continue
        if name not in names:
            names.append(name)
    return ", ".join(names) if names else "selected CUDA GPU(s)"


def normalize_training_precision(cfg: Any) -> None:
    """Normalize unsafe full-FP16 configurations to stable FP16 AMP training.

    Full-weight training keeps trainable parameters in FP32 and uses FP16 only
    for autocast compute. This lets the existing GradScaler path protect small
    gradients on V100-class hardware while retaining Tensor Core acceleration.
    """
    architecture = getattr(cfg, "architecture", None)
    environment = getattr(cfg, "environment", None)
    training = getattr(cfg, "training", None)
    if architecture is None or environment is None or training is None:
        return
    if int(getattr(training, "epochs", 0)) <= 0:
        return

    lora = bool(getattr(training, "lora", False))
    backbone_dtype = getattr(architecture, "backbone_dtype", None)

    if not lora and backbone_dtype == "float16":
        architecture.backbone_dtype = "float32"
        environment.mixed_precision = True
        environment.mixed_precision_dtype = "float16"
        logger.info(
            "Safe FP16 full-weight training enabled: trainable backbone weights "
            "were promoted from float16 to float32 while compute remains FP16 "
            "through mixed precision with dynamic gradient scaling."
        )
        backbone_dtype = "float32"

    bf16_supported = selected_cuda_supports_bfloat16(cfg)
    if bf16_supported is False:
        gpu_names = _selected_cuda_names(cfg)

        if getattr(environment, "mixed_precision_dtype", None) == "bfloat16":
            environment.mixed_precision = True
            environment.mixed_precision_dtype = "float16"
            logger.info(
                "%s does not provide native bfloat16 training support; switching "
                "Mixed Precision Dtype to float16 for AMP training.",
                gpu_names,
            )

        if backbone_dtype == "bfloat16":
            architecture.backbone_dtype = "float16" if lora else "float32"
            environment.mixed_precision = True
            environment.mixed_precision_dtype = "float16"
            logger.info(
                "%s cannot use a bfloat16 backbone safely; using %s backbone "
                "weights with float16 mixed-precision compute.",
                gpu_names,
                architecture.backbone_dtype,
            )


def build_deepspeed_config(cfg: Any) -> dict[str, Any]:
    """Build a DeepSpeed config whose compute dtype follows mixed precision."""
    mixed_precision = bool(getattr(cfg.environment, "mixed_precision", False))
    mixed_precision_dtype = getattr(cfg.environment, "mixed_precision_dtype", "float16")

    if mixed_precision:
        fp16_enabled = mixed_precision_dtype == "float16"
        bf16_enabled = mixed_precision_dtype == "bfloat16"
    else:
        # Preserve the legacy behavior for adapter/advanced configurations that
        # intentionally keep the backbone itself in a 16-bit dtype.
        fp16_enabled = cfg.architecture.backbone_dtype == "float16"
        bf16_enabled = cfg.architecture.backbone_dtype == "bfloat16"

    ds_config: dict[str, Any] = {
        "fp16": {
            "enabled": fp16_enabled,
            "loss_scale_window": 100,
        },
        "bf16": {
            "enabled": bf16_enabled,
        },
        "zero_force_ds_cpu_optimizer": False,
        "zero_optimization": {
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": cfg.environment.deepspeed_reduce_bucket_size,
        },
        "steps_per_print": 2000,
        "train_micro_batch_size_per_gpu": cfg.training.batch_size,
        "gradient_accumulation_steps": cfg.training.grad_accumulation,
        "wall_clock_breakdown": False,
    }

    if cfg.training.gradient_clip > 0:
        ds_config["gradient_clipping"] = cfg.training.gradient_clip

    if cfg.environment.deepspeed_method == "ZeRO2":
        ds_config["zero_optimization"].update(
            {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": (
                    cfg.environment.deepspeed_allgather_bucket_size
                ),
            }
        )
    elif cfg.environment.deepspeed_method == "ZeRO3":
        ds_config["zero_optimization"].update(
            {
                "stage": 3,
                "stage3_prefetch_bucket_size": (
                    cfg.environment.deepspeed_stage3_prefetch_bucket_size
                ),
                "stage3_param_persistence_threshold": (
                    cfg.environment.deepspeed_stage3_param_persistence_threshold
                ),
                "stage3_gather_16bit_weights_on_model_save": True,
            }
        )

    logger.info("DeepSpeed config: %s", ds_config)
    return ds_config


def install_precision_runtime_patch() -> None:
    """Install the DeepSpeed precision fix before train.py imports the function."""
    from llm_studio.src.utils import modeling_utils

    if getattr(modeling_utils, "_v100_precision_patch_installed", False):
        return

    modeling_utils.get_ds_config = build_deepspeed_config
    modeling_utils._v100_precision_patch_installed = True
