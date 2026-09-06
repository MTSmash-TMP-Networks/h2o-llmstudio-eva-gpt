"""Safe precision and low-memory DeepSpeed policy for CUDA training."""

from __future__ import annotations

import gc
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

_deepspeed_owns_mixed_precision = False
_ORIGINAL_CREATE_NLP_BACKBONE = None
_ORIGINAL_GET_OPTIMIZER = None
_ORIGINAL_WRAP_MODEL_DISTRIBUTED = None


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


def deepspeed_owns_mixed_precision() -> bool:
    """Return whether DeepSpeed, rather than torch autocast, owns precision."""
    return _deepspeed_owns_mixed_precision


def _set_deepspeed_precision_owner(cfg: Any) -> None:
    global _deepspeed_owns_mixed_precision

    environment = getattr(cfg, "environment", None)
    _deepspeed_owns_mixed_precision = bool(
        environment is not None
        and getattr(environment, "use_deepspeed", False)
        and getattr(environment, "mixed_precision", False)
    )


def _install_external_autocast_guard() -> None:
    """Disable H2O's outer CUDA autocast while DeepSpeed owns mixed precision.

    DeepSpeed's native FP16 mode already converts the model, keeps FP32 master
    optimizer state, performs dynamic loss scaling, and drives backward/step.
    Wrapping that engine in a second torch.cuda.amp.autocast context gives two
    independent mixed-precision controllers around the same forward pass.
    """
    from torch.cuda.amp import autocast as cuda_autocast

    if getattr(cuda_autocast, "_llmstudio_deepspeed_guard", False):
        return

    original_init = cuda_autocast.__init__

    def guarded_init(self, *args, **kwargs):
        if _deepspeed_owns_mixed_precision:
            if "enabled" in kwargs:
                kwargs["enabled"] = False
            elif args:
                args = (False, *args[1:])
            else:
                kwargs["enabled"] = False
        return original_init(self, *args, **kwargs)

    cuda_autocast.__init__ = guarded_init
    cuda_autocast._llmstudio_deepspeed_guard = True


def _is_deepspeed_causal_lm(cfg: Any) -> bool:
    """Return whether the current run can use the causal-LM DeepSpeed fast path."""
    environment = getattr(cfg, "environment", None)
    training = getattr(cfg, "training", None)
    return bool(
        environment is not None
        and training is not None
        and getattr(environment, "use_deepspeed", False)
        and not getattr(training, "lora", False)
        and getattr(cfg, "problem_type", "") == "text_causal_language_modeling"
    )


def _deepspeed_runtime_load_dtype(cfg: Any) -> torch.dtype | None:
    """Return a compact runtime model dtype while preserving FP32 master updates.

    The UI/config keeps ``backbone_dtype=float32`` for safe full-weight training.
    When DeepSpeed owns FP16/BF16 mixed precision it can start from a 16-bit
    trainable replica and create/manage the FP32 master optimizer weights itself.
    This avoids materializing a full FP32 CUDA replica immediately before
    ``deepspeed.initialize``.
    """
    if not _is_deepspeed_causal_lm(cfg):
        return None
    if not getattr(cfg.environment, "mixed_precision", False):
        return None
    if getattr(cfg.architecture, "backbone_dtype", None) != "float32":
        return None

    mixed_precision_dtype = getattr(cfg.environment, "mixed_precision_dtype", "float16")
    if mixed_precision_dtype == "float16":
        return torch.float16
    if mixed_precision_dtype == "bfloat16":
        return torch.bfloat16
    return None


def _dtype_overridden_model_class(model_class: Any, runtime_dtype: torch.dtype):
    """Proxy HF model factory calls while overriding only the runtime load dtype."""

    class RuntimeDtypeModelClass:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            kwargs["torch_dtype"] = runtime_dtype
            return model_class.from_pretrained(*args, **kwargs)

        @staticmethod
        def from_config(*args, **kwargs):
            kwargs["torch_dtype"] = runtime_dtype
            return model_class.from_config(*args, **kwargs)

    return RuntimeDtypeModelClass


def _current_rss_mb() -> float | None:
    """Return current Linux process RSS without adding another dependency."""
    try:
        with open("/proc/self/status", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return None


def _log_memory_snapshot(cfg: Any, stage: str) -> None:
    """Log host and CUDA memory around the otherwise opaque DeepSpeed startup."""
    rank = getattr(getattr(cfg, "environment", None), "_local_rank", 0)
    parts = [f"Rank {rank} memory at {stage}"]

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
            # Memory diagnostics must never prevent a training run from starting.
            pass

    logger.info("; ".join(parts))


def _memory_efficient_create_nlp_backbone(cfg: Any, model_class=None):
    """Load DeepSpeed's runtime replica directly in the compute dtype."""
    if _ORIGINAL_CREATE_NLP_BACKBONE is None:
        raise RuntimeError("Original create_nlp_backbone is unavailable.")

    if model_class is None:
        from transformers import AutoModel

        model_class = AutoModel

    runtime_dtype = _deepspeed_runtime_load_dtype(cfg)
    _log_memory_snapshot(cfg, "before backbone construction")
    if runtime_dtype is not None:
        logger.info(
            "DeepSpeed low-memory model load enabled: runtime backbone replica uses "
            "%s while DeepSpeed retains FP32 master optimizer weights.",
            str(runtime_dtype).replace("torch.", ""),
        )
        model_class = _dtype_overridden_model_class(model_class, runtime_dtype)

    result = _ORIGINAL_CREATE_NLP_BACKBONE(cfg, model_class=model_class)
    _log_memory_snapshot(cfg, "after backbone construction")
    return result


def _should_skip_outer_model_to(cfg: Any) -> bool:
    return _is_deepspeed_causal_lm(cfg) and deepspeed_owns_mixed_precision()


def _install_outer_model_to_guard(cfg: Any) -> None:
    """Let DeepSpeed own backbone placement instead of pre-moving the outer model."""
    if not _should_skip_outer_model_to(cfg):
        return

    architecture = getattr(cfg, "architecture", None)
    model_class = getattr(architecture, "model_class", None)
    if model_class is None or getattr(
        model_class, "_llmstudio_deepspeed_to_guard", False
    ):
        return

    original_to = model_class.to

    def guarded_to(self, *args, **kwargs):
        instance_cfg = getattr(self, "cfg", None)
        if instance_cfg is not None and _should_skip_outer_model_to(instance_cfg):
            _log_memory_snapshot(instance_cfg, "before DeepSpeed device placement")
            logger.info(
                "Rank %s skipping pre-DeepSpeed outer model.to(...); DeepSpeed "
                "owns backbone device placement and FP16/BF16 conversion.",
                getattr(instance_cfg.environment, "_local_rank", 0),
            )
            return self
        return original_to(self, *args, **kwargs)

    model_class.to = guarded_to
    model_class._llmstudio_deepspeed_to_guard = True
    model_class._llmstudio_original_to = original_to


def _get_optimizer_with_memory(model, cfg):
    if _ORIGINAL_GET_OPTIMIZER is None:
        raise RuntimeError("Original get_optimizer is unavailable.")
    if _is_deepspeed_causal_lm(cfg):
        _log_memory_snapshot(cfg, "before optimizer construction")
    optimizer = _ORIGINAL_GET_OPTIMIZER(model=model, cfg=cfg)
    if _is_deepspeed_causal_lm(cfg):
        _log_memory_snapshot(cfg, "after optimizer construction")
    return optimizer


def _wrap_model_distributed_with_memory(*args, **kwargs):
    if _ORIGINAL_WRAP_MODEL_DISTRIBUTED is None:
        raise RuntimeError("Original wrap_model_distributed is unavailable.")

    cfg = kwargs.get("cfg")
    if cfg is None and len(args) >= 6:
        cfg = args[5]

    if cfg is not None and _is_deepspeed_causal_lm(cfg):
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        _log_memory_snapshot(cfg, "immediately before deepspeed.initialize")
        logger.info(
            "Rank %s entering deepspeed.initialize after releasing Python/CUDA "
            "startup caches.",
            getattr(cfg.environment, "_local_rank", 0),
        )

    result = _ORIGINAL_WRAP_MODEL_DISTRIBUTED(*args, **kwargs)

    if cfg is not None and _is_deepspeed_causal_lm(cfg):
        _log_memory_snapshot(cfg, "after deepspeed.initialize")
    return result


def normalize_training_precision(cfg: Any) -> None:
    """Normalize unsafe full-FP16 configurations to stable FP16 AMP training.

    Normal DDP keeps trainable parameters in FP32 and uses FP16 autocast plus a
    GradScaler. DeepSpeed uses its native FP16 engine instead: the runtime model
    can stay in FP16 while DeepSpeed owns FP32 master optimizer state, dynamic loss
    scaling, backward and optimizer stepping.
    """
    global _deepspeed_owns_mixed_precision

    architecture = getattr(cfg, "architecture", None)
    environment = getattr(cfg, "environment", None)
    training = getattr(cfg, "training", None)
    if architecture is None or environment is None or training is None:
        _deepspeed_owns_mixed_precision = False
        return
    if int(getattr(training, "epochs", 0)) <= 0:
        _deepspeed_owns_mixed_precision = False
        return

    lora = bool(getattr(training, "lora", False))
    backbone_dtype = getattr(architecture, "backbone_dtype", None)

    if not lora and backbone_dtype == "float16":
        architecture.backbone_dtype = "float32"
        environment.mixed_precision = True
        environment.mixed_precision_dtype = "float16"
        logger.info(
            "Safe FP16 full-weight training enabled: the configuration uses an "
            "FP32 master-weight policy while compute remains FP16 through mixed "
            "precision. DeepSpeed may materialize its runtime replica directly in "
            "FP16 to reduce initialization memory."
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

    _set_deepspeed_precision_owner(cfg)
    if _deepspeed_owns_mixed_precision:
        logger.info(
            "DeepSpeed owns mixed precision for this run; external torch CUDA "
            "autocast is disabled to avoid nested AMP/loss-scaling control."
        )

    _install_outer_model_to_guard(cfg)


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
            "loss_scale": 0,
            "loss_scale_window": 100,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
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
    """Install precision and low-memory fixes before train.py imports helpers."""
    global _ORIGINAL_CREATE_NLP_BACKBONE
    global _ORIGINAL_GET_OPTIMIZER
    global _ORIGINAL_WRAP_MODEL_DISTRIBUTED

    from llm_studio.src.utils import modeling_utils

    _install_external_autocast_guard()

    if getattr(modeling_utils, "_v100_precision_patch_installed", False):
        return

    _ORIGINAL_CREATE_NLP_BACKBONE = modeling_utils.create_nlp_backbone
    _ORIGINAL_GET_OPTIMIZER = modeling_utils.get_optimizer
    _ORIGINAL_WRAP_MODEL_DISTRIBUTED = modeling_utils.wrap_model_distributed

    modeling_utils.get_ds_config = build_deepspeed_config
    modeling_utils.create_nlp_backbone = _memory_efficient_create_nlp_backbone
    modeling_utils.get_optimizer = _get_optimizer_with_memory
    modeling_utils.wrap_model_distributed = _wrap_model_distributed_with_memory
    modeling_utils._v100_precision_patch_installed = True
