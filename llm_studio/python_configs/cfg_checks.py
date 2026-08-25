import logging
import os

import torch

from llm_studio.app_utils.config import default_cfg
from llm_studio.python_configs.base import DefaultConfigProblemBase
from llm_studio.src.utils.export_utils import get_size_str
from llm_studio.src.utils.v100_precision import (
    install_precision_runtime_patch,
    normalize_training_precision,
)

logger = logging.getLogger(__name__)

# train.py imports cfg_checks before importing get_ds_config from modeling_utils.
# Installing here guarantees that DeepSpeed sees the same effective mixed-precision
# dtype as the normal DDP training path.
install_precision_runtime_patch()


def check_config_for_errors(cfg: DefaultConfigProblemBase) -> dict:
    """
    Checks the configuration for consistency.
        Parameters:
    - cfg (DefaultConfigProblemBase):
    The config object to be checked.

    Returns:
    A dictionary with two keys:
    - "title": A list of error titles.
    - "message": A list of error messages.
    """
    errors = check_for_common_errors(cfg)
    logging_errors = check_for_logging_errors(cfg)
    problem_type_errors = cfg.check()
    errors["title"].extend(problem_type_errors["title"])
    errors["message"].extend(problem_type_errors["message"])
    errors["type"].extend(problem_type_errors["type"])
    errors["title"].extend(logging_errors["title"])
    errors["message"].extend(logging_errors["message"])
    errors["type"].extend(logging_errors["type"])
    return errors


def check_for_common_errors(cfg: DefaultConfigProblemBase) -> dict:
    normalize_training_precision(cfg)

    errors: dict[str, list] = {"title": [], "message": [], "type": []}
    if not len(cfg.environment.gpus) > 0:
        errors["title"] += ["No GPU selected"]
        errors["message"] += [
            "Please select at least one GPU to start the experiment! "
        ]
        errors["type"].append("error")

    if len(cfg.environment.gpus) > torch.cuda.device_count():
        errors["title"] += ["More GPUs selected than available"]
        errors["message"] += [
            f"There are {cfg.environment.gpus} GPUs selected but only "
            f"{torch.cuda.device_count()} GPUs available."
            "This error can happen when you start from an experiment configuration "
            "that was created on a different machine. Please deselect all GPUs and "
            "select the GPUs you want to use again. "
        ]
        errors["type"].append("error")

    stats = os.statvfs(".")
    available_size = stats.f_frsize * stats.f_bavail
    if available_size < default_cfg.min_experiment_disk_space:
        errors["title"] += ["Not enough disk space."]
        errors["message"] += [
            f"Not enough disk space. Available space is {get_size_str(available_size)}."
            f" Required space is "
            f"{get_size_str(default_cfg.min_experiment_disk_space)}. "
            "Experiment has not started. "
            "Please ensure that you have enough disk space before "
            "starting the experiment."
        ]
        errors["type"].append("error")

    # see create_nlp_backbone
    if (
        cfg.architecture.backbone_dtype in ["int4", "int8"]
        and not cfg.architecture.pretrained
    ):
        errors["title"] += ["Quantization without pretrained weights."]
        errors["message"] += [
            "Quantization is only supported for pretrained models. "
            "Please enable pretrained model or disable quantization."
        ]
        errors["type"].append("error")

    if cfg.training.lora and not cfg.architecture.pretrained:
        errors["title"] += ["LoRA without pretrained weights."]
        errors["message"] += [
            "LoRA freezes the base model and only trains adapter weights. "
            "For training a model from scratch, please disable LoRA so all "
            "model weights are trainable."
        ]
        errors["type"].append("error")

    if (
        not cfg.training.lora
        and cfg.architecture.backbone_dtype == "float16"
        and cfg.training.epochs > 0
    ):
        errors["title"] += ["Unsafe pure float16 full-weight training."]
        errors["message"] += [
            "Full-weight float16 parameters can make training unstable. Use "
            "Backbone Dtype=float32 together with Mixed Precision=float16 so "
            "trainable weights stay in FP32 while forward/backward compute uses FP16."
        ]
        errors["type"].append("warning")

    if cfg.environment.use_deepspeed and cfg.architecture.backbone_dtype in [
        "int8",
        "int4",
    ]:
        errors["title"] += ["Deepspeed does not support quantization."]
        errors["message"] += [
            "Deepspeed does not support quantized backbone training. Use a float32 "
            "backbone with Mixed Precision set to float16 or bfloat16."
        ]
        errors["type"].append("error")
    if cfg.environment.use_deepspeed and len(cfg.environment.gpus) < 2:
        errors["title"] += ["Deepspeed not supported for single GPU."]
        errors["message"] += [
            "Deepspeed does not support single GPU training. "
            "Please select more than one GPU or disable deepspeed."
        ]
        errors["type"].append("error")
    return errors


def check_for_logging_errors(cfg: DefaultConfigProblemBase) -> dict:
    errors: dict[str, list] = {"title": [], "message": [], "type": []}

    if cfg.logging.logger == "W&B" and cfg.logging.log_step_size == "relative":
        errors["title"] += ["WandB relative step size logging not supported."]
        errors["message"] += [
            "WandB logging does not support relative step size logging. "
            "Please set log_step_size to 'absolute' when using WandB logging."
        ]
        errors["type"].append("error")

    return errors
