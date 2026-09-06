"""Repair stale quantization metadata on dense local EvaGPT model packages."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DENSE_FLOAT_DTYPES = {"BF16", "F16", "F32", "F64"}
_QUANTIZED_KEY_MARKERS = (
    ".blocks",
    ".scales",
    "qweight",
    "quant_state",
    "weight_scale",
)
_MAX_SAFETENSOR_HEADER_BYTES = 128 * 1024 * 1024


def _quantization_method(config: dict[str, Any]) -> str:
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        return ""
    method = quantization.get(
        "quant_method", quantization.get("quantization_method", "")
    )
    return str(method).strip().lower()


def _read_safetensor_header(weight_file: Path) -> dict[str, Any] | None:
    """Read only the JSON header; never map or materialize tensor payload bytes."""
    try:
        with weight_file.open("rb") as tensor_file:
            header_size_bytes = tensor_file.read(8)
            if len(header_size_bytes) != 8:
                return None
            header_size = int.from_bytes(
                header_size_bytes, byteorder="little", signed=False
            )
            if header_size <= 0 or header_size > _MAX_SAFETENSOR_HEADER_BYTES:
                return None
            header = json.loads(tensor_file.read(header_size).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return header if isinstance(header, dict) else None


def _looks_like_dense_safetensors(model_path: str) -> bool:
    """Inspect safetensor headers without materializing the model weights."""
    model_dir = Path(model_path)
    weight_files = sorted(model_dir.glob("*.safetensors"))
    if not weight_files:
        return False

    weight_count = 0
    for weight_file in weight_files:
        header = _read_safetensor_header(weight_file)
        if header is None:
            return False
        for key, tensor_metadata in header.items():
            if key == "__metadata__":
                continue
            lowered_key = key.lower()
            if any(marker in lowered_key for marker in _QUANTIZED_KEY_MARKERS):
                return False
            if not lowered_key.endswith(".weight"):
                continue
            if not isinstance(tensor_metadata, dict):
                return False
            dtype = str(tensor_metadata.get("dtype", "")).upper()
            if dtype not in _DENSE_FLOAT_DTYPES:
                return False
            weight_count += 1

    return weight_count > 0


def _write_config_atomically(config_path: Path, config: dict[str, Any]) -> None:
    temporary_path = config_path.with_name(f"{config_path.name}.tmp.{os.getpid()}")
    try:
        with temporary_path.open("w", encoding="utf-8") as config_file:
            json.dump(config, config_file, indent=2, ensure_ascii=False)
            config_file.write("\n")
        os.replace(temporary_path, config_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def repair_stale_dense_eva_quantization_config(model_path: str) -> bool:
    """Remove inherited MXFP4 metadata when the local weights are actually dense.

    Older MaTeLiX model creation runs could start from a Hugging Face base config,
    inherit its ``quantization_config``, then instantiate a new model, convert it to
    FP32 and save dense safetensors. The stale MXFP4 marker later makes Transformers
    attempt an unnecessary MXFP4 -> BF16 dequantization during every distributed
    rank startup. On V100 systems that creates a large host-RAM spike before
    DeepSpeed can take ownership of the model.

    The repair is deliberately conservative: it only applies to local EvaGPT-looking
    directories whose safetensor *weight* metadata is entirely dense floating point.
    Truly packed/quantized models are left untouched.
    """
    if not model_path or not os.path.isdir(model_path):
        return False

    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return False

    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(config, dict):
        return False
    if _quantization_method(config) != "mxfp4":
        return False

    from llm_studio.src.utils.local_model_utils import _looks_like_legacy_eva_model

    if not _looks_like_legacy_eva_model(model_path, config):
        return False
    if not _looks_like_dense_safetensors(model_path):
        return False

    config.pop("quantization_config", None)
    config.pop("_pre_quantization_dtype", None)

    rope_parameters = config.get("rope_parameters")
    if isinstance(rope_parameters, dict) and rope_parameters.get("rope_type") == "yarn":
        # Some inherited base configs also carried this non-YARN key. It is harmless
        # but newer Transformers warns about it on every rank.
        rope_parameters.pop("truncate", None)

    _write_config_atomically(config_path, config)
    logger.warning(
        "Repaired dense local EvaGPT config at %s: removed stale MXFP4 "
        "quantization metadata because the saved safetensor weights are dense.",
        config_path,
    )
    return True
