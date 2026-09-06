"""Helpers for local model metadata created by MaTeLiX AI Studio."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_EVA_MODEL_TYPE = "eva_gpt"
_EVA_CONFIG_KEYS = {
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
}


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _looks_like_legacy_eva_model(model_path: str, config: dict[str, Any]) -> bool:
    """Return True only when a local config is strongly identifiable as EvaGPT."""
    architectures = config.get("architectures") or []
    if isinstance(architectures, str):
        architectures = [architectures]

    architecture_match = any(
        "evagpt" in _normalized_name(str(architecture))
        for architecture in architectures
    )
    if architecture_match:
        return True

    model_name = _normalized_name(Path(model_path).name)
    name_match = "evagpt" in model_name
    structural_match = _EVA_CONFIG_KEYS.issubset(config)
    eva_attention_config = any(
        key in config for key in ("layer_types", "rope_parameters", "sliding_window")
    )
    return name_match and structural_match and eva_attention_config


def ensure_local_eva_model_type(model_path: str) -> bool:
    """Add a missing ``model_type`` to legacy local EvaGPT configs.

    Older models created by the MaTeLiX/EvaGPT model builder can contain a valid
    tokenizer and EvaGPT architecture settings while their ``config.json`` lacks
    Hugging Face's required ``model_type`` discriminator. ``AutoTokenizer`` and
    ``AutoConfig`` then reject the directory before training starts.

    The repair is intentionally restricted to local directories with strong EvaGPT
    signals. Remote Hugging Face model ids and unrelated local models are untouched.
    The update is written atomically so multiple distributed ranks may call this
    helper at startup without exposing a partially-written JSON file.
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
    if config.get("model_type"):
        return False
    if not _looks_like_legacy_eva_model(model_path, config):
        return False

    config["model_type"] = _EVA_MODEL_TYPE
    temporary_path = config_path.with_name(f"{config_path.name}.tmp.{os.getpid()}")
    try:
        with temporary_path.open("w", encoding="utf-8") as config_file:
            json.dump(config, config_file, indent=2, ensure_ascii=False)
            config_file.write("\n")
        os.replace(temporary_path, config_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    logger.warning(
        "Repaired legacy local EvaGPT config at %s by adding model_type=%s.",
        config_path,
        _EVA_MODEL_TYPE,
    )
    return True
