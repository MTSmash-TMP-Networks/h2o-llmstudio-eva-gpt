from types import SimpleNamespace

import torch

from llm_studio.src.utils.v100_precision import (
    build_deepspeed_config,
    install_precision_runtime_patch,
    normalize_training_precision,
)


def _cfg(
    *,
    backbone_dtype="float16",
    mixed_precision=True,
    mixed_precision_dtype="float16",
    lora=False,
):
    return SimpleNamespace(
        architecture=SimpleNamespace(
            backbone_dtype=backbone_dtype,
            pretrained=False,
        ),
        environment=SimpleNamespace(
            gpus=("0", "1", "2", "3"),
            mixed_precision=mixed_precision,
            mixed_precision_dtype=mixed_precision_dtype,
            deepspeed_method="ZeRO2",
            deepspeed_reduce_bucket_size=1_000_000,
            deepspeed_allgather_bucket_size=1_000_000,
            deepspeed_stage3_prefetch_bucket_size=1_000_000,
            deepspeed_stage3_param_persistence_threshold=100_000,
        ),
        training=SimpleNamespace(
            lora=lora,
            epochs=1,
            batch_size=2,
            grad_accumulation=4,
            gradient_clip=1.0,
        ),
    )


def _mock_v100(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (7, 0))
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda index: "Tesla V100-SXM2-32GB"
    )


def _mock_ampere(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "NVIDIA A100")


def test_full_float16_training_is_promoted_to_fp32_weights_with_fp16_amp():
    cfg = _cfg(
        backbone_dtype="float16",
        mixed_precision=False,
        mixed_precision_dtype="bfloat16",
        lora=False,
    )

    normalize_training_precision(cfg)

    assert cfg.architecture.backbone_dtype == "float32"
    assert cfg.environment.mixed_precision is True
    assert cfg.environment.mixed_precision_dtype == "float16"


def test_v100_falls_back_from_bfloat16_amp_to_float16(monkeypatch):
    _mock_v100(monkeypatch)
    cfg = _cfg(
        backbone_dtype="float32",
        mixed_precision=True,
        mixed_precision_dtype="bfloat16",
    )

    normalize_training_precision(cfg)

    assert cfg.architecture.backbone_dtype == "float32"
    assert cfg.environment.mixed_precision is True
    assert cfg.environment.mixed_precision_dtype == "float16"


def test_v100_converts_bfloat16_full_backbone_to_safe_fp16_amp(monkeypatch):
    _mock_v100(monkeypatch)
    cfg = _cfg(
        backbone_dtype="bfloat16",
        mixed_precision=False,
        mixed_precision_dtype="bfloat16",
        lora=False,
    )

    normalize_training_precision(cfg)

    assert cfg.architecture.backbone_dtype == "float32"
    assert cfg.environment.mixed_precision is True
    assert cfg.environment.mixed_precision_dtype == "float16"


def test_ampere_keeps_bfloat16_mixed_precision(monkeypatch):
    _mock_ampere(monkeypatch)
    cfg = _cfg(
        backbone_dtype="float32",
        mixed_precision=True,
        mixed_precision_dtype="bfloat16",
    )

    normalize_training_precision(cfg)

    assert cfg.architecture.backbone_dtype == "float32"
    assert cfg.environment.mixed_precision_dtype == "bfloat16"


def test_lora_float16_backbone_is_not_promoted_to_float32():
    cfg = _cfg(
        backbone_dtype="float16",
        mixed_precision=True,
        mixed_precision_dtype="float16",
        lora=True,
    )

    normalize_training_precision(cfg)

    assert cfg.architecture.backbone_dtype == "float16"
    assert cfg.environment.mixed_precision_dtype == "float16"


def test_deepspeed_uses_fp16_amp_with_float32_backbone():
    cfg = _cfg(
        backbone_dtype="float32",
        mixed_precision=True,
        mixed_precision_dtype="float16",
    )

    ds_config = build_deepspeed_config(cfg)

    assert ds_config["fp16"]["enabled"] is True
    assert ds_config["bf16"]["enabled"] is False
    assert ds_config["gradient_accumulation_steps"] == 4
    assert ds_config["zero_optimization"]["stage"] == 2


def test_deepspeed_uses_bfloat16_when_amp_requests_it():
    cfg = _cfg(
        backbone_dtype="float32",
        mixed_precision=True,
        mixed_precision_dtype="bfloat16",
    )

    ds_config = build_deepspeed_config(cfg)

    assert ds_config["fp16"]["enabled"] is False
    assert ds_config["bf16"]["enabled"] is True


def test_runtime_patch_replaces_legacy_deepspeed_dtype_selection():
    from llm_studio.src.utils import modeling_utils

    install_precision_runtime_patch()

    assert modeling_utils.get_ds_config is build_deepspeed_config
