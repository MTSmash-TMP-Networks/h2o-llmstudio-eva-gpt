from types import SimpleNamespace

import torch

from llm_studio.src.utils.v100_precision import (
    _deepspeed_runtime_load_dtype,
    _dtype_overridden_model_class,
    build_deepspeed_config,
    deepspeed_owns_mixed_precision,
    install_precision_runtime_patch,
    normalize_training_precision,
)


def _cfg(
    *,
    backbone_dtype="float16",
    mixed_precision=True,
    mixed_precision_dtype="float16",
    lora=False,
    use_deepspeed=False,
    problem_type="text_causal_language_modeling",
):
    return SimpleNamespace(
        problem_type=problem_type,
        architecture=SimpleNamespace(
            backbone_dtype=backbone_dtype,
            pretrained=False,
        ),
        environment=SimpleNamespace(
            gpus=("0", "1", "2", "3"),
            mixed_precision=mixed_precision,
            mixed_precision_dtype=mixed_precision_dtype,
            use_deepspeed=use_deepspeed,
            deepspeed_method="ZeRO2",
            deepspeed_reduce_bucket_size=1_000_000,
            deepspeed_allgather_bucket_size=1_000_000,
            deepspeed_stage3_prefetch_bucket_size=1_000_000,
            deepspeed_stage3_param_persistence_threshold=100_000,
            _local_rank=0,
            _device="cpu",
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
    assert deepspeed_owns_mixed_precision() is False


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
        use_deepspeed=True,
    )

    normalize_training_precision(cfg)
    ds_config = build_deepspeed_config(cfg)

    assert deepspeed_owns_mixed_precision() is True
    assert ds_config["fp16"]["enabled"] is True
    assert ds_config["fp16"]["loss_scale"] == 0
    assert ds_config["fp16"]["min_loss_scale"] == 1
    assert ds_config["bf16"]["enabled"] is False
    assert ds_config["gradient_accumulation_steps"] == 4
    assert ds_config["zero_optimization"]["stage"] == 2


def test_deepspeed_uses_bfloat16_when_amp_requests_it():
    cfg = _cfg(
        backbone_dtype="float32",
        mixed_precision=True,
        mixed_precision_dtype="bfloat16",
        use_deepspeed=True,
    )

    normalize_training_precision(cfg)
    ds_config = build_deepspeed_config(cfg)

    assert deepspeed_owns_mixed_precision() is True
    assert ds_config["fp16"]["enabled"] is False
    assert ds_config["bf16"]["enabled"] is True


def test_deepspeed_low_memory_replica_uses_fp16_with_fp32_policy():
    cfg = _cfg(
        backbone_dtype="float32",
        mixed_precision=True,
        mixed_precision_dtype="float16",
        use_deepspeed=True,
    )

    normalize_training_precision(cfg)

    assert cfg.architecture.backbone_dtype == "float32"
    assert _deepspeed_runtime_load_dtype(cfg) is torch.float16


def test_deepspeed_low_memory_replica_is_limited_to_full_weight_causal_lm():
    lora_cfg = _cfg(
        backbone_dtype="float32",
        mixed_precision=True,
        use_deepspeed=True,
        lora=True,
    )
    other_problem_cfg = _cfg(
        backbone_dtype="float32",
        mixed_precision=True,
        use_deepspeed=True,
        problem_type="text_dpo_modeling",
    )

    normalize_training_precision(lora_cfg)
    assert _deepspeed_runtime_load_dtype(lora_cfg) is None

    normalize_training_precision(other_problem_cfg)
    assert _deepspeed_runtime_load_dtype(other_problem_cfg) is None


def test_runtime_dtype_proxy_overrides_hf_factory_dtype():
    class DummyFactory:
        pretrained_kwargs = None
        config_kwargs = None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.pretrained_kwargs = kwargs
            return "pretrained"

        @classmethod
        def from_config(cls, *args, **kwargs):
            cls.config_kwargs = kwargs
            return "config"

    proxy = _dtype_overridden_model_class(DummyFactory, torch.float16)

    assert proxy.from_pretrained("model", torch_dtype=torch.float32) == "pretrained"
    assert proxy.from_config(object(), torch_dtype=torch.float32) == "config"
    assert DummyFactory.pretrained_kwargs["torch_dtype"] is torch.float16
    assert DummyFactory.config_kwargs["torch_dtype"] is torch.float16


def test_deepspeed_outer_model_to_is_left_to_engine():
    class DummyOuterModel(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.real_to_calls = 0

        def to(self, *args, **kwargs):
            self.real_to_calls += 1
            return super().to(*args, **kwargs)

    cfg = _cfg(
        backbone_dtype="float32",
        mixed_precision=True,
        mixed_precision_dtype="float16",
        use_deepspeed=True,
    )
    cfg.architecture.model_class = DummyOuterModel

    normalize_training_precision(cfg)
    model = DummyOuterModel(cfg)
    returned = model.to("meta")

    assert returned is model
    assert model.real_to_calls == 0
    assert model.weight.device.type == "cpu"


def test_deepspeed_disables_outer_cuda_autocast():
    from torch.cuda.amp import autocast

    install_precision_runtime_patch()
    cfg = _cfg(
        backbone_dtype="float32",
        mixed_precision=True,
        mixed_precision_dtype="float16",
        use_deepspeed=True,
    )
    normalize_training_precision(cfg)

    context = autocast(enabled=True)

    assert context._enabled is False


def test_ddp_keeps_outer_cuda_autocast_available():
    from torch.cuda.amp import autocast

    install_precision_runtime_patch()
    cfg = _cfg(
        backbone_dtype="float32",
        mixed_precision=True,
        mixed_precision_dtype="float16",
        use_deepspeed=False,
    )
    normalize_training_precision(cfg)

    context = autocast(enabled=False)

    assert deepspeed_owns_mixed_precision() is False
    assert context._enabled is False


def test_runtime_patch_replaces_legacy_deepspeed_dtype_selection():
    from llm_studio.src.utils import modeling_utils

    install_precision_runtime_patch()

    assert modeling_utils.get_ds_config is build_deepspeed_config
