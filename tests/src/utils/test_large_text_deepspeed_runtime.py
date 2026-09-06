from types import SimpleNamespace

import pandas as pd
import torch

from llm_studio.src.utils import large_text_deepspeed_runtime as runtime


def _cfg(*, workers=8):
    return SimpleNamespace(
        environment=SimpleNamespace(
            number_of_workers=workers,
            use_deepspeed=True,
            _distributed=True,
            _world_size=4,
            _local_rank=0,
        ),
        training=SimpleNamespace(lora=False),
        architecture=SimpleNamespace(pretrained_weights=None),
    )


def _marked_dataset():
    dataset = SimpleNamespace()
    setattr(dataset, runtime._RANK_PARTITIONED_MARKER, True)
    return dataset


def test_rank_partitioned_train_loader_forces_zero_workers_and_restores_cfg():
    cfg = _cfg(workers=8)
    dataset = _marked_dataset()
    observed = {}

    def original(*, train_ds, cfg):
        observed["workers"] = cfg.environment.number_of_workers
        return SimpleNamespace(dataset=train_ds, num_workers=observed["workers"])

    loader = runtime._force_zero_workers(original, dataset, cfg, kind="train")

    assert observed["workers"] == 0
    assert loader.num_workers == 0
    assert cfg.environment.number_of_workers == 8


def test_validation_dataset_inherits_rank_partition_marker(monkeypatch):
    cfg = _cfg()
    val_df = pd.DataFrame({"Text": ["a", "b"]})
    val_df.attrs[runtime._RANK_PARTITIONED_MARKER] = True
    built_dataset = SimpleNamespace()

    monkeypatch.setattr(
        runtime,
        "_ORIGINAL_GET_VAL_DATASET",
        lambda *, val_df, cfg: built_dataset,
    )

    result = runtime._get_val_dataset_rank_marker(val_df=val_df, cfg=cfg)

    assert result is built_dataset
    assert getattr(result, runtime._RANK_PARTITIONED_MARKER) is True


def test_rank_partitioned_deepspeed_keeps_existing_loaders(monkeypatch):
    cfg = _cfg()
    train_loader = SimpleNamespace(dataset=_marked_dataset())
    val_loader = SimpleNamespace(dataset=_marked_dataset())
    optimizer = object()
    scheduler = object()
    model = SimpleNamespace(backbone=object(), deepspeed_initialized=False)

    def init_deepspeed():
        model.deepspeed_initialized = True

    model.init_deepspeed = init_deepspeed
    captured = {}

    def fake_initialize(**kwargs):
        captured.update(kwargs)
        return "engine", "wrapped-optimizer", None, "wrapped-scheduler"

    from llm_studio.src.utils import modeling_utils

    monkeypatch.setattr(modeling_utils, "get_ds_config", lambda cfg: {"zero": 2})
    monkeypatch.setattr(runtime.deepspeed, "initialize", fake_initialize)
    monkeypatch.setattr(
        runtime,
        "_ORIGINAL_WRAP_MODEL_DISTRIBUTED",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("fallback called")),
    )

    result = runtime._wrap_model_distributed_low_memory(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        cfg=cfg,
    )

    assert captured["training_data"] is None
    assert captured["config_params"] == {"zero": 2}
    assert result[2] is train_loader
    assert result[3] is val_loader
    assert result[1] == "wrapped-optimizer"
    assert result[4] == "wrapped-scheduler"
    assert model.backbone == "engine"
    assert model.deepspeed_initialized is True


def test_unpartitioned_training_uses_original_distributed_wrapper(monkeypatch):
    cfg = _cfg()
    train_loader = SimpleNamespace(dataset=SimpleNamespace())
    val_loader = SimpleNamespace(dataset=SimpleNamespace())
    expected = ("model", "optimizer", "train", "val", "scheduler")
    captured = {}

    def original(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(runtime, "_ORIGINAL_WRAP_MODEL_DISTRIBUTED", original)

    result = runtime._wrap_model_distributed_low_memory(
        model="model",
        optimizer="optimizer",
        lr_scheduler="scheduler",
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        cfg=cfg,
    )

    assert result == expected
    assert captured["train_dataloader"] is train_loader


def test_large_distributed_checkpoint_uses_mmap(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.write_bytes(b"checkpoint-placeholder")
    cfg = _cfg()
    cfg.architecture.pretrained_weights = str(checkpoint_path)
    calls = []

    def fake_torch_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return {"model": {}}

    def original_loader(*, cfg, model, strict, weights_path):
        torch.load(
            weights_path or cfg.architecture.pretrained_weights,
            map_location="cpu",
        )
        return "loaded"

    monkeypatch.setattr(runtime, "_LARGE_CHECKPOINT_BYTES", 0)
    monkeypatch.setattr(runtime, "_ORIGINAL_LOAD_CHECKPOINT", original_loader)
    monkeypatch.setattr(torch, "load", fake_torch_load)

    result = runtime._load_checkpoint_low_memory(
        cfg=cfg,
        model=SimpleNamespace(),
        strict=True,
        weights_path=None,
    )

    assert result == "loaded"
    assert calls
    assert calls[0]["mmap"] is True
