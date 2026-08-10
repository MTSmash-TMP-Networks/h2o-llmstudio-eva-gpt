import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from llm_studio.src.utils.early_stop import (
    CHECKPOINT_MANIFEST_FILENAME,
    DEEPSPEED_RESUME_DIRNAME,
    EARLY_STOP_CHECKPOINT_DIRNAME,
    EARLY_STOP_POINTER_FILENAME,
    TRAINER_STATE_FILENAME,
    get_early_stop_checkpoint_path,
    is_clean_accumulation_boundary,
    is_early_stop_requested,
    save_early_stop_checkpoint,
)


def _cfg(tmp_path, *, save_checkpoint="last", use_deepspeed=False):
    return SimpleNamespace(
        output_directory=str(tmp_path),
        environment=SimpleNamespace(
            use_deepspeed=use_deepspeed,
            _distributed=False,
            _local_rank=0,
            _rank=0,
            _device="cpu",
            _curr_step=42,
        ),
        training=SimpleNamespace(
            save_checkpoint=save_checkpoint,
            lora=False,
            lora_unfreeze_layers=(),
        ),
    )


def _optimizer_and_scheduler(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    return optimizer, scheduler


def _load_training_state(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def test_early_stop_saves_model_optimizer_scheduler_scaler_and_progress(tmp_path):
    cfg = _cfg(tmp_path)
    model = torch.nn.Linear(3, 2)
    optimizer, scheduler = _optimizer_and_scheduler(model)
    scaler = MagicMock()
    scaler.state_dict.return_value = {"scale": 1024.0}
    stop_path = tmp_path / "stop_training"
    stop_path.touch()

    checkpoint_path = save_early_stop_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        cfg=cfg,
        epoch=2,
        iteration=7,
        best_val_metric=0.25,
        stop_training_path=str(stop_path),
    )

    assert checkpoint_path == str(tmp_path)
    assert (tmp_path / "checkpoint.pth").exists()
    assert (tmp_path / TRAINER_STATE_FILENAME).exists()
    assert (tmp_path / CHECKPOINT_MANIFEST_FILENAME).exists()
    assert not stop_path.exists()

    state = _load_training_state(tmp_path / TRAINER_STATE_FILENAME)
    assert state["epoch"] == 2
    assert state["iteration"] == 7
    assert state["current_step"] == 42
    assert state["best_val_metric"] == 0.25
    assert state["optimizer"] is not None
    assert state["scheduler"] is not None
    assert state["scaler"] == {"scale": 1024.0}
    assert "python_random_state" in state
    assert "numpy_random_state" in state
    assert "torch_random_state" in state

    manifest = json.loads(
        (tmp_path / CHECKPOINT_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["complete"] is True
    assert manifest["model_checkpoint"] == "checkpoint.pth"
    assert manifest["training_state"] == TRAINER_STATE_FILENAME


def test_early_stop_preserves_existing_best_checkpoint(tmp_path):
    cfg = _cfg(tmp_path, save_checkpoint="best")
    root_checkpoint = tmp_path / "checkpoint.pth"
    root_checkpoint.write_bytes(b"validated-best-model")
    model = torch.nn.Linear(3, 2)
    optimizer, scheduler = _optimizer_and_scheduler(model)
    stop_path = tmp_path / "stop_training"
    stop_path.touch()

    expected_path = tmp_path / EARLY_STOP_CHECKPOINT_DIRNAME
    assert get_early_stop_checkpoint_path(cfg) == str(expected_path)

    checkpoint_path = save_early_stop_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        cfg=cfg,
        epoch=1,
        iteration=4,
        best_val_metric=0.12,
        stop_training_path=str(stop_path),
    )

    assert checkpoint_path == str(expected_path)
    assert root_checkpoint.read_bytes() == b"validated-best-model"
    assert (expected_path / "checkpoint.pth").exists()
    assert (expected_path / TRAINER_STATE_FILENAME).exists()
    assert (
        tmp_path / EARLY_STOP_POINTER_FILENAME
    ).read_text(encoding="utf-8").strip() == EARLY_STOP_CHECKPOINT_DIRNAME


def test_best_mode_uses_root_checkpoint_as_fallback_before_first_validation(tmp_path):
    cfg = _cfg(tmp_path, save_checkpoint="best")
    assert get_early_stop_checkpoint_path(cfg) == str(tmp_path)


def test_distributed_stop_decision_is_synchronized_across_ranks(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.environment._distributed = True
    cfg.environment._rank = 1
    cfg.environment._local_rank = 1
    missing_marker = tmp_path / "stop_training"

    def mark_requested(tensor, op=None):
        tensor.fill_(1)

    with (
        patch.object(torch.distributed, "is_available", return_value=True),
        patch.object(torch.distributed, "is_initialized", return_value=True),
        patch.object(torch.distributed, "get_backend", return_value="gloo"),
        patch.object(torch.distributed, "all_reduce", side_effect=mark_requested),
    ):
        assert is_early_stop_requested(str(missing_marker), cfg) is True


def test_stop_waits_for_clean_gradient_accumulation_boundary():
    assert is_clean_accumulation_boundary(0, 4)
    assert not is_clean_accumulation_boundary(1, 4)
    assert not is_clean_accumulation_boundary(2, 4)
    assert not is_clean_accumulation_boundary(3, 4)
    assert is_clean_accumulation_boundary(4, 4)
    assert is_clean_accumulation_boundary(5, 1)


def test_deepspeed_early_stop_keeps_native_resume_checkpoint(tmp_path):
    cfg = _cfg(tmp_path, use_deepspeed=True)
    model = MagicMock()
    stop_path = tmp_path / "stop_training"
    stop_path.touch()

    with patch("llm_studio.src.utils.early_stop.save_checkpoint") as save_model:
        save_early_stop_checkpoint(
            model=model,
            optimizer=MagicMock(),
            scheduler=MagicMock(),
            scaler=None,
            cfg=cfg,
            epoch=3,
            iteration=9,
            best_val_metric=None,
            stop_training_path=str(stop_path),
        )

    save_model.assert_called_once()
    model.save_checkpoint.assert_called_once()
    args, kwargs = model.save_checkpoint.call_args
    assert args[0] == os.path.join(str(tmp_path), DEEPSPEED_RESUME_DIRNAME)
    assert kwargs["client_state"]["epoch"] == 3

    manifest = json.loads(
        (tmp_path / CHECKPOINT_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["training_state"] == DEEPSPEED_RESUME_DIRNAME


def test_train_source_uses_synchronized_complete_early_stop_path():
    source = Path("llm_studio/train.py").read_text(encoding="utf-8")
    ast.parse(source)

    assert "is_early_stop_requested(" in source
    assert "is_clean_accumulation_boundary(" in source
    assert "save_early_stop_checkpoint(" in source
    assert "optimizer=optimizer" in source
    assert "scheduler=scheduler" in source
    assert "scaler=scaler" in source
    assert "os.remove(stop_training_path)" not in source
