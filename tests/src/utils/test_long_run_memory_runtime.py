from types import SimpleNamespace

import numpy as np
import pytest

from llm_studio.src.utils import long_run_memory_runtime as runtime


def _cfg(*, problem_type="text_causal_language_modeling", rank=0):
    return SimpleNamespace(
        problem_type=problem_type,
        environment=SimpleNamespace(
            use_deepspeed=True,
            _local_rank=rank,
            _device="cpu",
        ),
    )


def test_prediction_text_is_converted_to_object_dtype(monkeypatch):
    long_text = "x" * 4096

    def original(self, output):
        return {"predicted_text": np.array([long_text, ""])}

    monkeypatch.setattr(runtime, "_ORIGINAL_POSTPROCESS_BATCH", original)

    result = runtime._postprocess_batch_predictions_object_strings(object(), {})

    assert result["predicted_text"].dtype == object
    assert result["predicted_text"].tolist() == [long_text, ""]


def test_causal_deepspeed_checkpoint_is_not_reloaded_into_cpu(monkeypatch, tmp_path):
    cfg = _cfg()
    calls = []

    class Model:
        def save_16bit_model(self, path, filename):
            calls.append((path, filename))
            return True

    monkeypatch.setattr(
        runtime,
        "_ORIGINAL_SAVE_CHECKPOINT",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("targeted Causal-LM path must not delegate")
        ),
    )
    monkeypatch.setattr(runtime, "_log_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "_release_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runtime.torch,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("saved checkpoint must not be torch.load'ed again")
        ),
    )

    runtime._save_checkpoint_low_memory(Model(), str(tmp_path), cfg)

    assert calls == [(str(tmp_path), "checkpoint.pth")]


def test_non_causal_checkpoint_keeps_original_path(monkeypatch, tmp_path):
    cfg = _cfg(problem_type="text_causal_regression_modeling")
    calls = []

    def original(**kwargs):
        calls.append(kwargs)
        return "saved"

    monkeypatch.setattr(runtime, "_ORIGINAL_SAVE_CHECKPOINT", original)

    result = runtime._save_checkpoint_low_memory(
        model="model",
        path=str(tmp_path),
        cfg=cfg,
    )

    assert result == "saved"
    assert calls == [{"model": "model", "path": str(tmp_path), "cfg": cfg}]


def test_validation_cleanup_runs_after_success(monkeypatch):
    cfg = _cfg()
    events = []

    def original(*args, **kwargs):
        events.append("eval")
        return 1.25, 2.5

    monkeypatch.setattr(runtime, "_ORIGINAL_RUN_EVAL", original)
    monkeypatch.setattr(runtime, "_log_memory", lambda cfg, stage: events.append(stage))
    monkeypatch.setattr(
        runtime,
        "_release_memory",
        lambda cfg, stage: events.append(stage),
    )

    result = runtime._run_eval_with_memory_cleanup(cfg=cfg, model=None)

    assert result == (1.25, 2.5)
    assert events == ["before validation", "eval", "after validation cleanup"]


def test_validation_cleanup_runs_after_exception(monkeypatch):
    cfg = _cfg()
    released = []

    def original(*args, **kwargs):
        raise RuntimeError("validation failed")

    monkeypatch.setattr(runtime, "_ORIGINAL_RUN_EVAL", original)
    monkeypatch.setattr(runtime, "_log_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runtime,
        "_release_memory",
        lambda cfg, stage: released.append(stage),
    )

    with pytest.raises(RuntimeError, match="validation failed"):
        runtime._run_eval_with_memory_cleanup(cfg=cfg, model=None)

    assert released == ["after validation cleanup"]
