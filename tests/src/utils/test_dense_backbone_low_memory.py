import torch

from llm_studio.src.utils import dense_backbone_low_memory, v100_precision


def test_runtime_proxy_enables_low_cpu_mem_usage(monkeypatch):
    calls = []

    def fake_dtype_factory(model_class, runtime_dtype):
        class RuntimeClass:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                calls.append(kwargs.copy())
                return "loaded"

            @staticmethod
            def from_config(*args, **kwargs):
                return "configured"

        return RuntimeClass

    monkeypatch.setattr(
        v100_precision, "_dtype_overridden_model_class", fake_dtype_factory
    )
    monkeypatch.setattr(dense_backbone_low_memory, "_INSTALLED", False)

    dense_backbone_low_memory.install_dense_backbone_low_memory()
    runtime_class = v100_precision._dtype_overridden_model_class(object, torch.float16)

    assert runtime_class.from_pretrained("model") == "loaded"
    assert calls[-1]["low_cpu_mem_usage"] is True
    assert runtime_class.from_config({}) == "configured"
