"""Reduce transient host RAM while loading dense DeepSpeed backbone replicas."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_INSTALLED = False


def install_dense_backbone_low_memory() -> None:
    """Add ``low_cpu_mem_usage`` to the existing DeepSpeed dtype proxy.

    ``v100_precision`` already wraps the Hugging Face model factory so a full-weight
    DeepSpeed run can materialize its runtime replica in FP16.  Add Transformers'
    low-CPU-memory loading mode to that same proxy so loading dense local safetensors
    does not first create another complete CPU state-dict copy on every rank.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from llm_studio.src.utils import v100_precision

    original_factory = v100_precision._dtype_overridden_model_class
    if getattr(original_factory, "_llmstudio_low_cpu_memory", False):
        _INSTALLED = True
        return

    def low_memory_factory(model_class: Any, runtime_dtype: Any):
        runtime_class = original_factory(model_class, runtime_dtype)

        class LowMemoryRuntimeModelClass:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                kwargs.setdefault("low_cpu_mem_usage", True)
                return runtime_class.from_pretrained(*args, **kwargs)

            @staticmethod
            def from_config(*args, **kwargs):
                return runtime_class.from_config(*args, **kwargs)

        return LowMemoryRuntimeModelClass

    low_memory_factory._llmstudio_low_cpu_memory = True
    v100_precision._dtype_overridden_model_class = low_memory_factory
    _INSTALLED = True
    logger.info(
        "DeepSpeed dense-backbone loader will use Transformers low_cpu_mem_usage "
        "to avoid a second full CPU state-dict copy per rank."
    )
