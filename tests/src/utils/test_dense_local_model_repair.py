import json

import torch
from safetensors.torch import save_file

from llm_studio.src.utils.dense_local_model_repair import (
    repair_stale_dense_eva_quantization_config,
)


def _write_config(model_dir):
    config = {
        "architectures": ["EvaGPTForCausalLM"],
        "model_type": "eva_gpt",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "rope_parameters": {
            "rope_type": "yarn",
            "factor": 2.0,
            "truncate": True,
        },
        "quantization_config": {
            "quant_method": "mxfp4",
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_repairs_stale_mxfp4_when_safetensors_are_dense(tmp_path):
    model_dir = tmp_path / "EvaGPT-Test"
    model_dir.mkdir()
    _write_config(model_dir)
    save_file(
        {"model.layers.0.weight": torch.ones(4, 4, dtype=torch.float32)},
        model_dir / "model.safetensors",
    )

    assert repair_stale_dense_eva_quantization_config(str(model_dir)) is True

    repaired = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    assert "quantization_config" not in repaired
    assert "truncate" not in repaired["rope_parameters"]


def test_does_not_repair_truly_quantized_weight_storage(tmp_path):
    model_dir = tmp_path / "EvaGPT-Quantized"
    model_dir.mkdir()
    _write_config(model_dir)
    save_file(
        {"model.layers.0.weight": torch.ones(4, 4, dtype=torch.uint8)},
        model_dir / "model.safetensors",
    )

    assert repair_stale_dense_eva_quantization_config(str(model_dir)) is False
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    assert config["quantization_config"]["quant_method"] == "mxfp4"


def test_does_not_touch_non_mxfp4_config(tmp_path):
    model_dir = tmp_path / "EvaGPT-Dense"
    model_dir.mkdir()
    _write_config(model_dir)
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["quantization_config"]["quant_method"] = "bitsandbytes"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    save_file(
        {"model.layers.0.weight": torch.ones(4, 4, dtype=torch.float32)},
        model_dir / "model.safetensors",
    )

    assert repair_stale_dense_eva_quantization_config(str(model_dir)) is False
