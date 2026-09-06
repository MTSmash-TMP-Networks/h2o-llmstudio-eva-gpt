import json

from transformers import AutoConfig

from llm_studio.src.utils.local_model_utils import ensure_local_eva_model_type


def _legacy_eva_config():
    return {
        "architectures": ["EvaGPTForCausalLM"],
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 131072,
        "sliding_window": 8192,
        "vocab_size": 32000,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }


def test_repairs_missing_model_type_and_unblocks_auto_config(tmp_path):
    model_dir = tmp_path / "EvaGPT-German-2B-Q22"
    model_dir.mkdir()
    config_path = model_dir / "config.json"
    config_path.write_text(json.dumps(_legacy_eva_config()), encoding="utf-8")

    assert ensure_local_eva_model_type(str(model_dir)) is True

    repaired = json.loads(config_path.read_text(encoding="utf-8"))
    assert repaired["model_type"] == "eva_gpt"

    config = AutoConfig.from_pretrained(str(model_dir))
    assert config.model_type == "eva_gpt"


def test_does_not_rewrite_unrelated_local_model(tmp_path):
    model_dir = tmp_path / "custom-model"
    model_dir.mkdir()
    config_path = model_dir / "config.json"
    config = _legacy_eva_config()
    config["architectures"] = ["CustomForCausalLM"]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert ensure_local_eva_model_type(str(model_dir)) is False
    unchanged = json.loads(config_path.read_text(encoding="utf-8"))
    assert "model_type" not in unchanged


def test_keeps_existing_model_type_unchanged(tmp_path):
    model_dir = tmp_path / "EvaGPT-existing"
    model_dir.mkdir()
    config_path = model_dir / "config.json"
    config = _legacy_eva_config()
    config["model_type"] = "llama"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert ensure_local_eva_model_type(str(model_dir)) is False
    unchanged = json.loads(config_path.read_text(encoding="utf-8"))
    assert unchanged["model_type"] == "llama"
