from dataclasses import fields

from llm_studio.app_utils.config import default_cfg
from llm_studio.python_configs.base import DefaultConfig
from llm_studio.python_configs.text_causal_language_modeling_config import (
    ConfigNLPCausalLMDataset,
    ConfigProblemBase,
)
from llm_studio.src.utils.config_utils import load_config_yaml, save_config_yaml


def test_train_text_column_is_loaded_from_dataset_configuration():
    assert "train_text_column" in default_cfg.dataset_keys


def test_train_text_column_refreshes_dataset_configuration():
    assert "train_text_column" in default_cfg.dataset_trigger_keys


def test_train_text_column_is_a_real_dataclass_field():
    field_names = {field.name for field in fields(ConfigNLPCausalLMDataset)}

    assert "train_text_column" in field_names
    assert "train_text_column" not in DefaultConfig.__annotations__


def test_train_text_column_can_be_disabled_in_constructor():
    cfg = ConfigNLPCausalLMDataset(train_text_column=False)

    assert cfg.train_text_column is False


def test_old_dataset_config_without_train_text_column_uses_default():
    cfg = ConfigNLPCausalLMDataset.from_dict({})

    assert cfg.train_text_column is True


def test_dataset_config_preserves_explicit_false():
    cfg = ConfigNLPCausalLMDataset.from_dict({"train_text_column": False})

    assert cfg.train_text_column is False


def test_train_text_column_false_survives_yaml_roundtrip(tmp_path):
    cfg = ConfigProblemBase(
        llm_backbone="unit-test",
        dataset=ConfigNLPCausalLMDataset(train_text_column=False),
    )

    config_path = tmp_path / "cfg.yaml"
    save_config_yaml(config_path, cfg)
    loaded_cfg = load_config_yaml(config_path)

    assert loaded_cfg.dataset.train_text_column is False
