from llm_studio.app_utils.config import default_cfg
from llm_studio.python_configs.text_causal_language_modeling_config import (
    ConfigNLPCausalLMDataset,
    ConfigProblemBase,
)
from llm_studio.src.utils.config_utils import load_config_yaml, save_config_yaml


def test_train_text_column_is_loaded_from_dataset_configuration():
    assert "train_text_column" in default_cfg.dataset_keys


def test_train_text_column_refreshes_dataset_configuration():
    assert "train_text_column" in default_cfg.dataset_trigger_keys


def test_train_text_column_false_survives_yaml_roundtrip(tmp_path):
    cfg = ConfigProblemBase(
        llm_backbone="unit-test",
        dataset=ConfigNLPCausalLMDataset(),
    )
    cfg.dataset.train_text_column = False

    config_path = tmp_path / "cfg.yaml"
    save_config_yaml(config_path, cfg)
    loaded_cfg = load_config_yaml(config_path)

    assert loaded_cfg.dataset.train_text_column is False
