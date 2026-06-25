import logging
import multiprocessing
import os
from dataclasses import dataclass, field
from typing import Any

import torch

import llm_studio.src.datasets.mixed_text_causal_language_modeling_ds
from llm_studio.app_utils.config import default_cfg
from llm_studio.python_configs.base import DefaultConfig, DefaultConfigProblemBase
from llm_studio.src import possible_values
from llm_studio.src.augmentations.nlp_aug import BaseNLPAug
from llm_studio.src.loggers import ExternalLoggers
from llm_studio.src.losses import text_causal_language_modeling_losses
from llm_studio.src.metrics import text_causal_language_modeling_metrics
from llm_studio.src.models import text_causal_language_modeling_model
from llm_studio.src.nesting import Dependency
from llm_studio.src.optimizers import Optimizers
from llm_studio.src.plots import text_causal_language_modeling_plots
from llm_studio.src.schedulers import Schedulers
from llm_studio.src.utils.data_utils import sanity_check
from llm_studio.src.utils.modeling_utils import generate_experiment_name

logger = logging.getLogger(__name__)


@dataclass
class ConfigNLPCausalLMDataset(DefaultConfig):
    dataset_class: Any = (
        llm_studio.src.datasets.mixed_text_causal_language_modeling_ds.CustomDataset
    )

    personalize: bool = False
    chatbot_name: str = "h2oGPT"
    chatbot_author: str = "MaTeLiX AI"

    train_dataframe: str = "/path/to/train.csv"
    validation_strategy: str = "automatic"
    validation_dataframe: str = ""
    validation_size: float = 0.01

    data_sample: float = 1.0
    data_sample_choice: tuple[str, ...] = ("Train", "Validation")

    system_column: str = "system"
    prompt_column: tuple[str, ...] = ("instruction", "input")
    prompt_column_separator: str = "\\n\\n"
    answer_column: str = "output"
    parent_id_column: str = "parent_id"
    id_column: str = "id"

    text_system_start: str = "<|system|>"
    text_prompt_start: str = "<|prompt|>"
    text_answer_separator: str = "<|answer|>"

    add_eos_token_to_system: bool = True
    add_eos_token_to_prompt: bool = True
    add_eos_token_to_answer: bool = True
    limit_chained_samples: bool = False
    mask_prompt_labels: bool = True
    mask_prompt_user_text_only: bool = True
    only_last_answer: bool = False

    _allowed_file_extensions: tuple[str, ...] = (
        "csv",
        "CSV",
        "pq",
        "PQ",
        "parquet",
        "PARQUET",
    )

    def __post_init__(self):
        self.prompt_column = (
            tuple(
                self.prompt_column,
            )
            if isinstance(self.prompt_column, str)
            else tuple(self.prompt_column)
        )
        super().__post_init__()

        self._possible_values["train_dataframe"] = possible_values.Files(
            prefer_with=lambda path: "train" in path
        )
        self._possible_values["validation_strategy"] = possible_values.String(
            values=(
                ("custom", "Custom holdout validation"),
                ("automatic", "Automatic holdout validation"),
            ),
            allow_custom=False,
        )
        self._possible_values["validation_dataframe"] = possible_values.Files(
            add_none=True, prefer_with=lambda path: "val" in path
        )
        self._possible_values["validation_size"] = (0.01, 0.95, 0.01)
        self._possible_values["data_sample"] = (0.01, 1, 0.01)
        self._possible_values["data_sample_choice"] = ["Train", "Validation"]
        self._possible_values["system_column"] = possible_values.Columns(
            prefer_with=lambda column: column in ("system",), add_none=True
        )
        self._possible_values["prompt_column"] = possible_values.Columns(
            prefer_with=lambda column: (
                column in ("instruction", "prompt", "question", "input", "user")
            )
        )
        self._possible_values["answer_column"] = possible_values.Columns(
            prefer_with=lambda column: (
                column in ("answer", "output", "response", "assistant", "chosen")
            )
        )
        self._possible_values["parent_id_column"] = possible_values.Columns(
            prefer_with=lambda column: column in ("parent", "parent_id"), add_none=True
        )

        self._possible_values["id_column"] = possible_values.Columns(
            prefer_with=lambda column: column in ("id", "ID", "index"), add_none=True
        )

        self._nesting.add(
            ["chatbot_name", "chatbot_author"],
            [Dependency(key="personalize", value=True, is_set=True)],
        )

        self._nesting.add(
            ["validation_dataframe"],
            [Dependency(key="validation_strategy", value="custom", is_set=True)],
        )

        self._nesting.add(
            ["validation_size"],
            [Dependency(key="validation_strategy", value="automatic", is_set=True)],
        )

        self._nesting.add(
            ["data_sample_choice"],
            [Dependency(key="data_sample", value=1, is_set=False)],
        )

        self._nesting.add(
            ["limit_chained_samples"],
            [Dependency(key="parent_id_column", value="None", is_set=False)],
        )

        self._nesting.add(
            ["id_column"],
            [Dependency(key="parent_id_column", value="None", is_set=False)],
        )

        self._nesting.add(
            ["text_system_start", "add_eos_token_to_system"],
            [Dependency(key="system_column", value="None", is_set=False)],
        )

        self._nesting.add(
            ["mask_prompt_user_text_only"],
            [Dependency(key="mask_prompt_labels", value=True, is_set=True)],
        )

        self._nesting.add(
            ["only_last_answer"],
            [
                Dependency(key="parent_id_column", value="None", is_set=False),
                Dependency(key="mask_prompt_labels", value=True, is_set=True),
            ],
        )

        self._visibility["dataset_class"] = -1


@dataclass
class ConfigNLPCausalLMTraining(DefaultConfig):
    loss_class: Any = text_causal_language_modeling_losses.Losses
    loss_function: str = "TokenAveragedCrossEntropy"
    optimizer: str = "AdamW"

    learning_rate: float = 0.0001
    differential_learning_rate_layers: tuple[str, ...] = ()
    differential_learning_rate: float = 0.00001
    freeze_layers: tuple[str, ...] = ()

    attention_implementation: str = "auto"
    batch_size: int = 2
    drop_last_batch: bool = True
    epochs: int = 1
    schedule: str = "Cosine"
    min_learning_rate_ratio: float = 0.0
    warmup_epochs: float = 0.0

    weight_decay: float = 0.0
    gradient_clip: float = 0.0
    grad_accumulation: int = 1

    lora: bool = True
    use_dora: bool = False
    lora_r: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    use_rslora: bool = False
    lora_target_modules: str = ""
    lora_unfreeze_layers: tuple[str, ...] = ()

    save_checkpoint: str = "last"
    evaluation_epochs: float = 1.0
    evaluate_before_training: bool = False
    train_validation_data: bool = False

    def __post_init__(self):
        super().__post_init__()
        self._possible_values["loss_function"] = self.loss_class.names()
        self._possible_values["optimizer"] = Optimizers.names()
        self._possible_values["learning_rate"] = possible_values.Number(
            step=1e-9, min=1e-9
        )
        self._possible_values["differential_learning_rate_layers"] = (
            possible_values.String(
                values=("backbone", "embed", "head"),
                allow_custom=True,
                placeholder="Select optional layers...",
            )
        )
        self._possible_values["differential_learning_rate"] = self._possible_values[
            "learning_rate"
        ]
        self._possible_values["freeze_layers"] = possible_values.String(
            values=("embed", "layer", "head"),
            allow_custom=True,
            placeholder="Select optional layers to freeze...",
        )
        self._possible_values["attention_implementation"] = possible_values.String(
            values=(
                ("auto", "Auto"),
                ("eager", "Eager"),
                ("flash_attention_2", "Flash Attention 2"),
                ("sdpa", "SDPA"),
            ),
            allow_custom=False,
        )

        self._possible_values["batch_size"] = (1, 256, 1)
        self._possible_values["epochs"] = (0, 10, 1)
        self._possible_values["schedule"] = Schedulers.names()
        self._possible_values["min_learning_rate_ratio"] = (0.0, 0.1, 0.0001)
        self._possible_values["warmup_epochs"] = (0.0, 5.0, 0.05)

        self._possible_values["weight_decay"] = possible_values.Number(step=1e-5, min=0)
        self._possible_values["gradient_clip"] = (0.0, 10.0, 0.1)
        self._possible_values["grad_accumulation"] = (1, 32, 1)

        self._possible_values["lora_r"] = (1, 256, 1)
        self._possible_values["lora_alpha"] = (1, 256, 1)
        self._possible_values["lora_dropout"] = (0.0, 0.5, 0.01)
        self._possible_values["lora_unfreeze_layers"] = possible_values.String(
            values=("embed", "head"),
            allow_custom=True,
            placeholder="Select optional layers to unfreeze...",
        )
