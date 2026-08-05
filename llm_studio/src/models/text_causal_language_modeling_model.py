import logging
import os
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from llm_studio.src.metrics.text_causal_language_modeling_metrics import Perplexity
from llm_studio.src.utils.data_utils import batch_padding
from llm_studio.src.utils.modeling_utils import (
    create_nlp_backbone,
    forward,
    generate,
    prepare_lora,
)

logger = logging.getLogger(__name__)

_PERPLEXITY_GENERATION_DEFAULT_SAMPLES = 8
_PERPLEXITY_GENERATION_EXTRA_TOKENS = 8
_PERPLEXITY_GENERATION_SAMPLES_ENV = "H2O_LLM_STUDIO_PERPLEXITY_GENERATION_SAMPLES"


def get_perplexity_generation_indices(
    total_samples: int, requested_samples: int
) -> tuple[int, ...]:
    """Return deterministic, evenly spaced validation sample indices."""
    total_samples = max(int(total_samples), 0)
    requested_samples = min(max(int(requested_samples), 0), total_samples)
    if requested_samples == 0:
        return ()
    if requested_samples == 1:
        return (total_samples // 2,)
    if requested_samples == total_samples:
        return tuple(range(total_samples))

    last_index = total_samples - 1
    return tuple(
        round(sample_number * last_index / (requested_samples - 1))
        for sample_number in range(requested_samples)
    )


def _slice_batch_rows(batch: Any, row_indices: torch.Tensor, batch_size: int) -> Any:
    """Select batch rows while leaving scalar and metadata values unchanged."""
    if isinstance(batch, dict):
        return {
            key: _slice_batch_rows(value, row_indices, batch_size)
            for key, value in batch.items()
        }
    if isinstance(batch, torch.Tensor) and batch.ndim > 0:
        if batch.shape[0] == batch_size:
            return batch.index_select(0, row_indices.to(batch.device))
    return batch


class Model(nn.Module):
    """
    Model for causal language modeling problem type.
    """

    def __init__(self, cfg: Any):
        """
        Args:
            cfg: config with all the hyperparameters
        """

        super(Model, self).__init__()

        self.cfg = cfg
        self.backbone, self.backbone_config = create_nlp_backbone(
            cfg, model_class=AutoModelForCausalLM
        )

        if cfg.training.lora:
            self.backbone = prepare_lora(cfg, self.backbone)

        self.loss_fn = self.cfg.training.loss_class.get(
            self.cfg.training.loss_function
        )(self.cfg)

        if self.cfg.prediction.metric == "Perplexity":
            self.perplexity = Perplexity(self.cfg, reduce=False)

    def _get_deepspeed_engine(self):
        """
        Return the active DeepSpeed engine.

        In the non-LoRA path, the backbone itself is replaced by the engine.
        In the LoRA path, the wrapped base model is replaced by the engine.
        """
        if self.cfg.training.lora:
            return self.backbone.base_model.model
        return self.backbone

    def init_deepspeed(self):
        engine = self._get_deepspeed_engine()

        self.backward = engine.backward
        self.step = engine.step
        self.save_checkpoint = engine.save_checkpoint
        self.save_16bit_model = engine.save_16bit_model
        if self.cfg.training.lora:
            self.backbone.base_model.model.config = engine.module.config
            self.backbone.base_model.model.generation_config = (
                engine.module.generation_config
            )
        else:
            self.backbone.config = engine.module.config
            self.backbone.generation_config = engine.module.generation_config

    def generate(self, batch: dict, cfg: Any, streamer=None):
        if cfg.environment.use_deepspeed and cfg.training.lora:
            return generate(self.backbone.base_model.model, batch, cfg, streamer)
        else:
            return generate(self.backbone, batch, cfg, streamer)

    def _get_requested_perplexity_generation_samples(self) -> int:
        configured = getattr(
            self.cfg.prediction,
            "perplexity_generation_samples",
            None,
        )
        if configured is not None:
            return max(int(configured), 0)

        raw_value = os.getenv(
            _PERPLEXITY_GENERATION_SAMPLES_ENV,
            str(_PERPLEXITY_GENERATION_DEFAULT_SAMPLES),
        )
        try:
            return max(int(raw_value), 0)
        except ValueError:
            logger.warning(
                "Ignoring invalid %s=%r; using %s insight samples.",
                _PERPLEXITY_GENERATION_SAMPLES_ENV,
                raw_value,
                _PERPLEXITY_GENERATION_DEFAULT_SAMPLES,
            )
            return _PERPLEXITY_GENERATION_DEFAULT_SAMPLES

    def _get_perplexity_generation_mask(self, batch: dict) -> torch.Tensor:
        batch_size = int(batch["input_ids"].shape[0])
        requested_samples = self._get_requested_perplexity_generation_samples()
        sample_indices = batch.get("validation_sample_index")
        sample_counts = batch.get("validation_sample_count")

        if sample_indices is None or sample_counts is None:
            # Compatibility fallback for custom datasets that do not expose their
            # validation positions. Generate only the first requested samples.
            current_step = int(getattr(self.cfg.environment, "_curr_val_step", 0))
            first_index = max(current_step - batch_size, 0)
            mask_values = [
                sample_index < requested_samples
                for sample_index in range(first_index, first_index + batch_size)
            ]
            return torch.tensor(
                mask_values,
                dtype=torch.bool,
                device=batch["input_ids"].device,
            )

        total_samples = int(sample_counts.reshape(-1)[0].item())
        selected_indices = set(
            get_perplexity_generation_indices(total_samples, requested_samples)
        )
        sample_indices_cpu = sample_indices.detach().cpu().reshape(-1).tolist()
        mask = torch.tensor(
            [sample_index in selected_indices for sample_index in sample_indices_cpu],
            dtype=torch.bool,
            device=batch["input_ids"].device,
        )

        log_key = (total_samples, len(selected_indices))
        if (
            getattr(self, "_perplexity_generation_log_key", None) != log_key
            and getattr(self.cfg.environment, "_local_rank", 0) == 0
        ):
            logger.info(
                "Perplexity will be calculated for all %s validation samples; "
                "autoregressive Prediction Insights will be generated for %s "
                "evenly spaced samples. Set %s=0 to disable or another integer "
                "to change the sample count.",
                total_samples,
                len(selected_indices),
                _PERPLEXITY_GENERATION_SAMPLES_ENV,
            )
            self._perplexity_generation_log_key = log_key
        return mask

    def _get_perplexity_generation_limit(self, batch: dict) -> int:
        configured_limit = max(int(self.cfg.prediction.max_length_inference), 1)
        minimum_length = max(int(self.cfg.prediction.min_length_inference), 1)
        answer_attention_mask = batch.get("answer_attention_mask")
        if answer_attention_mask is None or answer_attention_mask.numel() == 0:
            return configured_limit

        target_length = int(answer_attention_mask.sum(dim=1).max().item())
        target_aware_limit = max(
            minimum_length,
            target_length + _PERPLEXITY_GENERATION_EXTRA_TOKENS,
        )
        return min(configured_limit, target_aware_limit)

    def _generate_perplexity_insight_samples(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(batch["input_ids"].shape[0])
        generation_mask = self._get_perplexity_generation_mask(batch)
        selected_rows = torch.nonzero(generation_mask, as_tuple=False).flatten()

        generation_backbone = (
            self._get_deepspeed_engine()
            if self.cfg.environment.use_deepspeed
            else self.backbone
        )
        generation_config = generation_backbone.generation_config
        pad_token_id = generation_config.pad_token_id
        if pad_token_id is None:
            pad_token_id = getattr(self.backbone_config, "pad_token_id", 0) or 0

        if len(selected_rows) == 0:
            return (
                torch.full((batch_size, 1), int(pad_token_id), dtype=torch.long),
                generation_mask.detach().cpu(),
            )

        selected_batch = _slice_batch_rows(batch, selected_rows, batch_size)
        max_new_tokens = self._get_perplexity_generation_limit(selected_batch)
        previous_max_new_tokens = generation_config.max_new_tokens
        generation_config.max_new_tokens = max_new_tokens
        try:
            generated_ids = self.generate(selected_batch, self.cfg).detach().cpu()
        finally:
            generation_config.max_new_tokens = previous_max_new_tokens

        output_width = max(int(generated_ids.shape[1]), 1)
        full_generated_ids = torch.full(
            (batch_size, output_width),
            int(pad_token_id),
            dtype=generated_ids.dtype,
        )
        full_generated_ids[selected_rows.detach().cpu()] = generated_ids
        return full_generated_ids, generation_mask.detach().cpu()

    def forward(
        self,
        batch: dict,
        padding: bool = True,
    ) -> dict:
        # disable cache if gradient checkpointing is enabled
        if self.cfg.architecture.gradient_checkpointing:
            self.backbone.config.use_cache = False

        outputs: dict = {}
        mask_key = "attention_mask"
        pad_keys = [
            "input_ids",
            "attention_mask",
            "special_tokens_mask",
            "labels",
        ]

        if padding:
            batch = batch_padding(
                self.cfg,
                batch,
                self.training,
                mask_key=mask_key,
                pad_keys=pad_keys,
                padding_side=self.cfg.tokenizer._padding_side,
            )

        output = forward(self.backbone, batch["input_ids"], batch["attention_mask"])

        if "labels" in batch:
            loss = self.loss_fn(output.logits, batch["labels"])
            outputs["loss"] = loss

        if not self.training and self.cfg.prediction.metric == "Perplexity":
            outputs["perplexity"] = self.perplexity(output.logits, batch["labels"])
            (
                outputs["predicted_answer_ids"],
                outputs["prediction_generated"],
            ) = self._generate_perplexity_insight_samples(batch)

        # enable cache again if gradient checkpointing is enabled
        if self.cfg.architecture.gradient_checkpointing:
            self.backbone.config.use_cache = True

        return outputs
