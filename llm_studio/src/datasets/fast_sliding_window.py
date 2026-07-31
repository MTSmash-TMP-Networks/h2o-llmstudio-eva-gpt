"""Fast and exact long-sample indexing for causal language-model datasets."""

from __future__ import annotations

import codecs
import logging
import time
from typing import Any

import numpy as np
import torch

from llm_studio.src.datasets import text_causal_language_modeling_ds as base_ds
from llm_studio.src.datasets.conversation_chain_handler import PLAIN_TEXT_PROMPT
from llm_studio.src.datasets.sliding_window_cache import (
    get_cache_path,
    load_index,
    save_index,
)

logger = logging.getLogger(__name__)
_SAMPLE_BATCH_SIZE = 256
_TOKENIZER_BATCH_SIZE = 512


class FastSlidingWindowDataset(base_ds.CustomDataset):
    """Use tokenizer batches and a persistent cache to build long-sample indices."""

    def _build_sample_index(self) -> list[tuple[int, int | None, int]]:
        strategy = getattr(self.cfg.tokenizer, "long_sample_strategy", "Truncate")
        if strategy not in ("Truncate", "Sliding Window", "Skip"):
            strategy = "Truncate"
        sample_count = len(self.conversation_chain_handler)
        if self.mode != "train" or strategy == "Truncate":
            return [(idx, None, 0) for idx in range(sample_count)]

        cache_path = get_cache_path(self, strategy)
        cached = load_index(cache_path, sample_count)
        if cached is not None:
            logger.info("Loaded %s long-sample index entries from cache.", len(cached))
            return cached

        started_at = time.perf_counter()
        sample_lengths = self._compute_sample_lengths_batched()
        sample_index, skipped, windows = self._index_from_lengths(
            sample_lengths, strategy
        )
        save_index(cache_path, sample_index)
        logger.info(
            "Built %s long-sample index entries (%s skipped, %s windows) in %.2fs.",
            len(sample_index),
            skipped,
            windows,
            time.perf_counter() - started_at,
        )
        return sample_index

    def _index_from_lengths(
        self, sample_lengths: list[int], strategy: str
    ) -> tuple[list[tuple[int, int | None, int]], int, int]:
        max_length = int(self.cfg.tokenizer.max_length)
        overlap = int(getattr(self.cfg.tokenizer, "sliding_window_overlap", 0))
        index: list[tuple[int, int | None, int]] = []
        skipped = 0
        windows_created = 0
        for sample_idx, sample_length in enumerate(sample_lengths):
            if sample_length <= max_length:
                index.append((sample_idx, None, 0))
            elif strategy == "Skip":
                skipped += 1
            else:
                windows = self._get_sliding_window_starts_and_prefix_masks(
                    sample_length, max_length, overlap
                )
                index.extend(
                    (sample_idx, start, prefix_mask)
                    for start, prefix_mask in windows
                )
                windows_created += len(windows)
        return index, skipped, windows_created

    def _compute_sample_lengths_batched(self) -> list[int]:
        """Compute the exact serial-encoding lengths without creating tensors."""
        sample_count = len(self.conversation_chain_handler)
        max_length = int(self.cfg.tokenizer.max_length)
        result: list[int] = []

        for first in range(0, sample_count, _SAMPLE_BATCH_SIZE):
            last = min(first + _SAMPLE_BATCH_SIZE, sample_count)
            sample_lengths = [0] * (last - first)
            texts: list[str] = []
            records: list[tuple[str, int, int]] = []

            for local_idx, sample_idx in enumerate(range(first, last)):
                sample = self._prepare_input_text_dict(sample_idx)
                systems = sample.get("systems", [])
                prompts = sample.get("prompts", [])
                answers = sample.get("answers", [])
                if systems and systems[0]:
                    texts.append(systems[0])
                    records.append(("direct", local_idx, -1))
                for turn_idx, prompt in enumerate(prompts):
                    for part in self._prompt_parts(prompt):
                        texts.append(part)
                        records.append(("prompt", local_idx, turn_idx))
                for answer in answers:
                    if answer:
                        texts.append(answer)
                        records.append(("direct", local_idx, -1))

            prompt_lengths: dict[tuple[int, int], int] = {}
            for record, token_length in zip(
                records, self._batch_token_lengths(texts), strict=True
            ):
                kind, local_idx, turn_idx = record
                token_length = min(token_length, max_length)
                if kind == "direct":
                    sample_lengths[local_idx] += token_length
                else:
                    key = (local_idx, turn_idx)
                    prompt_lengths[key] = prompt_lengths.get(key, 0) + token_length
            for (local_idx, _), prompt_length in prompt_lengths.items():
                sample_lengths[local_idx] += min(prompt_length, max_length)
            result.extend(sample_lengths)

            if last < sample_count and last % 10000 == 0:
                logger.info(
                    "Indexed token lengths for %s/%s samples.", last, sample_count
                )
        return result

    def _prompt_parts(self, prompt: str) -> list[str]:
        if prompt == PLAIN_TEXT_PROMPT:
            return []
        parts = [
            codecs.decode(self.cfg.dataset.text_prompt_start, "unicode_escape"),
            prompt,
        ]
        if self.cfg.dataset.add_eos_token_to_prompt:
            parts.append(self.cfg.tokenizer._tokenizer_eos_token)
        parts.append(
            codecs.decode(self.cfg.dataset.text_answer_separator, "unicode_escape")
        )
        return [part for part in parts if part]

    def _batch_token_lengths(self, texts: list[str]) -> list[int]:
        lengths: list[int] = []
        for first in range(0, len(texts), _TOKENIZER_BATCH_SIZE):
            batch = texts[first : first + _TOKENIZER_BATCH_SIZE]
            batch_lengths = self._try_batch_lengths(batch)
            if batch_lengths is None:
                batch_lengths = [self._single_length(text) for text in batch]
            lengths.extend(batch_lengths)
        return lengths

    def _try_batch_lengths(self, texts: list[str]) -> list[int] | None:
        if len(texts) < 2:
            return None
        try:
            input_ids = self.tokenizer(
                texts,
                add_special_tokens=False,
                truncation=True,
                max_length=int(self.cfg.tokenizer.max_length),
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
        except (TypeError, ValueError, KeyError, AttributeError):
            return None

        if isinstance(input_ids, torch.Tensor):
            if input_ids.ndim != 2 or input_ids.shape[0] != len(texts):
                return None
            sequences: list[Any] = list(input_ids)
        elif isinstance(input_ids, (list, tuple)):
            if len(input_ids) != len(texts):
                return None
            if input_ids and isinstance(input_ids[0], (int, np.integer)):
                return None
            sequences = list(input_ids)
        else:
            return None
        return [len(sequence) for sequence in sequences]

    def _single_length(self, text: str) -> int:
        return len(
            self.encode(
                self.tokenizer,
                text,
                int(self.cfg.tokenizer.max_length),
                "right",
            )["input_ids"]
        )


def install_fast_sliding_window() -> None:
    """Replace the exported base class once while preserving subclass behavior."""
    if base_ds.CustomDataset is not FastSlidingWindowDataset:
        base_ds.CustomDataset = FastSlidingWindowDataset
