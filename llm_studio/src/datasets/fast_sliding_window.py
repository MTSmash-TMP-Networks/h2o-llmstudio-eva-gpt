"""Fast and exact long-sample indexing for causal language-model datasets."""

from __future__ import annotations

import codecs
import logging
import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _SampleLayout:
    """Token length and half-open spans containing trainable labels."""

    length: int
    trainable_spans: tuple[tuple[int, int], ...]


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
        sample_layouts = self._compute_sample_layouts_batched()
        sample_index, skipped, windows, all_masked = self._index_from_layouts(
            sample_layouts, strategy
        )
        save_index(cache_path, sample_index)
        logger.info(
            "Built %s supervised long-sample index entries "
            "(%s skipped, %s windows, %s all-masked samples removed) in %.2fs.",
            len(sample_index),
            skipped,
            windows,
            all_masked,
            time.perf_counter() - started_at,
        )
        return sample_index

    def _index_from_layouts(
        self, sample_layouts: list[_SampleLayout], strategy: str
    ) -> tuple[list[tuple[int, int | None, int]], int, int, int]:
        max_length = int(self.cfg.tokenizer.max_length)
        overlap = int(getattr(self.cfg.tokenizer, "sliding_window_overlap", 0))
        index: list[tuple[int, int | None, int]] = []
        skipped = 0
        windows_created = 0
        all_masked = 0

        for sample_idx, layout in enumerate(sample_layouts):
            if layout.length <= max_length:
                if self._window_has_shifted_target(
                    layout,
                    window_start=0,
                    prefix_label_mask_len=0,
                    max_length=max_length,
                ):
                    index.append((sample_idx, None, 0))
                else:
                    all_masked += 1
                continue

            if strategy == "Skip":
                skipped += 1
                continue

            windows = self._get_supervised_windows(layout, max_length, overlap)
            if not windows:
                all_masked += 1
                continue
            index.extend(
                (sample_idx, start, prefix_mask) for start, prefix_mask in windows
            )
            windows_created += len(windows)

        return index, skipped, windows_created, all_masked

    def _get_supervised_windows(
        self, layout: _SampleLayout, max_length: int, overlap: int
    ) -> list[tuple[int, int]]:
        """Return only windows that contribute at least one shifted target label."""
        base_windows = self._get_sliding_window_starts_and_prefix_masks(
            layout.length, max_length, overlap
        )
        starts = {start for start, _ in base_windows}

        # Add a target-anchored candidate so the first supervised chat window keeps
        # visible prompt/context tokens before its first answer label whenever the
        # sample contains such context.
        last_start = max(layout.length - max_length, 0)
        context_tokens = min(max(overlap, 1), max(max_length - 1, 0))
        for span_start, _ in layout.trainable_spans:
            has_context_candidate = any(
                start < span_start < start + max_length for start in starts
            )
            if span_start > 0 and context_tokens > 0 and not has_context_candidate:
                starts.add(min(last_start, max(0, span_start - context_tokens)))

        windows: list[tuple[int, int]] = []
        previous_start: int | None = None
        for current_start in sorted(starts):
            prefix_label_mask_len = 0
            if previous_start is not None:
                previous_end = previous_start + max_length
                duplicate_prefix_len = max(0, previous_end - current_start)
                prefix_label_mask_len = min(duplicate_prefix_len, max_length)

            if not self._window_has_shifted_target(
                layout,
                window_start=current_start,
                prefix_label_mask_len=prefix_label_mask_len,
                max_length=max_length,
            ):
                continue

            windows.append((current_start, prefix_label_mask_len))
            previous_start = current_start

        return windows

    @staticmethod
    def _window_has_shifted_target(
        layout: _SampleLayout,
        window_start: int,
        prefix_label_mask_len: int,
        max_length: int,
    ) -> bool:
        """Check the labels that remain after overlap masking and causal shifting."""
        # Causal language-model loss compares logits[:-1] with labels[1:]. A target
        # only at local position zero would therefore still produce an all--100 loss.
        target_start = window_start + max(prefix_label_mask_len, 1)
        target_end = min(window_start + max_length, layout.length)
        if target_start >= target_end:
            return False
        return any(
            max(span_start, target_start) < min(span_end, target_end)
            for span_start, span_end in layout.trainable_spans
        )

    def _compute_sample_lengths_batched(self) -> list[int]:
        """Compute exact serial-encoding lengths without creating tensors."""
        return [layout.length for layout in self._compute_sample_layouts_batched()]

    def _compute_sample_layouts_batched(self) -> list[_SampleLayout]:
        """Compute exact lengths and trainable label spans in tokenizer batches."""
        sample_count = len(self.conversation_chain_handler)
        max_length = int(self.cfg.tokenizer.max_length)
        result: list[_SampleLayout] = []

        for first in range(0, sample_count, _SAMPLE_BATCH_SIZE):
            last = min(first + _SAMPLE_BATCH_SIZE, sample_count)
            prepared_samples = [
                self._prepare_input_text_dict(sample_idx)
                for sample_idx in range(first, last)
            ]
            texts: list[str] = []
            records: list[tuple[int, int, str, int]] = []

            for local_idx, sample in enumerate(prepared_samples):
                turns = list(
                    zip(
                        sample.get("systems", []),
                        sample.get("prompts", []),
                        sample.get("answers", []),
                        strict=False,
                    )
                )
                for turn_idx, (system, prompt, answer) in enumerate(turns):
                    if turn_idx == 0 and system:
                        texts.append(system)
                        records.append((local_idx, turn_idx, "system", 0))
                    for part_idx, (part, _) in enumerate(
                        self._prompt_parts_with_masks(prompt)
                    ):
                        texts.append(part)
                        records.append((local_idx, turn_idx, "prompt", part_idx))
                    if answer:
                        texts.append(answer)
                        records.append((local_idx, turn_idx, "answer", 0))

            token_lengths = {
                record: min(token_length, max_length)
                for record, token_length in zip(
                    records, self._batch_token_lengths(texts), strict=True
                )
            }

            for local_idx, sample in enumerate(prepared_samples):
                turns = list(
                    zip(
                        sample.get("systems", []),
                        sample.get("prompts", []),
                        sample.get("answers", []),
                        strict=False,
                    )
                )
                position = 0
                trainable_spans: list[tuple[int, int]] = []
                last_turn_idx = len(turns) - 1

                for turn_idx, (_, prompt, _) in enumerate(turns):
                    eligible_turn = (
                        not bool(getattr(self.cfg.dataset, "only_last_answer", False))
                        or turn_idx == last_turn_idx
                    )
                    mask_prompt_labels = bool(
                        getattr(self.cfg.dataset, "mask_prompt_labels", True)
                    )
                    mask_user_only = bool(
                        getattr(
                            self.cfg.dataset,
                            "mask_prompt_user_text_only",
                            False,
                        )
                    )

                    if turn_idx == 0:
                        system_length = token_lengths.get(
                            (local_idx, turn_idx, "system", 0), 0
                        )
                        system_trainable = not mask_prompt_labels or (
                            eligible_turn and mask_user_only
                        )
                        position = self._append_segment(
                            position,
                            system_length,
                            system_trainable,
                            trainable_spans,
                        )

                    prompt_parts = self._prompt_parts_with_masks(prompt)
                    prompt_part_lengths = [
                        token_lengths.get(
                            (local_idx, turn_idx, "prompt", part_idx), 0
                        )
                        for part_idx in range(len(prompt_parts))
                    ]
                    left_trim = max(sum(prompt_part_lengths) - max_length, 0)
                    for (_, mask_user_text), part_length in zip(
                        prompt_parts, prompt_part_lengths, strict=True
                    ):
                        trimmed = min(left_trim, part_length)
                        left_trim -= trimmed
                        kept_length = part_length - trimmed
                        prompt_part_trainable = not mask_prompt_labels or (
                            eligible_turn and mask_user_only and not mask_user_text
                        )
                        position = self._append_segment(
                            position,
                            kept_length,
                            prompt_part_trainable,
                            trainable_spans,
                        )

                    answer_length = token_lengths.get(
                        (local_idx, turn_idx, "answer", 0), 0
                    )
                    answer_trainable = not mask_prompt_labels or eligible_turn
                    position = self._append_segment(
                        position,
                        answer_length,
                        answer_trainable,
                        trainable_spans,
                    )

                result.append(
                    _SampleLayout(
                        length=position,
                        trainable_spans=tuple(trainable_spans),
                    )
                )

            if last < sample_count and last % 10000 == 0:
                logger.info(
                    "Indexed supervised token layouts for %s/%s samples.",
                    last,
                    sample_count,
                )
        return result

    @staticmethod
    def _append_segment(
        position: int,
        length: int,
        trainable: bool,
        trainable_spans: list[tuple[int, int]],
    ) -> int:
        if length <= 0:
            return position
        end = position + length
        if trainable:
            if trainable_spans and trainable_spans[-1][1] == position:
                trainable_spans[-1] = (trainable_spans[-1][0], end)
            else:
                trainable_spans.append((position, end))
        return end

    def _prompt_parts_with_masks(self, prompt: str) -> list[tuple[str, bool]]:
        if prompt == PLAIN_TEXT_PROMPT:
            return []
        parts = [
            (
                codecs.decode(self.cfg.dataset.text_prompt_start, "unicode_escape"),
                False,
            ),
            (prompt, True),
        ]
        if self.cfg.dataset.add_eos_token_to_prompt:
            parts.append((self.cfg.tokenizer._tokenizer_eos_token, False))
        parts.append(
            (
                codecs.decode(
                    self.cfg.dataset.text_answer_separator, "unicode_escape"
                ),
                False,
            )
        )
        return [(part, mask_user_text) for part, mask_user_text in parts if part]

    def _prompt_parts(self, prompt: str) -> list[str]:
        return [part for part, _ in self._prompt_parts_with_masks(prompt)]

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
