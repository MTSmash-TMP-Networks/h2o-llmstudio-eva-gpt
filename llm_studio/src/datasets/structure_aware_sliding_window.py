"""Structure-aware long-sample windows for causal language-model training."""

from __future__ import annotations

import logging
import time

import torch

from llm_studio.src.datasets import text_causal_language_modeling_ds as base_ds
from llm_studio.src.datasets.fast_sliding_window import (
    FastSlidingWindowDataset,
    _SampleLayout,
)
from llm_studio.src.datasets.sliding_window_cache import (
    get_cache_path,
    load_index,
    save_index,
)

logger = logging.getLogger(__name__)

# The anchor replaces only already duplicated overlap tokens. New target tokens are
# never removed. Keep enough immediate overlap for fluent local transitions and use
# the remaining duplicated prefix for system/current-prompt context.
_ANCHOR_MAX_TOKENS = 256
_MIN_LOCAL_OVERLAP_TOKENS = 64
_CACHE_STRATEGY_VERSION = "structure-aware-v1"


class StructureAwareSlidingWindowDataset(FastSlidingWindowDataset):
    """Prefer semantic boundaries and repeat masked chat context in later windows."""

    def _build_sample_index(self) -> list[tuple[int, int | None, int]]:
        strategy = getattr(self.cfg.tokenizer, "long_sample_strategy", "Truncate")
        if strategy not in ("Truncate", "Sliding Window", "Skip"):
            strategy = "Truncate"
        sample_count = len(self.conversation_chain_handler)
        if self.mode != "train" or strategy == "Truncate":
            return [(idx, None, 0) for idx in range(sample_count)]

        # Use a versioned cache namespace. Mixed-text datasets keep their public
        # dataset class name, so changing only the implementation class would not
        # otherwise invalidate their old fixed-stride indices.
        cache_strategy = f"{strategy}:{_CACHE_STRATEGY_VERSION}"
        cache_path = get_cache_path(self, cache_strategy)
        cached = load_index(cache_path, sample_count)
        if cached is not None:
            logger.info(
                "Loaded %s structure-aware long-sample index entries from cache.",
                len(cached),
            )
            return cached

        started_at = time.perf_counter()
        sample_layouts = self._compute_sample_layouts_batched()
        sample_index, skipped, windows, all_masked = self._index_from_layouts(
            sample_layouts, strategy
        )
        save_index(cache_path, sample_index)
        logger.info(
            "Built %s structure-aware supervised long-sample index entries "
            "(%s skipped, %s windows, %s all-masked samples removed) in %.2fs.",
            len(sample_index),
            skipped,
            windows,
            all_masked,
            time.perf_counter() - started_at,
        )
        return sample_index

    def _get_structure_aware_starts(
        self,
        layout: _SampleLayout,
        max_length: int,
        overlap: int,
    ) -> list[int]:
        """Choose exact window starts, preferring supervised segment boundaries.

        Trainable span starts correspond to answer/target segment starts in chat
        datasets. A boundary is used only when it preserves continuous coverage and
        a useful amount of local overlap. Fixed token strides remain the fallback for
        long single answers and raw text.
        """
        if max_length <= 0 or layout.length <= max_length:
            return [0]

        overlap = min(max(overlap, 0), max(max_length - 1, 0))
        stride = max(max_length - overlap, 1)
        last_start = max(layout.length - max_length, 0)
        preferred = sorted(
            {
                span_start
                for span_start, _ in layout.trainable_spans
                if 0 < span_start < last_start
            }
        )
        minimum_overlap = min(overlap, _MIN_LOCAL_OVERLAP_TOKENS)

        starts = [0]
        while starts[-1] < last_start:
            previous_start = starts[-1]
            nominal_start = min(previous_start + stride, last_start)
            if nominal_start == last_start:
                starts.append(last_start)
                break

            # Never introduce a gap. For configured overlap, keep at least the
            # smaller of that overlap and the local-context reserve.
            lower = previous_start + 1
            upper = min(
                last_start,
                previous_start + max_length - minimum_overlap,
            )
            if upper < lower:
                next_start = nominal_start
            else:
                candidates = [
                    candidate
                    for candidate in preferred
                    if lower <= candidate <= upper
                ]
                next_start = (
                    min(candidates, key=lambda value: (abs(value - nominal_start), value))
                    if candidates
                    else nominal_start
                )

            if next_start <= previous_start:
                next_start = nominal_start
            starts.append(min(next_start, last_start))

        if starts[-1] != last_start:
            starts.append(last_start)
        return list(dict.fromkeys(starts))

    def _get_supervised_windows(
        self, layout: _SampleLayout, max_length: int, overlap: int
    ) -> list[tuple[int, int]]:
        """Return structure-aware windows with at least one shifted target label."""
        starts = set(self._get_structure_aware_starts(layout, max_length, overlap))

        # Keep the existing target-anchored safeguard: the first useful answer
        # window should retain visible context before its first trainable label.
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

    def _get_system_anchor_ids(self, original_idx: int) -> torch.Tensor:
        cache = getattr(self, "_structure_system_anchor_cache", None)
        if cache is None:
            cache = {}
            self._structure_system_anchor_cache = cache
        if original_idx in cache:
            return cache[original_idx]

        prepared = self._prepare_input_text_dict(original_idx)
        systems = prepared.get("systems", [])
        system = systems[0] if systems else ""
        if system:
            system_ids = self.encode(
                self.tokenizer,
                system,
                int(self.cfg.tokenizer.max_length),
                "right",
            )["input_ids"]
        else:
            system_ids = torch.empty(0, dtype=torch.long)
        cache[original_idx] = system_ids
        return system_ids

    @staticmethod
    def _find_target_turn(
        labels: torch.Tensor,
        prompt_encodings: list[torch.Tensor],
        answer_encodings: list[torch.Tensor],
        window_start: int,
        prefix_label_mask_len: int,
        max_length: int,
        system_length: int,
    ) -> tuple[int, int, int] | None:
        """Return target turn index, prompt start, and answer start."""
        search_start = window_start + max(prefix_label_mask_len, 1)
        search_end = min(window_start + max_length, len(labels))
        if search_start >= search_end:
            return None
        target_offsets = torch.nonzero(
            labels[search_start:search_end] != -100,
            as_tuple=False,
        ).flatten()
        if len(target_offsets) == 0:
            return None
        target_position = search_start + int(target_offsets[0])

        position = 0
        for turn_idx, (prompt_encoding, answer_encoding) in enumerate(
            zip(prompt_encodings, answer_encodings, strict=False)
        ):
            prompt_start = position + (system_length if turn_idx == 0 else 0)
            answer_start = position + len(prompt_encoding)
            turn_end = answer_start + len(answer_encoding)
            if position <= target_position < turn_end:
                return turn_idx, prompt_start, answer_start
            position = turn_end
        return None

    @staticmethod
    def _fit_anchor_parts(
        system_ids: torch.Tensor,
        prompt_ids: torch.Tensor,
        budget: int,
    ) -> torch.Tensor:
        """Fit system and current prompt into the available duplicated prefix."""
        if budget <= 0:
            return torch.empty(0, dtype=torch.long)
        if len(system_ids) == 0:
            return prompt_ids[-budget:].clone()
        if len(prompt_ids) == 0:
            return system_ids[:budget].clone()

        # Reserve a small but meaningful prefix for the system role/instruction and
        # give the current user prompt priority for the remaining budget. Prompt
        # tails retain the answer separator/assistant marker.
        system_budget = min(len(system_ids), max(16, budget // 4), budget)
        prompt_budget = max(budget - system_budget, 0)
        if len(prompt_ids) < prompt_budget:
            system_budget = min(len(system_ids), budget - len(prompt_ids))
            prompt_budget = min(len(prompt_ids), budget - system_budget)

        parts = []
        if system_budget:
            parts.append(system_ids[:system_budget])
        if prompt_budget:
            parts.append(prompt_ids[-prompt_budget:])
        return torch.cat(parts).clone() if parts else torch.empty(0, dtype=torch.long)

    def _get_structure_anchor(
        self,
        original_idx: int,
        window_start: int,
        prefix_label_mask_len: int,
        labels: torch.Tensor,
        prompt_encodings: list[torch.Tensor],
        answer_encodings: list[torch.Tensor],
    ) -> torch.Tensor:
        if window_start <= 0 or prefix_label_mask_len <= 0:
            return torch.empty(0, dtype=torch.long)

        minimum_overlap = min(prefix_label_mask_len, _MIN_LOCAL_OVERLAP_TOKENS)
        budget = min(
            _ANCHOR_MAX_TOKENS,
            max(prefix_label_mask_len - minimum_overlap, 0),
        )
        if budget <= 0:
            return torch.empty(0, dtype=torch.long)

        system_ids = self._get_system_anchor_ids(original_idx)
        target_turn = self._find_target_turn(
            labels=labels,
            prompt_encodings=prompt_encodings,
            answer_encodings=answer_encodings,
            window_start=window_start,
            prefix_label_mask_len=prefix_label_mask_len,
            max_length=int(self.cfg.tokenizer.max_length),
            system_length=len(system_ids),
        )
        if target_turn is None:
            return torch.empty(0, dtype=torch.long)

        turn_idx, prompt_start, _ = target_turn
        prompt_ids = prompt_encodings[turn_idx]
        if turn_idx == 0 and len(system_ids) and len(prompt_ids) >= len(system_ids):
            if torch.equal(prompt_ids[: len(system_ids)], system_ids):
                prompt_ids = prompt_ids[len(system_ids) :]

        system_missing = len(system_ids) > 0 and window_start >= len(system_ids)
        prompt_missing = prompt_start < window_start
        if not system_missing and not prompt_missing:
            return torch.empty(0, dtype=torch.long)

        return self._fit_anchor_parts(
            system_ids if system_missing else torch.empty(0, dtype=torch.long),
            prompt_ids if prompt_missing else torch.empty(0, dtype=torch.long),
            budget,
        )

    def _compose_structure_aware_window(
        self,
        original_idx: int,
        window_start: int,
        prefix_label_mask_len: int,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        prompt_encodings: list[torch.Tensor],
        answer_encodings: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_length = int(self.cfg.tokenizer.max_length)
        window_end = window_start + max_length
        window_input_ids = input_ids[window_start:window_end].clone()
        window_labels = labels[window_start:window_end].clone()

        anchor_ids = self._get_structure_anchor(
            original_idx=original_idx,
            window_start=window_start,
            prefix_label_mask_len=prefix_label_mask_len,
            labels=labels,
            prompt_encodings=prompt_encodings,
            answer_encodings=answer_encodings,
        )
        anchor_length = min(
            len(anchor_ids),
            prefix_label_mask_len,
            len(window_input_ids),
        )
        if anchor_length:
            anchor_ids = anchor_ids[:anchor_length]
            window_input_ids = torch.cat(
                [anchor_ids, window_input_ids[anchor_length:]]
            )
            window_labels = torch.cat(
                [
                    torch.full(
                        (anchor_length,),
                        -100,
                        dtype=window_labels.dtype,
                    ),
                    window_labels[anchor_length:],
                ]
            )

        # The entire duplicated prefix, including inserted anchor tokens, is visible
        # context only. It never contributes duplicate loss.
        if prefix_label_mask_len > 0:
            window_labels[: min(prefix_label_mask_len, len(window_labels))] = -100
        return window_input_ids, window_labels

    def __getitem__(self, idx: int) -> dict:
        """Read one sample and add masked structural context to later windows."""
        original_idx, window_start, prefix_label_mask_len = self.sample_index[idx]
        use_long_sample_windows = (
            self.mode == "train"
            and getattr(self.cfg.tokenizer, "long_sample_strategy", "Truncate")
            == "Sliding Window"
        )
        input_ids, labels, prompt_encodings, answer_encodings = (
            self._get_input_ids_labels_and_encodings(
                original_idx,
                augment=not use_long_sample_windows,
                trim_to_max_length=not use_long_sample_windows,
            )
        )

        if window_start is not None:
            input_ids, labels = self._compose_structure_aware_window(
                original_idx=original_idx,
                window_start=window_start,
                prefix_label_mask_len=prefix_label_mask_len,
                input_ids=input_ids,
                labels=labels,
                prompt_encodings=prompt_encodings,
                answer_encodings=answer_encodings,
            )

        sample = self.pad_labels(labels, self.cfg.tokenizer.max_length)
        sample.update(
            self.pad_tokens(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_length=self.cfg.tokenizer.max_length,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        )

        sample.update(
            self.pad_tokens(
                answer_encodings[-1],
                attention_mask=torch.ones_like(answer_encodings[-1]),
                max_length=self.cfg.tokenizer.max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                direction="right",
                prefix="answer_",
            )
        )

        if window_start is not None:
            prompt_input_ids = input_ids[labels == -100]
        else:
            answer_encodings[-1] = torch.empty(0)
            prompt_input_ids = torch.cat(
                [
                    torch.cat([prompt_encoding, answer_encoding])
                    for prompt_encoding, answer_encoding in zip(
                        prompt_encodings, answer_encodings, strict=False
                    )
                ]
            )
        sample.update(
            self.pad_tokens(
                prompt_input_ids,
                attention_mask=torch.ones_like(prompt_input_ids),
                max_length=self.cfg.tokenizer.max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                prefix="prompt_",
            )
        )
        return sample


def install_structure_aware_sliding_window() -> None:
    """Install the structure-aware dataset class after the fast index layer."""
    if base_ds.CustomDataset is StructureAwareSlidingWindowDataset:
        return
    if issubclass(base_ds.CustomDataset, FastSlidingWindowDataset):
        base_ds.CustomDataset = StructureAwareSlidingWindowDataset
