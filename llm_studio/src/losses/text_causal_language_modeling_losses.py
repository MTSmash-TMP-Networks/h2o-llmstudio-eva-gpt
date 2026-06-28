import logging
from collections.abc import KeysView
from typing import Any

from torch import nn

logger = logging.getLogger(__name__)


IGNORE_INDEX = -100


def _zero_loss_like(logits):
    """
    Return a finite zero loss that stays connected to the computation graph.

    CrossEntropyLoss can return NaN when all labels in a batch are ignored.
    This helper lets the training loop continue safely for such batches while
    still producing a tensor that supports backward().
    """
    return logits.sum() * 0.0


def _has_trainable_labels(labels):
    """Return True when at least one label contributes to the loss."""
    return labels.ne(IGNORE_INDEX).any()


class TokenAveragedCrossEntropyLoss(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self._empty_label_batches = 0

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        if not _has_trainable_labels(shift_labels):
            self._empty_label_batches += 1
            if self._empty_label_batches <= 5:
                logger.warning(
                    "Skipping loss for batch because all shifted labels are masked "
                    "with -100. This usually means an empty answer, too short answer, "
                    "or a truncated/sliding-window sample without trainable answer "
                    "tokens."
                )
            return _zero_loss_like(shift_logits)

        return self.loss_fn(shift_logits, shift_labels)


class SampleAveragedCrossEntropyLoss(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self._empty_label_batches = 0

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        losses = []
        for i in range(labels.shape[0]):
            sample_logits = shift_logits[i].contiguous()
            sample_labels = shift_labels[i].contiguous()

            sample_logits = sample_logits.view(-1, sample_logits.size(-1))
            sample_labels = sample_labels.view(-1)

            if not _has_trainable_labels(sample_labels):
                continue

            losses.append(self.loss_fn(sample_logits, sample_labels))

        if not losses:
            self._empty_label_batches += 1
            if self._empty_label_batches <= 5:
                logger.warning(
                    "Skipping loss for batch because all samples only contain "
                    "masked labels (-100)."
                )
            return _zero_loss_like(shift_logits)

        return sum(losses) / len(losses)


class Losses:
    """Losses factory."""

    _losses = {
        "TokenAveragedCrossEntropy": TokenAveragedCrossEntropyLoss,
        "SampleAveragedCrossEntropy": SampleAveragedCrossEntropyLoss,
    }

    @classmethod
    def names(cls) -> KeysView:
        return cls._losses.keys()

    @classmethod
    def get(cls, name: str) -> Any:
        """Access to Losses.

        Args:
            name: losses name
        Returns:
            A class to build the Losses
        """
        return cls._losses.get(name, TokenAveragedCrossEntropyLoss)
