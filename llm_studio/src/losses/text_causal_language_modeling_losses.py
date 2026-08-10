import logging
from collections.abc import KeysView
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from llm_studio.src.schedulers import report_training_loss

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
        self.report_loss = getattr(cfg.training, "schedule", "") == "LossAwareCosine"

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

        loss = self.loss_fn(shift_logits, shift_labels)
        if self.training and self.report_loss:
            report_training_loss(loss)
        return loss


class StableTokenCrossEntropyLoss(nn.Module):
    """Stable token loss with optional label smoothing and z-loss.

    Label smoothing is opt-in. Keeping the default at zero avoids introducing an
    artificial positive training-loss floor that can hide continued model
    improvement. A non-zero value can still be supplied through
    ``stable_loss_label_smoothing`` when deliberate smoothing is desired.
    """

    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        training_cfg = cfg.training
        self.label_smoothing = float(
            getattr(training_cfg, "stable_loss_label_smoothing", 0.0)
        )
        self.z_loss_coefficient = float(
            getattr(training_cfg, "stable_loss_z_loss_coefficient", 0.0)
        )
        self._empty_label_batches = 0
        self.report_loss = getattr(cfg.training, "schedule", "") == "LossAwareCosine"

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        valid_mask = flat_labels.ne(IGNORE_INDEX)

        if not valid_mask.any():
            self._empty_label_batches += 1
            if self._empty_label_batches <= 5:
                logger.warning(
                    "Skipping stable loss because all shifted labels are masked "
                    "with -100."
                )
            return _zero_loss_like(flat_logits)

        cross_entropy = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=IGNORE_INDEX,
            label_smoothing=self.label_smoothing,
        )

        if self.z_loss_coefficient > 0:
            log_z = torch.logsumexp(flat_logits.float(), dim=-1)
            z_loss = log_z[valid_mask].square().mean().to(cross_entropy.dtype)
            loss = cross_entropy + self.z_loss_coefficient * z_loss
        else:
            loss = cross_entropy

        if self.training and self.report_loss:
            report_training_loss(loss)
        return loss


class SampleAveragedCrossEntropyLoss(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self._empty_label_batches = 0
        self.report_loss = getattr(cfg.training, "schedule", "") == "LossAwareCosine"

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

        loss = sum(losses) / len(losses)
        if self.training and self.report_loss:
            report_training_loss(loss)
        return loss


class Losses:
    """Losses factory."""

    _losses = {
        "TokenAveragedCrossEntropy": TokenAveragedCrossEntropyLoss,
        "StableTokenCrossEntropy": StableTokenCrossEntropyLoss,
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