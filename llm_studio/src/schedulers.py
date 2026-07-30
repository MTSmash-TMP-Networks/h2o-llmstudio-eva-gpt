import logging
import math
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from transformers import get_constant_schedule_with_warmup

logger = logging.getLogger(__name__)


class _TrainingLossMonitor:
    """Collect detached training losses between optimizer updates."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.loss_sum = 0.0
        self.loss_count = 0
        self.device = None

    def report(self, loss: torch.Tensor) -> None:
        detached = loss.detach()
        if not torch.isfinite(detached):
            return
        self.loss_sum += float(detached.float().item())
        self.loss_count += 1
        self.device = detached.device

    def consume(self) -> float | None:
        distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if self.loss_count == 0 and not distributed:
            return None

        device = self.device
        if device is None:
            backend = torch.distributed.get_backend() if distributed else None
            if backend == "nccl":
                device = torch.device("cuda", torch.cuda.current_device())
            else:
                device = torch.device("cpu")

        values = torch.tensor(
            [self.loss_sum, float(self.loss_count)],
            dtype=torch.float64,
            device=device,
        )
        if distributed:
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)

        self.reset()
        if values[1].item() <= 0:
            return None
        return float((values[0] / values[1]).item())


_TRAINING_LOSS_MONITOR = _TrainingLossMonitor()


def report_training_loss(loss: torch.Tensor) -> None:
    """Report a training loss for the loss-aware scheduler."""
    _TRAINING_LOSS_MONITOR.report(loss)


class LossAwareCosineScheduler(LRScheduler):
    """Cosine decay with a bounded, loss-driven learning-rate controller.

    A slow EMA provides a stable loss reference while a faster EMA detects the
    current trend. The scheduler lowers the learning rate on plateaus and strong
    loss spikes. After a reduction, it can gradually recover the learning rate
    when the loss resumes a sustained downward trend. Recovery is capped at the
    normal cosine schedule, so the adaptive controller can never exceed the
    configured base learning rate.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_learning_rate_ratio: float = 0.0,
        last_epoch: int = -1,
        ema_beta: float = 0.98,
        fast_ema_beta: float = 0.9,
        reduction_factor: float = 0.7,
        spike_reduction_factor: float = 0.5,
        recovery_factor: float = 1.05,
        spike_ratio: float = 1.25,
        improvement_threshold: float = 0.002,
        recovery_threshold: float = 0.003,
        plateau_patience: int | None = None,
        recovery_patience: int | None = None,
        cooldown_steps: int | None = None,
    ):
        self.num_warmup_steps = max(0, int(num_warmup_steps))
        self.num_training_steps = max(1, int(num_training_steps))
        self.min_learning_rate_ratio = float(min_learning_rate_ratio)
        self.ema_beta = float(ema_beta)
        self.fast_ema_beta = float(fast_ema_beta)
        self.reduction_factor = float(reduction_factor)
        self.spike_reduction_factor = float(spike_reduction_factor)
        self.recovery_factor = float(recovery_factor)
        self.spike_ratio = float(spike_ratio)
        self.improvement_threshold = float(improvement_threshold)
        self.recovery_threshold = float(recovery_threshold)
        self.plateau_patience = plateau_patience or max(
            50, self.num_training_steps // 200
        )
        self.recovery_patience = recovery_patience or max(
            5, self.plateau_patience // 4
        )
        self.cooldown_steps = cooldown_steps or max(20, self.plateau_patience // 2)
        self.loss_ema: float | None = None
        self.fast_loss_ema: float | None = None
        self.best_loss_ema = math.inf
        self.bad_steps = 0
        self.good_steps = 0
        self.cooldown_remaining = 0
        self.adaptive_scale = 1.0
        self.loss_observations = 0
        self.minimum_observations = max(20, min(100, self.plateau_patience // 2))
        _TRAINING_LOSS_MONITOR.reset()
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    def _cosine_factor(self, step: int) -> float:
        if step < self.num_warmup_steps:
            return float(step) / float(max(1, self.num_warmup_steps))

        progress = float(step - self.num_warmup_steps) / float(
            max(1, self.num_training_steps - self.num_warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        return max(
            self.min_learning_rate_ratio,
            0.5 * (1.0 + math.cos(math.pi * progress)),
        )

    def _update_from_loss(self, loss: float | None) -> None:
        if loss is None or not math.isfinite(loss):
            return

        self.loss_observations += 1
        previous_ema = self.loss_ema
        if previous_ema is None:
            self.loss_ema = loss
            self.fast_loss_ema = loss
        else:
            self.loss_ema = self.ema_beta * previous_ema + (
                1.0 - self.ema_beta
            ) * loss
            self.fast_loss_ema = self.fast_ema_beta * self.fast_loss_ema + (
                1.0 - self.fast_ema_beta
            ) * loss

        if self.last_epoch < self.num_warmup_steps:
            self.best_loss_ema = min(self.best_loss_ema, self.loss_ema)
            return

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        if self.loss_observations < self.minimum_observations:
            self.best_loss_ema = min(self.best_loss_ema, self.loss_ema)
            return

        is_spike = previous_ema is not None and loss > previous_ema * self.spike_ratio
        if is_spike and self.cooldown_remaining == 0:
            self._reduce_scale(self.spike_reduction_factor, "loss spike")
            return

        new_best = self.loss_ema < self.best_loss_ema * (
            1.0 - self.improvement_threshold
        )
        if new_best:
            self.best_loss_ema = self.loss_ema

        trend_denominator = max(abs(self.loss_ema), 1e-12)
        trend = (self.fast_loss_ema - self.loss_ema) / trend_denominator
        is_improving = trend < -self.recovery_threshold

        if is_improving:
            self.bad_steps = 0
            self.good_steps += 1
            if (
                self.adaptive_scale < 1.0
                and self.cooldown_remaining == 0
                and self.good_steps >= self.recovery_patience
            ):
                self._increase_scale(
                    self.recovery_factor, "sustained loss improvement"
                )
            return

        self.good_steps = 0
        if new_best:
            self.bad_steps = 0
            return

        self.bad_steps += 1
        if self.bad_steps >= self.plateau_patience and self.cooldown_remaining == 0:
            self._reduce_scale(self.reduction_factor, "loss plateau")

    def _reduce_scale(self, factor: float, reason: str) -> None:
        old_scale = self.adaptive_scale
        self.adaptive_scale = max(
            self.min_learning_rate_ratio,
            self.adaptive_scale * factor,
        )
        self.bad_steps = 0
        self.good_steps = 0
        self.cooldown_remaining = self.cooldown_steps
        if self.adaptive_scale < old_scale:
            logger.info(
                "Loss-aware scheduler reduced LR scale %.6f -> %.6f (%s, "
                "loss_ema=%.6f, fast_loss_ema=%.6f)",
                old_scale,
                self.adaptive_scale,
                reason,
                self.loss_ema,
                self.fast_loss_ema,
            )

    def _increase_scale(self, factor: float, reason: str) -> None:
        old_scale = self.adaptive_scale
        self.adaptive_scale = min(1.0, self.adaptive_scale * factor)
        self.bad_steps = 0
        self.good_steps = 0
        self.cooldown_remaining = max(1, self.cooldown_steps // 2)
        if self.adaptive_scale > old_scale:
            logger.info(
                "Loss-aware scheduler recovered LR scale %.6f -> %.6f (%s, "
                "loss_ema=%.6f, fast_loss_ema=%.6f)",
                old_scale,
                self.adaptive_scale,
                reason,
                self.loss_ema,
                self.fast_loss_ema,
            )

    def get_lr(self) -> list[float]:
        factor = max(
            self.min_learning_rate_ratio,
            self._cosine_factor(self.last_epoch) * self.adaptive_scale,
        )
        return [base_lr * factor for base_lr in self.base_lrs]

    def step(self, epoch: int | None = None) -> None:
        if epoch is None:
            self._update_from_loss(_TRAINING_LOSS_MONITOR.consume())
        super().step(epoch)


def constant_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, **kwargs
) -> LambdaLR:
    return get_constant_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=num_warmup_steps
    )


# adjusted from transformers
def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_learning_rate_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            min_learning_rate_ratio,
            0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# adjusted from transformers
def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_learning_rate_ratio: float = 0.0,
    last_epoch: int = -1,
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            min_learning_rate_ratio,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class Schedulers:
    """Schedulers factory."""

    _schedulers = {
        "Cosine": get_cosine_schedule_with_warmup,
        "LossAwareCosine": LossAwareCosineScheduler,
        "Linear": get_linear_schedule_with_warmup,
        "Constant": constant_schedule_with_warmup,
    }

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._schedulers.keys())

    @classmethod
    def get(cls, name: str) -> Any:
        """Access to Schedulers."""
        return cls._schedulers.get(name)
