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
        self.loss_sum: torch.Tensor | None = None
        self.loss_count = 0

    def report(self, loss: torch.Tensor) -> None:
        detached = loss.detach().float()
        if detached.numel() != 1:
            detached = detached.mean()
        if not torch.isfinite(detached):
            return
        self.loss_sum = detached if self.loss_sum is None else self.loss_sum + detached
        self.loss_count += 1

    def consume(self) -> float | None:
        distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if self.loss_count == 0 and not distributed:
            return None

        if self.loss_sum is not None:
            device = self.loss_sum.device
            loss_sum = self.loss_sum
        else:
            backend = torch.distributed.get_backend() if distributed else None
            if backend == "nccl":
                device = torch.device("cuda", torch.cuda.current_device())
            else:
                device = torch.device("cpu")
            loss_sum = torch.zeros((), dtype=torch.float32, device=device)

        values = torch.stack(
            [
                loss_sum.to(dtype=torch.float64),
                torch.tensor(
                    float(self.loss_count), dtype=torch.float64, device=device
                ),
            ]
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
    """Cosine decay with a conservative loss-driven learning-rate controller.

    A slow EMA is the stable reference and a faster EMA tracks the recent trend.
    Reductions require either a sustained fast-EMA spike or a genuine plateau.
    Relative trend detection uses a denominator floor so tiny absolute changes at
    already-low loss values are not amplified into false percentage spikes. The
    adaptive multiplier also has its own lower bound, separate from the cosine LR
    floor, so noisy batches can never collapse the effective learning rate to zero.
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
        reduction_factor: float = 0.85,
        spike_reduction_factor: float = 0.9,
        recovery_factor: float = 1.1,
        spike_ratio: float = 1.08,
        spike_patience: int = 8,
        improvement_threshold: float = 0.002,
        recovery_threshold: float = 0.003,
        min_adaptive_scale: float = 0.2,
        trend_denominator_floor: float = 0.1,
        plateau_patience: int | None = None,
        recovery_patience: int | None = None,
        cooldown_steps: int | None = None,
    ):
        self.num_warmup_steps = max(0, int(num_warmup_steps))
        self.num_training_steps = max(1, int(num_training_steps))
        self.min_learning_rate_ratio = min(
            max(float(min_learning_rate_ratio), 0.0), 1.0
        )
        self.ema_beta = min(max(float(ema_beta), 0.0), 0.999999)
        self.fast_ema_beta = min(max(float(fast_ema_beta), 0.0), 0.999999)
        self.reduction_factor = min(max(float(reduction_factor), 0.0), 1.0)
        self.spike_reduction_factor = min(
            max(float(spike_reduction_factor), 0.0), 1.0
        )
        self.recovery_factor = max(float(recovery_factor), 1.0)
        self.spike_ratio = max(float(spike_ratio), 1.0)
        self.spike_patience = max(1, int(spike_patience))
        self.improvement_threshold = max(float(improvement_threshold), 0.0)
        self.recovery_threshold = max(float(recovery_threshold), 0.0)
        self.min_adaptive_scale = min(max(float(min_adaptive_scale), 0.0), 1.0)
        self.trend_denominator_floor = max(float(trend_denominator_floor), 1e-12)
        self.plateau_patience = plateau_patience or max(
            100, self.num_training_steps // 150
        )
        self.recovery_patience = recovery_patience or max(
            5, self.plateau_patience // 5
        )
        self.cooldown_steps = cooldown_steps or max(50, self.plateau_patience // 2)
        self.loss_ema: float | None = None
        self.fast_loss_ema: float | None = None
        self.best_loss_ema = math.inf
        self.bad_steps = 0
        self.good_steps = 0
        self.spike_steps = 0
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

    def _bounded_adaptive_scale(self) -> float:
        return min(1.0, max(self.min_adaptive_scale, self.adaptive_scale))

    def _combined_factor(self, step: int) -> float:
        return max(
            self.min_learning_rate_ratio,
            self._cosine_factor(step) * self._bounded_adaptive_scale(),
        )

    def _effective_lr_range(self) -> tuple[float, float]:
        factor = self._combined_factor(self.last_epoch)
        learning_rates = [base_lr * factor for base_lr in self.base_lrs]
        return min(learning_rates), max(learning_rates)

    def _reset_detection_counters(self) -> None:
        self.bad_steps = 0
        self.good_steps = 0
        self.spike_steps = 0

    def _relative_loss_trend(self) -> float:
        if self.loss_ema is None or self.fast_loss_ema is None:
            return 0.0
        denominator = max(abs(self.loss_ema), self.trend_denominator_floor)
        return (self.fast_loss_ema - self.loss_ema) / denominator

    def _update_from_loss(self, loss: float | None) -> None:
        if loss is None or not math.isfinite(loss):
            return

        self.loss_observations += 1
        if self.loss_ema is None:
            self.loss_ema = loss
            self.fast_loss_ema = loss
        else:
            self.loss_ema = self.ema_beta * self.loss_ema + (
                1.0 - self.ema_beta
            ) * loss
            self.fast_loss_ema = self.fast_ema_beta * self.fast_loss_ema + (
                1.0 - self.fast_ema_beta
            ) * loss

        new_best = self.loss_ema < self.best_loss_ema * (
            1.0 - self.improvement_threshold
        )
        if new_best:
            self.best_loss_ema = self.loss_ema

        if self.last_epoch < self.num_warmup_steps:
            self._reset_detection_counters()
            return

        if self.loss_observations < self.minimum_observations:
            self._reset_detection_counters()
            return

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self._reset_detection_counters()
            return

        relative_trend = self._relative_loss_trend()
        spike_threshold = max(self.spike_ratio - 1.0, 0.0)

        if relative_trend > spike_threshold:
            self.spike_steps += 1
            self.bad_steps = 0
            self.good_steps = 0
            if self.spike_steps >= self.spike_patience:
                self._reduce_scale(
                    self.spike_reduction_factor,
                    f"sustained loss spike ({self.spike_steps} observations)",
                )
            return

        self.spike_steps = 0
        if relative_trend < -self.recovery_threshold:
            self.bad_steps = 0
            self.good_steps += 1
            if (
                self._bounded_adaptive_scale() < 1.0
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
        if self.bad_steps >= self.plateau_patience:
            self._reduce_scale(self.reduction_factor, "loss plateau")

    def _reduce_scale(self, factor: float, reason: str) -> None:
        old_scale = self._bounded_adaptive_scale()
        self.adaptive_scale = max(
            self.min_adaptive_scale,
            old_scale * factor,
        )
        self._reset_detection_counters()
        self.cooldown_remaining = self.cooldown_steps
        if self.loss_ema is not None:
            self.best_loss_ema = self.loss_ema
        if self.adaptive_scale < old_scale:
            min_lr, max_lr = self._effective_lr_range()
            logger.info(
                "Loss-aware scheduler reduced LR scale %.6f -> %.6f (%s, "
                "loss_ema=%.6f, fast_loss_ema=%.6f, trend=%+.4f, "
                "effective_lr=%.3e..%.3e)",
                old_scale,
                self.adaptive_scale,
                reason,
                self.loss_ema,
                self.fast_loss_ema,
                self._relative_loss_trend(),
                min_lr,
                max_lr,
            )

    def _increase_scale(self, factor: float, reason: str) -> None:
        old_scale = self._bounded_adaptive_scale()
        self.adaptive_scale = min(1.0, old_scale * factor)
        self._reset_detection_counters()
        self.cooldown_remaining = max(1, self.cooldown_steps // 2)
        if self.adaptive_scale > old_scale:
            min_lr, max_lr = self._effective_lr_range()
            logger.info(
                "Loss-aware scheduler recovered LR scale %.6f -> %.6f (%s, "
                "loss_ema=%.6f, fast_loss_ema=%.6f, trend=%+.4f, "
                "effective_lr=%.3e..%.3e)",
                old_scale,
                self.adaptive_scale,
                reason,
                self.loss_ema,
                self.fast_loss_ema,
                self._relative_loss_trend(),
                min_lr,
                max_lr,
            )

    def get_lr(self) -> list[float]:
        factor = self._combined_factor(self.last_epoch)
        return [base_lr * factor for base_lr in self.base_lrs]

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        legacy_state = "min_adaptive_scale" not in state_dict
        pre_low_loss_guard_state = "trend_denominator_floor" not in state_dict
        super().load_state_dict(state_dict)

        if legacy_state:
            self.reduction_factor = max(self.reduction_factor, 0.85)
            self.recovery_factor = max(self.recovery_factor, 1.1)

        if pre_low_loss_guard_state:
            self.trend_denominator_floor = max(
                getattr(self, "trend_denominator_floor", 0.0), 0.1
            )
            self.spike_reduction_factor = max(self.spike_reduction_factor, 0.9)
            self.spike_patience = max(getattr(self, "spike_patience", 0), 8)
            self.min_adaptive_scale = max(
                getattr(self, "min_adaptive_scale", 0.0), 0.2
            )
            self.recovery_factor = max(self.recovery_factor, 1.1)
        else:
            self.trend_denominator_floor = max(
                float(getattr(self, "trend_denominator_floor", 0.1)), 1e-12
            )

        self.adaptive_scale = self._bounded_adaptive_scale()
        self.spike_steps = getattr(self, "spike_steps", 0)

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
