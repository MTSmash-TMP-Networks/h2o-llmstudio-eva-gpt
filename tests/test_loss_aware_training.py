from types import SimpleNamespace

import torch

from llm_studio.src.losses.text_causal_language_modeling_losses import (
    StableTokenCrossEntropyLoss,
)
from llm_studio.src.schedulers import (
    LossAwareCosineScheduler,
    _TRAINING_LOSS_MONITOR,
    report_training_loss,
)


def _cfg(**training_values):
    return SimpleNamespace(training=SimpleNamespace(**training_values))


def test_stable_token_loss_is_finite_and_supports_backward():
    loss_fn = StableTokenCrossEntropyLoss(_cfg())
    logits = torch.randn(2, 6, 17, requires_grad=True)
    labels = torch.randint(0, 17, (2, 6))

    loss = loss_fn(logits, labels)

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_stable_token_loss_handles_fully_masked_batch():
    loss_fn = StableTokenCrossEntropyLoss(_cfg())
    logits = torch.randn(2, 6, 17, requires_grad=True)
    labels = torch.full((2, 6), -100)

    loss = loss_fn(logits, labels)

    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None


def test_loss_aware_scheduler_reduces_lr_scale_on_plateau():
    _TRAINING_LOSS_MONITOR.reset()
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = LossAwareCosineScheduler(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=1000,
        plateau_patience=5,
        cooldown_steps=2,
    )
    scheduler.minimum_observations = 1

    for _ in range(7):
        optimizer.step()
        report_training_loss(torch.tensor(2.0))
        scheduler.step()

    assert scheduler.adaptive_scale < 1.0
    assert optimizer.param_groups[0]["lr"] < 1e-3


def test_loss_aware_scheduler_reduces_lr_scale_on_spike():
    _TRAINING_LOSS_MONITOR.reset()
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = LossAwareCosineScheduler(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=1000,
        plateau_patience=100,
        cooldown_steps=2,
    )
    scheduler.minimum_observations = 1

    optimizer.step()
    report_training_loss(torch.tensor(1.0))
    scheduler.step()
    initial_scale = scheduler.adaptive_scale

    optimizer.step()
    report_training_loss(torch.tensor(2.0))
    scheduler.step()

    assert scheduler.adaptive_scale < initial_scale


def test_loss_aware_scheduler_recovers_lr_after_sustained_improvement():
    _TRAINING_LOSS_MONITOR.reset()
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = LossAwareCosineScheduler(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=1000,
        plateau_patience=5,
        recovery_patience=2,
        cooldown_steps=1,
        recovery_factor=1.2,
        recovery_threshold=0.001,
        ema_beta=0.9,
        fast_ema_beta=0.2,
    )
    scheduler.minimum_observations = 1
    scheduler.adaptive_scale = 0.5

    for loss_value in (2.0, 1.8, 1.6, 1.4, 1.2):
        optimizer.step()
        report_training_loss(torch.tensor(loss_value))
        scheduler.step()

    assert scheduler.adaptive_scale > 0.5
    assert scheduler.adaptive_scale <= 1.0
