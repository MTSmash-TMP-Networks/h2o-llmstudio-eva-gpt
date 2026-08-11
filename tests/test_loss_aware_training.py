from types import SimpleNamespace

import torch
from torch.nn import functional as F

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


def _make_scheduler(**kwargs):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = LossAwareCosineScheduler(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=1000,
        **kwargs,
    )
    scheduler.minimum_observations = 1
    return optimizer, scheduler


def _step_with_loss(optimizer, scheduler, loss_value):
    optimizer.step()
    report_training_loss(torch.as_tensor(loss_value, dtype=torch.float32))
    scheduler.step()


def test_stable_token_loss_is_finite_and_supports_backward():
    loss_fn = StableTokenCrossEntropyLoss(_cfg())
    logits = torch.randn(2, 6, 17, requires_grad=True)
    labels = torch.randint(0, 17, (2, 6))

    loss = loss_fn(logits, labels)

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_stable_token_loss_defaults_to_unsmoothed_cross_entropy():
    loss_fn = StableTokenCrossEntropyLoss(_cfg())
    logits = torch.randn(2, 6, 17, requires_grad=True)
    labels = torch.randint(0, 17, (2, 6))
    labels[0, 2] = -100

    loss = loss_fn(logits, labels)
    expected = F.cross_entropy(
        logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
        labels[..., 1:].contiguous().view(-1),
        ignore_index=-100,
        label_smoothing=0.0,
    )

    assert loss_fn.label_smoothing == 0.0
    assert torch.allclose(loss, expected)


def test_stable_token_loss_honors_explicit_label_smoothing():
    loss_fn = StableTokenCrossEntropyLoss(_cfg(stable_loss_label_smoothing=0.01))
    logits = torch.randn(2, 6, 17, requires_grad=True)
    labels = torch.randint(0, 17, (2, 6))

    loss = loss_fn(logits, labels)
    expected = F.cross_entropy(
        logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
        labels[..., 1:].contiguous().view(-1),
        ignore_index=-100,
        label_smoothing=0.01,
    )

    assert loss_fn.label_smoothing == 0.01
    assert torch.allclose(loss, expected)


def test_stable_token_loss_handles_fully_masked_batch():
    loss_fn = StableTokenCrossEntropyLoss(_cfg())
    logits = torch.randn(2, 6, 17, requires_grad=True)
    labels = torch.full((2, 6), -100)

    loss = loss_fn(logits, labels)

    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None


def test_training_loss_monitor_averages_without_storing_graph():
    _TRAINING_LOSS_MONITOR.reset()
    loss_a = torch.tensor(2.0, requires_grad=True)
    loss_b = torch.tensor(4.0, requires_grad=True)

    report_training_loss(loss_a)
    report_training_loss(loss_b)

    assert _TRAINING_LOSS_MONITOR.loss_sum is not None
    assert not _TRAINING_LOSS_MONITOR.loss_sum.requires_grad
    assert _TRAINING_LOSS_MONITOR.consume() == 3.0


def test_training_loss_monitor_reduces_non_scalar_losses():
    _TRAINING_LOSS_MONITOR.reset()

    report_training_loss(torch.tensor([2.0, 4.0]))

    assert _TRAINING_LOSS_MONITOR.consume() == 3.0


def test_loss_aware_scheduler_uses_low_loss_safe_defaults():
    _TRAINING_LOSS_MONITOR.reset()
    _, scheduler = _make_scheduler()

    assert scheduler.spike_reduction_factor == 0.9
    assert scheduler.spike_patience == 8
    assert scheduler.min_adaptive_scale == 0.2
    assert scheduler.trend_denominator_floor == 0.1


def test_observed_low_loss_gap_is_not_magnified_into_false_spike():
    _TRAINING_LOSS_MONITOR.reset()
    _, scheduler = _make_scheduler()
    scheduler.loss_ema = 0.053792
    scheduler.fast_loss_ema = 0.060804

    relative_trend = scheduler._relative_loss_trend()

    assert abs(relative_trend - 0.07012) < 1e-6
    assert relative_trend < scheduler.spike_ratio - 1.0


def test_observed_low_loss_gap_does_not_accumulate_spike_counter():
    _TRAINING_LOSS_MONITOR.reset()
    _, scheduler = _make_scheduler(plateau_patience=100)
    scheduler.loss_ema = 0.053792
    scheduler.fast_loss_ema = 0.060804
    scheduler.best_loss_ema = 0.053792
    scheduler.loss_observations = scheduler.minimum_observations

    scheduler._update_from_loss(0.053792)

    assert scheduler.spike_steps == 0
    assert scheduler.adaptive_scale == 1.0


def test_loss_aware_scheduler_reduces_lr_scale_on_plateau():
    _TRAINING_LOSS_MONITOR.reset()
    optimizer, scheduler = _make_scheduler(
        plateau_patience=5,
        cooldown_steps=2,
        spike_ratio=10.0,
    )

    for _ in range(7):
        _step_with_loss(optimizer, scheduler, 2.0)

    assert scheduler.adaptive_scale < 1.0
    assert optimizer.param_groups[0]["lr"] < 1e-3


def test_single_noisy_batch_does_not_reduce_lr_scale():
    _TRAINING_LOSS_MONITOR.reset()
    optimizer, scheduler = _make_scheduler(
        plateau_patience=100,
        spike_ratio=1.05,
        spike_patience=3,
        ema_beta=0.98,
        fast_ema_beta=0.2,
    )

    for _ in range(5):
        _step_with_loss(optimizer, scheduler, 1.0)
    initial_scale = scheduler.adaptive_scale

    _step_with_loss(optimizer, scheduler, 2.0)

    assert scheduler.adaptive_scale == initial_scale
    assert scheduler.spike_steps == 1


def test_sustained_ema_spike_reduces_lr_scale():
    _TRAINING_LOSS_MONITOR.reset()
    optimizer, scheduler = _make_scheduler(
        plateau_patience=100,
        cooldown_steps=2,
        spike_ratio=1.05,
        spike_patience=3,
        ema_beta=0.98,
        fast_ema_beta=0.2,
    )

    for _ in range(5):
        _step_with_loss(optimizer, scheduler, 1.0)
    initial_scale = scheduler.adaptive_scale

    for _ in range(4):
        _step_with_loss(optimizer, scheduler, 2.0)

    assert scheduler.adaptive_scale < initial_scale


def test_real_low_loss_spike_still_reduces_lr_scale():
    _TRAINING_LOSS_MONITOR.reset()
    optimizer, scheduler = _make_scheduler(
        plateau_patience=100,
        cooldown_steps=2,
        spike_patience=3,
        ema_beta=0.98,
        fast_ema_beta=0.2,
    )

    for _ in range(5):
        _step_with_loss(optimizer, scheduler, 0.05)
    initial_scale = scheduler.adaptive_scale

    for _ in range(4):
        _step_with_loss(optimizer, scheduler, 0.15)

    assert scheduler.adaptive_scale < initial_scale


def test_adaptive_scale_never_falls_below_its_own_floor():
    _TRAINING_LOSS_MONITOR.reset()
    _, scheduler = _make_scheduler(min_adaptive_scale=0.1)

    for _ in range(50):
        scheduler._reduce_scale(0.5, "test")

    assert scheduler.adaptive_scale == 0.1


def test_default_adaptive_scale_never_falls_below_point_two():
    _TRAINING_LOSS_MONITOR.reset()
    _, scheduler = _make_scheduler()

    for _ in range(50):
        scheduler._reduce_scale(0.5, "test")

    assert scheduler.adaptive_scale == 0.2


def test_cooldown_does_not_accumulate_plateau_or_spike_counters():
    _TRAINING_LOSS_MONITOR.reset()
    optimizer, scheduler = _make_scheduler(
        plateau_patience=2,
        cooldown_steps=3,
        spike_ratio=10.0,
    )
    scheduler._reduce_scale(0.85, "test")

    for _ in range(3):
        _step_with_loss(optimizer, scheduler, 2.0)

    assert scheduler.bad_steps == 0
    assert scheduler.good_steps == 0
    assert scheduler.spike_steps == 0


def test_reduction_resets_plateau_reference():
    _TRAINING_LOSS_MONITOR.reset()
    _, scheduler = _make_scheduler()
    scheduler.loss_ema = 1.25
    scheduler.fast_loss_ema = 1.25
    scheduler.best_loss_ema = 0.9

    scheduler._reduce_scale(0.85, "test")

    assert scheduler.best_loss_ema == 1.25


def test_loss_aware_scheduler_recovers_lr_after_sustained_improvement():
    _TRAINING_LOSS_MONITOR.reset()
    optimizer, scheduler = _make_scheduler(
        plateau_patience=5,
        recovery_patience=2,
        cooldown_steps=1,
        recovery_factor=1.2,
        recovery_threshold=0.001,
        ema_beta=0.9,
        fast_ema_beta=0.2,
        spike_ratio=10.0,
    )
    scheduler.adaptive_scale = 0.5

    for loss_value in (2.0, 1.8, 1.6, 1.4, 1.2):
        _step_with_loss(optimizer, scheduler, loss_value)

    assert scheduler.adaptive_scale > 0.5
    assert scheduler.adaptive_scale <= 1.0


def test_pre_low_loss_guard_state_is_migrated_to_safe_defaults():
    _TRAINING_LOSS_MONITOR.reset()
    _, source_scheduler = _make_scheduler()
    old_state = source_scheduler.state_dict()
    old_state.pop("trend_denominator_floor")
    old_state["adaptive_scale"] = 0.107132
    old_state["min_adaptive_scale"] = 0.1
    old_state["spike_reduction_factor"] = 0.8
    old_state["spike_patience"] = 5

    _, scheduler = _make_scheduler()
    scheduler.load_state_dict(old_state)

    assert scheduler.trend_denominator_floor == 0.1
    assert scheduler.min_adaptive_scale == 0.2
    assert scheduler.adaptive_scale == 0.2
    assert scheduler.spike_reduction_factor == 0.9
    assert scheduler.spike_patience == 8


def test_legacy_collapsed_scheduler_state_is_migrated_safely():
    _TRAINING_LOSS_MONITOR.reset()
    _, scheduler = _make_scheduler()
    legacy_state = scheduler.state_dict()
    legacy_state.pop("trend_denominator_floor")
    legacy_state.pop("min_adaptive_scale")
    legacy_state.pop("spike_patience")
    legacy_state.pop("spike_steps")
    legacy_state["adaptive_scale"] = 1e-5
    legacy_state["reduction_factor"] = 0.7
    legacy_state["spike_reduction_factor"] = 0.5
    legacy_state["recovery_factor"] = 1.05

    scheduler.load_state_dict(legacy_state)

    assert scheduler.adaptive_scale == 0.2
    assert scheduler.reduction_factor >= 0.85
    assert scheduler.spike_reduction_factor >= 0.9
    assert scheduler.recovery_factor >= 1.1
    assert scheduler.spike_patience >= 8
    assert scheduler.trend_denominator_floor >= 0.1
