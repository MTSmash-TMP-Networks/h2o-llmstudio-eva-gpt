from types import SimpleNamespace

from llm_studio.src.utils import data_utils, modeling_utils


class Loader:
    def __init__(self, dataset, batch_size=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = 0


def _cfg(*, problem_type="text_causal_language_modeling", workers=6):
    return SimpleNamespace(
        problem_type=problem_type,
        environment=SimpleNamespace(
            use_deepspeed=True,
            _distributed=True,
            _local_rank=0,
            _world_size=4,
            number_of_workers=workers,
            find_unused_parameters=False,
        ),
        training=SimpleNamespace(lora=False),
        architecture=SimpleNamespace(gradient_checkpointing=False),
    )


def test_core_deepspeed_causal_lm_never_passes_training_data(monkeypatch):
    cfg = _cfg()
    train_loader = Loader(object())
    val_loader = Loader(object())
    optimizer = object()
    scheduler = object()
    model = SimpleNamespace(backbone=object(), deepspeed_initialized=False)

    def init_deepspeed():
        model.deepspeed_initialized = True

    model.init_deepspeed = init_deepspeed
    captured = {}

    def fake_initialize(**kwargs):
        captured.update(kwargs)
        return "engine", "wrapped-optimizer", None, "wrapped-scheduler"

    monkeypatch.setattr(modeling_utils, "get_ds_config", lambda cfg: {"zero": 2})
    monkeypatch.setattr(modeling_utils.deepspeed, "initialize", fake_initialize)

    result = modeling_utils.wrap_model_distributed(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        cfg=cfg,
    )

    assert captured["training_data"] is None
    assert result[2] is train_loader
    assert result[3] is val_loader
    assert result[1] == "wrapped-optimizer"
    assert result[4] == "wrapped-scheduler"
    assert model.backbone == "engine"
    assert model.deepspeed_initialized is True


def test_core_worker_count_is_zero_for_distributed_deepspeed_causal_lm():
    cfg = _cfg(workers=8)

    assert data_utils._effective_dataloader_workers(cfg) == 0
    assert cfg.environment.number_of_workers == 8


def test_core_worker_count_keeps_config_for_non_deepspeed():
    cfg = _cfg(workers=5)
    cfg.environment.use_deepspeed = False

    assert data_utils._effective_dataloader_workers(cfg) == 5


def test_core_worker_count_keeps_config_for_non_causal_problem():
    cfg = _cfg(problem_type="text_sequence_classification", workers=3)

    assert data_utils._effective_dataloader_workers(cfg) == 3
