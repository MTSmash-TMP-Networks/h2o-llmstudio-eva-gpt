from types import SimpleNamespace

from llm_studio.src.utils import large_chat_deepspeed_runtime as runtime


class SizedDataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


class Sampler:
    def __init__(self):
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


class Loader:
    def __init__(self, dataset, workers=8):
        self.dataset = dataset
        self.num_workers = workers
        self.sampler = Sampler()
        self.batch_size = 4

    def __len__(self):
        return 10

    def __iter__(self):
        return iter(())


def _cfg(*, workers=8):
    return SimpleNamespace(
        problem_type="text_causal_language_modeling",
        environment=SimpleNamespace(
            number_of_workers=workers,
            use_deepspeed=True,
            _distributed=True,
            _world_size=4,
            _local_rank=0,
        ),
        training=SimpleNamespace(lora=False),
    )


def test_large_chat_train_loader_forces_zero_workers_and_restores_cfg(monkeypatch):
    cfg = _cfg(workers=8)
    dataset = SizedDataset(360_104)
    observed = {}

    def original(*, train_ds, cfg):
        observed["workers"] = cfg.environment.number_of_workers
        return Loader(train_ds, workers=observed["workers"])

    monkeypatch.setattr(runtime, "_ORIGINAL_GET_TRAIN_DATALOADER", original)
    monkeypatch.setattr(runtime, "_current_rss_bytes", lambda: 11 * 1024**3)

    loader = runtime._get_train_dataloader_low_memory(train_ds=dataset, cfg=cfg)

    assert observed["workers"] == 0
    assert loader.num_workers == 0
    assert cfg.environment.number_of_workers == 8
    assert getattr(loader, runtime._PRESERVE_LOADER_MARKER) is True


def test_preserved_loader_advances_distributed_sampler_each_epoch():
    raw_loader = Loader(SizedDataset(360_104), workers=0)
    loader = runtime._EpochAwareLoaderProxy(raw_loader)

    iter(loader)
    iter(loader)
    iter(loader)

    assert raw_loader.sampler.epochs == [0, 1, 2]


def test_large_chat_validation_loader_forces_zero_workers(monkeypatch):
    cfg = _cfg(workers=6)
    dataset = SizedDataset(157_243)
    observed = {}

    def original(*, val_ds, cfg):
        observed["workers"] = cfg.environment.number_of_workers
        return Loader(val_ds, workers=observed["workers"])

    monkeypatch.setattr(runtime, "_ORIGINAL_GET_VAL_DATALOADER", original)
    monkeypatch.setattr(runtime, "_current_rss_bytes", lambda: 9 * 1024**3)

    loader = runtime._get_val_dataloader_low_memory(val_ds=dataset, cfg=cfg)

    assert observed["workers"] == 0
    assert loader.num_workers == 0
    assert cfg.environment.number_of_workers == 6


def test_large_chat_deepspeed_keeps_existing_loaders(monkeypatch):
    cfg = _cfg()
    train_loader = runtime._EpochAwareLoaderProxy(Loader(SizedDataset(360_104), 0))
    val_loader = Loader(SizedDataset(157_243), 0)
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

    from llm_studio.src.utils import modeling_utils

    monkeypatch.setattr(modeling_utils, "get_ds_config", lambda cfg: {"zero": 2})
    monkeypatch.setattr(runtime.deepspeed, "initialize", fake_initialize)
    monkeypatch.setattr(
        runtime,
        "_ORIGINAL_WRAP_MODEL_DISTRIBUTED",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("fallback called")),
    )

    result = runtime._wrap_model_distributed_low_memory(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        cfg=cfg,
    )

    assert captured["training_data"] is None
    assert captured["config_params"] == {"zero": 2}
    assert result[2] is train_loader
    assert result[3] is val_loader
    assert result[1] == "wrapped-optimizer"
    assert result[4] == "wrapped-scheduler"
    assert model.backbone == "engine"
    assert model.deepspeed_initialized is True


def test_wrap_defensively_rebuilds_workerful_loaders_without_marker(monkeypatch):
    cfg = _cfg(workers=7)
    train_dataset = SizedDataset(128)
    val_dataset = SizedDataset(64)
    train_loader = Loader(train_dataset, workers=7)
    val_loader = Loader(val_dataset, workers=7)
    optimizer = object()
    scheduler = object()
    model = SimpleNamespace(backbone=object(), deepspeed_initialized=False)

    def init_deepspeed():
        model.deepspeed_initialized = True

    model.init_deepspeed = init_deepspeed
    built = []

    def original_train(*, train_ds, cfg):
        built.append(("train", cfg.environment.number_of_workers))
        return Loader(train_ds, workers=cfg.environment.number_of_workers)

    def original_val(*, val_ds, cfg):
        built.append(("validation", cfg.environment.number_of_workers))
        return Loader(val_ds, workers=cfg.environment.number_of_workers)

    captured = {}

    def fake_initialize(**kwargs):
        captured.update(kwargs)
        return "engine", "wrapped-optimizer", None, "wrapped-scheduler"

    from llm_studio.src.utils import modeling_utils

    monkeypatch.setattr(runtime, "_ORIGINAL_GET_TRAIN_DATALOADER", original_train)
    monkeypatch.setattr(runtime, "_ORIGINAL_GET_VAL_DATALOADER", original_val)
    monkeypatch.setattr(modeling_utils, "get_ds_config", lambda cfg: {"zero": 2})
    monkeypatch.setattr(runtime.deepspeed, "initialize", fake_initialize)
    monkeypatch.setattr(
        runtime,
        "_ORIGINAL_WRAP_MODEL_DISTRIBUTED",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("fallback called")),
    )

    result = runtime._wrap_model_distributed_low_memory(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        cfg=cfg,
    )

    assert built == [("train", 0), ("validation", 0)]
    assert result[2].num_workers == 0
    assert result[3].num_workers == 0
    assert getattr(result[2], runtime._PRESERVE_LOADER_MARKER) is True
    assert captured["training_data"] is None
    assert cfg.environment.number_of_workers == 7


def test_small_deepspeed_chat_dataset_is_also_workerless(monkeypatch):
    cfg = _cfg(workers=8)
    dataset = SizedDataset(128)
    observed = {}

    def original(*, train_ds, cfg):
        observed["workers"] = cfg.environment.number_of_workers
        return Loader(train_ds, workers=observed["workers"])

    monkeypatch.setattr(runtime, "_ORIGINAL_GET_TRAIN_DATALOADER", original)
    monkeypatch.setattr(runtime, "_current_rss_bytes", lambda: 128 * 1024**2)

    result = runtime._get_train_dataloader_low_memory(train_ds=dataset, cfg=cfg)

    assert observed["workers"] == 0
    assert result.num_workers == 0
    assert getattr(result, runtime._PRESERVE_LOADER_MARKER) is True
    assert cfg.environment.number_of_workers == 8


def test_non_deepspeed_chat_keeps_original_worker_setting(monkeypatch):
    cfg = _cfg(workers=5)
    cfg.environment.use_deepspeed = False
    dataset = SizedDataset(128)
    expected = Loader(dataset, workers=5)
    calls = []

    def original(*, train_ds, cfg):
        calls.append((train_ds, cfg.environment.number_of_workers))
        return expected

    monkeypatch.setattr(runtime, "_ORIGINAL_GET_TRAIN_DATALOADER", original)

    result = runtime._get_train_dataloader_low_memory(train_ds=dataset, cfg=cfg)

    assert result is expected
    assert calls == [(dataset, 5)]


def test_rank_partitioned_dataset_is_left_to_existing_parquet_runtime(monkeypatch):
    cfg = _cfg()
    dataset = SizedDataset(360_104)
    setattr(dataset, runtime._RANK_PARTITIONED_MARKER, True)
    expected = Loader(dataset, workers=8)

    monkeypatch.setattr(
        runtime,
        "_ORIGINAL_GET_TRAIN_DATALOADER",
        lambda *, train_ds, cfg: expected,
    )
    monkeypatch.setattr(runtime, "_current_rss_bytes", lambda: 12 * 1024**3)

    result = runtime._get_train_dataloader_low_memory(train_ds=dataset, cfg=cfg)

    assert result is expected


def test_same_custom_train_validation_source_uses_automatic_split(monkeypatch):
    cfg = _cfg()
    cfg.dataset = SimpleNamespace(
        validation_strategy="custom",
        train_dataframe="/data/eva.csv",
        validation_dataframe="/data/eva.csv",
    )
    observed = []
    expected = (object(), object())

    def original(cfg):
        observed.append(cfg.dataset.validation_strategy)
        return expected

    monkeypatch.setattr(runtime, "_ORIGINAL_LOAD_TRAIN_VALID_DATA", original)

    result = runtime._load_train_valid_data_low_memory(cfg)

    assert result is expected
    assert observed == ["automatic"]
    assert cfg.dataset.validation_strategy == "custom"


def test_different_custom_train_validation_sources_keep_custom_path(monkeypatch):
    cfg = _cfg()
    cfg.dataset = SimpleNamespace(
        validation_strategy="custom",
        train_dataframe="/data/train.csv",
        validation_dataframe="/data/validation.csv",
    )
    observed = []
    expected = (object(), object())

    def original(cfg):
        observed.append(cfg.dataset.validation_strategy)
        return expected

    monkeypatch.setattr(runtime, "_ORIGINAL_LOAD_TRAIN_VALID_DATA", original)

    result = runtime._load_train_valid_data_low_memory(cfg)

    assert result is expected
    assert observed == ["custom"]
