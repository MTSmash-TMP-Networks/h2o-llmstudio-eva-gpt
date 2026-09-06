import logging
import os
import re
import shutil
from contextlib import contextmanager

import pandas as pd
from datasets import load_dataset
from h2o_wave import Q, ui
from huggingface_hub import HfApi, hf_hub_download

from llm_studio.app_utils.huggingface_parquet import (
    is_parquet_directory,
    limit_parquet_directory_reads,
    parquet_directory_row_count,
    write_parquet_directory_metadata,
)
from llm_studio.app_utils.utils import get_data_dir, get_valid_temp_data_folder
from llm_studio.app_utils.wave_utils import busy_dialog

logger = logging.getLogger(__name__)

_ORIGINAL_DATASET_IMPORT = None
_PATCH_INSTALLED = False


def _clean_optional_value(value: object) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _safe_filename_part(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value or "dataset"


def _find_native_parquet_files(
    dataset_name: str,
    config: str | None,
    split: str,
    token: str | None,
) -> list[str]:
    """Find Parquet shards already stored in the Hugging Face dataset repo."""
    if config is None:
        return []

    api = HfApi(token=token)
    repo_files = api.list_repo_files(repo_id=dataset_name, repo_type="dataset")
    prefix = f"{config.rstrip('/')}/"
    split_prefix = f"{split}-"

    return sorted(
        path
        for path in repo_files
        if path.startswith(prefix)
        and os.path.basename(path).startswith(split_prefix)
        and path.lower().endswith(".parquet")
    )


def _download_native_parquet_dataset(
    dataset_name: str,
    config: str,
    split: str,
    token: str | None,
    repo_files: list[str],
    target_dir: str,
) -> None:
    """Download existing HF Parquet shards without rebuilding the full dataset."""
    for repo_file in repo_files:
        hf_hub_download(
            repo_id=dataset_name,
            filename=repo_file,
            repo_type="dataset",
            token=token,
            local_dir=target_dir,
        )

    metadata = {
        "source": "huggingface",
        "dataset": dataset_name,
        "config": config,
        "split": split,
        "files": repo_files,
    }

    # Wikimedia Wikipedia is already a clean continued-pretraining corpus. Expose
    # only its article body and map it to LLM Studio's raw-text column. This avoids
    # accidentally selecting id/url/title as supervised prompt/answer columns.
    if dataset_name == "wikimedia/wikipedia":
        metadata["columns"] = ["text"]
        metadata["column_aliases"] = {"text": "Text"}

    write_parquet_directory_metadata(target_dir, metadata)


async def huggingface_download_with_config(
    q: Q, huggingface_dataset: str, huggingface_split: str
) -> tuple[str, str]:
    """Download a Hugging Face dataset with optional config/subset support.

    Large datasets that already publish native Parquet shards are downloaded as a
    logical sharded dataframe. This avoids `load_dataset(...).to_parquet(...)`,
    which can otherwise duplicate many gigabytes and make the Wave UI appear stuck.
    Other datasets keep the existing datasets.load_dataset() fallback.
    """

    huggingface_path = f"{get_data_dir(q)}/tmp"
    huggingface_path = get_valid_temp_data_folder(q, huggingface_path)

    if os.path.exists(huggingface_path):
        shutil.rmtree(huggingface_path)
    os.makedirs(huggingface_path, exist_ok=True)

    token = _clean_optional_value(q.client["dataset/import/huggingface_api_token"])
    config = _clean_optional_value(q.client["dataset/import/huggingface_config"])
    split = _clean_optional_value(huggingface_split) or "train"

    filename_parts = [huggingface_dataset.split("/")[-1]]
    if config is not None:
        filename_parts.append(config)
    filename_parts.append(split)
    filename = _safe_filename_part("_".join(filename_parts))

    await busy_dialog(
        q=q,
        title="Preparing Hugging Face dataset",
        text="Checking dataset files and configuration...",
    )

    native_parquet_files = _find_native_parquet_files(
        dataset_name=huggingface_dataset,
        config=config,
        split=split,
        token=token,
    )

    if native_parquet_files and config is not None:
        # The .parquet suffix intentionally marks this directory as one logical
        # dataframe while the original HF shard filenames remain untouched inside.
        dataset_path = os.path.join(huggingface_path, f"{filename}.parquet")
        os.makedirs(dataset_path, exist_ok=True)
        logger.info(
            "Downloading %s native Parquet shards for %s/%s (%s)",
            len(native_parquet_files),
            huggingface_dataset,
            config,
            split,
        )
        await busy_dialog(
            q=q,
            title="Downloading Hugging Face dataset",
            text=(
                f"Downloading {len(native_parquet_files)} Parquet files directly. "
                "Large datasets can take a while, but no second full-size dataset "
                "copy will be created afterwards."
            ),
        )
        _download_native_parquet_dataset(
            dataset_name=huggingface_dataset,
            config=config,
            split=split,
            token=token,
            repo_files=native_parquet_files,
            target_dir=dataset_path,
        )
        return huggingface_path, filename

    await busy_dialog(
        q=q,
        title="Loading Hugging Face dataset",
        text="No native Parquet shard set found; using the standard dataset loader...",
    )
    load_kwargs = {"split": split, "token": token}
    if config is None:
        dataset = load_dataset(huggingface_dataset, **load_kwargs)
    else:
        dataset = load_dataset(huggingface_dataset, config, **load_kwargs)

    dataset_path = os.path.join(huggingface_path, f"{filename}.pq")
    dataset.to_parquet(dataset_path)
    return huggingface_path, filename


@contextmanager
def _fast_row_count_for_import():
    """Avoid loading a multi-GB Parquet directory just to store its row count."""
    from llm_studio.app_utils.sections import dataset as dataset_section

    original = dataset_section.read_dataframe_drop_missing_labels

    def row_count_reader(path, cfg):
        if is_parquet_directory(path):
            return pd.DataFrame(index=pd.RangeIndex(parquet_directory_row_count(path)))
        return original(path, cfg)

    dataset_section.read_dataframe_drop_missing_labels = row_count_reader
    try:
        yield
    finally:
        dataset_section.read_dataframe_drop_missing_labels = original


async def dataset_import_with_huggingface_config(
    q: Q,
    step: int,
    edit: bool | None = False,
    error: str | None = "",
    warning: str | None = "",
    info: str | None = "",
    allow_merge: bool = True,
) -> None:
    """Wrap the existing import wizard and add scalable HF dataset support."""

    if _ORIGINAL_DATASET_IMPORT is None:
        raise RuntimeError("Hugging Face import extension is not installed.")

    # Import preview/sanity checks need only a representative sample. Training later
    # still reads the complete logical dataframe.
    if step == 5:
        with limit_parquet_directory_reads(max_rows=2000):
            await _ORIGINAL_DATASET_IMPORT(
                q,
                step=step,
                edit=edit,
                error=error,
                warning=warning,
                info=info,
                allow_merge=allow_merge,
            )
    elif step == 6:
        with _fast_row_count_for_import():
            await _ORIGINAL_DATASET_IMPORT(
                q,
                step=step,
                edit=edit,
                error=error,
                warning=warning,
                info=info,
                allow_merge=allow_merge,
            )
    else:
        await _ORIGINAL_DATASET_IMPORT(
            q,
            step=step,
            edit=edit,
            error=error,
            warning=warning,
            info=info,
            allow_merge=allow_merge,
        )

    if step != 1 or q.client["dataset/import/source"] != "Huggingface":
        return

    if q.client["dataset/import/huggingface_config"] is None:
        q.client["dataset/import/huggingface_config"] = ""

    card = q.page["dataset/import"]
    items = list(card.items or [])
    config_field_name = "dataset/import/huggingface_config"

    if any(getattr(item, "name", None) == config_field_name for item in items):
        return

    config_field = ui.textbox(
        name=config_field_name,
        label="Hugging Face config / subset",
        value=q.client[config_field_name],
        required=False,
        tooltip=(
            "Optional dataset configuration/subset, for example 20231101.de "
            "for wikimedia/wikipedia. Leave empty for datasets without configs."
        ),
    )

    insert_at = min(3, len(items))
    items.insert(insert_at, config_field)
    card.items = items


def install_huggingface_import_extension() -> None:
    """Install Hugging Face config/subset and large-dataset import support."""

    global _ORIGINAL_DATASET_IMPORT, _PATCH_INSTALLED

    if _PATCH_INSTALLED:
        return

    from llm_studio.app_utils import handlers
    from llm_studio.app_utils.sections import dataset as dataset_section

    _ORIGINAL_DATASET_IMPORT = dataset_section.dataset_import
    dataset_section.huggingface_download = huggingface_download_with_config
    dataset_section.dataset_import = dataset_import_with_huggingface_config
    handlers.dataset_import = dataset_import_with_huggingface_config

    _PATCH_INSTALLED = True
    logger.info("Hugging Face large dataset import support enabled")
