import logging
import os
import re
import shutil
from contextlib import contextmanager

import pandas as pd
from datasets import load_dataset
from h2o_wave import Q, ui
from huggingface_hub import HfApi, hf_hub_download

from llm_studio.app_utils.config import default_cfg
from llm_studio.app_utils.huggingface_parquet import (
    is_parquet_directory,
    limit_parquet_directory_reads,
    parquet_directory_row_count,
    write_parquet_directory_metadata,
)
from llm_studio.app_utils.sections.common import clean_dashboard
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


async def _render_huggingface_import_form(
    q: Q,
    error: str | None = "",
    warning: str | None = "",
    info: str | None = "",
) -> None:
    """Render the Hugging Face source form directly.

    The normal dataset importer builds a source form and the previous extension then
    mutated that already-rendered Wave card to inject the config/subset textbox.
    On a source dropdown trigger this can leave the Wave request in a permanent
    loading state. Build the complete Hugging Face form in one pass instead.
    """

    await clean_dashboard(q, mode="full")
    q.client["nav/active"] = "dataset/import"
    q.client["dataset/import/source"] = "Huggingface"

    if q.client["dataset/import/huggingface_split"] is None:
        q.client["dataset/import/huggingface_split"] = "train"
    if q.client["dataset/import/huggingface_api_token"] is None:
        q.client["dataset/import/huggingface_api_token"] = q.client[
            "default_huggingface_api_token"
        ]
    if q.client["dataset/import/huggingface_config"] is None:
        q.client["dataset/import/huggingface_config"] = ""

    import_choices = [
        ui.choice("Upload", "Upload"),
        ui.choice("Local", "Local"),
        ui.choice("S3", "AWS S3"),
        ui.choice("Azure", "Azure Datalake"),
        ui.choice("H2O-Drive", "H2O-Drive"),
        ui.choice("Kaggle", "Kaggle"),
        ui.choice("Huggingface", "Hugging Face"),
    ]

    items = [
        ui.text_l("Import dataset"),
        ui.dropdown(
            name="dataset/import/source",
            label="Source",
            value="Huggingface",
            choices=import_choices,
            trigger=True,
            tooltip="Source of dataset import",
        ),
        ui.textbox(
            name="dataset/import/huggingface_dataset",
            label="Hugging Face dataset",
            value=q.client["dataset/import/huggingface_dataset"],
            required=True,
            placeholder="wikimedia/wikipedia",
            tooltip="Name of the Hugging Face dataset, for example wikimedia/wikipedia",
        ),
        ui.textbox(
            name="dataset/import/huggingface_config",
            label="Hugging Face config / subset",
            value=q.client["dataset/import/huggingface_config"],
            required=False,
            placeholder="20231101.de",
            tooltip=(
                "Optional dataset configuration/subset, for example 20231101.de "
                "for wikimedia/wikipedia. Leave empty for datasets without configs."
            ),
        ),
        ui.textbox(
            name="dataset/import/huggingface_split",
            label="Split",
            value=q.client["dataset/import/huggingface_split"],
            required=True,
            password=False,
            tooltip="Split of the dataset, usually train",
        ),
        ui.textbox(
            name="dataset/import/huggingface_api_token",
            label="Hugging Face API token",
            value=q.client["dataset/import/huggingface_api_token"],
            required=False,
            password=True,
            tooltip="Optional Hugging Face API token",
        ),
    ]

    allowed_types = ", ".join(default_cfg.allowed_file_extensions)
    allowed_types = " or".join(allowed_types.rsplit(",", 1))
    items += [
        ui.message_bar(
            type="info",
            text=(info or "")
            + "Hugging Face datasets are imported as CSV/Parquet-compatible data. "
            + f"Supported local representations: {allowed_types}.",
        ),
        ui.message_bar(type="error", text=error or ""),
        ui.message_bar(type="warning", text=warning or ""),
    ]

    q.page["dataset/import"] = ui.form_card(box="content", items=items)
    q.client.delete_cards.add("dataset/import")

    q.page["dataset/import/footer"] = ui.form_card(
        box="footer",
        items=[
            ui.inline(
                items=[
                    ui.button(
                        name="dataset/import/2", label="Continue", primary=True
                    ),
                    ui.button(name="dataset/list", label="Abort"),
                ],
                justify="start",
            )
        ],
    )
    q.client.delete_cards.add("dataset/import/footer")

    q.client["dataset/import/id"] = None
    q.client["dataset/import/cfg_file"] = None


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

    # Render the Hugging Face source form atomically. Do not mutate the form card
    # after the original Wave handler returns; that was the cause of the endless
    # loading state when switching the Source dropdown to Hugging Face.
    if step == 1 and q.client["dataset/import/source"] == "Huggingface":
        await _render_huggingface_import_form(
            q=q,
            error=error,
            warning=warning,
            info=info,
        )
        return

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
