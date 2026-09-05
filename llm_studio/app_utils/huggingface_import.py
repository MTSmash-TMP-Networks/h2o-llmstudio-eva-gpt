import logging
import os
import re
import shutil

from datasets import load_dataset
from h2o_wave import Q, ui

from llm_studio.app_utils.utils import get_data_dir, get_valid_temp_data_folder

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


async def huggingface_download_with_config(
    q: Q, huggingface_dataset: str, huggingface_split: str
) -> tuple[str, str]:
    """Download a Hugging Face dataset with optional config/subset support.

    The optional config is read from ``dataset/import/huggingface_config``.
    Datasets are written directly to Parquet to avoid materializing the whole
    dataset as a pandas DataFrame during import.
    """

    huggingface_path = f"{get_data_dir(q)}/tmp"
    huggingface_path = get_valid_temp_data_folder(q, huggingface_path)

    if os.path.exists(huggingface_path):
        shutil.rmtree(huggingface_path)
    os.makedirs(huggingface_path, exist_ok=True)

    token = _clean_optional_value(q.client["dataset/import/huggingface_api_token"])
    config = _clean_optional_value(q.client["dataset/import/huggingface_config"])

    load_kwargs = {
        "split": huggingface_split,
        "token": token,
    }
    if config is None:
        dataset = load_dataset(huggingface_dataset, **load_kwargs)
    else:
        dataset = load_dataset(huggingface_dataset, config, **load_kwargs)

    filename_parts = [huggingface_dataset.split("/")[-1]]
    if config is not None:
        filename_parts.append(config)
    filename_parts.append(huggingface_split)
    filename = _safe_filename_part("_".join(filename_parts))
    dataset_path = os.path.join(huggingface_path, f"{filename}.pq")

    # Hugging Face Dataset.to_parquet writes Arrow batches directly and avoids
    # the extra full-size pandas copy used by the original importer.
    dataset.to_parquet(dataset_path)

    return huggingface_path, filename


async def dataset_import_with_huggingface_config(
    q: Q,
    step: int,
    edit: bool | None = False,
    error: str | None = "",
    warning: str | None = "",
    info: str | None = "",
    allow_merge: bool = True,
) -> None:
    """Wrap the existing import wizard and add an optional HF config field."""

    if _ORIGINAL_DATASET_IMPORT is None:
        raise RuntimeError("Hugging Face import extension is not installed.")

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

    # Avoid duplicate fields when Wave redraws the source form.
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

    # Hugging Face form order is: title, source, dataset, split, token, messages.
    # Insert the config directly after the dataset name.
    insert_at = min(3, len(items))
    items.insert(insert_at, config_field)
    card.items = items


def install_huggingface_import_extension() -> None:
    """Install the Hugging Face config/subset extension into the Wave app."""

    global _ORIGINAL_DATASET_IMPORT, _PATCH_INSTALLED

    if _PATCH_INSTALLED:
        return

    from llm_studio.app_utils import handlers
    from llm_studio.app_utils.sections import dataset as dataset_section

    _ORIGINAL_DATASET_IMPORT = dataset_section.dataset_import

    # The original dataset_import function resolves huggingface_download from
    # its module globals, so replacing it here also affects recursive wizard
    # transitions performed by the original function.
    dataset_section.huggingface_download = huggingface_download_with_config
    dataset_section.dataset_import = dataset_import_with_huggingface_config

    # handlers imported dataset_import by name, therefore patch that reference
    # as well so UI events enter the wrapped wizard.
    handlers.dataset_import = dataset_import_with_huggingface_config

    _PATCH_INSTALLED = True
    logger.info("Hugging Face dataset config/subset import support enabled")
