import logging
import os

from llm_studio.app_utils.sections.chat_update import is_app_blocked_while_streaming
from llm_studio.src.utils.logging_utils import initialize_logging

os.environ["MKL_THREADING_LAYER"] = "GNU"

from h2o_wave import Q, app, copy_expando, main, ui  # noqa: F401

from llm_studio.app_utils.experiment_training_mode_fix import (
    install_experiment_training_mode_fix,
)
from llm_studio.app_utils.handlers import handle
from llm_studio.app_utils.huggingface_import import install_huggingface_import_extension
from llm_studio.app_utils.huggingface_parquet import install_parquet_directory_support
from llm_studio.app_utils.initializers import initialize_app, initialize_client
from llm_studio.app_utils.sections.common import heap_redact, interface
from llm_studio.app_utils.text_only_training import install_text_only_training_mode

install_parquet_directory_support()
install_huggingface_import_extension()
handle = install_text_only_training_mode(handle)
install_experiment_training_mode_fix()

logger = logging.getLogger(__name__)


def on_startup() -> None:
    initialize_logging()
    logger.info("Starting MaTeLiX AI Studio")


@app("/", on_startup=on_startup)
async def serve(q: Q) -> None:
    """Serving function."""

    # Chat is still being streamed but user clicks on another button.
    # Wait until streaming has been completed
    if await is_app_blocked_while_streaming(q):
        return

    await initialize_app(q)

    copy_expando(q.args, q.client)

    await initialize_client(q)

    # Training mode is import-specific. Do not leak a previous Text-only choice
    # into the next newly imported or edited dataset; edit mode will infer the
    # persisted representation from the dataset configuration again.
    if q.args.__wave_submission_name__ in ("dataset/import", "dataset/edit"):
        q.client["dataset/import/training_mode"] = None
        q.client["dataset/import/text_column"] = None

    await handle(q)

    if not q.args["experiment/display/chat/chatbot"]:
        await interface(q)

    await heap_redact(q)
    await q.page.save()
