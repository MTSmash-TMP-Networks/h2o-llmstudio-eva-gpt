import os
import subprocess

from h2o_wave import Q, ui

from llm_studio.app_utils.sections.common import clean_dashboard
from llm_studio.app_utils.utils import get_output_dir


async def create_model(q: Q) -> None:
    """UI helper to guide tokenizer + dense model bootstrapping from uploaded datasets."""

    q.client["nav/active"] = "experiment/create_model"
    await clean_dashboard(q, mode="full")

    datasets_df = q.client.app_db.get_datasets_df()
    dataset_choices = []

    if datasets_df.shape[0] > 0:
        dataset_choices = [
            ui.choice(name=str(row.id), label=f"{row.name} ({row.path})")
            for _, row in datasets_df.iterrows()
        ]

    def _client_value(key: str, default: int) -> str:
        value = q.client[key] if key in q.client else default
        return str(value or default)

    vocab_size = _client_value("experiment/create_model/vocab_size", 128256)
    max_lines = _client_value("experiment/create_model/max_lines", 500000)
    hidden_size = _client_value("experiment/create_model/hidden_size", 2048)
    intermediate_size = _client_value("experiment/create_model/intermediate_size", 8192)
    num_hidden_layers = _client_value("experiment/create_model/num_hidden_layers", 16)
    num_attention_heads = _client_value("experiment/create_model/num_attention_heads", 32)
    num_key_value_heads = _client_value("experiment/create_model/num_key_value_heads", 8)
    dataset_value = q.client["experiment/create_model/dataset_id"]
    if not dataset_value and dataset_choices:
        dataset_value = dataset_choices[0].name

    tokenizer_log_text = ""
    tokenizer_log_path = q.client["experiment/create_model/tokenizer_log_path"]
    if tokenizer_log_path:
        tokenizer_log_text = tail_log(tokenizer_log_path)

    model_name = q.client["experiment/create_model/model_name"] or "eva-mini131k-eva_gpt-dense-fp32"
    default_model_dir = os.path.join(get_output_dir(q), model_name)
    default_tokenizer_dir = os.path.join(get_output_dir(q), f"{model_name}_tokenizer_fast")

    q.page["experiment/create_model"] = ui.form_card(
        box="content",
        items=[
            ui.text_xl("Create model"),
            ui.text(
                "Build a tokenizer from an uploaded dataset and initialize a dense EvaGPT model."
            ),
            ui.message_bar(
                type="info",
                text=(
                    "This workflow configures and generates ready-to-run training commands for tokenizer and dense-model initialization."
                ),
            ),
            ui.dropdown(
                name="experiment/create_model/dataset_id",
                label="Dataset",
                choices=dataset_choices,
                value=dataset_value,
                required=False,
            ),
            ui.textbox(
                name="experiment/create_model/model_name",
                label="Model name",
                value=model_name,
            ),
            ui.textbox(
                name="experiment/create_model/tokenizer_dir",
                label="Tokenizer output directory",
                value=default_tokenizer_dir,
            ),
            ui.textbox(
                name="experiment/create_model/model_dir",
                label="Model output directory",
                value=default_model_dir,
            ),
            ui.text_l("Tokenizer settings"),
            ui.textbox(
                name="experiment/create_model/vocab_size",
                label="Token count (vocab size)",
                value=vocab_size,
            ),
            ui.textbox(
                name="experiment/create_model/max_lines",
                label="Max sampled lines",
                value=max_lines,
            ),
            ui.text_l("Model architecture settings"),
            ui.textbox(
                name="experiment/create_model/hidden_size",
                label="Hidden size",
                value=hidden_size,
            ),
            ui.textbox(
                name="experiment/create_model/intermediate_size",
                label="Intermediate size",
                value=intermediate_size,
            ),
            ui.textbox(
                name="experiment/create_model/num_hidden_layers",
                label="Layer count",
                value=num_hidden_layers,
            ),
            ui.textbox(
                name="experiment/create_model/num_attention_heads",
                label="Attention heads",
                value=num_attention_heads,
            ),
            ui.textbox(
                name="experiment/create_model/num_key_value_heads",
                label="KV heads",
                value=num_key_value_heads,
            ),
            ui.buttons(
                items=[
                    ui.button(
                        name="experiment/create_model/start_tokenizer",
                        label="Start tokenizer training",
                        primary=True,
                    ),
                    ui.button(
                        name="experiment/create_model/start_model_init",
                        label="Start model creation",
                    ),
                    ui.button(
                        name="experiment/create_model/refresh_logs",
                        label="Refresh logs",
                    ),
                ]
            ),
            ui.text_l("Tokenizer training log"),
            ui.text(
                f"Log file: {tokenizer_log_path}"
                if tokenizer_log_path
                else "Start tokenizer training to see logs here."
            ),
            ui.text(
                f"```text\n{tokenizer_log_text or 'No log output yet.'}\n```"
            ),
            ui.separator(),
            ui.text_l("Run pipeline"),
            ui.text("1) Train tokenizer:"),
            ui.text("python scripts/create_model/train_tokenizer.py --csv <path-to-dataset.csv> --tokenizer-dir ./tokenizer --tokenizer-fast-dir ./tokenizer_fast"),
            ui.text("2) Initialize dense EvaGPT model:"),
            ui.text("python scripts/create_model/initialize_eva_model.py --tokenizer-src ./tokenizer_fast --out-dir ./eva-mini131k-eva_gpt-dense-fp32"),
            ui.separator(),
            ui.text_l("Next steps"),
            ui.text(
                "1) Use your selected dataset as CSV input for tokenizer training.\n"
                "2) Train SentencePiece and export slow/fast tokenizers.\n"
                "3) Initialize EvaGPT config with long context and save model + tokenizer."
            ),
            ui.text(
                "Tip: Keep your scripts in the project and parametrize paths via UI values above."
            ),
        ],
    )
    q.client.delete_cards.add("experiment/create_model")

    if datasets_df.shape[0] == 0:
        q.client["notification_bar"] = (
            "No datasets found. Please import a dataset first, then open Create model again."
        )

    # Ensure directories are displayed as normalized paths for consistency.
    q.client["experiment/create_model/tokenizer_dir"] = os.path.normpath(
        q.client["experiment/create_model/tokenizer_dir"] or "./tokenizer_fast"
    )
    q.client["experiment/create_model/model_dir"] = os.path.normpath(
        q.client["experiment/create_model/model_dir"] or "./eva-mini131k-eva_gpt-dense-fp32"
    )


def start_background_command(cmd: list[str], log_path: str) -> int:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a")
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )
    return process.pid


def tail_log(log_path: str, max_lines: int = 100) -> str:
    if not os.path.exists(log_path):
        return ""
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()[-max_lines:]
    return "".join(lines).strip()
