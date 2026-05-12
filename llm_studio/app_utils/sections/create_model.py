import os
import subprocess

from h2o_wave import Q, ui

from llm_studio.app_utils.sections.common import clean_dashboard


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
                value=(dataset_choices[0].name if dataset_choices else None),
                required=False,
            ),
            ui.textbox(
                name="experiment/create_model/tokenizer_dir",
                label="Tokenizer output directory",
                value="./tokenizer_fast",
            ),
            ui.textbox(
                name="experiment/create_model/model_dir",
                label="Model output directory",
                value="./eva-mini131k-eva_gpt-dense-fp32",
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
                ]
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


def start_background_command(cmd: list[str]) -> int:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process.pid
