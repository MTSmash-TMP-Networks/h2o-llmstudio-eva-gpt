import os
import shutil
import signal
import subprocess
import time

import pandas as pd
from h2o_wave import Q, ui

from llm_studio.app_utils.sections.common import clean_dashboard
from llm_studio.app_utils.utils import get_output_dir
from llm_studio.app_utils.wave_utils import ui_table_from_df


async def create_model(q: Q) -> None:
    """UI helper to guide tokenizer + dense model bootstrapping from uploaded datasets."""

    q.client["nav/active"] = "experiment/create_model"
    await clean_dashboard(q, mode="full")

    datasets_df = q.client.app_db.get_datasets_df()
    dataset_choices = []

    if datasets_df.shape[0] > 0:
        dataset_choices = [
            ui.choice(name=str(row.id), label=str(row["name"]))
            for _, row in datasets_df.iterrows()
        ]

    def _client_value(key: str, default: int) -> str:
        value = q.client[key] if key in q.client else default
        return str(value or default)

    vocab_size = _client_value("experiment/create_model/vocab_size", 128256)
    use_pretrained_tokenizer = bool(
        q.client["experiment/create_model/use_pretrained_tokenizer"]
    )
    pretrained_tokenizer_model = (
        q.client["experiment/create_model/pretrained_tokenizer_model"] or ""
    )
    max_lines = _client_value("experiment/create_model/max_lines", 500000)
    hidden_size = _client_value("experiment/create_model/hidden_size", 2048)
    intermediate_size = _client_value("experiment/create_model/intermediate_size", 8192)
    num_hidden_layers = _client_value("experiment/create_model/num_hidden_layers", 16)
    num_attention_heads = _client_value("experiment/create_model/num_attention_heads", 32)
    num_key_value_heads = _client_value("experiment/create_model/num_key_value_heads", 8)
    attn_implementation = (
        q.client["experiment/create_model/attn_implementation"] or "eager"
    )
    dataset_value = q.client["experiment/create_model/dataset_id"]
    if not dataset_value and dataset_choices:
        dataset_value = dataset_choices[0].name

    model_name = q.client["experiment/create_model/model_name"] or "eva-mini131k-eva_gpt-dense-fp32"
    models_df = _get_created_models_df(get_output_dir(q))
    created_model_choices = [
        ui.choice(name=str(row["path"]), label=str(row["name"]))
        for _, row in models_df.iterrows()
    ]
    selected_model_path = q.client["experiment/create_model/selected_model_path"]
    if not selected_model_path and created_model_choices:
        selected_model_path = created_model_choices[0].name

    items = [
        ui.text_xl("Create model"),
        ui.text(
            "Build a tokenizer from an uploaded dataset and initialize a dense EvaGPT model."
        ),
    ]

    if not models_df.empty:
        items.extend(
            [
                ui.separator(),
                ui.text_l("Created models"),
                ui.dropdown(
                    name="experiment/create_model/selected_model_path",
                    label="Model for new experiment",
                    choices=created_model_choices,
                    value=selected_model_path,
                    required=False,
                ),
                ui.buttons(
                    items=[
                        ui.button(
                            name="experiment/create_model/create_experiment",
                            label="Create experiment from selected model",
                            primary=True,
                        )
                    ]
                ),
                ui_table_from_df(
                    q=q,
                    df=models_df,
                    name="experiment/create_model/models_table",
                    sortables=["name"],
                    filterables=["name", "path"],
                    searchables=["name", "path"],
                    link_col="name",
                    height="250px",
                    min_widths={"name": "220", "path": "500", "actions": "80"},
                    actions={"experiment/create_model/delete_model": "Delete"},
                ),
            ]
        )

    items.extend(
        [
            ui.separator(),
            ui.text_l("Create new model"),
            ui.dropdown(
                name="experiment/create_model/dataset_id",
                label="Dataset",
                choices=dataset_choices,
                value=dataset_value,
                required=False,
            )
            if not use_pretrained_tokenizer
            else ui.text(""),
            ui.textbox(
                name="experiment/create_model/model_name",
                label="Model name",
                value=model_name,
            ),
            ui.text_l("Tokenizer settings"),
            ui.toggle(
                name="experiment/create_model/use_pretrained_tokenizer",
                label="Use pretrained Hugging Face tokenizer",
                value=use_pretrained_tokenizer,
                trigger=True,
            ),
            ui.textbox(
                name="experiment/create_model/pretrained_tokenizer_model",
                label="Hugging Face tokenizer model",
                value=pretrained_tokenizer_model,
                placeholder="z.B. meta-llama/Llama-3.2-1B",
                required=use_pretrained_tokenizer,
            )
            if use_pretrained_tokenizer
            else ui.text(""),
            ui.textbox(
                name="experiment/create_model/vocab_size",
                label="Token count (vocab size)",
                value=vocab_size,
                disabled=use_pretrained_tokenizer,
            )
            if not use_pretrained_tokenizer
            else ui.text(""),
            ui.textbox(
                name="experiment/create_model/max_lines",
                label="Max sampled lines",
                value=max_lines,
                disabled=use_pretrained_tokenizer,
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
            ui.dropdown(
                name="experiment/create_model/attn_implementation",
                label="Attention implementation",
                choices=[
                    ui.choice(name="eager", label="eager"),
                    ui.choice(name="sdpa", label="sdpa"),
                ],
                value=attn_implementation,
                required=True,
            ),
            ui.buttons(
                items=[
                    ui.button(
                        name="experiment/create_model/start_pipeline",
                        label="Start create model",
                        primary=True,
                    ),
                    ui.button(
                        name="experiment/create_model/logs",
                        label="Open logs",
                    ),
                ]
            ),
            ui.separator(),
            ui.text_l("Run pipeline"),
            ui.text("1) Train tokenizer (or use pretrained Hugging Face tokenizer):"),
            ui.text("python scripts/create_model/train_tokenizer.py --csv <path-to-dataset.csv> --tokenizer-dir ./tokenizer --tokenizer-fast-dir ./tokenizer_fast"),
            ui.text("2) Initialize dense EvaGPT model:"),
            ui.text("python scripts/create_model/initialize_eva_model.py --tokenizer-src ./tokenizer_fast|<hf-model-id> --out-dir ./eva-mini131k-eva_gpt-dense-fp32"),
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
        ]
    )

    q.page["experiment/create_model"] = ui.form_card(
        box="content",
        items=items,
    )
    q.client.delete_cards.add("experiment/create_model")

    if datasets_df.shape[0] == 0:
        q.client["notification_bar"] = (
            "No datasets found. Please import a dataset first, then open Create model again."
        )


async def create_model_logs(q: Q, follow: bool = False) -> None:
    q.client["nav/active"] = "experiment/create_model"

    async def _render_logs() -> bool:
        await clean_dashboard(q, mode="full")

        tokenizer_log_path = q.client["experiment/create_model/tokenizer_log_path"]
        model_log_path = q.client["experiment/create_model/model_log_path"]
        pipeline_pid = q.client["experiment/create_model/pipeline_pid"]
        pipeline_started_at = q.client["experiment/create_model/pipeline_started_at"]

        tokenizer_log_text = tail_log(tokenizer_log_path) if tokenizer_log_path else ""
        model_log_text = tail_log(model_log_path) if model_log_path else ""
        is_running = _is_process_running(pipeline_pid)

        q.page["experiment/create_model/logs"] = ui.form_card(
            box="content",
            items=[
                ui.text_xl("Create model logs"),
                ui.buttons(
                    items=[
                        ui.button(
                            name="experiment/create_model/logs",
                            label="Refresh",
                            primary=True,
                        ),
                        ui.button(name="experiment/create_model", label="Back"),
                    ]
                ),
                ui.text_l("Tokenizer log"),
                ui.text(
                    f"Log file: {tokenizer_log_path}"
                    if tokenizer_log_path
                    else "No tokenizer log file yet."
                ),
                ui.text(f"```text\n{tokenizer_log_text or 'No log output yet.'}\n```"),
                ui.separator(),
                ui.text_l("Model initialization log"),
                ui.text(
                    f"Log file: {model_log_path}"
                    if model_log_path
                    else "No model log file yet."
                ),
                ui.text(f"```text\n{model_log_text or 'No log output yet.'}\n```"),
            ],
        )
        q.client.delete_cards.add("experiment/create_model/logs")
        await q.page.save()

        if not is_running:
            q.client["experiment/create_model/pipeline_pid"] = None
        return is_running

    is_running = await _render_logs()
    while follow and is_running:
        await q.sleep(2)
        is_running = await _render_logs()



def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _is_process_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _estimate_remaining_seconds(start_time: float | None, is_running: bool) -> int | None:
    if not start_time or not is_running:
        return 0
    elapsed = max(0.0, time.time() - start_time)
    estimated_total = 45 * 60
    return max(0, int(estimated_total - elapsed))


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


def _get_created_models_df(output_dir: str) -> pd.DataFrame:
    if not os.path.isdir(output_dir):
        return pd.DataFrame(columns=["name", "path"])

    model_rows = []
    for entry in sorted(os.listdir(output_dir)):
        path = os.path.join(output_dir, entry)
        config_path = os.path.join(path, "config.json")
        if os.path.isdir(path) and os.path.isfile(config_path):
            model_rows.append({"name": entry, "path": path})

    return pd.DataFrame(model_rows, columns=["name", "path"])


def delete_created_model(model_path: str) -> bool:
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isdir(model_path) or not os.path.isfile(config_path):
        return False

    shutil.rmtree(model_path, ignore_errors=False)
    return True


def tail_log(log_path: str, max_lines: int = 100) -> str:
    if not os.path.exists(log_path):
        return ""
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()[-max_lines:]
    return "".join(lines).strip()
