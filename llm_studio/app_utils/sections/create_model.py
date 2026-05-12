import os

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
                    "This workflow currently provides an in-UI assistant and path setup. "
                    "Training and model creation still run as Python jobs in your environment."
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
        q.client.get("experiment/create_model/tokenizer_dir", "./tokenizer_fast")
    )
    q.client["experiment/create_model/model_dir"] = os.path.normpath(
        q.client.get("experiment/create_model/model_dir", "./eva-mini131k-eva_gpt-dense-fp32")
    )
