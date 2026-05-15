<img width="898" height="120" alt="Screenshot 2026-05-15 at 21-40-33 MaTeLiX AI Studio" align="center" src="https://github.com/user-attachments/assets/686b3ce7-7289-470f-af52-115dc6826284" />

<h3 align="center">
    <p>EVA GPT Studio is a practical GUI + CLI framework for fine-tuning,
    evaluating, and serving large language models.</p>
</h3>

> This repository started from H2O LLM Studio and has evolved into an EVA-focused fork.

## Overview

EVA GPT Studio is designed for teams that want fast iteration on LLM experiments without rebuilding the entire training stack from scratch.

Core capabilities:

- Fine-tune instruction/chat models via a no-code graphical interface.
- Run training and inference workflows from CLI for automation.
- Support modern adaptation techniques (LoRA, quantization-aware workflows, multi-GPU training).
- Track experiments and compare runs with integrated visual tooling.
- Chat with trained checkpoints directly in the app.
- Export and publish trained artifacts.

## Table of Contents

- [Quick Start](#quick-start)
- [System Requirements](#system-requirements)
- [Installation](#installation)
  - [Recommended (uv)](#recommended-uv)
  - [Alternative (pip)](#alternative-pip)
- [Run the GUI](#run-the-gui)
- [Run with Docker](#run-with-docker)
- [Run via CLI](#run-via-cli)
- [Data Format](#data-format)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

```bash
make setup
make llmstudio
```

Then open: <http://localhost:10101/>

## System Requirements

Minimum recommended environment:

- Ubuntu 20.04+ (Linux recommended)
- Python 3.10
- NVIDIA GPU + recent drivers
- CUDA-compatible PyTorch runtime
- 24 GB+ VRAM for larger models

CPU-only workflows may work for limited scenarios (for example preprocessing or lightweight evaluation) but training performance will be significantly reduced.

## Installation

### Recommended (uv)

```bash
make setup
```

This creates a local environment and installs the project dependencies.

### Alternative (pip)

```bash
pip install -r requirements.txt
# optional acceleration package
pip install flash-attn==2.8.3 --no-build-isolation
```

## Run the GUI

```bash
make llmstudio
```

If you run outside the default `uv` setup, start manually:

```bash
H2O_WAVE_MAX_REQUEST_SIZE=25MB \
H2O_WAVE_NO_LOG=true \
H2O_WAVE_PRIVATE_DIR="/download/@output/download" \
wave run llm_studio.app
```

App URL: <http://localhost:10101/>

## Run with Docker

```bash
mkdir -p "$(pwd)/llmstudio_mnt"
chmod 777 "$(pwd)/llmstudio_mnt"

docker pull h2oairelease/h2oai-llmstudio-app:latest

docker run \
  --runtime=nvidia \
  --shm-size=64g \
  --init \
  --rm \
  -it \
  -u "$(id -u):$(id -g)" \
  -p 10101:10101 \
  -v "$(pwd)/llmstudio_mnt:/mount" \
  h2oairelease/h2oai-llmstudio-app:latest
```

Then open: <http://localhost:10101/>

## Run via CLI

Train from a YAML config:

```bash
uv run python llm_studio/train.py -Y {path_to_config_yaml}
```

Distributed training:

```bash
bash distributed_train.sh {NUM_GPUS} -Y {path_to_config_yaml}
```

Interactive prompt against a trained experiment:

```bash
uv run python llm_studio/prompt.py -e {experiment_name}
```

Publish an experiment to Hugging Face:

```bash
uv run python llm_studio/publish_to_hugging_face.py \
  -p {path_to_experiment} \
  -d {device} \
  -a {api_key} \
  -u {user_id} \
  -m {model_name} \
  -s {safe_serialization}
```

## Data Format

See the docs under [`documentation/`](documentation/) and in-app dataset tooling for expected column schemas and task-specific formats.

## Project Structure

- `llm_studio/` — main application and training framework
- `examples/` — example configs and data snippets
- `documentation/` — project docs source
- `tests/` — unit/UI tests

## Troubleshooting

Remote/proxy environments may require:

```bash
export H2O_WAVE_ALLOWED_ORIGINS="*"
```

For timeout issues:

```bash
export H2O_WAVE_APP_CONNECT_TIMEOUT="15"
export H2O_WAVE_APP_WRITE_TIMEOUT="15"
export H2O_WAVE_APP_READ_TIMEOUT="15"
export H2O_WAVE_APP_POOL_TIMEOUT="15"
```

## Contributing

Contributions are welcome. Please start with [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0. See [LICENSE](LICENSE).
