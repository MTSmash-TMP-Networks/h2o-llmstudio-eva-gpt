import os

# Set this before importing any other modules to be on the safe side
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import logging
import sys
import time

import psutil

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def check_for_done(process_queue):
    """Checks for finished process ids

    Args:
        process_queue: list of process ids
    Returns:
        (True, process_idx) if there is any finished process
        (False, False) if there is not finished processes
    """

    for i, pid in enumerate(process_queue):
        zombie = False
        try:
            p = psutil.Process(pid)
            zombie = p.status() == "zombie"
        except psutil.NoSuchProcess:
            pass
        if not psutil.pid_exists(pid) or zombie:
            return True, i

    return False, False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-Y", "--yaml", help="yaml filename", type=(str), default=argparse.SUPPRESS
    )
    parser.add_argument(
        "-Q",
        "--process-queue",
        help="process queue to wait for",
        default=argparse.SUPPRESS,
    )
    parser_args, _ = parser.parse_known_args(sys.argv)

    process_queue = []
    if "process_queue" in parser_args and parser_args.process_queue != "":
        process_queue = [int(x) for x in parser_args.process_queue.split(",")]

    while True:
        if len(process_queue) == 0:
            break
        done, num = check_for_done(process_queue)
        if done:
            process_queue.pop(num)
        else:
            time.sleep(30)

    # delayed imports from llm_studio, only after we want to start training
    import subprocess

    import torch

    from llm_studio.src.utils.config_utils import load_config_yaml
    from llm_studio.src.utils.dense_local_model_repair import (
        repair_stale_dense_eva_quantization_config,
    )
    from llm_studio.src.utils.exceptions import (
        LLMAugmentationsException,
        LLMDataException,
        LLMMetricException,
        LLMModelException,
        LLMTrainingException,
    )
    from llm_studio.src.utils.gpu_utils import is_oom_error
    from llm_studio.src.utils.large_text_arrow_memory import (
        install_large_text_arrow_memory,
    )
    from llm_studio.src.utils.large_text_deepspeed_runtime import (
        install_large_text_deepspeed_runtime,
    )
    from llm_studio.src.utils.local_model_utils import ensure_local_eva_model_type
    from llm_studio.src.utils.logging_utils import initialize_logging, write_flag
    from llm_studio.src.utils.sharded_parquet_training import (
        install_sharded_parquet_training_support,
    )
    from llm_studio.src.utils.utils import kill_child_processes_and_current

    # The Wave app installs sharded-Parquet support in its own process. Training is
    # launched as a separate Python process, so install the core reader here too.
    # This preserves logical column aliases such as source `text` -> trainer `Text`.
    install_sharded_parquet_training_support()

    # Keep the multi-million-row raw text corpus in Arrow buffers. Without this,
    # to_pandas()/astype(str) creates hundreds of thousands of Python string objects
    # per rank before the model is even loaded.
    install_large_text_arrow_memory()

    # Large rank-partitioned text corpora must keep their already prepared loader
    # instead of letting DeepSpeed repartition/fork it again. Install this before
    # train.py imports the runtime helpers so the low-memory wrappers are captured.
    install_large_text_deepspeed_runtime()

    from llm_studio.train import run
    from llm_studio.src.utils.dense_backbone_low_memory import (
        install_dense_backbone_low_memory,
    )

    # Importing train.py installs the existing V100/DeepSpeed precision wrapper.
    # Extend that wrapper afterwards with Transformers low_cpu_mem_usage.
    install_dense_backbone_low_memory()

    cfg = load_config_yaml(parser_args.yaml)

    # Backward compatibility for locally created EvaGPT models whose older
    # config.json omitted Hugging Face's model_type discriminator. Repair this
    # before either AutoTokenizer or AutoConfig sees the local backbone path.
    ensure_local_eva_model_type(cfg.llm_backbone)

    # Older dense FP32 models created from a pretrained base config could retain
    # that base model's MXFP4 metadata even though their safetensors are dense. On
    # V100 this makes Transformers dequantize every rank to BF16 during startup.
    repair_stale_dense_eva_quantization_config(cfg.llm_backbone)

    flag_path = os.path.join(cfg.output_directory, "flags{}.json")

    # Check if DDP
    if "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank == 0:
            write_flag(flag_path.format(""), "status", "running")
    else:
        write_flag(flag_path.format(""), "status", "running")
        local_rank = 0

    try:
        run(cfg=cfg)
    except Exception as exception:
        initialize_logging(cfg)
        write_flag(flag_path.format(local_rank), "status", "failed")
        if is_oom_error(exception):
            logging.error(
                "GPU Out-of-Memory (OOM) error occurred. "
                "Please, reduce the batch size, or input data size, "
                "or model size. Or try gradient checkpointing.",
                exc_info=True,
            )
            write_flag(flag_path.format(local_rank), "info", "OOM error")

            logging.info(
                "<pre>"
                + subprocess.check_output(["nvidia-smi"]).decode("utf-8")
                + "</pre>"
            )

            if torch.cuda.is_available():
                logging.info(
                    "<pre>" + torch.cuda.memory_summary().replace("-", "=") + "</pre>"
                )

        elif isinstance(exception, LLMDataException):
            logging.error(
                "Data error occurred during H2O LLM Studio run:", exc_info=True
            )
            write_flag(flag_path.format(local_rank), "info", "Data error")
        elif isinstance(exception, LLMTrainingException):
            logging.error(
                "Training error occurred during H2O LLM Studio run:", exc_info=True
            )
            write_flag(flag_path.format(local_rank), "info", "Training error")
        elif isinstance(exception, LLMMetricException):
            logging.error(
                "Validation metric failed. Please make sure selected validation "
                "metric is suitable for your current problem setup.",
                exc_info=True,
            )
            write_flag(flag_path.format(local_rank), "info", "Metric error")
        elif isinstance(exception, LLMAugmentationsException):
            logging.error(
                "Custom augmentations error occurred during H2O LLM Studio run:",
                exc_info=True,
            )
            write_flag(flag_path.format(local_rank), "info", "Augmentations error")
        elif isinstance(exception, LLMModelException):
            logging.error(
                "Model error occurred during H2O LLM Studio run:",
                exc_info=True,
            )
            write_flag(flag_path.format(local_rank), "info", "Model error")
        else:
            logging.error(
                "Exception occurred during H2O LLM Studio run:", exc_info=True
            )
            write_flag(flag_path.format(local_rank), "info", "See logs")

        # Clean up any potential processes for this experiment
        kill_child_processes_and_current()
