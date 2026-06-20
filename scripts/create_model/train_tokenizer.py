import argparse
import json
import os
import random
import re
import shutil
from pathlib import Path

import pandas as pd
import sentencepiece as spm
from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaTokenizerFast

# (trimmed constants)
FENCE_RE = re.compile(r"```([a-zA-Z0-9_+\-#.]*)\n(.*?)```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}\b-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
LONG_NUMBER_RE = re.compile(r"\b\d{8,}\b")


def _clean_text(t: str) -> str:
    t = "" if t is None else str(t)
    if t.lower() == "nan":
        return ""
    t = t.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in t.split("\n")]
    lines = [x for x in lines if x]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _mask_noise(t: str) -> str:
    return LONG_NUMBER_RE.sub("<LONGNUM>", UUID_RE.sub("<UUID>", t or ""))


def _normalize_line(t: str, max_chars: int) -> str:
    t = _mask_noise((t or "").replace("\n", "\\n"))
    t = re.sub(r"[ \t]+", " ", t).strip()
    return t[:max_chars]


def _extract_codeblocks(text: str) -> list[str]:
    return [_mask_noise(code.strip()) for _, code in FENCE_RE.findall(text or "") if code.strip()]


def _extract_inline_code(text: str) -> list[str]:
    return [_mask_noise(s.strip()) for s in INLINE_CODE_RE.findall(text or "") if s.strip()]


def _resolve_table_path(data_path: str, dataset_name: str | None = None) -> str:
    path = Path(data_path)
    if path.is_file():
        return str(path)
    if path.is_dir():
        candidates = sorted([*path.rglob("*.csv"), *path.rglob("*.pq"), *path.rglob("*.parquet")])
        if not candidates:
            raise FileNotFoundError(f"No CSV/Parquet files found in directory: {path}")

        def _is_meta_file(p: Path) -> bool:
            name = p.name.lower()
            return name.startswith("__meta_info__") or "meta" in p.parts

        preferred = [p for p in candidates if not _is_meta_file(p)]
        if dataset_name:
            target = dataset_name.lower()
            matched = [p for p in preferred if p.name.lower() == target or p.stem.lower() == target]
            if len(matched) == 1:
                return str(matched[0])
            if not matched:
                raise ValueError(
                    f"No dataset matched --dataset-name={dataset_name!r} in directory '{path}'. "
                    + "Available files: "
                    + ", ".join(str(p.relative_to(path)) for p in preferred[:20])
                )
            candidates = matched

        if len(preferred) == 1:
            return str(preferred[0])
        if len(preferred) > 1:
            for default_name in ("train", "train_full"):
                train_named = [p for p in preferred if p.stem.lower() == default_name]
                if len(train_named) == 1:
                    return str(train_named[0])
            candidates = preferred

        if len(candidates) == 1:
            return str(candidates[0])
        raise ValueError(
            f"Expected one CSV/Parquet file in directory '{path}', found {len(candidates)}: "
            + ", ".join(str(p.relative_to(path)) for p in candidates[:20])
        )
    raise FileNotFoundError(f"Input path does not exist: {path}")


def _select_text_columns(columns: list[str], text_column: str | None) -> list[str]:
    cols = [c for c in columns if str(c).strip() and not str(c).startswith("Unnamed:")]
    if text_column:
        if text_column not in cols:
            raise ValueError(f"Text column {text_column!r} not found. Available columns: {cols}")
        return [text_column]
    if len(cols) == 1:
        return cols
    raise ValueError(
        "Tokenizer training expects exactly one text column. "
        "Pass --text-column for CSV/Parquet files with multiple columns."
    )


def _write_text_rows(row_iter, columns: list[str], out_txt: str, max_line_chars: int) -> None:
    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as out:
        for chunk in row_iter:
            lines = []
            for _, row in chunk.iterrows():
                text = "\n".join(_clean_text(row.get(c, "")) for c in columns if _clean_text(row.get(c, "")))
                if text:
                    lines.append(_normalize_line(text, max_line_chars))
                lines.extend(_normalize_line(x, max_line_chars) for x in _extract_codeblocks(text))
                lines.extend(_normalize_line(x, max_line_chars) for x in _extract_inline_code(text))
            if lines:
                out.write("\n".join(x for x in lines if x) + "\n")


def build_training_file(table_path: str, out_txt: str, max_line_chars: int, text_column: str | None = None) -> None:
    ext = Path(table_path).suffix.lower()
    if ext == ".csv":
        header = pd.read_csv(table_path, nrows=0)
        row_iter = pd.read_csv(table_path, chunksize=20_000)
    elif ext in {".pq", ".parquet"}:
        df = pd.read_parquet(table_path)
        header = df.head(0)
        row_iter = (df,)
    else:
        raise ValueError(f"Unsupported dataset extension '{ext}' for file: {table_path}")

    cols = _select_text_columns(list(header.columns), text_column)
    _write_text_rows(row_iter, cols, out_txt, max_line_chars)


def build_training_file_from_hf(
    dataset_path: str,
    out_txt: str,
    max_line_chars: int,
    text_column: str,
    dataset_name: str | None = None,
    split: str = "train",
) -> None:
    dataset = load_dataset(dataset_path, dataset_name, split=split)
    if text_column not in dataset.column_names:
        raise ValueError(
            f"Text column {text_column!r} not found in Hugging Face dataset. "
            f"Available columns: {dataset.column_names}"
        )

    def _chunks():
        batch_size = 20_000
        for start in range(0, len(dataset), batch_size):
            yield pd.DataFrame(dataset[start : start + batch_size])

    _write_text_rows(_chunks(), [text_column], out_txt, max_line_chars)


def reservoir_sample(input_path: str, output_path: str, max_lines: int, seed: int) -> None:
    if max_lines <= 0:
        shutil.copyfile(input_path, output_path)
        return

    random.seed(seed)
    reservoir: list[str] = []
    total = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            if len(reservoir) < max_lines:
                reservoir.append(line)
            else:
                j = random.randint(0, total - 1)
                if j < max_lines:
                    reservoir[j] = line
    random.shuffle(reservoir)
    with open(output_path, "w", encoding="utf-8") as out:
        out.write("\n".join(reservoir) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="Path to CSV/Parquet file or directory containing dataset file(s)")
    source.add_argument("--hf-dataset", help="Hugging Face dataset path, for example wikitext")
    p.add_argument("--dataset-name", default=None, help="Optional CSV filename/stem, or Hugging Face dataset config name")
    p.add_argument("--hf-split", default="train", help="Hugging Face dataset split to read")
    p.add_argument("--text-column", default=None, help="Text column to train from. Required for Hugging Face datasets and multi-column tables")
    p.add_argument("--tokenizer-dir", default="./tokenizer")
    p.add_argument("--tokenizer-fast-dir", default="./tokenizer_fast")
    p.add_argument("--model-prefix", default="tokenizer")
    p.add_argument("--vocab-size", type=int, default=128256)
    p.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help="Maximum number of lines sampled for training. Use 0 to train on all available lines.",
    )
    p.add_argument("--max-line-chars", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-type", choices=("bpe", "unigram"), default="bpe")
    p.add_argument("--character-coverage", type=float, default=0.99995)
    p.add_argument("--input-sentence-size", type=int, default=0)
    p.add_argument("--num-threads", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument("--shuffle-input-sentence", action="store_true", default=True)
    p.add_argument("--no-shuffle-input-sentence", dest="shuffle_input_sentence", action="store_false")
    args = p.parse_args()

    if args.hf_dataset and not args.text_column:
        raise ValueError("--text-column is required when using --hf-dataset")

    data_dir = os.getcwd()
    if args.csv:
        args.csv = _resolve_table_path(args.csv, args.dataset_name)
        data_dir = os.path.dirname(args.csv)

    train_txt = os.path.join(data_dir, "train_data_from_text_source.txt")
    sampled_txt = os.path.join(data_dir, "train_data_sampled.txt")
    if args.hf_dataset:
        build_training_file_from_hf(args.hf_dataset, train_txt, args.max_line_chars, args.text_column, args.dataset_name, args.hf_split)
    else:
        build_training_file(args.csv, train_txt, args.max_line_chars, args.text_column)
    reservoir_sample(train_txt, sampled_txt, args.max_lines, seed=args.seed)

    trainer_kwargs = {
        "input": sampled_txt,
        "model_prefix": args.model_prefix,
        "vocab_size": args.vocab_size,
        "model_type": args.model_type,
        "character_coverage": args.character_coverage,
        "num_threads": args.num_threads,
        "byte_fallback": True,
        "split_by_whitespace": False,
        "split_by_unicode_script": False,
        "shuffle_input_sentence": args.shuffle_input_sentence,
        "unk_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "pad_id": 3,
        "unk_piece": "<unk>",
        "bos_piece": "<s>",
        "eos_piece": "</s>",
        "pad_piece": "<pad>",
    }
    if args.input_sentence_size > 0:
        trainer_kwargs["input_sentence_size"] = args.input_sentence_size

    spm.SentencePieceTrainer.train(**trainer_kwargs)

    tok_cfg = {"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>", "tokenizer_class": "LlamaTokenizer"}
    with open("tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(tok_cfg, f, indent=2)
    with open("special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in tok_cfg.items() if k.endswith("_token")}, f, indent=2)

    os.makedirs(args.tokenizer_dir, exist_ok=True)
    for fn in [f"{args.model_prefix}.model", f"{args.model_prefix}.vocab", "tokenizer_config.json", "special_tokens_map.json"]:
        shutil.move(fn, os.path.join(args.tokenizer_dir, fn))

    os.makedirs(args.tokenizer_fast_dir, exist_ok=True)
    LlamaTokenizer.from_pretrained(args.tokenizer_dir)
    fast = LlamaTokenizerFast.from_pretrained(args.tokenizer_dir)
    fast.save_pretrained(args.tokenizer_fast_dir)


if __name__ == "__main__":
    main()
