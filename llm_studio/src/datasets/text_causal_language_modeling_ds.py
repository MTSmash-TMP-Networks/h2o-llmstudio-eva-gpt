import codecs
import collections.abc
import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from llm_studio.src.datasets.conversation_chain_handler import ConversationChainHandler
from llm_studio.src.datasets.text_utils import get_tokenizer

logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    """Dataset for Causal Language modeling."""

    def __init__(self, df: pd.DataFrame, cfg: Any, mode: str = "train"):
        """
        Args:
            df: input DataFrame
            cfg: config with all the hyperparameters
            mode: dataset mode. One of {"train", "validation"}
        """
        self.cfg = cfg
        self.mode = mode
        self.df = df.copy()
        self.tokenizer = get_tokenizer(self.cfg)
        self.conversation_chain_handler = ConversationChainHandler(self.df, cfg)
        self.sample_index = self._build_sample_index()

    def __len__(self) -> int:
        return len(self.sample_index)

    def _build_sample_index(self) -> list[tuple[int, int | None, int]]:
        """Build index entries for truncation, skip, and sliding-window modes."""
        strategy = getattr(self.cfg.tokenizer, "long_sample_strategy", "Truncate")
        if strategy not in ["Truncate", "Sliding Window", "Skip"]:
            strategy = "Truncate"
        max_length = self.cfg.tokenizer.max_length
        sample_index: list[tuple[int, int | None, int]] = []

        if self.mode != "train" or strategy == "Truncate":
            return [
                (idx, None, 0) for idx in range(len(self.conversation_chain_handler))
            ]

        skipped_samples = 0
        created_windows = 0
        for idx in range(len(self.conversation_chain_handler)):
            input_ids, _, _, _ = self._get_input_ids_labels_and_encodings(
                idx, augment=False, trim_to_max_length=False
            )
            sample_length = len(input_ids)

            if sample_length <= max_length:
                sample_index.append((idx, None, 0))
            elif strategy == "Skip":
                skipped_samples += 1
            elif strategy == "Sliding Window":
                windows = self._get_sliding_window_starts_and_prefix_masks(
                    sample_length=sample_length,
                    max_length=max_length,
                    overlap=getattr(self.cfg.tokenizer, "sliding_window_overlap", 0),
                )
                sample_index.extend(
                    (idx, start, prefix_label_mask_len)
                    for start, prefix_label_mask_len in windows
                )
                created_windows += len(windows)
            else:
                sample_index.append((idx, None, 0))

        if strategy == "Skip":
            logger.info(
                "Skipped %s train samples longer than tokenizer.max_length=%s.",
                skipped_samples,
                max_length,
            )
        elif strategy == "Sliding Window":
            logger.info(
                "Created %s sliding-window train samples from long samples with "
                "tokenizer.max_length=%s.",
                created_windows,
                max_length,
            )

        return sample_index

    @staticmethod
    def _get_sliding_window_starts_and_prefix_masks(
        sample_length: int,
        max_length: int,
        overlap: int,
    ) -> list[tuple[int, int]]:
        """Return window starts and duplicated-prefix label mask lengths.

        Overlapping tokens remain in ``input_ids`` so the model can use them as
        left context across window boundaries. Their labels are masked in later
        windows because those tokens were already trained in the preceding
        window. The final window is shifted to include the sample end, so it can
        duplicate more tokens than the configured overlap.
        """
        if max_length <= 0:
            return [(0, 0)]
        if sample_length <= max_length:
            return [(0, 0)]

        overlap = min(max(overlap, 0), max(max_length - 1, 0))
        stride = max(max_length - overlap, 1)
        last_start = max(sample_length - max_length, 0)
        starts = list(range(0, last_start + 1, stride))
        if not starts or starts[-1] != last_start:
            starts.append(last_start)

        windows: list[tuple[int, int]] = []
        previous_start: int | None = None
        for current_start in starts:
            prefix_label_mask_len = 0
            if previous_start is not None:
                previous_end = previous_start + max_length
                duplicate_prefix_len = max(0, previous_end - current_start)
                prefix_label_mask_len = min(duplicate_prefix_len, max_length)
            windows.append((current_start, prefix_label_mask_len))
            previous_start = current_start

        return windows

    def _prepare_input_text_dict(self, idx: int) -> dict[str, list[str]]:
        input_text_dict = self.conversation_chain_handler[idx]
        input_text_dict["systems"] = [
            self.parse_system(self.cfg, system) for system in input_text_dict["systems"]
        ]
        input_text_dict["prompts"] = [
            self.parse_prompt_body(self.cfg, prompt)
            for prompt in input_text_dict["prompts"]
        ]
        input_text_dict["answers"] = [
            self.parse_answer(self.cfg, answer) for answer in input_text_dict["answers"]
        ]
        return input_text_dict

    def _get_input_ids_labels_and_encodings(
        self, idx: int, augment: bool = True, trim_to_max_length: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        input_text_dict = self._prepare_input_text_dict(idx)
        _, prompt_encodings, answer_encodings, prompt_label_masks = self.get_encodings(
            input_text_dict=input_text_dict,
            augment=augment,
            trim_to_max_length=trim_to_max_length,
        )
        input_ids = torch.cat(
            [
                torch.cat([prompt_encoding, answer_encoding])
                for prompt_encoding, answer_encoding in zip(
                    prompt_encodings, answer_encodings, strict=False
                )
            ]
        )
        labels = self._get_raw_labels(
            prompt_encodings, answer_encodings, prompt_label_masks
        )
        return input_ids, labels, prompt_encodings, answer_encodings

    def __getitem__(self, idx: int) -> dict:
        """Reads a single text observation."""
        original_idx, window_start, prefix_label_mask_len = self.sample_index[idx]
        use_long_sample_windows = (
            self.mode == "train"
            and getattr(self.cfg.tokenizer, "long_sample_strategy", "Truncate")
            == "Sliding Window"
        )
        input_ids, labels, prompt_encodings, answer_encodings = (
            self._get_input_ids_labels_and_encodings(
                original_idx,
                augment=not use_long_sample_windows,
                trim_to_max_length=not use_long_sample_windows,
            )
        )

        if window_start is not None:
            window_end = window_start + self.cfg.tokenizer.max_length
            input_ids = input_ids[window_start:window_end]
            labels = labels[window_start:window_end]
            if prefix_label_mask_len > 0:
                # Keep duplicated overlap tokens visible as context in input_ids,
                # but remove them from the loss in later windows so repeated
                # text is not over-weighted during training.
                labels[: min(prefix_label_mask_len, len(labels))] = -100

        sample = self.pad_labels(labels, self.cfg.tokenizer.max_length)
        sample.update(
            self.pad_tokens(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_length=self.cfg.tokenizer.max_length,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        )

        # get answer encodings
        sample.update(
            self.pad_tokens(
                answer_encodings[-1],
                attention_mask=torch.ones_like(answer_encodings[-1]),
                max_length=self.cfg.tokenizer.max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                direction="right",
                prefix="answer_",
            )
        )

        if window_start is not None:
            # For sliding-window train samples, the train-data insight table should
            # describe the same token window that is actually trained on.  The
            # regular inference prompt (full conversation with the last answer
            # removed) can be unrelated to a specific window and makes the GUI
            # show prompt/answer text that does not match the tokenized text.
            prompt_input_ids = input_ids[labels == -100]
        else:
            # Remove last answer from encoding to create the prompt for inference
            answer_encodings[-1] = torch.empty(0)
            prompt_input_ids = torch.cat(
                [
                    torch.cat([prompt_encoding, answer_encoding])
                    for prompt_encoding, answer_encoding in zip(
                        prompt_encodings, answer_encodings, strict=False
                    )
                ]
            )
        sample.update(
            self.pad_tokens(
                prompt_input_ids,
                attention_mask=torch.ones_like(prompt_input_ids),
                max_length=self.cfg.tokenizer.max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                prefix="prompt_",
            )
        )

        return sample

    @staticmethod
    def parse_prompt_body(cfg: Any, prompt: str):
        return f"{prompt}"

    @staticmethod
    def parse_prompt(cfg: Any, prompt: str):
        prompt = (
            f"{codecs.decode(cfg.dataset.text_prompt_start, 'unicode_escape')}{prompt}"
        )
        if cfg.dataset.add_eos_token_to_prompt:
            prompt += cfg.tokenizer._tokenizer_eos_token
        prompt = (
            f"{prompt}"
            f"{codecs.decode(cfg.dataset.text_answer_separator, 'unicode_escape')}"
        )
        return prompt

    @staticmethod
    def parse_answer(cfg: Any, answer: str):
        if cfg.dataset.add_eos_token_to_answer:
            answer += cfg.tokenizer._tokenizer_eos_token
        return answer

    @staticmethod
    def parse_system(cfg: Any, system: str):
        # no system tokens if empty
        if system == "":
            return system
        system = (
            f"{codecs.decode(cfg.dataset.text_system_start, 'unicode_escape')}{system}"
        )
        if cfg.dataset.add_eos_token_to_system:
            system += cfg.tokenizer._tokenizer_eos_token
        return system

    @staticmethod
    def batch_to_device(
        batch: dict | list | torch.Tensor, device: str
    ) -> dict | list | torch.Tensor | str:
        """Function to send the batch to the device specified

        Args:
            batch: input batch
            device: device to send the data to
        Returns:
            batch with the elements on the device specified
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, (list, tuple)) and all(
            isinstance(item, str) for item in batch
        ):
            # Do not move list of strings to device
            return batch
        elif isinstance(batch, collections.abc.Mapping):
            return {
                key: CustomDataset.batch_to_device(value, device)
                for key, value in batch.items()
            }
        elif isinstance(batch, collections.abc.Sequence):
            return [CustomDataset.batch_to_device(value, device) for value in batch]
        else:
            raise ValueError(f"Can not move {type(batch)} to device.")

    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
        """
        Preprocesses the input dataframe

        Args:
            df: the full training dataframe
            cfg: config
        Returns:
            the processed dataframe
        """

        def personalize(text):
            text = text.replace("Open Assistant", cfg.dataset.chatbot_name)
            text = text.replace("Open-Assistant", cfg.dataset.chatbot_name)
            text = text.replace("open-assistant", cfg.dataset.chatbot_name)
            text = text.replace("OpenAssistant", cfg.dataset.chatbot_name)
            text = text.replace("open assistant", cfg.dataset.chatbot_name)
            text = text.replace("Open Assistand", cfg.dataset.chatbot_name)
            text = text.replace("Open Assitant", cfg.dataset.chatbot_name)
            text = text.replace("Open Assistent", cfg.dataset.chatbot_name)
            text = text.replace("Open Assisstant", cfg.dataset.chatbot_name)
            text = text.replace("Open Assitent", cfg.dataset.chatbot_name)
            text = text.replace("Open Assitiant", cfg.dataset.chatbot_name)
            text = text.replace("Open Assistiant", cfg.dataset.chatbot_name)
            text = text.replace("Open Assitan ", cfg.dataset.chatbot_name + " ")
            text = text.replace("Open Assistan ", cfg.dataset.chatbot_name + " ")
            text = text.replace("Open Asistant", cfg.dataset.chatbot_name)
            text = text.replace("Open Assiant", cfg.dataset.chatbot_name)
            text = text.replace("Assistant", cfg.dataset.chatbot_name)
            text = text.replace("ChatGPT", cfg.dataset.chatbot_name)
            text = text.replace("LAION AI", cfg.dataset.chatbot_author)
            text = text.replace("LAION-AI", cfg.dataset.chatbot_author)
            text = text.replace("LAION,", cfg.dataset.chatbot_author + ",")
            text = text.replace("LAION.ai", cfg.dataset.chatbot_author)
            text = text.replace("LAION.", cfg.dataset.chatbot_author + ".")
            text = text.replace("LAION", cfg.dataset.chatbot_author)
            text = text.replace("Laion AI", cfg.dataset.chatbot_author)
            text = text.replace("OpenAI", cfg.dataset.chatbot_author)
            text = text.replace("Open AI", cfg.dataset.chatbot_author)
            text = text.replace("openai", cfg.dataset.chatbot_author)
            text = text.replace("open ai", cfg.dataset.chatbot_author)

            return text

        if cfg.dataset.personalize:
            for prompt_col in cfg.dataset.prompt_column:
                df[prompt_col] = df[prompt_col].apply(personalize)
            df[cfg.dataset.answer_column] = df[cfg.dataset.answer_column].apply(
                personalize
            )

        return df

    def get_train_collate_fn(self):
        """
        Returns train batch collate function for the PyTorch Dataloader.
        By default returns None that uses the default PyTorch collate
        """

        return None

    def get_validation_collate_fn(self):
        """
        Return validation batch collate function for the PyTorch Dataloader.
        By default returns None that uses the default PyTorch collate
        """

        return None

    def postprocess_batch_predictions(self, output: dict) -> dict:
        if "predicted_answer_ids" in output.keys():
            predicted_text = [
                self.tokenizer.decode(ids, skip_special_tokens=True).strip()
                for ids in output["predicted_answer_ids"]
            ]

            output["predicted_text"] = np.array(predicted_text)
            del output["predicted_answer_ids"]
        return output

    @staticmethod
    def clean_output(
        output: dict,
        cfg: Any,
    ):
        output["predicted_text"] = output["predicted_text"].tolist()
        for j in range(len(output["predicted_text"])):
            curr_text = output["predicted_text"][j].strip()
            for stop_token in cfg.tokenizer._stop_words:
                if curr_text.find(stop_token) != -1:
                    curr_text = curr_text[: curr_text.find(stop_token)]
            output["predicted_text"][j] = curr_text.strip()

        return output

    def postprocess_output(self, cfg, df: pd.DataFrame, output: dict) -> dict:
        if not cfg.prediction.metric == "Perplexity":
            output = self.clean_output(output, cfg)

        output["target_text"] = self.conversation_chain_handler.answers

        metric_func, _, _ = cfg.prediction.metric_class.get(cfg.prediction.metric)

        if "GPT" in cfg.prediction.metric:
            metrics, explanations = metric_func(
                cfg,
                output,
                df,
                raw_results=True,
            )
            output["explanations"] = explanations
        else:
            metrics = metric_func(
                cfg,
                output,
                df,
            )
        output["metrics"] = metrics

        return output

    def format_output(
        self, cfg, df: pd.DataFrame, output: dict
    ) -> tuple[dict, pd.DataFrame]:
        output = {
            key: value
            for key, value in output.items()
            if key not in ["loss", "target", "losses"]
        }
        output.pop("target_text", None)

        # in case limit_chained_samples is True, only last answer is predicted
        end_conversation_ids = (
            self.conversation_chain_handler.get_conversation_end_ids()
        )

        if "predicted_text" in output.keys():
            output["predicted_text"] = np.array(output["predicted_text"])

        if "logits" in output.keys():
            output["logits"] = np.array(output["logits"].float())

        if isinstance(cfg.dataset.prompt_column, tuple):
            for col in cfg.dataset.prompt_column:
                output[col] = df.loc[end_conversation_ids, col].values
        else:
            output[cfg.dataset.prompt_column] = df.loc[
                end_conversation_ids, cfg.dataset.prompt_column
            ].values

        if "predicted_text" in output.keys():
            col_name = cfg.dataset.answer_column
            if isinstance(col_name, list):
                col_name = ", ".join(col_name)
            df[f"pred_{col_name}"] = (
                "NO ANSWER GENERATED. ONLY LAST ANSWER OF A CONVERSATION IS PREDICTED."
            )
            df.loc[end_conversation_ids, f"pred_{col_name}"] = output["predicted_text"]
        return output, df

    @classmethod
    def sanity_check(cls, df: pd.DataFrame, cfg: Any, mode: str = "train"):
        """
        Quick check whether Dataframe and configurations are correctly set.
        """
        has_parent_column = (
            cfg.dataset.parent_id_column not in ("None", None)
            and cfg.dataset.parent_id_column in df.columns
        )

        if has_parent_column:
            assert cfg.dataset.id_column != cfg.dataset.parent_id_column, (
                "'Id Column' should be different from 'Parent column'"
            )

        if has_parent_column and cfg.dataset.id_column in df.columns:
            assert (
                df[cfg.dataset.parent_id_column] != df[cfg.dataset.id_column]
            ).all(), (
                f"Parent id column:{cfg.dataset.parent_id_column}"
                " is the same as id column for some rows"
            )
            assert (df[cfg.dataset.parent_id_column].fillna("") == "").sum() > 0, (
                "Did not find any conversation chain. "
                "Please ensure that some parent ids are empty."
                "\n"
                "Conversations are chained using parent id, "
                "start conversation record should have empty parent id."
                "\n"
                f"Parent id column checked:{cfg.dataset.parent_id_column}"
            )

        assert cfg.dataset.answer_column in df.columns, (
            f"Answer column {cfg.dataset.answer_column} not found in the "
            f"{mode} DataFrame."
        )
        assert df.shape[0] == df[[cfg.dataset.answer_column]].dropna().shape[0], (
            f"The {mode} DataFrame"
            f" column {cfg.dataset.answer_column}"
            " contains missing values."
        )
        if has_parent_column:
            assert cfg.dataset.id_column in df.columns, (
                "When using Parent Column, set 'Id Column' in the previous screen. "
            )

        if has_parent_column and df[cfg.dataset.parent_id_column].notna().any():
            # Comprehensive checks for conversation chaining
            parent_id_list = df[cfg.dataset.parent_id_column].tolist()
            id_list = df[cfg.dataset.id_column].tolist()
            # Track if any valid parent_id is found in the id_list
            found_valid_parent = False

            # 1. Check if at least one parent_id element is present in id_list
            for pid in parent_id_list:
                if pid is not None and not pd.isna(pid) and pid in id_list:
                    found_valid_parent = True
                    break

            # If no valid parent_id is found in the id_list, raise an error
            if not found_valid_parent:
                raise AssertionError(
                    "None of the Parent IDs in the 'Parent Id Column' were found "
                    "in the 'Id Column'. "
                    "Please ensure that at least one ID in 'Parent Id Column' "
                    "is present in the 'Id Column'. "
                    f"Checked 'Parent Id Column': '{cfg.dataset.parent_id_column}', "
                    f"and 'Id Column': '{cfg.dataset.id_column}'."
                )
            # 2. Check if all elements in id_list are unique
            if len(id_list) != len(set(id_list)):
                raise AssertionError("ID list contains duplicate values.")
            # 3. Check if parent_id[i] is not the same as id_list[i] (self-referencing)
            for i in range(len(id_list)):
                if parent_id_list[i] == id_list[i]:
                    raise AssertionError(f"ID '{id_list[i]}' is self-referencing.")
            # 4. Check if there is at least one root element (where parent_id is None)
            if not (None in parent_id_list or pd.isna(parent_id_list).any()):
                raise AssertionError(
                    "There must be at least one root element (with no parent). "
                    "Currently, all records have a parent."
                )

            # 5. Check for circular references
            def find_ancestor(node, parent_map):
                seen = set()
                while node in parent_map:
                    if node in seen:
                        raise AssertionError(
                            f"Circular reference detected for ID '{node}'."
                        )
                    seen.add(node)
                    node = parent_map[node]
                return False

            parent_map = {
                id_list[i]: parent_id_list[i]
                for i in range(len(id_list))
                if parent_id_list[i] is not None
            }

            for id_ in id_list:
                find_ancestor(id_, parent_map)

    def _get_raw_labels(
        self, prompt_encodings, answer_encodings, prompt_label_masks=None
    ):
        labels = torch.cat(
            [
                torch.cat([prompt_encoding, answer_encoding])
                for prompt_encoding, answer_encoding in zip(
                    prompt_encodings, answer_encodings, strict=False
                )
            ]
        ).clone()

        if self.cfg.dataset.mask_prompt_labels:
            masks = []
            for idx, (prompt_encoding, answer_encoding) in enumerate(
                zip(prompt_encodings, answer_encodings, strict=False)
            ):
                if (
                    not self.cfg.dataset.only_last_answer
                    or idx == len(answer_encodings) - 1
                ):
                    if (
                        getattr(self.cfg.dataset, "mask_prompt_user_text_only", False)
                        and prompt_label_masks is not None
                    ):
                        prompt_mask = prompt_label_masks[idx]
                    else:
                        prompt_mask = torch.ones_like(prompt_encoding)
                    mask = [
                        prompt_mask,
                        torch.zeros_like(answer_encoding),
                    ]
                else:
                    mask = [
                        torch.ones_like(prompt_encoding),
                        torch.ones_like(answer_encoding),
                    ]
                masks.append(torch.cat(mask))
            masks = torch.cat(masks).to(torch.bool)
            labels.masked_fill_(masks, -100)
        return labels

    def get_labels(self, prompt_encodings, answer_encodings):
        labels = self._get_raw_labels(prompt_encodings, answer_encodings)
        return self.pad_labels(labels, self.cfg.tokenizer.max_length)

    @staticmethod
    def pad_labels(labels: torch.Tensor, max_length: int):
        if max_length < len(labels):
            labels = labels[-max_length:]

        sample = dict(labels=torch.full((max_length,), -100))
        sample["labels"][-len(labels) :] = labels
        return sample

    def get_encodings(
        self,
        input_text_dict: dict[str, list[str]],
        augment: bool = True,
        trim_to_max_length: bool = True,
    ):
        """
        Get encodings for a single conversation history.
        Args:
            input_text_dict: A dictionary containing the input text for a single sample.
            Contains the keys "systems", "prompts", "answers".
            System may be an empty string.
        """
        encodings = [
            self._get_sample_encoding(system, prompt, answer)
            for idx, (system, prompt, answer) in enumerate(
                zip(
                    input_text_dict["systems"],
                    input_text_dict["prompts"],
                    input_text_dict["answers"],
                    strict=False,
                )
            )
        ]

        if self.mode == "train" and augment:
            encodings = self.augment_data(encodings)

        if trim_to_max_length:
            encodings = self._trim_encodings_to_max_length(encodings)

        system_encoding = encodings[0][0]
        prompt_encodings = [encoding[1] for encoding in encodings]
        answer_encodings = [encoding[2] for encoding in encodings]
        prompt_label_masks = [encoding[3] for encoding in encodings]
        # concatenate system encoding with root prompt encoding
        prompt_encodings[0] = torch.cat([system_encoding, prompt_encodings[0]])
        prompt_label_masks[0] = torch.cat(
            [
                torch.zeros_like(system_encoding, dtype=torch.bool),
                prompt_label_masks[0],
            ]
        )
        return (
            system_encoding,
            prompt_encodings,
            answer_encodings,
            prompt_label_masks,
        )

    def _trim_encodings_to_max_length(self, encodings: list[list[torch.Tensor]]):
        """
        Trim oldest conversation turns so the concatenated prompt/answer tokens fit
        into tokenizer.max_length as much as possible before final padding/truncation.
        """
        max_length = self.cfg.tokenizer.max_length
        start_idx = 0
        total_len = sum(len(prompt) + len(answer) for _, prompt, answer, _ in encodings)
        while start_idx < len(encodings) - 1 and total_len > max_length:
            _, prompt, answer, _ = encodings[start_idx]
            total_len -= len(prompt) + len(answer)
            start_idx += 1

        return encodings[start_idx:]

    def augment_data(self, encodings):
        parent_encodings = encodings[:-1]
        # randomly skip parent
        parent_encodings = [
            encoding
            for idx, encoding in enumerate(parent_encodings)
            if np.random.random() > self.cfg.augmentation.skip_parent_probability
        ]
        # randomly replace parent with another parent
        if np.random.random() < self.cfg.augmentation.random_parent_probability:
            idx = np.random.randint(len(self.conversation_chain_handler.prompts))
            parent_encodings = [
                self._get_sample_encoding(
                    self.parse_system(
                        self.cfg, self.conversation_chain_handler.systems[idx]
                    ),
                    self.parse_prompt_body(
                        self.cfg, self.conversation_chain_handler.prompts[idx]
                    ),
                    self.parse_answer(
                        self.cfg, self.conversation_chain_handler.answers[idx]
                    ),
                )
            ] + parent_encodings[1:]
        encodings = parent_encodings + [encodings[-1]]
        return encodings

    def _get_sample_encoding(self, system: str, prompt: str, answer: str) -> list:
        if len(system) > 0:
            system_encoding = self.encode(
                self.tokenizer, system, self.cfg.tokenizer.max_length, "right"
            )["input_ids"]
        else:
            system_encoding = torch.empty(0, dtype=torch.long)
        prompt_encoding, prompt_label_mask = self._get_prompt_encoding_and_mask(prompt)
        answer_encoding = self.encode(
            self.tokenizer, answer, self.cfg.tokenizer.max_length, "right"
        )["input_ids"]

        return [system_encoding, prompt_encoding, answer_encoding, prompt_label_mask]

    def _get_prompt_encoding_and_mask(
        self, prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_parts = [
            (
                codecs.decode(self.cfg.dataset.text_prompt_start, "unicode_escape"),
                False,
            ),
            (prompt, True),
        ]
        if self.cfg.dataset.add_eos_token_to_prompt:
            prompt_parts.append((self.cfg.tokenizer._tokenizer_eos_token, False))
        prompt_parts.append(
            (
                codecs.decode(self.cfg.dataset.text_answer_separator, "unicode_escape"),
                False,
            )
        )

        encodings = []
        masks = []
        for text, mask_user_text in prompt_parts:
            if text == "":
                continue
            input_ids = self.encode(
                self.tokenizer, text, self.cfg.tokenizer.max_length, "left"
            )["input_ids"]
            encodings.append(input_ids)
            # Keep chat-format tokens trainable; mask only the natural user text.
            masks.append(torch.full_like(input_ids, mask_user_text, dtype=torch.bool))

        if not encodings:
            return torch.empty(0), torch.empty(0, dtype=torch.bool)

        prompt_encoding = torch.cat(encodings)[-self.cfg.tokenizer.max_length :]
        prompt_label_mask = torch.cat(masks)[-self.cfg.tokenizer.max_length :]
        return prompt_encoding, prompt_label_mask

    @staticmethod
    def pad_tokens(
        input_ids,
        attention_mask,
        max_length,
        pad_token_id,
        direction="left",
        prefix="",
    ):
        sample = {}

        if max_length < len(input_ids):
            logger.info(f"Input exceeds max_length of {max_length}, truncating sample.")
            input_ids = input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]

        if len(input_ids) > 0:
            if direction == "left":
                sample[f"{prefix}input_ids"] = torch.full((max_length,), pad_token_id)
                sample[f"{prefix}input_ids"][-len(input_ids) :] = input_ids
                sample[f"{prefix}attention_mask"] = torch.zeros(max_length)
                sample[f"{prefix}attention_mask"][-len(input_ids) :] = attention_mask
            else:
                sample[f"{prefix}input_ids"] = torch.full((max_length,), pad_token_id)
                sample[f"{prefix}input_ids"][: len(input_ids)] = input_ids
                sample[f"{prefix}attention_mask"] = torch.zeros(max_length)
                sample[f"{prefix}attention_mask"][: len(input_ids)] = attention_mask
        else:
            # Pad everything if empty (continued pretraining)
            sample[f"{prefix}input_ids"] = torch.full((max_length,), pad_token_id)
            sample[f"{prefix}attention_mask"] = torch.zeros(max_length)

        return sample

    @staticmethod
    def encode(tokenizer, text: str, max_length: int, truncation_side: str) -> dict:
        encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        encodings["input_ids"] = encodings["input_ids"][0]
        encodings["attention_mask"] = encodings["attention_mask"][0]
        if truncation_side == "right":
            encodings["input_ids"] = encodings["input_ids"][:max_length]
            encodings["attention_mask"] = encodings["attention_mask"][:max_length]
        else:
            encodings["input_ids"] = encodings["input_ids"][-max_length:]
            encodings["attention_mask"] = encodings["attention_mask"][-max_length:]
        return encodings
