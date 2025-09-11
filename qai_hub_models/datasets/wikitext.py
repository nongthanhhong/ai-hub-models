# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math

import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from qai_hub_models.datasets.common import BaseDataset, DatasetMetadata, DatasetSplit


def collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return batch[0]["input_ids"], batch[0]["attention_mask"], batch[0]["input_ids"]


class WikiText(BaseDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        block_size: int = 128,
        context_length: int = 4096,
        split: DatasetSplit = DatasetSplit.TEST,
        num_samples: int = 0,
    ):
        self.block_size = block_size
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.num_samples = num_samples

        if split == DatasetSplit.TEST:
            self.split_str = "test"
        elif split == DatasetSplit.TRAIN:
            self.split_str = "train"
        else:
            raise ValueError(
                "Wikitext dataset currently only supports `test` and `train` split"
            )

        raw_dataset = self.load_raw_dataset()

        # This is necessary because calibrating the model on data with tokens for the "\n\n" separator between texts
        # Causes a big drop in quantization accuracy
        separator = "\n\n" if split == DatasetSplit.TEST else self.tokenizer.bos_token
        
        # Fix: Limit the concatenated text to avoid exceeding model's max_position_embeddings
        # Get model's maximum sequence length from tokenizer config
        max_model_length = getattr(self.tokenizer, 'model_max_length', 131072)
        if max_model_length > 1000000:  # Some tokenizers have very large default values
            max_model_length = 131072  # Use Llama 3.2's actual limit
            
        # Take only the first portion of text that won't exceed the limit
        concatenated_text = separator.join(raw_dataset["text"])
        
        # Rough estimate: limit text to 80% of max length to account for tokenization expansion
        max_chars = int(max_model_length * 0.8 * 4)  # Rough char-to-token ratio
        if len(concatenated_text) > max_chars:
            concatenated_text = concatenated_text[:max_chars]
            print(f"WARNING: WikiText dataset truncated to {max_chars} characters to fit model limits")
        
        self.tokens = self.tokenizer(
            concatenated_text,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=max_model_length,
            truncation=True,
        )

    def load_raw_dataset(self) -> Dataset:
        return load_dataset(
            path="wikitext", name="wikitext-2-raw-v1", split=self.split_str
        )

    def __len__(self) -> int:
        if self.num_samples != 0:
            return self.num_samples
        if self.split_str == "train":
            # 80k samples to be passed for calibration and advanced algorithms like Sequential MSE.
            return 20
        return math.ceil(len(self.tokens["input_ids"][0]) / self.context_length)

    def __getitem__(self, index: int):
        num_tokens = self.tokens["input_ids"].shape[-1]
        start_index = index * self.context_length
        end_index = min((index + 1) * self.context_length, num_tokens)
        return {
            "input_ids": self.tokens["input_ids"][:, start_index:end_index],
            "attention_mask": self.tokens["attention_mask"][:, start_index:end_index],
        }

    def _download_data(self) -> None:
        pass

    @staticmethod
    def default_samples_per_job() -> int:
        """
        The default value for how many samples to run in each inference job.
        """
        return 1

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://huggingface.co/datasets/mindchain/wikitext2",
            split_description="test split",
        )
