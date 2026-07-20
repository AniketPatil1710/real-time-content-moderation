"""torch Dataset wrapper for tokenized comment data. Phase 1 (exercised in Phase 3 on GPU).

Depends on torch/transformers, which are only installed for GPU training —
not part of the lightweight local Phase 1/2 environment.
"""

from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.data.preprocess import load_label_names


class ToxicCommentsDataset(Dataset):
    """Tokenizes comment text on the fly and returns multi-label float targets.

    label_names controls both which columns are read and their output order;
    defaults to configs/labels.json, the single source of truth for label order.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        label_names: list[str] | None = None,
    ) -> None:
        self.label_names = label_names or load_label_names()
        self.texts = dataframe["comment_text"].tolist()
        self.labels = dataframe[self.label_names].to_numpy(dtype="float32")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)
        return item
