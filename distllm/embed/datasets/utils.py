"""Dataset utilities for batching sequences."""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset
from transformers import BatchEncoding
from transformers import PreTrainedTokenizer


class InMemoryDataset(Dataset):
    """Holds the data in memory for efficient batching."""

    def __init__(
        self,
        data: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        self.data = data
        self.metadata = metadata

        # Ensure that the metadata is the same length as the data
        if metadata is not None:
            assert len(data) == len(metadata)

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        """Get an item from the dataset."""
        return self.data[idx]


class DataCollator:
    """Data collator for batching sequences."""

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        """Initialize the data collator."""
        self.tokenizer = tokenizer

    def __call__(self, batch: list[str]) -> BatchEncoding:
        """Collate the batch of sequences."""
        return self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
