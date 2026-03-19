from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..tokenizer import Tokenizer


def url(split: str) -> str:
    assert split in ["train", "valid"]
    return f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-{split}.txt?download=true"


def text_path(data_dir: Path, split: str) -> Path:
    assert split in ["train", "valid"]
    return data_dir / f"tiny_stories-{split}.txt"


def npz_path(data_dir: Path, split: str, tag: str | None) -> Path:
    assert split in ["train", "valid"]
    tag = f"-{tag}" if tag else ""
    return data_dir / f"tiny_stories-{split}{tag}.npz"


def read_examples(file: Path | str) -> list[str]:
    file = Path(file)
    corpus = file.read_text()
    seqs = corpus.split("<|endoftext|>")
    seqs = [s.strip() for s in seqs]
    return seqs


def encode(seqs: list[str], tok: Tokenizer, *, verbose: bool = True):
    eos = tok.get_control_token("<|EOS|>")
    tokens = [tok.encode(seq) + [eos] for seq in seqs]
    tokens = [np.array(seq, np.uint16) for seq in tokens]
    # right-pad to length of longest
    pad = tok.get_control_token("<|pad|>")
    max_len = max(len(t) for t in tokens)
    # prealloc array with padding tokens
    output = np.full((len(tokens), max_len), pad, np.uint16)
    length = np.array([len(seq) for seq in tokens], np.uint16)
    for i, seq in enumerate(tokens):
        output[i, : len(seq)] = seq

    if verbose:
        print("example count:", len(tokens))
        print("encoded length (token count):", output.size)
        print("encoded length (kilobytes):", output.size * 2 / 1_024)

    return output, length


class TinyStoriesIterableDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        data_file: Path | str,
        context_size: int,
        pad_token_id: int,
        seed: int = 0,
    ):
        self.context_size = context_size
        self.pad_token_id = pad_token_id
        self.seed = seed

        with np.load(data_file, allow_pickle=False) as npz:
            self.data = np.asarray(npz["data"])
            self.unpadded_length = np.asarray(npz["unpadded_length"], dtype=np.int64)

        if self.data.ndim != 2:
            raise ValueError(f"expected rank-2 dataset, got shape {self.data.shape}")
        if self.unpadded_length.ndim != 1:
            raise ValueError(
                f"expected rank-1 length array, got shape {self.unpadded_length.shape}"
            )
        if self.data.shape[0] != self.unpadded_length.shape[0]:
            raise ValueError(
                "data and unpadded_length must agree on example count: "
                f"{self.data.shape[0]} != {self.unpadded_length.shape[0]}"
            )

        self.valid_examples = np.flatnonzero(self.unpadded_length >= 2)
        if self.valid_examples.size == 0:
            raise ValueError("dataset does not contain any examples with length >= 2")

        weights = self.unpadded_length[self.valid_examples] - 1
        weights = weights.astype(np.float64)
        self.sample_probs = weights / weights.sum()

        self._rng: np.random.Generator | None = None
        self._pending_state_dict: dict[str, Any] | None = None
        self._samples_emitted = 0

    def __iter__(self):
        self._ensure_rng()
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_rng()
        assert self._rng is not None

        context_size = self.context_size
        pad = self.pad_token_id
        inputs = torch.full((context_size,), pad, dtype=torch.long)
        targets = torch.full((context_size,), pad, dtype=torch.long)

        example_idx = int(
            self._rng.choice(
                self.valid_examples,
                size=(),
                replace=True,
                p=self.sample_probs,
            )
        )
        length = int(self.unpadded_length[example_idx])
        max_start = length - 1
        start = 0 if max_start <= 1 else int(self._rng.integers(0, max_start))
        stop = min(length, start + context_size + 1)
        window = self.data[example_idx, start:stop]
        usable = window.size - 1
        if usable > 0:
            inputs[:usable] = torch.as_tensor(window[:-1], dtype=torch.long)
            targets[:usable] = torch.as_tensor(window[1:], dtype=torch.long)

        self._samples_emitted += 1
        return inputs, targets

    def state_dict(self) -> dict[str, Any]:
        self._ensure_rng()
        assert self._rng is not None
        return {
            "rng_state": self._rng.bit_generator.state,
            "samples_emitted": self._samples_emitted,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._pending_state_dict = dict(state_dict)

    def _ensure_rng(self):
        if self._rng is not None:
            return

        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        self._rng = np.random.default_rng(self.seed + worker_id)
        self._samples_emitted = 0

        if self._pending_state_dict is None:
            return

        self._rng.bit_generator.state = self._pending_state_dict["rng_state"]
        self._samples_emitted = int(self._pending_state_dict["samples_emitted"])
        self._pending_state_dict = None
