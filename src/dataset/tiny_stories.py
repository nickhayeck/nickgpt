from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

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


@dataclass
class TinyStoriesDataLoaderState:
    data_file: Path | str
    batch_size: int
    context_size: int
    pad_token_id: int
    seed: int = 0
    rng_state: dict[str, Any] | None = None
    batches_emitted: int = 0
    samples_emitted: int = 0

    def __post_init__(self):
        self.data_file = str(self.data_file)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.context_size <= 0:
            raise ValueError(f"context_size must be positive, got {self.context_size}")


class TinyStoriesDataLoader:
    def __init__(self, state: TinyStoriesDataLoaderState):
        self.state = state

        with np.load(self.state.data_file, allow_pickle=False) as npz:
            self.data = np.asarray(npz["data"], dtype=np.int64)
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

        self.rng = np.random.default_rng(self.state.seed)
        if self.state.rng_state is not None:
            self.rng.bit_generator.state = self.state.rng_state

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = self.state.batch_size
        context_size = self.state.context_size
        pad = self.state.pad_token_id

        inputs = np.full((batch_size, context_size), pad, dtype=np.int64)
        targets = np.full((batch_size, context_size), pad, dtype=np.int64)

        example_indices = self.rng.choice(
            self.valid_examples,
            size=batch_size,
            replace=True,
            p=self.sample_probs,
        )
        for row_idx, example_idx in enumerate(example_indices):
            length = int(self.unpadded_length[example_idx])
            max_start = length - 1
            start = 0 if max_start <= 1 else int(self.rng.integers(0, max_start))
            stop = min(length, start + context_size + 1)
            window = self.data[example_idx, start:stop]
            usable = window.size - 1
            if usable <= 0:
                continue

            inputs[row_idx, :usable] = window[:-1]
            targets[row_idx, :usable] = window[1:]

        self.state.rng_state = self.rng.bit_generator.state  # type: ignore
        self.state.batches_emitted += 1
        self.state.samples_emitted += batch_size

        return torch.from_numpy(inputs).long(), torch.from_numpy(targets).long()
