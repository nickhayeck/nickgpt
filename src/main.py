import asyncio
from pathlib import Path
from time import time

import numpy as np

from . import tokenizer
from .dataset import tiny_stories
from .dataset.download import download_files
from .pretraining import experiment


def dataset_download(
    data_dir: Path = Path("artifacts/data"),
    redownload: bool = False,
):
    """Downloads the TinyStories dataset to `data_dir`"""
    data_dir.mkdir(parents=True, exist_ok=True)
    files = dict[str, Path]()

    for split in ["train", "valid"]:
        p = tiny_stories.text_path(data_dir, split)
        if redownload or not p.exists():
            url = tiny_stories.url(split)
            files[url] = p

    if not files:
        print(
            f"all files already in {data_dir}. pass --redownload if you want to download them again."
        )
        return

    asyncio.run(
        download_files(
            files,
            concurrency=len(files),
        )
    )


def vocab_build(
    data_dir: Path = Path("artifacts/data"),
    vocab_file: Path = Path("artifacts/data/vocab.pkl"),
    max_size: int = 4096,
    min_frequency: int = 2,
    split_on: str | None = "<|endoftext|>",
    quiet: bool = False,
):
    """
    Builds the vocabulary representation using tokenizer.build(...) and saves it to `vocab_file`.

    The default vocab size is 4096 (this is pretty good for tiny-stories).
    """
    start_time = time()

    text = tiny_stories.text_path(data_dir, "train").read_text()
    text = text.split(split_on) if split_on else [text]
    tok = tokenizer.build(
        text,
        max_size=max_size,
        min_frequency=min_frequency,
    )
    tokenizer.save(tok, vocab_file)

    if not quiet:
        end_time = time() - start_time
        print(f'Wrote to: "{vocab_file}". Completed in {end_time:.2f}sec.')


def dataset_build(
    data_dir: Path = Path("artifacts/data"),
    vocab_file: Path = Path("artifacts/data/vocab.pkl"),
    max_examples: int | None = None,
    quiet: bool = False,
):
    """Tokenizes the dataset using `vocab_file` and saves it to `data_dir` as a compressed numpy array"""
    tag = f"examples={max_examples}" if max_examples else None

    for split in ["train", "valid"]:
        text_path = tiny_stories.text_path(data_dir, split)
        npz_path = tiny_stories.npz_path(data_dir, split, tag)

        examples = tiny_stories.read_examples(text_path)[:max_examples]
        tok = tokenizer.load(vocab_file)
        output, length = tiny_stories.encode(examples, tok, verbose=not quiet)

        with open(npz_path, "wb") as fp:
            np.savez_compressed(fp, data=output, unpadded_length=length)


def pretrain_run(
    config: str,
    pretraining_dir: Path = Path("artifacts/pretraining"),
    quiet: bool = False,
):
    cfg = experiment.get_config(config)
    experiment.run(
        cfg,
        artifact_root=pretraining_dir,
        quiet=quiet,
    )


def pretrain_resume(
    config: str,
    checkpoint: Path,
    pretraining_dir: Path = Path("artifacts/pretraining"),
    quiet: bool = False,
):
    cfg = experiment.get_config(config)
    experiment.resume(
        cfg,
        checkpoint,
        artifact_root=pretraining_dir,
        quiet=quiet,
    )


if __name__ == "__main__":
    import typer

    cli = typer.Typer(add_completion=False)

    cli.command()(dataset_download)
    cli.command()(vocab_build)
    cli.command()(dataset_build)
    cli.command()(pretrain_run)
    cli.command()(pretrain_resume)

    cli()
