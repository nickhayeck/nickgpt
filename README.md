# nickgpt

`nickgpt` is a small PyTorch language-model project built around the TinyStories dataset. It includes:

- dataset download and preprocessing utilities
- a simple tokenizer/vocabulary builder
- a decoder-only GPT-style model
- named pretraining configs and a checkpointed training loop

The main entrypoint is [`src/main.py`](src/main.py). Data artifacts are written under `artifacts/data/`, and pretraining runs are written under `artifacts/pretraining/`.

## Quickstart

```bash
# clone the repo and install dependencies
git clone https://github.com/nickhayeck/nickgpt.git
cd nickgpt/
uv sync

# download the TinyStories train/valid text files into artifacts/data/
python -m src.main dataset-download

# build the vocabulary from the training split
# the default vocab size is 4096, which matches the bundled configs
python -m src.main vocab-build

# build a tiny tokenized dataset first so you can smoke-test the full pipeline
# this writes tiny_stories-{train,valid}-examples=32.npz
python -m src.main dataset-build --max-examples 32

# run the small overfit config
# this is the fastest way to verify that preprocessing, checkpoints, and training work
python -m src.main pretrain-run overfit-batch

# once the smoke test works, build the full tokenized dataset
python -m src.main dataset-build

# run the main 2M-parameter pretraining config
python -m src.main pretrain-run nickgpt-2m

# resume a run from a saved checkpoint if needed
python -m src.main pretrain-resume nickgpt-2m artifacts/pretraining/<run>/checkpoints/latest.pt
```

The bundled training configs live in `src/pretraining/configs/`. Right now the two main ones are:

- `overfit-batch`: a small smoke test that should overfit a tiny dataset quickly
- `nickgpt-2m`: the main TinyStories pretraining run

## Rough Timings

These are rough measurements, not guarantees.

| Step | Command | Rough time | Notes |
| --- | --- | --- | --- |
| Download dataset | `python -m src.main dataset-download` | Depends on network | Downloads about 2 GB |
| Build vocabulary | `python -m src.main vocab-build` | ~10 min | Reads `artifacts/data/tiny_stories-train.txt` |
| Build full tokenized dataset | `python -m src.main dataset-build` | ~22 min | Train split: ~2M sequences, ~4B tokens. Valid split: ~22k sequences, ~24M tokens |
| Smoke-test pretraining | `python -m src.main pretrain-run overfit-batch` | A few minutes | Measured on an RTX 5060 Ti |
| Main 2M pretraining run | `python -m src.main pretrain-run nickgpt-2m` | ~8 hr | Measured on an RTX 5060 Ti |
