```bash
# setup the repo and install packages
git clone https://github.com/nickhayeck/nickgpt.git
cd nickgpt/
uv sync

# download data and build vocab
python -m src.main dataset-download
python -m src.main vocab-build

# pretrain on one batch of the dataset (just as a smoke test)
python -m src.main dataset-build --max-examples 32
python -m src.main pretrain-run "overfit_batch"

# build the full dataset and pretrain a 2m GPT
python -m src.main dataset-build
python -m src.main pretrain-run "nickgpt-2m"

```

### timings
- dataset-download (depends on wifi; downloads ~2GB)
- vocab-build artifacts/data/tiny_stories-train.txt (10min)
- dataset-build (22 min; training: ~2m sequences, ~4b tokens; valid: ~22k sequences, ~24m tokens)
- pretrain-run "overfit-batch"