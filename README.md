python -m src.main dataset-download (depends on wifi; downloads ~2GB)
python -m src.main vocab-build artifacts/data/tiny_stories-train.txt (10min)
python -m src.main dataset-build (22 min; training: ~2m sequences, ~4b tokens; valid: ~22k sequences, ~24m tokens)
python -m src.main pretrain-run "overfit-batch"