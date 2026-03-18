from .. import experiment

# 2M parameter model
model_config: dict = {
    "vocab_size": 4096,
    "context_size": 256,
    "embedding_dim": 128,
    "attn_heads": 4,
    "num_blocks": 7,
    "dropout": 0.0,
}
train_data_config = {
    "data_file": "artifacts/data/tiny_stories-train.npz",
    "batch_size": 128,
    "context_size": 256,
}
valid_data_config = {
    "data_file": "artifacts/data/tiny_stories-val.npz",
    "batch_size": 128,
    "context_size": 256,
}
optim_config = {
    "lr": 1e-3,
    "weight_decay": 0.01,
}

experiment.register_config(
    experiment.Config(
        name="nickgpt-2m",
        train_data_config=train_data_config,
        valid_data_config=valid_data_config,
        model_config=model_config,
        optimizer_kind="adamw",
        optimizer_config=optim_config,
        max_steps=5000,
        logging_frequency=10,
        checkpoint_frequency=1000,
        validation_frequency=100,
        validation_batches=32,
    )
)
