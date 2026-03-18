from .. import experiment

train_data_config = {
    "data_file": "artifacts/data/tiny_stories-train-examples=32.npz",
    "batch_size": 32,
    "context_size": 256,
}
valid_data_config = {
    "data_file": "artifacts/data/tiny_stories-valid-examples=32.npz",
    "batch_size": 32,
    "context_size": 256,
}
model_config = {
    "vocab_size": 4096,
    "context_size": 256,
    "embedding_dim": 64,
    "attn_heads": 4,
    "num_blocks": 2,
}
optimizer_config = {
    "lr": 3e-3,
    "weight_decay": 0.0,
}

experiment.register_config(
    experiment.Config(
        # meta
        name="overfit_batch",
        # data and model
        train_data_config=train_data_config,
        valid_data_config=train_data_config,  # ignore basically
        model_config=model_config,
        # optimizer and scheduler
        optimizer_kind="adamw",
        optimizer_config=optimizer_config,
        # trainer parameters
        max_steps=2000,
        logging_frequency=50,
        checkpoint_frequency=2001,
        validation_frequency=50,
        validation_batches=1,
    )
)
