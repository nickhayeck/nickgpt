import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .state import initialize, load_checkpoint
from .trainer import train


@dataclass(kw_only=True)
class Config:
    # meta
    name: str
    # data and model
    train_data_config: dict[str, Any]
    valid_data_config: dict[str, Any]
    model_config: dict[str, Any]
    # optimizer and scheduler
    optimizer_kind: str = "adamw"
    optimizer_config: dict[str, Any] | None = None
    scheduler_kind: str = "constant"
    scheduler_config: dict[str, Any] | None = None
    # trainer parameters
    max_steps: int
    logging_frequency: int
    checkpoint_frequency: int
    validation_frequency: int
    validation_batches: int
    # runtime
    device: str | None = None
    precision: str | None = "auto"
    compile_mode: str | None = "default"


def run(
    config: Config,
    artifact_root: Path = Path("artifacts/"),
    quiet: bool = False,
):
    run_dir = _run_dir(artifact_root, config)
    _record_config(run_dir, config)
    state = initialize(
        train_data_config=config.train_data_config,
        valid_data_config=config.valid_data_config,
        model_config=config.model_config,
        optimizer_config=config.optimizer_config,
        scheduler_config=config.scheduler_config,
        optimizer_kind=config.optimizer_kind,
        scheduler_kind=config.scheduler_kind,
        device=config.device,
        precision=config.precision,
    )

    _log(quiet, f"starting experiment in {run_dir}")

    train(
        run_dir,
        state,
        max_steps=config.max_steps,
        logging_frequency=config.logging_frequency,
        checkpoint_frequency=config.checkpoint_frequency,
        validation_frequency=config.validation_frequency,
        validation_batches=config.validation_batches,
        compile_mode=config.compile_mode,
    )

    _log(quiet, "done!")


def resume(
    config: Config,
    checkpoint: Path,
    artifact_root: Path = Path("artifacts/"),
    quiet: bool = False,
):
    run_dir = _run_dir(artifact_root, config)
    _record_config(run_dir, config)
    chkpt = load_checkpoint(
        checkpoint,
        device=config.device,
        precision=config.precision,
    )

    _log(quiet, f'resuming checkpoint "{checkpoint}" into "{run_dir}"')

    train(
        run_dir,
        chkpt,
        max_steps=config.max_steps,
        logging_frequency=config.logging_frequency,
        checkpoint_frequency=config.checkpoint_frequency,
        validation_frequency=config.validation_frequency,
        validation_batches=config.validation_batches,
        compile_mode=config.compile_mode,
    )

    _log(quiet, "done!")


_CONFIG_REGISTRY: dict[str, Config] = {}


def register_config(config: Config) -> None:
    if config.name in _CONFIG_REGISTRY:
        raise KeyError(f'duplicate config name: "{config.name!r}".')
    _CONFIG_REGISTRY[config.name] = config


def get_config(name: str) -> Config:
    try:
        return _CONFIG_REGISTRY[name]
    except KeyError as e:
        available = ", ".join(sorted(_CONFIG_REGISTRY)) or "(no registered configs)"
        raise KeyError(
            f"Unknown config {name!r}. Available configs: {available}"
        ) from e


def get_configs() -> list[str]:
    return list(_CONFIG_REGISTRY.keys())


def _run_dir(artifact_root: Path, config: Config):
    run_dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = artifact_root / f"{run_dt}_{config.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _record_config(run_dir: Path, config: Config):
    (run_dir / "config.pkl").write_bytes(pickle.dumps(config))


def _log(quiet: bool, v: str):
    if not quiet:
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{dt}][INFO] {v}")
