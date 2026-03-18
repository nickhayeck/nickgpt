from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as nnf
from torch.utils.tensorboard import SummaryWriter

from .. import model as llm_model
from ..dataset.tiny_stories import TinyStoriesDataLoader, TinyStoriesDataLoaderState


@dataclass
class ModelState:
    config: dict[str, Any]
    state_dict: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerState:
    kind: str = "adamw"
    config: dict[str, Any] = field(default_factory=dict)
    state_dict: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerState:
    kind: str = "constant"
    config: dict[str, Any] = field(default_factory=dict)
    state_dict: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingState:
    train_dataset: TinyStoriesDataLoaderState
    valid_dataset: TinyStoriesDataLoaderState
    global_step: int
    model: ModelState
    optimizer: OptimizerState
    scheduler: SchedulerState


def init_train_state(
    *,
    train_data_config: dict[str, Any],
    valid_data_config: dict[str, Any],
    model_config: dict[str, Any],
    optimizer_config: dict[str, Any] | None = None,
    scheduler_config: dict[str, Any] | None = None,
    optimizer_kind: str = "adamw",
    scheduler_kind: str = "constant",
) -> TrainingState:
    train_data_config = dict(train_data_config)
    model_config = dict(model_config)
    optimizer_config = dict(optimizer_config or {})
    scheduler_config = dict(scheduler_config or {})

    train_dataset = _build_dataset_state(train_data_config)
    valid_dataset = _build_dataset_state(valid_data_config)
    model = build_model(ModelState(config=model_config))
    optimizer_state = OptimizerState(kind=optimizer_kind, config=optimizer_config)
    scheduler_state = SchedulerState(kind=scheduler_kind, config=scheduler_config)
    optimizer = _build_optimizer(model, optimizer_state)
    scheduler = _build_scheduler(optimizer, scheduler_state)

    return TrainingState(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        global_step=0,
        model=ModelState(
            config=model_config, state_dict=copy.deepcopy(model.state_dict())
        ),
        optimizer=OptimizerState(
            kind=optimizer_kind,
            config=optimizer_config,
            state_dict=copy.deepcopy(optimizer.state_dict()),
        ),
        scheduler=SchedulerState(
            kind=scheduler_kind,
            config=scheduler_config,
            state_dict=copy.deepcopy(scheduler.state_dict()),
        ),
    )


def save_checkpoint(state: TrainingState, file: Path | str):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, file)


def load_checkpoint(file: Path | str) -> TrainingState:
    state = torch.load(file, map_location="cpu", weights_only=False)
    if not isinstance(state, TrainingState):
        raise TypeError(f"expected TrainingState checkpoint, got {type(state)!r}")
    return state


def build_model(model_state: ModelState) -> llm_model.GPT:
    config = dict(model_state.config)
    config["device"] = torch.device("cpu")
    return llm_model.GPT(**config)


def train(
    run_dir: Path | str,
    state: TrainingState,
    *,
    max_steps: int,
    logging_frequency: int,
    checkpoint_frequency: int,
    validation_frequency: int,
    validation_batches: int,
) -> TrainingState:
    if checkpoint_frequency <= 0:
        raise ValueError(
            f"checkpoint_freq must be positive, got {checkpoint_frequency}"
        )
    if logging_frequency <= 0:
        raise ValueError(f"log_freq must be positive, got {logging_frequency}")

    run_dir = Path(run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    train_loader = TinyStoriesDataLoader(state.train_dataset)
    valid_loader = TinyStoriesDataLoader(state.valid_dataset)
    model = build_model(state.model)
    optimizer = _build_optimizer(model, state.optimizer)
    scheduler = _build_scheduler(optimizer, state.scheduler)

    if state.model.state_dict:
        model.load_state_dict(state.model.state_dict)
    if state.optimizer.state_dict:
        optimizer.load_state_dict(state.optimizer.state_dict)
    if state.scheduler.state_dict:
        scheduler.load_state_dict(state.scheduler.state_dict)

    model.train()
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    try:
        while state.global_step < max_steps:
            inputs, targets = train_loader.next_batch()
            logits = model(inputs)
            loss = nnf.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=state.train_dataset.pad_token_id,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            scheduler.step()

            state.global_step += 1
            last_step = state.global_step == max_steps

            if state.global_step % logging_frequency == 0 or last_step:
                lr = float(optimizer.param_groups[0]["lr"])
                writer.add_scalar("train/loss", float(loss.item()), state.global_step)
                writer.add_scalar("train/lr", lr, state.global_step)
                writer.add_scalar("train/grad_norm", grad_norm, state.global_step)
                writer.flush()

            if state.global_step % validation_frequency == 0 or last_step:
                valid_loss = _evaluate_loss(
                    model,
                    valid_loader,
                    pad_token_id=state.valid_dataset.pad_token_id,
                    num_batches=validation_batches,
                )
                writer.add_scalar("valid/loss", valid_loss, state.global_step)
                writer.flush()

            if state.global_step % checkpoint_frequency == 0 or last_step:
                _sync_training_state(state, model, optimizer, scheduler)
                step_file = checkpoints_dir / f"step_{state.global_step:08d}.pt"
                save_checkpoint(state, step_file)
                save_checkpoint(state, checkpoints_dir / "latest.pt")
    finally:
        _sync_training_state(state, model, optimizer, scheduler)
        save_checkpoint(state, checkpoints_dir / "latest.pt")
        writer.close()

    return state


def _build_dataset_state(config: dict[str, Any]):
    return TinyStoriesDataLoaderState(
        **config,
        pad_token_id=257,  # ControlToken(name='<|pad|>')
    )


def _build_optimizer(
    model: torch.nn.Module,
    optimizer_state: OptimizerState,
) -> torch.optim.Optimizer:
    kind = optimizer_state.kind.lower()
    if kind != "adamw":
        raise ValueError(f"unsupported optimizer kind: {optimizer_state.kind}")

    config: dict[str, Any] = {"lr": 3e-4}
    config.update(optimizer_state.config)
    return torch.optim.AdamW(model.parameters(), **config)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_state: SchedulerState,
) -> torch.optim.lr_scheduler.LRScheduler:
    kind = scheduler_state.kind.lower()
    config = dict(scheduler_state.config)

    if kind == "constant":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda _: 1.0,
        )

    if kind == "linear_warmup":
        warmup_steps = int(config.get("warmup_steps", 0))
        if warmup_steps <= 0:
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda _: 1.0,
            )

        def lr_lambda(step: int) -> float:
            return min((step + 1) / warmup_steps, 1.0)

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
        )

    raise ValueError(f"unsupported scheduler kind: {scheduler_state.kind}")


def _sync_training_state(
    state: TrainingState,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
):
    state.model.state_dict = copy.deepcopy(model.state_dict())
    state.optimizer.state_dict = copy.deepcopy(optimizer.state_dict())
    state.scheduler.state_dict = copy.deepcopy(scheduler.state_dict())


def _evaluate_loss(
    model: torch.nn.Module,
    dataloader: TinyStoriesDataLoader,
    *,
    pad_token_id: int,
    num_batches: int,
) -> float:
    losses: list[float] = []
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for _ in range(num_batches):
                inputs, targets = dataloader.next_batch()
                logits = model(inputs)
                loss = nnf.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    ignore_index=pad_token_id,
                )
                losses.append(float(loss.item()))
    finally:
        if was_training:
            model.train()

    return sum(losses) / len(losses)
