from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as nnf
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .state import Training, save_checkpoint


def train(
    run_dir: Path | str,
    training: Training,
    *,
    max_steps: int,
    logging_frequency: int,
    checkpoint_frequency: int,
    validation_frequency: int,
    validation_batches: int,
    compile_mode: str | None,
) -> Training:
    # setup data for training
    train_iter = iter(training.train_data.loader)
    valid_iter = iter(training.valid_data.loader)

    # setup model for training
    model = training.model
    if compile_mode is not None:
        model.compile(mode=compile_mode)
    model.train()

    # setup directory and logging sink
    checkpoints_dir, tensorboard_dir = _setup_dir(run_dir)
    logger = SummaryWriter(log_dir=str(tensorboard_dir))
    
    # training loop.
    try:
        while training.global_step < max_steps:
            inputs, targets = _get_batch(train_iter, training.device)
            with _autocast_context(training.device, training.amp_dtype):
                logits = model(inputs)
                loss = nnf.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    ignore_index=training.train_data.pad_token_id,
                )

            # perform the backprop + optimizer steps
            if training.scaler.is_enabled():
                grad_norm = _scaled_step(training, loss, model)
            else:
                grad_norm = _unscaled_step(training, loss, model)

            training.global_step += 1

            # logging, validation, checkpointing
            last_step = training.global_step == max_steps

            if training.global_step % logging_frequency == 0 or last_step:
                logger.add_scalar(
                    "train/loss",
                    float(loss.item()),
                    training.global_step,
                )
                logger.add_scalar(
                    "train/lr",
                    float(training.optimizer.param_groups[0]["lr"]),
                    training.global_step,
                )
                logger.add_scalar(
                    "train/grad_norm",
                    _tensor_to_float(grad_norm),
                    training.global_step,
                )
                logger.flush()

            if training.global_step % validation_frequency == 0 or last_step:
                valid_loss = _evaluate_loss(
                    model,
                    valid_iter,
                    device=training.device,
                    amp_dtype=training.amp_dtype,
                    pad_token_id=training.valid_data.pad_token_id,
                    num_batches=validation_batches,
                )
                logger.add_scalar("valid/loss", valid_loss, training.global_step)
                logger.flush()

            if training.global_step % checkpoint_frequency == 0 or last_step:
                step_file = checkpoints_dir / f"step_{training.global_step:08d}.pt"
                save_checkpoint(training, step_file)
                save_checkpoint(training, checkpoints_dir / "latest.pt")
    finally:
        save_checkpoint(training, checkpoints_dir / "latest.pt")
        logger.close()

    return training


DataIterator = Iterator[tuple[torch.Tensor, torch.Tensor]]


def _evaluate_loss(
    model: torch.nn.Module,
    dataloader_iter: DataIterator,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    pad_token_id: int,
    num_batches: int,
) -> float:
    losses: list[float] = []
    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            for _ in range(num_batches):
                inputs, targets = _get_batch(dataloader_iter, device)
                with _autocast_context(device, amp_dtype):
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


def _setup_dir(run_dir: Path | str):
    run_dir = Path(run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    return checkpoints_dir, tensorboard_dir


def _get_batch(it: DataIterator, device: torch.device):
    non_blocking = device.type == "cuda"
    inputs, targets = next(it)
    inputs = inputs.to(device, non_blocking=non_blocking)
    targets = targets.to(device, non_blocking=non_blocking)
    return inputs, targets


def _unscaled_step(training: Training, loss: torch.Tensor, model: nn.Module):
    """standard torch grad + optimizer step WITHOUT gradient scaling"""
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    training.optimizer.step()
    training.scheduler.step()

    training.optimizer.zero_grad()
    return grad_norm


def _scaled_step(training: Training, loss: torch.Tensor, model: nn.Module):
    """torch grad + optimizer step WITH gradient scaling"""
    training.scaler.scale(loss).backward()
    training.scaler.unscale_(training.optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    scale_before_step = training.scaler.get_scale()
    training.scaler.step(training.optimizer)
    training.scaler.update()

    if training.scaler.get_scale() >= scale_before_step:
        training.scheduler.step()

    training.optimizer.zero_grad()
    return grad_norm


def _autocast_context(
    device: torch.device,
    amp_dtype: torch.dtype | None,
):
    if amp_dtype is None:
        return nullcontext()
    return torch.amp.autocast_mode.autocast(device_type=device.type, dtype=amp_dtype)


def _tensor_to_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)
