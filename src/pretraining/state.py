from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import grad_scaler
from torchdata.stateful_dataloader import StatefulDataLoader

from .. import model as llm_model
from ..dataset.tiny_stories import TinyStoriesIterableDataset


@dataclass
class DataPipeline:
    config: dict[str, Any]
    loader: StatefulDataLoader
    kind: str = "tiny_stories"

    @property
    def pad_token_id(self) -> int:
        return int(self.config["dataset"]["pad_token_id"])

    @classmethod
    def initialize(
        cls,
        config: dict[str, Any],
    ) -> DataPipeline:
        config = dict(config)
        loader = _build_dataloader(config)
        return cls(
            config=config,
            loader=loader,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "config": copy.deepcopy(self.config),
            "loader_state_dict": copy.deepcopy(self.loader.state_dict()),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        saved_kind = state_dict.get("kind", "tiny_stories")
        saved_config = copy.deepcopy(state_dict["config"])
        if saved_kind != self.kind:
            raise ValueError(
                f"data pipeline kind mismatch: {saved_kind!r} != {self.kind!r}"
            )
        if saved_config != self.config:
            raise ValueError("data pipeline config mismatch")

        self.loader = _build_dataloader(self.config)
        loader_state_dict = state_dict.get("loader_state_dict", {})
        if loader_state_dict:
            self.loader.load_state_dict(loader_state_dict)


@dataclass
class Training:
    train_data: DataPipeline
    valid_data: DataPipeline
    model_config: dict[str, Any]
    optimizer_kind: str
    optimizer_config: dict[str, Any]
    scheduler_kind: str
    scheduler_config: dict[str, Any]
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: grad_scaler.GradScaler
    device: torch.device
    amp_dtype: torch.dtype | None
    global_step: int = 0

    @classmethod
    def initialize(
        cls,
        *,
        train_data_config: dict[str, Any],
        valid_data_config: dict[str, Any],
        model_config: dict[str, Any],
        optimizer_config: dict[str, Any] | None = None,
        scheduler_config: dict[str, Any] | None = None,
        optimizer_kind: str = "adamw",
        scheduler_kind: str = "constant",
        device: str | None = None,
        precision: str | None = "auto",
    ) -> Training:
        resolved_device = _resolve_device(device)
        amp_dtype = _resolve_amp_dtype(resolved_device, precision)

        train_data = DataPipeline.initialize(train_data_config)
        valid_data = DataPipeline.initialize(valid_data_config)

        model_config = dict(model_config)
        optimizer_config = dict(optimizer_config or {})
        scheduler_config = dict(scheduler_config or {})

        model = build_model(model_config, device=resolved_device)
        optimizer = _build_optimizer(model, optimizer_kind, optimizer_config)
        scheduler = _build_scheduler(optimizer, scheduler_kind, scheduler_config)
        scaler = _build_grad_scaler(device=resolved_device, amp_dtype=amp_dtype)

        return cls(
            train_data=train_data,
            valid_data=valid_data,
            model_config=model_config,
            optimizer_kind=optimizer_kind,
            optimizer_config=optimizer_config,
            scheduler_kind=scheduler_kind,
            scheduler_config=scheduler_config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=resolved_device,
            amp_dtype=amp_dtype,
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, Any],
        *,
        device: str | None = None,
        precision: str | None = "auto",
    ) -> Training:
        training = cls.initialize(
            train_data_config=state_dict["train_data"]["config"],
            valid_data_config=state_dict["valid_data"]["config"],
            model_config=state_dict["model"]["config"],
            optimizer_kind=state_dict["optimizer"]["kind"],
            optimizer_config=state_dict["optimizer"]["config"],
            scheduler_kind=state_dict["scheduler"]["kind"],
            scheduler_config=state_dict["scheduler"]["config"],
            device=device,
            precision=precision,
        )
        training.load_state_dict(state_dict)
        return training

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "global_step": int(self.global_step),
            "train_data": self.train_data.state_dict(),
            "valid_data": self.valid_data.state_dict(),
            "model": {
                "config": copy.deepcopy(self.model_config),
                "state_dict": _clone_to_cpu(self.model.state_dict()),
            },
            "optimizer": {
                "kind": self.optimizer_kind,
                "config": copy.deepcopy(self.optimizer_config),
                "state_dict": _clone_to_cpu(self.optimizer.state_dict()),
            },
            "scheduler": {
                "kind": self.scheduler_kind,
                "config": copy.deepcopy(self.scheduler_config),
                "state_dict": copy.deepcopy(self.scheduler.state_dict()),
            },
            "scaler": {
                "state_dict": (
                    copy.deepcopy(self.scaler.state_dict())
                    if self.scaler.is_enabled()
                    else {}
                )
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._validate_state_dict(state_dict)

        self.global_step = int(state_dict["global_step"])
        self.train_data.load_state_dict(state_dict["train_data"])
        self.valid_data.load_state_dict(state_dict["valid_data"])
        self.model.load_state_dict(state_dict["model"]["state_dict"])
        self.optimizer.load_state_dict(state_dict["optimizer"]["state_dict"])
        _move_optimizer_state_to_device(self.optimizer, self.device)
        self.scheduler.load_state_dict(state_dict["scheduler"]["state_dict"])

        scaler_state_dict = state_dict.get("scaler", {}).get("state_dict", {})
        if scaler_state_dict and self.scaler.is_enabled():
            self.scaler.load_state_dict(scaler_state_dict)

    def _validate_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict["model"]["config"] != self.model_config:
            raise ValueError("model config mismatch")
        if state_dict["optimizer"]["kind"] != self.optimizer_kind:
            raise ValueError("optimizer kind mismatch")
        if state_dict["optimizer"]["config"] != self.optimizer_config:
            raise ValueError("optimizer config mismatch")
        if state_dict["scheduler"]["kind"] != self.scheduler_kind:
            raise ValueError("scheduler kind mismatch")
        if state_dict["scheduler"]["config"] != self.scheduler_config:
            raise ValueError("scheduler config mismatch")


def initialize(**kwargs: Any) -> Training:
    return Training.initialize(**kwargs)


def save_checkpoint(training: Training, file: Path | str) -> None:
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(training.state_dict(), file)


def load_checkpoint(
    file: Path | str,
    *,
    device: str | None = None,
    precision: str | None = "auto",
) -> Training:
    payload = torch.load(file, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"expected Training checkpoint dict, got {type(payload)!r}")
    return Training.from_state_dict(payload, device=device, precision=precision)


def build_model(
    model_config: dict[str, Any],
    *,
    device: torch.device,
):
    config = dict(model_config)
    config["device"] = device
    return llm_model.GPT(**config)


def _build_dataloader(config: dict[str, Any]) -> StatefulDataLoader:
    return StatefulDataLoader(
        TinyStoriesIterableDataset(**config["dataset"]),
        **config["loader"],
    )


def _build_optimizer(
    model: nn.Module,
    optimizer_kind: str,
    optimizer_config: dict[str, Any],
) -> torch.optim.Optimizer:
    kind = optimizer_kind.lower()
    if kind != "adamw":
        raise ValueError(f"unsupported optimizer kind: {optimizer_kind}")

    config: dict[str, Any] = {"lr": 3e-4}
    config.update(optimizer_config)
    return torch.optim.AdamW(model.parameters(), **config)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_kind: str,
    scheduler_config: dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler:
    kind = scheduler_kind.lower()
    config = dict(scheduler_config)

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

    raise ValueError(f"unsupported scheduler kind: {scheduler_kind}")


def _build_grad_scaler(
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> grad_scaler.GradScaler:
    return grad_scaler.GradScaler(
        device=device.type,
        enabled=(device.type == "cuda" and amp_dtype == torch.float16),
    )


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_amp_dtype(
    device: torch.device,
    precision: str | None,
) -> torch.dtype | None:
    if precision is None or precision == "float32":
        return None
    if precision == "auto":
        if device.type != "cuda":
            return None
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    try:
        amp_dtype = mapping[precision]
    except KeyError as e:
        available = ", ".join(["auto", "float32", *mapping])
        raise ValueError(
            f"unsupported precision {precision!r}. Available: {available}"
        ) from e

    if device.type == "cpu":
        raise ValueError("AMP precision overrides are only supported on accelerators")
    return amp_dtype


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for param_state in optimizer.state.values():
        for key, value in list(param_state.items()):
            param_state[key] = _move_tensors_to_device(value, device)


def _move_tensors_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_tensors_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_tensors_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_device(v, device) for v in value)
    return value


def _clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _clone_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(v) for v in value)
    return copy.deepcopy(value)


def _default_num_workers() -> int:
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 1:
        return 0
    return min(4, cpu_count)
