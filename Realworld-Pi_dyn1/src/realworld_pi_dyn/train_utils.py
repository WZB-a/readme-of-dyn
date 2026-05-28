from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def save_checkpoint(path: str | Path, model: torch.nn.Module, config: dict, metrics: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config, "metrics": metrics or {}}, path)


def load_state(path: str | Path, model: torch.nn.Module, map_location: str | torch.device = "cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    return ckpt


def default_loader(dataset, config, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=shuffle,
        num_workers=config.train.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def model_forward(model, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    name = model.__class__.__name__
    if name == "DynamicTokenizer":
        return model(
            h_s=batch["h_s"],
            h_tau_raw=batch["h_tau_raw"],
            robot_state=batch["robot_state"],
            base_action=batch["base_action"],
            action_summary=batch["action_summary"],
            chunk_index=batch["chunk_index"],
            time_to_exec=batch["time_to_exec"],
            robot_state_tau=batch.get("robot_state_tau"),
        )
    if name == "PumaLitePredictor":
        return model(
            object_feature_current=batch["object_feature_current"],
            robot_state_current=batch["robot_state"],
            flow_feature=batch["flow_feature"],
            base_action=batch["base_action"],
            action_summary=batch["action_summary"],
            chunk_index=batch["chunk_index"],
            time_to_exec=batch["time_to_exec"],
            task_id=batch["task_id"],
            base_chunk=batch.get("base_chunk"),
            pi05_feature=batch.get("pi05_feature"),
        )
    if name == "CorrectionHead":
        return model(
            h_hat_tau=batch["h_hat_tau"],
            base_action=batch["base_action"],
            robot_state=batch["robot_state"],
            action_summary=batch["action_summary"],
            chunk_index=batch["chunk_index"],
            time_to_exec=batch["time_to_exec"],
            task_id=batch["task_id"],
        )
    raise TypeError(f"Unsupported model type: {name}")


def run_training(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: Callable[[dict, dict], torch.Tensor],
    device: torch.device,
    config: dict,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    early_stop_patience: int,
    out_dir: str | Path,
) -> dict[str, float | int]:
    model.to(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    history = []
    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_loader, loss_fn, device, optimizer, grad_clip)
        val_loss = _run_epoch(model, val_loader, loss_fn, device, None, grad_clip)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            save_checkpoint(out_dir / "best.pt", model, config, {"best_val": best_val, "best_epoch": best_epoch})
        else:
            stale += 1
            if stale >= early_stop_patience:
                break
    save_checkpoint(out_dir / "last.pt", model, config, {"best_val": best_val, "best_epoch": best_epoch})
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"best_val": best_val, "best_epoch": best_epoch}


def _run_epoch(model, loader, loss_fn, device, optimizer, grad_clip):
    train = optimizer is not None
    model.train(train)
    total = 0.0
    count = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        batch_size = next(iter(batch.values())).shape[0]
        with torch.set_grad_enabled(train):
            outputs = model_forward(model, batch)
            loss = loss_fn(outputs, batch)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total += float(loss.detach().cpu()) * batch_size
        count += batch_size
    return total / max(1, count)
