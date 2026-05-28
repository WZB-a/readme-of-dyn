from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from realworld_pi_dyn.audit import assert_action_space_ready
from realworld_pi_dyn.config import ATGConfig
from realworld_pi_dyn.data import ATGWindowDataset
from realworld_pi_dyn.factory import make_correction_head
from realworld_pi_dyn.models import correction_loss
from realworld_pi_dyn.safety import compute_delta_stats
from realworld_pi_dyn.train_utils import default_loader
from realworld_pi_dyn.train_utils import resolve_device
from realworld_pi_dyn.train_utils import run_training
from realworld_pi_dyn.train_utils import seed_everything


class CurrentOnlyDataset(ATGWindowDataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample["h_hat_tau"] = sample["h_s"].new_zeros(sample["h_s"].shape)
        return sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="outputs/current_correction_baseline")
    args = parser.parse_args()
    config = ATGConfig.from_file(args.config)
    assert_action_space_ready(config)
    seed_everything(config.train.seed)
    train_ds = CurrentOnlyDataset(config, config.data.train_split)
    val_ds = CurrentOnlyDataset(config, config.data.val_split)
    _write_safety_stats(train_ds, Path(args.out_dir) / "delta_stats.npz")
    model = make_correction_head(config)
    run_training(
        model=model,
        train_loader=default_loader(train_ds, config, True),
        val_loader=default_loader(val_ds, config, False),
        loss_fn=lambda outputs, batch: correction_loss(outputs, batch, config.loss.lambda_residual_l2),
        device=resolve_device(config.train.device),
        config=config.to_dict(),
        epochs=config.train.epochs,
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        grad_clip=config.train.grad_clip,
        early_stop_patience=config.train.early_stop_patience,
        out_dir=args.out_dir,
    )


def _write_safety_stats(dataset: ATGWindowDataset, out_path: Path) -> None:
    base_actions = []
    expert_actions = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        base_actions.append(sample["base_action"].numpy())
        expert_actions.append(sample["expert_action"].numpy())
    compute_delta_stats(np.stack(base_actions, axis=0), np.stack(expert_actions, axis=0), out_path)


if __name__ == "__main__":
    main()
