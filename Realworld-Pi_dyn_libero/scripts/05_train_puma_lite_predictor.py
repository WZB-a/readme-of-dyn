from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from libero_pi_dyn.config import ATGConfig
from libero_pi_dyn.data import ATGWindowDataset
from libero_pi_dyn.factory import make_predictor
from libero_pi_dyn.models import predictor_loss
from libero_pi_dyn.train_utils import default_loader
from libero_pi_dyn.train_utils import resolve_device
from libero_pi_dyn.train_utils import run_training
from libero_pi_dyn.train_utils import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="outputs/predictor")
    args = parser.parse_args()
    config = ATGConfig.from_file(args.config)
    seed_everything(config.train.seed)
    train_ds = ATGWindowDataset(config, config.data.train_split, require_teacher=True)
    val_ds = ATGWindowDataset(config, config.data.val_split, require_teacher=True)
    model = make_predictor(config)
    run_training(
        model=model,
        train_loader=default_loader(train_ds, config, True),
        val_loader=default_loader(val_ds, config, False),
        loss_fn=lambda outputs, batch: predictor_loss(outputs, batch, config.loss.lambda_delta, config.loss.lambda_l2),
        device=resolve_device(config.train.device),
        config=config.to_dict(),
        epochs=config.train.epochs,
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        grad_clip=config.train.grad_clip,
        early_stop_patience=config.train.early_stop_patience,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
