from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import _bootstrap  # noqa: F401
from libero_pi_dyn.config import ATGConfig
from libero_pi_dyn.data import ATGWindowDataset
from libero_pi_dyn.factory import make_predictor
from libero_pi_dyn.train_utils import load_state
from libero_pi_dyn.train_utils import model_forward
from libero_pi_dyn.train_utils import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()
    config = ATGConfig.from_file(args.config)
    device = resolve_device(config.train.device)
    model = make_predictor(config).to(device).eval()
    load_state(args.ckpt, model, device)
    splits = [args.split] if args.split else [config.data.train_split, config.data.val_split]
    root = Path(config.data.cache_root)
    for split in splits:
        ds = ATGWindowDataset(config, split, require_teacher=True)
        per_episode = defaultdict(list)
        per_episode_idx = defaultdict(list)
        with torch.inference_mode():
            for idx in tqdm(range(len(ds)), desc=f"h_hat_tau {split}"):
                batch = {k: v.unsqueeze(0).to(device) for k, v in ds[idx].items()}
                out = model_forward(model, batch)
                row = ds.rows[idx]
                per_episode[str(row["episode_id"])].append(out["h_hat_tau"].squeeze(0).cpu().numpy())
                per_episode_idx[str(row["episode_id"])].append(int(row["pair_index_in_episode"]))
        out_dir = root / "h_hat_tau" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for episode_id, values in per_episode.items():
            order = np.argsort(np.asarray(per_episode_idx[episode_id]))
            np.savez(out_dir / f"{episode_id}.npz", h_hat_tau=np.stack(values, axis=0)[order], sample_aligned=np.array(True))


if __name__ == "__main__":
    main()
