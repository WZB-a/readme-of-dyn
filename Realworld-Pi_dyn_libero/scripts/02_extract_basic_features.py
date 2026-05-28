from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

import _bootstrap  # noqa: F401
from libero_pi_dyn.config import ATGConfig
from libero_pi_dyn.data import NpzCache
from libero_pi_dyn.data import raw_episode_paths
from libero_pi_dyn.features import basic_episode_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()
    config = ATGConfig.from_file(args.config)
    splits = [args.split] if args.split else [config.data.train_split, config.data.val_split]
    root = Path(config.data.cache_root)
    for split in splits:
        out_dir = root / "features" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for raw_path in tqdm(raw_episode_paths(config, split), desc=f"features {split}"):
            raw = NpzCache(raw_path)
            features = basic_episode_features(
                state=raw.get_sequence("state"),
                actions=raw.get_sequence("expert_action") if raw.has("expert_action") else raw.get_sequence("actions"),
                d_h_tau=config.model.d_h_tau,
                d_obj=config.model.d_obj,
                d_flow=config.model.d_flow,
                state_dim=config.model.state_dim,
                action_dim=config.model.action_dim,
            )
            np.savez(out_dir / f"{raw_path.stem}.npz", **features)


if __name__ == "__main__":
    main()
