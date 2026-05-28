from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

import _bootstrap  # noqa: F401
from realworld_pi_dyn.config import ATGConfig
from realworld_pi_dyn.data import NpzCache
from realworld_pi_dyn.data import raw_episode_paths
from realworld_pi_dyn.openpi_bridge import create_openpi_policy
from realworld_pi_dyn.openpi_bridge import infer_action_chunk
from realworld_pi_dyn.openpi_bridge import raw_to_openpi_observation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()
    config = ATGConfig.from_file(args.config)
    policy = create_openpi_policy(config)
    splits = [args.split] if args.split else [config.data.train_split, config.data.val_split]
    root = Path(config.data.cache_root)
    for split in splits:
        out_dir = root / "pi05_chunks" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for raw_path in tqdm(raw_episode_paths(config, split), desc=f"pi05 chunks {split}"):
            raw = NpzCache(raw_path)
            states = raw.get_sequence("state")
            chunks = []
            available_t2 = []
            for t in range(states.shape[0]):
                obs = raw_to_openpi_observation(raw, t, config)
                chunk = infer_action_chunk(policy, obs, config.base.action_output_key)
                chunks.append(chunk.astype(np.float32))
                available_t2.append(t + config.data.latency_steps)
            np.savez(out_dir / f"{raw_path.stem}.npz", base_chunk=np.stack(chunks, axis=0), available_t2=np.asarray(available_t2, dtype=np.int32))


if __name__ == "__main__":
    main()
