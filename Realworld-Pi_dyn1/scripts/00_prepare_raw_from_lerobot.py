from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from tqdm import tqdm

import _bootstrap  # noqa: F401
from realworld_pi_dyn.config import ATGConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--with-video", action="store_true", help="Also cache front_view/wrist_view frames for pi0.5 inference.")
    args = parser.parse_args()
    config = ATGConfig.from_file(args.config)
    dataset_root = Path(config.resolved_dataset_root())
    episodes = _read_episodes(dataset_root)
    train_n = min(config.data.train_episodes, len(episodes))
    val_n = min(config.data.val_episodes, max(0, len(episodes) - train_n))
    splits = {
        config.data.train_split: episodes[:train_n],
        config.data.val_split: episodes[train_n : train_n + val_n],
    }
    root = Path(config.data.cache_root)
    for split, split_episodes in splits.items():
        out_dir = root / "raw" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for ep in tqdm(split_episodes, desc=f"prepare raw {split}"):
            episode_index = int(ep["episode_index"])
            parquet_path = dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
            table = pl.read_parquet(parquet_path)
            if config.data.max_frames_per_episode is not None:
                table = table.head(int(config.data.max_frames_per_episode))
            payload = {
                "state": np.asarray(table[config.data.raw_state_key].to_list(), dtype=np.float32),
                "actions": np.asarray(table[config.data.raw_action_key].to_list(), dtype=np.float32),
                "timestamp": np.asarray(table["timestamp"].to_numpy(), dtype=np.float32) if "timestamp" in table.columns else np.arange(len(table), dtype=np.float32) * config.data.control_dt,
                "task_id": np.asarray(int(table["task_index"][0]) if "task_index" in table.columns else config.data.task_id_default, dtype=np.int64),
                "prompt": np.asarray((ep.get("tasks") or ["complete the task"])[0]),
            }
            if args.with_video:
                payload["front_view"] = _read_video(dataset_root, "front_view", episode_index, len(table))
                payload["wrist_view"] = _read_video(dataset_root, "wrist_view", episode_index, len(table))
            np.savez_compressed(out_dir / f"episode_{episode_index:06d}.npz", **payload)


def _read_episodes(dataset_root: Path) -> list[dict]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    with episodes_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_video(dataset_root: Path, key: str, episode_index: int, expected_frames: int) -> np.ndarray:
    path = dataset_root / "videos" / "chunk-000" / key / f"episode_{episode_index:06d}.mp4"
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while len(frames) < expected_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"Could not read video frames: {path}")
    while len(frames) < expected_frames:
        frames.append(frames[-1])
    return np.stack(frames[:expected_frames], axis=0).astype(np.uint8)


if __name__ == "__main__":
    main()
