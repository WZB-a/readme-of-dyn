from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

import _bootstrap  # noqa: F401
from libero_pi_dyn.config import ATGConfig


DEFAULT_TASK_FILE = "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--hdf5", default=None, help="LIBERO demo hdf5. Defaults to one libero_spatial task.")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--max-train-episodes", type=int, default=None)
    parser.add_argument("--max-val-episodes", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None, help="Keep stored demo image size unless set.")
    args = parser.parse_args()

    config = ATGConfig.from_file(args.config)
    dataset_root = Path(config.resolved_dataset_root())
    hdf5_path = Path(args.hdf5) if args.hdf5 else dataset_root / DEFAULT_TASK_FILE
    if not hdf5_path.exists():
        raise FileNotFoundError(hdf5_path)

    train_n = args.max_train_episodes if args.max_train_episodes is not None else config.data.train_episodes
    val_n = args.max_val_episodes if args.max_val_episodes is not None else config.data.val_episodes

    root = Path(config.data.cache_root)
    with h5py.File(hdf5_path, "r") as f:
        demos = _sorted_demos(f)
        prompt = _read_prompt(f)
        train_demos = demos[: min(train_n, len(demos))]
        val_demos = demos[len(train_demos) : len(train_demos) + min(val_n, max(0, len(demos) - len(train_demos)))]

        for split, split_demos in ((config.data.train_split, train_demos), (config.data.val_split, val_demos)):
            out_dir = root / "raw" / split
            out_dir.mkdir(parents=True, exist_ok=True)
            for demo_name in tqdm(split_demos, desc=f"prepare LIBERO raw {split}"):
                payload = _episode_payload(
                    f["data"][demo_name],
                    prompt=prompt,
                    task_id=args.task_id,
                    control_dt=config.data.control_dt,
                    max_frames=config.data.max_frames_per_episode,
                )
                np.savez_compressed(out_dir / f"{hdf5_path.stem}_{demo_name}.npz", **payload)


def _sorted_demos(f: h5py.File) -> list[str]:
    demos = list(f["data"].keys())
    return sorted(demos, key=lambda name: int(name.split("_")[-1]))


def _read_prompt(f: h5py.File) -> str:
    attrs = f["data"].attrs
    if "problem_info" in attrs:
        try:
            info = json.loads(attrs["problem_info"])
            if info.get("language_instruction"):
                return str(info["language_instruction"])
        except json.JSONDecodeError:
            pass
    return "complete the LIBERO task"


def _episode_payload(
    demo: h5py.Group,
    *,
    prompt: str,
    task_id: int,
    control_dt: float,
    max_frames: int | None,
) -> dict[str, np.ndarray]:
    obs = demo["obs"]
    horizon = int(demo["actions"].shape[0])
    if max_frames is not None:
        horizon = min(horizon, int(max_frames))

    ee_pos = np.asarray(obs["ee_pos"][:horizon], dtype=np.float32)
    ee_ori = np.asarray(obs["ee_ori"][:horizon], dtype=np.float32)
    gripper = np.asarray(obs["gripper_states"][:horizon], dtype=np.float32)
    state = np.concatenate([ee_pos, ee_ori, gripper], axis=-1)
    actions = np.asarray(demo["actions"][:horizon], dtype=np.float32)

    return {
        "state": state,
        "robot_state": state,
        "actions": actions,
        "expert_action": actions,
        "image": np.asarray(obs["agentview_rgb"][:horizon], dtype=np.uint8),
        "wrist_image": np.asarray(obs["eye_in_hand_rgb"][:horizon], dtype=np.uint8),
        "timestamp": np.arange(horizon, dtype=np.float32) * float(control_dt),
        "task_id": np.asarray(task_id, dtype=np.int64),
        "prompt": np.asarray(prompt),
    }


if __name__ == "__main__":
    main()
