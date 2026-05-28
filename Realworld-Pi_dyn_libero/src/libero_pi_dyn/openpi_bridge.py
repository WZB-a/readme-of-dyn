from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from libero_pi_dyn.config import ATGConfig
from libero_pi_dyn.data import NpzCache


def add_openpi_to_path(openpi_root: str | Path) -> Path:
    root = Path(openpi_root).expanduser().resolve()
    src = root / "src"
    client_src = root / "packages" / "openpi-client" / "src"
    for path in (str(src), str(client_src), str(root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return root


def create_openpi_policy(config: ATGConfig):
    base = config.resolved_base()
    add_openpi_to_path(base.openpi_root)
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    train_config = openpi_config.get_config(base.openpi_config_name)
    return policy_config.create_trained_policy(train_config, base.checkpoint_dir)


def raw_to_openpi_observation(raw: NpzCache, t: int, config: ATGConfig) -> dict[str, Any]:
    obs: dict[str, Any] = {}
    if "image" in raw.data:
        image = raw.data["image"]
        obs["observation/image"] = image[int(np.clip(t, 0, image.shape[0] - 1))]
    if "wrist_image" in raw.data:
        wrist = raw.data["wrist_image"]
        obs["observation/wrist_image"] = wrist[int(np.clip(t, 0, wrist.shape[0] - 1))]
    if "front_view" in raw.data:
        front = raw.data["front_view"]
        frame = front[int(np.clip(t, 0, front.shape[0] - 1))]
        obs.setdefault("observation/image", frame)
        obs.setdefault("observation/front_image", frame)
    if "wrist_view" in raw.data:
        wrist = raw.data["wrist_view"]
        obs.setdefault("observation/wrist_image", wrist[int(np.clip(t, 0, wrist.shape[0] - 1))])
    if "images" in raw.data and "observation/front_image" not in obs:
        images = raw.data["images"]
        frame = images[int(np.clip(t, 0, images.shape[0] - 1))]
        if frame.ndim == 4:
            obs["observation/image"] = frame[0]
            obs["observation/front_image"] = frame[0]
            obs["observation/wrist_image"] = frame[min(1, frame.shape[0] - 1)]
        else:
            obs["observation/image"] = frame
            obs["observation/front_image"] = frame
            obs["observation/wrist_image"] = frame
    state = raw.get("state", t)
    obs["observation/state"] = state
    obs["state"] = state
    obs["prompt"] = raw.get_prompt(config.base.prompt_key)
    return obs


def infer_action_chunk(policy, obs: dict[str, Any], action_key: str) -> np.ndarray:
    result = policy.infer(obs)
    if action_key not in result:
        keys = ", ".join(sorted(result.keys()))
        raise KeyError(f"Policy output missing {action_key!r}; available keys: {keys}")
    actions = np.asarray(result[action_key], dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected action chunk [H, A], got {actions.shape}")
    return actions
