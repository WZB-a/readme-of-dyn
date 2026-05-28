from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from libero_pi_dyn.config import ATGConfig
from libero_pi_dyn.features import numpy_action_summary
from libero_pi_dyn.features import pad_or_trim


ALIASES = {
    "base_chunk": ("base_chunk", "actions", "action"),
    "h_s": ("h_s", "latent", "features"),
    "h_tau_raw": ("h_tau_raw", "future_latent", "h_tau"),
    "h_tau_teacher": ("h_tau_teacher", "teacher"),
    "h_hat_tau": ("h_hat_tau", "h_hat"),
    "object_feature_current": ("object_feature_current", "object_feature", "h_s"),
    "flow_feature": ("flow_feature", "flow"),
    "robot_state": ("robot_state", "state", "state8", "observation.state"),
    "expert_action": ("expert_action", "action", "actions"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def raw_episode_paths(config: ATGConfig, split: str) -> list[Path]:
    return sorted((Path(config.data.cache_root) / "raw" / split).glob("*.npz"))


class NpzCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = np.load(self.path, allow_pickle=True)

    def has(self, key: str) -> bool:
        return any(candidate in self.data for candidate in ALIASES.get(key, (key,)))

    def get(self, key: str, index: int | None = None, dim: int | None = None, default: float = 0.0) -> np.ndarray:
        for candidate in ALIASES.get(key, (key,)):
            if candidate in self.data:
                arr = np.asarray(self.data[candidate], dtype=np.float32)
                if index is not None and arr.ndim >= 2:
                    arr = arr[int(np.clip(index, 0, arr.shape[0] - 1))]
                return pad_or_trim(arr, dim) if dim is not None else arr.astype(np.float32)
        if dim is None:
            raise KeyError(f"{self.path} missing key {key}")
        return np.full((dim,), default, dtype=np.float32)

    def get_sequence(self, key: str) -> np.ndarray:
        for candidate in ALIASES.get(key, (key,)):
            if candidate in self.data:
                return np.asarray(self.data[candidate], dtype=np.float32)
        raise KeyError(f"{self.path} missing sequence {key}")

    def get_prompt(self, key: str, default: str = "complete the task") -> str:
        if key not in self.data:
            return default
        value = self.data[key]
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return str(value.item())
            return str(value.reshape(-1)[0])
        return str(value)


class ATGWindowDataset(Dataset):
    def __init__(self, config: ATGConfig, split: str, *, require_teacher: bool = False, require_h_hat: bool = False):
        self.config = config
        self.split = split
        self.require_teacher = require_teacher
        self.require_h_hat = require_h_hat
        root = Path(config.data.cache_root)
        self.rows = read_jsonl(root / "pair_index" / f"{split}.jsonl")
        self.feature_dir = root / "features" / split
        self.chunk_dir = root / "pi05_chunks" / split
        self.teacher_dir = root / "teacher" / split
        self.h_hat_dir = root / "h_hat_tau" / split

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        episode_id = str(row["episode_id"])
        t1 = int(row.get("t1", row.get("t3", 0)))
        t3 = int(row["t3"])
        t4 = int(row["t4"])
        k = int(row["k"])
        dt = float(row.get("dt", (t4 - t3) * self.config.data.control_dt))

        features = NpzCache(self.feature_dir / f"{episode_id}.npz")
        chunks = NpzCache(self.chunk_dir / f"{episode_id}.npz")
        base_chunk = chunks.get_sequence("base_chunk")
        if base_chunk.ndim == 3:
            base_chunk = base_chunk[int(np.clip(t1, 0, base_chunk.shape[0] - 1))]
        k = int(np.clip(k, 0, base_chunk.shape[0] - 1))
        base_action = pad_or_trim(base_chunk[k], self.config.model.action_dim)
        sample = {
            "object_feature_current": features.get("object_feature_current", t3, self.config.model.d_obj),
            "robot_state": features.get("robot_state", t3, self.config.model.state_dim),
            "robot_state_tau": features.get("robot_state", t4, self.config.model.state_dim),
            "flow_feature": features.get("flow_feature", t3, self.config.model.d_flow),
            "base_chunk": base_chunk.astype(np.float32),
            "base_action": base_action,
            "action_summary": numpy_action_summary(base_chunk, k, self.config.model.d_action_summary),
            "chunk_index": np.array(k, dtype=np.float32),
            "time_to_exec": np.array(dt, dtype=np.float32),
            "task_id": np.array(int(row.get("task_id", self.config.data.task_id_default)), dtype=np.int64),
            "h_s": features.get("h_s", t3, self.config.model.d_h_tau),
            "h_tau_raw": features.get("h_tau_raw", t4, self.config.model.d_h_tau),
            "expert_action": features.get("expert_action", t4, self.config.model.action_dim),
        }
        pair_index_in_episode = int(row.get("pair_index_in_episode", 0))
        if self.require_teacher:
            sample["h_tau_teacher"] = _load_pair_value(
                self.teacher_dir / f"{episode_id}.npz", "h_tau_teacher", pair_index_in_episode, self.config.model.d_h_tau
            )
        if self.require_h_hat:
            sample["h_hat_tau"] = _load_pair_value(
                self.h_hat_dir / f"{episode_id}.npz", "h_hat_tau", pair_index_in_episode, self.config.model.d_h_tau
            )
        return {key: torch.as_tensor(value) for key, value in sample.items()}


def _load_pair_value(path: Path, key: str, idx: int, dim: int) -> np.ndarray:
    cache = NpzCache(path)
    index = idx if "sample_aligned" in cache.data else None
    return cache.get(key, index, dim)


def build_pair_index(config: ATGConfig, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in raw_episode_paths(config, split):
        episode_id = raw_path.stem
        pair_index_in_episode = 0
        raw = NpzCache(raw_path)
        states = raw.get_sequence("robot_state") if raw.has("robot_state") else raw.get_sequence("state")
        horizon = int(states.shape[0])
        task_id = int(raw.data["task_id"].item()) if "task_id" in raw.data and np.asarray(raw.data["task_id"]).shape == () else config.data.task_id_default
        for t3 in range(horizon):
            t1 = max(0, t3 - config.data.latency_steps)
            t2 = t1 + config.data.latency_steps
            for step in config.data.future_steps:
                t4 = t3 + int(step)
                if t4 >= horizon:
                    continue
                k = t4 - t1
                if k not in config.data.chunk_indices:
                    continue
                rows.append(
                    {
                        "domain_id": 0,
                        "task_id": task_id,
                        "episode_id": episode_id,
                        "t1": t1,
                        "t2": t2,
                        "t3": t3,
                        "T": int(step),
                        "t4": t4,
                        "k": k,
                        "dt": float(step) * config.data.control_dt,
                        "pair_index_in_episode": pair_index_in_episode,
                    }
                )
                pair_index_in_episode += 1
    return rows
