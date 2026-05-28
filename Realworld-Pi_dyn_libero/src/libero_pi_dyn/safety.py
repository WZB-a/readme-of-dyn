from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np


@dataclasses.dataclass
class SafetyInfo:
    used_fallback: bool
    clipped_per_dim: bool
    clipped_norm: bool
    reason: str = ""


class ResidualSafetyFilter:
    def __init__(self, per_dim_limit: np.ndarray | float | None = None, norm_limit: float | None = None):
        self.per_dim_limit = None if per_dim_limit is None else np.asarray(per_dim_limit, dtype=np.float32)
        self.norm_limit = None if norm_limit is None else float(norm_limit)

    @classmethod
    def from_stats(cls, path: str | Path, multiplier: float = 1.5) -> "ResidualSafetyFilter":
        data = np.load(path, allow_pickle=False)
        return cls(
            per_dim_limit=np.asarray(data["abs_target_delta_p95"], dtype=np.float32) * float(multiplier),
            norm_limit=float(data["target_delta_norm_p95"]) * float(multiplier),
        )

    def filter(self, delta: np.ndarray, *, fallback_on_nan: bool = True) -> tuple[np.ndarray, SafetyInfo]:
        delta = np.asarray(delta, dtype=np.float32)
        if fallback_on_nan and not np.all(np.isfinite(delta)):
            return np.zeros_like(delta), SafetyInfo(True, False, False, "non_finite_delta")
        out = delta.copy()
        clipped_per_dim = False
        if self.per_dim_limit is not None:
            limit = np.broadcast_to(self.per_dim_limit, out.shape)
            before = out.copy()
            out = np.clip(out, -limit, limit)
            clipped_per_dim = bool(np.any(before != out))
        clipped_norm = False
        if self.norm_limit is not None:
            norm = float(np.linalg.norm(out))
            if norm > self.norm_limit and norm > 1.0e-8:
                out *= self.norm_limit / norm
                clipped_norm = True
        return out, SafetyInfo(False, clipped_per_dim, clipped_norm)


def compute_delta_stats(base_actions: np.ndarray, expert_actions: np.ndarray, out_path: str | Path) -> None:
    target_delta = np.asarray(expert_actions, dtype=np.float32) - np.asarray(base_actions, dtype=np.float32)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        abs_target_delta_p95=np.percentile(np.abs(target_delta), 95, axis=0).astype(np.float32),
        target_delta_norm_p95=np.float32(np.percentile(np.linalg.norm(target_delta, axis=-1), 95)),
    )
