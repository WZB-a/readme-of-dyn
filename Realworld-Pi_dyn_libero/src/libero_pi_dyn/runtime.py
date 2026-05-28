from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from libero_pi_dyn.config import ATGConfig
from libero_pi_dyn.factory import make_correction_head
from libero_pi_dyn.factory import make_predictor
from libero_pi_dyn.features import numpy_action_summary
from libero_pi_dyn.features import pad_or_trim
from libero_pi_dyn.safety import ResidualSafetyFilter
from libero_pi_dyn.train_utils import load_state
from libero_pi_dyn.train_utils import resolve_device


class BasicRuntimeFeatureBuilder:
    def __init__(self, config: ATGConfig):
        self.config = config

    def __call__(
        self,
        obs_t3: dict[str, Any] | None,
        robot_state_t3: np.ndarray,
        base_chunk: np.ndarray,
        k: int,
        time_to_exec: float,
        prompt: str = "",
    ) -> dict[str, np.ndarray]:
        del obs_t3, prompt
        state = pad_or_trim(robot_state_t3, self.config.model.state_dim)
        chunk = np.asarray(base_chunk, dtype=np.float32)
        base_action = pad_or_trim(chunk[int(np.clip(k, 0, chunk.shape[0] - 1))], self.config.model.action_dim)
        object_feature = np.zeros((self.config.model.d_obj,), dtype=np.float32)
        object_feature[: min(object_feature.shape[0], state.shape[0])] = state[: min(object_feature.shape[0], state.shape[0])]
        return {
            "object_feature_current": object_feature,
            "robot_state": state,
            "flow_feature": np.zeros((self.config.model.d_flow,), dtype=np.float32),
            "base_action": base_action,
            "action_summary": numpy_action_summary(chunk, k, self.config.model.d_action_summary),
            "chunk_index": np.array(float(k), dtype=np.float32),
            "time_to_exec": np.array(float(time_to_exec), dtype=np.float32),
            "task_id": np.array(0, dtype=np.int64),
        }


class RealtimeATGCorrector:
    def __init__(
        self,
        *,
        config: ATGConfig,
        predictor_ckpt: str | Path,
        correction_ckpt: str | Path,
        safety_stats: str | Path | None = None,
        device: str = "cuda",
        feature_builder: BasicRuntimeFeatureBuilder | None = None,
    ):
        self.config = config
        self.device = resolve_device(device)
        self.predictor = make_predictor(config).to(self.device).eval()
        self.correction_head = make_correction_head(config).to(self.device).eval()
        load_state(predictor_ckpt, self.predictor, self.device)
        load_state(correction_ckpt, self.correction_head, self.device)
        self.safety = (
            ResidualSafetyFilter.from_stats(safety_stats, config.safety.stats_multiplier)
            if safety_stats is not None
            else ResidualSafetyFilter(config.safety.per_dim_limit, config.safety.norm_limit)
        )
        self.feature_builder = feature_builder or BasicRuntimeFeatureBuilder(config)

    @torch.inference_mode()
    def correct(
        self,
        obs_t3: dict[str, Any] | None,
        robot_state_t3: np.ndarray,
        base_chunk: np.ndarray,
        k: int,
        time_to_exec: float,
        prompt: str = "",
    ) -> tuple[np.ndarray, dict[str, Any]]:
        chunk = np.asarray(base_chunk, dtype=np.float32)
        base_action = pad_or_trim(chunk[int(np.clip(k, 0, chunk.shape[0] - 1))], self.config.model.action_dim)
        features = self.feature_builder(obs_t3, robot_state_t3, chunk, k, time_to_exec, prompt)
        batch = {key: torch.as_tensor(value, device=self.device).unsqueeze(0) for key, value in features.items()}
        pred = self.predictor(
            object_feature_current=batch["object_feature_current"],
            robot_state_current=batch["robot_state"],
            flow_feature=batch["flow_feature"],
            base_action=batch["base_action"],
            action_summary=batch["action_summary"],
            chunk_index=batch["chunk_index"],
            time_to_exec=batch["time_to_exec"],
            task_id=batch["task_id"],
            base_chunk=torch.as_tensor(chunk, device=self.device).unsqueeze(0),
        )
        corr = self.correction_head(
            h_hat_tau=pred["h_hat_tau"],
            base_action=batch["base_action"],
            robot_state=batch["robot_state"],
            action_summary=batch["action_summary"],
            chunk_index=batch["chunk_index"],
            time_to_exec=batch["time_to_exec"],
            task_id=batch["task_id"],
        )
        delta = corr["delta_action"].squeeze(0).detach().cpu().numpy()
        delta_safe, info = self.safety.filter(delta, fallback_on_nan=self.config.safety.fallback_on_nan)
        return base_action + delta_safe, {"delta_raw": delta, "delta_safe": delta_safe, "safety": info.__dict__}
