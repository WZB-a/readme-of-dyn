from __future__ import annotations

import math

import numpy as np
import torch


def pad_or_trim(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.shape[0] == dim:
        return x
    if x.shape[0] > dim:
        return x[:dim].copy()
    out = np.zeros((dim,), dtype=np.float32)
    out[: x.shape[0]] = x
    return out


def numpy_action_summary(base_chunk: np.ndarray, k: int, out_dim: int) -> np.ndarray:
    chunk = np.asarray(base_chunk, dtype=np.float32)
    if chunk.ndim != 2:
        raise ValueError(f"base_chunk must be [H, A], got {chunk.shape}")
    k = int(np.clip(k, 0, chunk.shape[0] - 1))
    action_k = chunk[k]
    prefix = chunk[: k + 1].mean(axis=0)
    suffix = chunk[k:].mean(axis=0)
    velocity = np.zeros_like(action_k) if k == 0 else action_k - chunk[k - 1]
    return pad_or_trim(np.concatenate([action_k, prefix, suffix, velocity], axis=0), out_dim)


def sinusoidal_scalar(values: torch.Tensor, dim: int) -> torch.Tensor:
    values = values.reshape(-1, 1).to(dtype=torch.float32)
    if dim <= 0:
        return values.new_zeros((values.shape[0], 0))
    half = max(1, dim // 2)
    freqs = torch.exp(
        torch.arange(half, device=values.device, dtype=values.dtype) * (-math.log(10000.0) / max(1, half - 1))
    )
    angles = values * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[-1]))
    return emb[:, :dim]


def basic_episode_features(
    *,
    state: np.ndarray,
    actions: np.ndarray,
    d_h_tau: int,
    d_obj: int,
    d_flow: int,
    state_dim: int,
    action_dim: int,
) -> dict[str, np.ndarray]:
    state = np.asarray(state, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if state.ndim != 2 or actions.ndim != 2:
        raise ValueError(f"state/actions must be rank-2, got {state.shape=} {actions.shape=}")
    t = min(state.shape[0], actions.shape[0])
    state = state[:t]
    actions = actions[:t]
    robot_state = np.stack([pad_or_trim(s, state_dim) for s in state], axis=0)
    expert_action = np.stack([pad_or_trim(a, action_dim) for a in actions], axis=0)

    object_feature = np.zeros((t, d_obj), dtype=np.float32)
    object_feature[:, : min(d_obj, state_dim)] = robot_state[:, : min(d_obj, state_dim)]
    flow = np.zeros((t, d_flow), dtype=np.float32)
    if t > 1:
        delta_state = np.diff(robot_state, axis=0, prepend=robot_state[:1])
        flow[:, : min(d_flow, state_dim)] = delta_state[:, : min(d_flow, state_dim)]

    latent_input = np.concatenate([robot_state, expert_action, flow[:, : min(flow.shape[1], state_dim)]], axis=-1)
    h = np.stack([pad_or_trim(row, d_h_tau) for row in latent_input], axis=0)
    return {
        "h_s": h,
        "h_tau_raw": h.copy(),
        "object_feature_current": object_feature,
        "flow_feature": flow,
        "robot_state": robot_state,
        "expert_action": expert_action,
    }
