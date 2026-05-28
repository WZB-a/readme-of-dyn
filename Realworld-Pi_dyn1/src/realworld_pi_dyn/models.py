from __future__ import annotations

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from realworld_pi_dyn.features import sinusoidal_scalar


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(max(1, num_layers - 1)):
        layers += [nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, dim: int, commitment_cost: float):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.dim = int(dim)
        self.commitment_cost = float(commitment_cost)
        self.embedding = nn.Embedding(self.codebook_size, self.dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z: Tensor) -> dict[str, Tensor]:
        flat = z.reshape(-1, self.dim)
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1).unsqueeze(0)
        )
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view_as(z)
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(z, quantized.detach())
        quantized_st = z + (quantized - z).detach()
        return {
            "quantized": quantized_st,
            "indices": indices.view(*z.shape[:-1]),
            "vq_loss": codebook_loss + self.commitment_cost * commitment_loss,
        }


class DynamicTokenizer(nn.Module):
    def __init__(
        self,
        *,
        d_h_tau: int,
        state_dim: int,
        action_dim: int,
        d_action_summary: int,
        d_model: int,
        codebook_size: int,
        num_latent_tokens: int,
        num_ego_tokens: int,
        dropout: float,
        lambda_vq: float,
    ):
        super().__init__()
        self.num_latent_tokens = int(num_latent_tokens)
        self.num_ego_tokens = int(num_ego_tokens)
        if self.num_ego_tokens <= 0 or self.num_ego_tokens >= self.num_latent_tokens:
            raise ValueError("num_ego_tokens must be in [1, num_latent_tokens - 1]")
        self.h_s_proj = nn.Linear(d_h_tau, d_model)
        self.h_tau_proj = nn.Linear(d_h_tau, d_model)
        self.delta_h_proj = nn.Linear(d_h_tau, d_model)
        self.q_s_proj = nn.Linear(state_dim, d_model)
        self.q_tau_proj = nn.Linear(state_dim, d_model)
        self.delta_q_proj = nn.Linear(state_dim, d_model)
        self.base_action_proj = nn.Linear(action_dim, d_model)
        self.action_summary_proj = nn.Linear(d_action_summary, d_model)
        self.time_proj = nn.Linear(18, d_model)
        self.dynamics_queries = nn.Parameter(torch.randn(1, self.num_latent_tokens, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model, 4, d_model * 4, dropout, batch_first=True, activation="gelu")
        self.transition_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.query_attn = nn.MultiheadAttention(d_model, 4, dropout=dropout, batch_first=True)
        self.to_code = nn.Linear(d_model, d_model)
        self.vq = VectorQuantizer(codebook_size, d_model, lambda_vq)
        self.decoder = _mlp(d_h_tau + d_model * 2 + state_dim + action_dim + d_action_summary + 18, d_model, d_h_tau, 3, dropout)
        self.robot_head = _mlp(d_model, d_model, state_dim, 2, dropout)

    def forward(
        self,
        h_s: Tensor,
        h_tau_raw: Tensor,
        robot_state: Tensor,
        base_action: Tensor,
        action_summary: Tensor,
        chunk_index: Tensor,
        time_to_exec: Tensor,
        robot_state_tau: Tensor | None = None,
    ) -> dict[str, Tensor]:
        delta_h = h_tau_raw - h_s
        if robot_state_tau is None:
            robot_state_tau = robot_state
        delta_q = robot_state_tau - robot_state
        time_feature = torch.cat([sinusoidal_scalar(time_to_exec, 16), time_to_exec.reshape(-1, 1), chunk_index.float().reshape(-1, 1)], dim=-1)
        transition_tokens = torch.stack(
            [
                self.h_s_proj(h_s),
                self.h_tau_proj(h_tau_raw),
                self.delta_h_proj(delta_h),
                self.q_s_proj(robot_state),
                self.q_tau_proj(robot_state_tau),
                self.delta_q_proj(delta_q),
                self.base_action_proj(base_action),
                self.action_summary_proj(action_summary),
                self.time_proj(time_feature),
            ],
            dim=1,
        )
        encoded = self.transition_encoder(transition_tokens)
        queries = self.dynamics_queries.expand(h_s.shape[0], -1, -1)
        e_all, _ = self.query_attn(queries, encoded, encoded, need_weights=False)
        vq = self.vq(self.to_code(e_all))
        z_all = vq["quantized"]
        z_ego_tokens = z_all[:, : self.num_ego_tokens]
        z_env_tokens = z_all[:, self.num_ego_tokens :]
        z_ego = z_ego_tokens.mean(dim=1)
        z_env = z_env_tokens.mean(dim=1)
        h_tau_teacher = self.decoder(torch.cat([h_s, z_ego, z_env, robot_state, base_action, action_summary, time_feature], dim=-1))
        delta_q_hat = self.robot_head(z_ego)
        return {
            "h_tau_teacher": h_tau_teacher,
            "z_ego": z_ego,
            "z_env": z_env,
            "z_indices": vq["indices"],
            "delta_q_hat": delta_q_hat,
            "delta_q_target": delta_q.detach(),
            "recon_loss": 1.0 - F.cosine_similarity(h_tau_teacher, h_tau_raw, dim=-1).mean() + 0.05 * F.mse_loss(h_tau_teacher, h_tau_raw),
            "robot_loss": F.smooth_l1_loss(delta_q_hat, delta_q.detach()),
            "vq_loss": vq["vq_loss"],
        }


class PumaLitePredictor(nn.Module):
    def __init__(
        self,
        *,
        d_obj: int,
        state_dim: int,
        d_flow: int,
        action_dim: int,
        d_action_summary: int,
        d_h_tau: int,
        d_model: int,
        d_pi05: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        num_tasks: int,
        task_embedding_dim: int,
    ):
        super().__init__()
        self.task_embedding = nn.Embedding(num_tasks, task_embedding_dim)
        self.d_model = d_model
        self.object_proj = nn.Linear(d_obj, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.flow_proj = nn.Linear(d_flow, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.chunk_action_proj = nn.Linear(action_dim, d_model)
        self.action_summary_proj = nn.Linear(d_action_summary, d_model)
        self.time_proj = nn.Linear(34, d_model)
        self.task_proj = nn.Linear(task_embedding_dim, d_model)
        self.pi05_proj = nn.Linear(d_pi05, d_model) if d_pi05 > 0 else None
        self.future_query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.world_queries = nn.Parameter(torch.zeros(1, 4, d_model))
        layer = nn.TransformerEncoderLayer(d_model, num_heads, d_model * 4, dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Sequential(nn.LayerNorm(d_model), _mlp(d_model, d_model, d_h_tau, 2, dropout))

    def forward(
        self,
        object_feature_current: Tensor,
        robot_state_current: Tensor,
        flow_feature: Tensor,
        base_action: Tensor,
        action_summary: Tensor,
        chunk_index: Tensor,
        time_to_exec: Tensor,
        task_id: Tensor,
        base_chunk: Tensor | None = None,
        pi05_feature: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch = base_action.shape[0]
        time_feature = torch.cat(
            [
                sinusoidal_scalar(chunk_index, 16),
                sinusoidal_scalar(time_to_exec, 16),
                time_to_exec.reshape(-1, 1),
                chunk_index.float().reshape(-1, 1),
            ],
            dim=-1,
        )
        tokens = [
            self.future_query.expand(batch, -1, -1),
            self.world_queries.expand(batch, -1, -1),
            self.object_proj(object_feature_current).unsqueeze(1),
            self.state_proj(robot_state_current).unsqueeze(1),
            self.flow_proj(flow_feature).unsqueeze(1),
            self.action_proj(base_action).unsqueeze(1),
            self.action_summary_proj(action_summary).unsqueeze(1),
            self.time_proj(time_feature).unsqueeze(1),
            self.task_proj(self.task_embedding(task_id.long().clamp_min(0) % self.task_embedding.num_embeddings)).unsqueeze(1),
        ]
        if base_chunk is not None:
            tokens.append(self.chunk_action_proj(base_chunk).contiguous())
        if self.pi05_proj is not None and pi05_feature is not None:
            tokens.append(self.pi05_proj(pi05_feature).unsqueeze(1))
        h = self.encoder(torch.cat(tokens, dim=1))[:, 0]
        return {"h_hat_tau": self.out(h)}


class CorrectionHead(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        state_dim: int,
        d_h_tau: int,
        d_action_summary: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_tasks: int,
        task_embedding_dim: int,
    ):
        super().__init__()
        self.task_embedding = nn.Embedding(num_tasks, task_embedding_dim)
        self.net = _mlp(d_h_tau + action_dim + state_dim + d_action_summary + 34 + task_embedding_dim, hidden_dim, action_dim, num_layers, dropout)

    def forward(
        self,
        h_hat_tau: Tensor,
        base_action: Tensor,
        robot_state: Tensor,
        action_summary: Tensor,
        chunk_index: Tensor,
        time_to_exec: Tensor,
        task_id: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if task_id is None:
            task_id = torch.zeros(base_action.shape[0], dtype=torch.long, device=base_action.device)
        time_feature = torch.cat(
            [
                sinusoidal_scalar(chunk_index, 16),
                sinusoidal_scalar(time_to_exec, 16),
                time_to_exec.reshape(-1, 1),
                chunk_index.float().reshape(-1, 1),
            ],
            dim=-1,
        )
        task = self.task_embedding(task_id.long().clamp_min(0) % self.task_embedding.num_embeddings)
        return {"delta_action": self.net(torch.cat([h_hat_tau, base_action, robot_state, action_summary, time_feature, task], dim=-1))}


def predictor_loss(outputs: dict[str, Tensor], batch: dict[str, Tensor], lambda_delta: float, lambda_l2: float) -> Tensor:
    pred = outputs["h_hat_tau"]
    target = batch["h_tau_teacher"]
    loss = F.smooth_l1_loss(pred, target)
    loss = loss + lambda_delta * F.smooth_l1_loss(pred - batch["h_s"], target - batch["h_s"])
    if lambda_l2 > 0:
        loss = loss + lambda_l2 * pred.pow(2).mean()
    return loss


def correction_loss(outputs: dict[str, Tensor], batch: dict[str, Tensor], lambda_residual_l2: float) -> Tensor:
    target_delta = batch["expert_action"] - batch["base_action"]
    delta = outputs["delta_action"]
    return F.smooth_l1_loss(delta, target_delta) + lambda_residual_l2 * delta.pow(2).mean()
