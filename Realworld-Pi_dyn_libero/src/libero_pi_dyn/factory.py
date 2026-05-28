from __future__ import annotations

from libero_pi_dyn.config import ATGConfig
from libero_pi_dyn.models import CorrectionHead
from libero_pi_dyn.models import DynamicTokenizer
from libero_pi_dyn.models import PumaLitePredictor


def make_tokenizer(config: ATGConfig) -> DynamicTokenizer:
    return DynamicTokenizer(
        d_h_tau=config.model.d_h_tau,
        state_dim=config.model.state_dim,
        action_dim=config.model.action_dim,
        d_action_summary=config.model.d_action_summary,
        d_model=config.model.d_model,
        codebook_size=config.model.codebook_size,
        num_latent_tokens=config.model.num_latent_tokens,
        num_ego_tokens=config.model.num_ego_tokens,
        dropout=config.model.dropout,
        lambda_vq=config.loss.lambda_vq,
    )


def make_predictor(config: ATGConfig) -> PumaLitePredictor:
    return PumaLitePredictor(
        d_obj=config.model.d_obj,
        state_dim=config.model.state_dim,
        d_flow=config.model.d_flow,
        action_dim=config.model.action_dim,
        d_action_summary=config.model.d_action_summary,
        d_h_tau=config.model.d_h_tau,
        d_model=config.model.d_model,
        d_pi05=config.model.d_pi05,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        num_tasks=config.model.num_tasks,
        task_embedding_dim=config.model.task_embedding_dim,
    )


def make_correction_head(config: ATGConfig) -> CorrectionHead:
    return CorrectionHead(
        action_dim=config.model.action_dim,
        state_dim=config.model.state_dim,
        d_h_tau=config.model.d_h_tau,
        d_action_summary=config.model.d_action_summary,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        num_tasks=config.model.num_tasks,
        task_embedding_dim=config.model.task_embedding_dim,
    )
