from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from realworld_pi_dyn.config import ATGConfig
from realworld_pi_dyn.factory import make_correction_head
from realworld_pi_dyn.factory import make_predictor
from realworld_pi_dyn.factory import make_tokenizer


def test_model_shapes():
    config = ATGConfig()
    b = 3
    tokenizer = make_tokenizer(config)
    out = tokenizer(
        h_s=torch.zeros(b, config.model.d_h_tau),
        h_tau_raw=torch.zeros(b, config.model.d_h_tau),
        robot_state=torch.zeros(b, config.model.state_dim),
        robot_state_tau=torch.ones(b, config.model.state_dim),
        base_action=torch.zeros(b, config.model.action_dim),
        action_summary=torch.zeros(b, config.model.d_action_summary),
        chunk_index=torch.zeros(b),
        time_to_exec=torch.ones(b),
    )
    assert out["h_tau_teacher"].shape == (b, config.model.d_h_tau)
    predictor = make_predictor(config)
    pred = predictor(
        object_feature_current=torch.zeros(b, config.model.d_obj),
        robot_state_current=torch.zeros(b, config.model.state_dim),
        flow_feature=torch.zeros(b, config.model.d_flow),
        base_action=torch.zeros(b, config.model.action_dim),
        action_summary=torch.zeros(b, config.model.d_action_summary),
        chunk_index=torch.zeros(b),
        time_to_exec=torch.ones(b),
        task_id=torch.zeros(b, dtype=torch.long),
        base_chunk=torch.zeros(b, 10, config.model.action_dim),
    )
    assert pred["h_hat_tau"].shape == (b, config.model.d_h_tau)
    correction = make_correction_head(config)
    corr = correction(
        h_hat_tau=pred["h_hat_tau"],
        base_action=torch.zeros(b, config.model.action_dim),
        robot_state=torch.zeros(b, config.model.state_dim),
        action_summary=torch.zeros(b, config.model.d_action_summary),
        chunk_index=torch.zeros(b),
        time_to_exec=torch.ones(b),
        task_id=torch.zeros(b, dtype=torch.long),
    )
    assert corr["delta_action"].shape == (b, config.model.action_dim)
