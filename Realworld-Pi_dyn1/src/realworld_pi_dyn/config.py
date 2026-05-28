from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass
class BaseConfig:
    openpi_root: str = "../RealWorld-Pi"
    selected_preset: str = "yuanxingball"
    openpi_config_name: str = "pi05_yuanxingball_20260518_lora"
    checkpoint_dir: str = "../RealWorld-Pi/checkpoints/yuanxingball_20260518_lora/pi05_yuanxingball_20260518_lora/yuanxingball_20260518_lora_v1/19999"
    prompt_key: str = "prompt"
    action_output_key: str = "actions"
    checkpoint_presets: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ModelConfig:
    action_dim: int = 7
    state_dim: int = 7
    d_obj: int = 128
    d_flow: int = 64
    d_pi05: int = 0
    d_model: int = 256
    d_h_tau: int = 256
    d_action_summary: int = 128
    hidden_dim: int = 512
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.05
    num_tasks: int = 64
    task_embedding_dim: int = 32
    codebook_size: int = 128
    num_latent_tokens: int = 4
    num_ego_tokens: int = 2
    tokenizer_warmup_epochs: int = 0
    use_pi05_feature: bool = False


@dataclasses.dataclass
class DataConfig:
    cache_root: str = "caches/atg_vla"
    dataset_root: str = "/data1/vla-data/processed/PI/data/yuanxingball_20260518"
    raw_action_key: str = "actions"
    raw_state_key: str = "state"
    train_split: str = "train"
    val_split: str = "val"
    train_episodes: int = 160
    val_episodes: int = 40
    max_frames_per_episode: int | None = None
    future_steps: tuple[int, ...] = (1, 2, 4, 6)
    chunk_indices: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    control_dt: float = 0.05
    latency_steps: int = 2
    task_id_default: int = 0
    require_action_space_audit: bool = True
    action_space: str = "raw_7d_xyz_axisangle_gripper"


@dataclasses.dataclass
class LossConfig:
    lambda_delta: float = 0.5
    lambda_l2: float = 0.05
    lambda_vq: float = 0.25
    lambda_recon: float = 1.0
    lambda_robot: float = 0.1
    lambda_residual_l2: float = 1.0e-4


@dataclasses.dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 50
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    early_stop_patience: int = 8
    num_workers: int = 4
    seed: int = 7
    device: str = "cuda"


@dataclasses.dataclass
class SafetyConfig:
    per_dim_limit: float | None = None
    norm_limit: float | None = None
    stats_multiplier: float = 1.5
    fallback_on_nan: bool = True


@dataclasses.dataclass
class ATGConfig:
    base: BaseConfig = dataclasses.field(default_factory=BaseConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    safety: SafetyConfig = dataclasses.field(default_factory=SafetyConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> "ATGConfig":
        path = Path(path)
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw) if path.suffix.lower() == ".json" else yaml.safe_load(raw)
        if not isinstance(data, dict):
            raise ValueError(f"Config must be a mapping: {path}")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ATGConfig":
        return cls(
            base=_coerce(BaseConfig, data.get("base", {})),
            model=_coerce(ModelConfig, data.get("model", {})),
            data=_coerce(DataConfig, data.get("data", {})),
            loss=_coerce(LossConfig, data.get("loss", {})),
            train=_coerce(TrainConfig, data.get("train", {})),
            safety=_coerce(SafetyConfig, data.get("safety", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def resolved_base(self) -> BaseConfig:
        preset = self.base.checkpoint_presets.get(self.base.selected_preset)
        if not preset:
            return self.base
        values = dataclasses.asdict(self.base)
        values.update({k: v for k, v in preset.items() if k in {"openpi_config_name", "checkpoint_dir"}})
        return BaseConfig(**values)

    def resolved_dataset_root(self) -> str:
        preset = self.base.checkpoint_presets.get(self.base.selected_preset)
        if preset and "dataset_root" in preset:
            return str(preset["dataset_root"])
        return self.data.dataset_root


def _coerce(cls: type, values: dict[str, Any]):
    names = {field.name for field in dataclasses.fields(cls)}
    filtered = {key: value for key, value in dict(values).items() if key in names}
    for key in ("future_steps", "chunk_indices"):
        if key in filtered and isinstance(filtered[key], list):
            filtered[key] = tuple(filtered[key])
    return cls(**filtered)
