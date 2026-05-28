# Cache Schema

The independent project uses cache files instead of modifying `RealWorld-Pi`.

## Raw Episode Cache

Location:

```text
caches/atg_vla/raw/<split>/<episode_id>.npz
```

Required keys:

```text
images: uint8 [T, H, W, 3] or [T, C, H, W, 3]
front_view: optional uint8 [T, H, W, 3]
wrist_view: optional uint8 [T, H, W, 3]
state: float32 [T, state_dim]
actions: float32 [T, action_dim]
prompt: optional string scalar or object array
task_id: optional int scalar
timestamp: optional float32 [T]
```

If `timestamp` is missing, scripts use `data.control_dt`.

The realworld LeRobot datasets found locally use 7D state/action:

```text
[eef_pos.x, eef_pos.y, eef_pos.z, eef_axisangle.ax, eef_axisangle.ay, eef_axisangle.az, gripper.pos]
```

This matches the RealWorld-Pi configs for `pi05_tank_lora`, `pi05_yuanxingball_20260518_lora`, `pi05_clean_table_lora`, and `pi05_yuanxingcup_lora`.

## Dyn1 Training Order

`Realworld-Pi_dyn1` follows the document order:

```text
current correction baseline
dynamic tokenizer -> teacher cache
PUMA-style predictor -> h_hat_tau cache
predicted h_hat_tau correction
```

The baseline is not the final method. It checks whether residual targets and action cache are meaningful before using future latent conditioning.

## Pair Index

Location:

```text
caches/atg_vla/pair_index/<split>.jsonl
```

Each row stores:

```text
domain_id, task_id, episode_id, t1, t2, t3, T, t4, k, dt
```

## pi0.5 Chunk Cache

Location:

```text
caches/atg_vla/pi05_chunks/<split>/<episode_id>.npz
```

Keys:

```text
base_chunk: float32 [T, H, action_dim]
available_t2: int32 [T]
```

## Feature Cache

Location:

```text
caches/atg_vla/features/<split>/<episode_id>.npz
```

Keys:

```text
h_s: float32 [T, d_h_tau]
h_tau_raw: float32 [T, d_h_tau]
object_feature_current: float32 [T, d_obj]
flow_feature: float32 [T, d_flow]
robot_state: float32 [T, state_dim]
expert_action: float32 [T, action_dim]
```

The provided basic feature extractor is deliberately simple. Replace it with a frozen pi0.5 or vision encoder extractor when that interface is confirmed.
