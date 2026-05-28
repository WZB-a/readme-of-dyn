# Realworld-Pi_dyn

Independent ATG-VLA project layered on top of `../RealWorld-Pi` as a frozen pi0.5 base VLA.

This project follows the deployment document in `../readme-of-dyn/ATG_VLA_three_modules_engineering_deployment_zh.md`:

- Do not modify the pi0.5 network.
- Use an already LoRA-finetuned pi0.5 policy as the base VLA.
- Train dynamic tokenizer offline.
- Train predictor to produce `h_hat_tau`.
- Train correction head to output residual `delta_action`.
- Deploy with safety clipping and base-action fallback.

## Layout

```text
Realworld-Pi_dyn/
  configs/atg_vla.yaml
  src/realworld_pi_dyn/
  scripts/
  docs/cache_schema.md
  tests/
```

## Minimal Pipeline

```bash
cd /data3/yinmenghao/code/openpi/Realworld-Pi_dyn

python scripts/00_make_pair_index.py --config configs/atg_vla.yaml
python scripts/01_cache_pi05_chunks.py --config configs/atg_vla.yaml
python scripts/02_extract_basic_features.py --config configs/atg_vla.yaml
python scripts/03_train_dynamic_tokenizer.py --config configs/atg_vla.yaml
python scripts/04_export_teacher_cache.py --config configs/atg_vla.yaml --ckpt outputs/tokenizer/best.pt
python scripts/05_train_puma_lite_predictor.py --config configs/atg_vla.yaml
python scripts/06_export_h_hat_tau_cache.py --config configs/atg_vla.yaml --ckpt outputs/predictor/best.pt
python scripts/07_train_current_correction_baseline.py --config configs/atg_vla.yaml
python scripts/07_train_correction_head.py --config configs/atg_vla.yaml
```

For your existing LeRobot datasets, first prepare the raw cache:

```bash
python scripts/00_prepare_raw_from_lerobot.py --config configs/atg_vla.yaml

# Required before caching pi0.5 chunks, because pi0.5 needs images:
python scripts/00_prepare_raw_from_lerobot.py --config configs/atg_vla.yaml --with-video
```

Before training `07_train_correction_head.py`, set `data.action_space` in `configs/atg_vla.yaml` to a confirmed value such as `raw_8d_xyz_quat_gripper` or `normalized_7d_xyz_axisangle_gripper`. If it remains `unknown`, the script stops by design.

## Base VLA

`01_cache_pi05_chunks.py` loads the base policy from `../RealWorld-Pi` through `openpi.training.config` and `openpi.policies.policy_config`. Set:

- `base.openpi_root`
- `base.openpi_config_name`
- `base.checkpoint_dir`
- `base.prompt_key`

The base policy output is cached under `caches/atg_vla/pi05_chunks/<split>/<episode_id>.npz`.

To switch base VLA, change `base.selected_preset` in `configs/atg_vla.yaml`. Current presets include `yuanxingball`, `tank`, `clean_table`, and `yuanxingcup`.

## Dyn1 Model Notes

This `Realworld-Pi_dyn1` version keeps the runnable pipeline from `Realworld-Pi_dyn`, but changes model internals to better match the deployment document:

- Dynamic tokenizer: transition tokens, learnable dynamics queries, shared VQ, ego/env slot split, robot auxiliary loss.
- Predictor: PUMA-style lightweight future/world queries, flow/action-summary/base-chunk tokens, `h_hat_tau` head only by default.
- Correction: document-aligned small MLP, plus a current-only correction baseline script.

The full DOMINO/PUMA VLM, GroundingSAM, image/BEV reconstruction, token CE, and online tokenizer are intentionally not included in first-pass real-robot training because the project has 4 tasks x 200 trajectories and limited training budget.

## Data

The scripts expect episode `.npz` files under:

```text
caches/atg_vla/raw/train/*.npz
caches/atg_vla/raw/val/*.npz
```

See `docs/cache_schema.md` for required keys and shapes.
