# Realworld-Pi_dyn_libero

Independent LIBERO variant of the ATG-VLA three-module project. It does not modify `../RealWorld-Pi`, `../Realworld-Pi_dyn`, or `../Realworld-Pi_dyn1`.

This version keeps the document-aligned three modules:

- Dynamic tokenizer: offline teacher latent `h_tau_teacher`.
- PUMA-lite predictor: predicts `h_hat_tau` from current visual/state/flow/action context.
- Action correction: small MLP residual head over 7D LIBERO actions.

For official `pi05_libero`, the action is 7D `[dx, dy, dz, dax, day, daz, gripper]`, while the observation state is 8D `[x, y, z, ax, ay, az, gripper_qpos_0, gripper_qpos_1]`. This follows `RealWorld-Pi/src/openpi/policies/libero_policy.py`.

## Small Single-Task Online Test

Start the pi0.5 LIBERO policy server in one terminal:

```bash
cd /data3/yinmenghao/code/openpi/Realworld-Pi_dyn_libero
CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ../RealWorld-Pi/.venv/bin/python scripts/serve_pi05_libero_policy.py \
  --checkpoint-dir /data1/vla-data/openpi/openpi-assets/checkpoints/pi05_libero \
  --port 8000
```

Run one short LIBERO rollout from the LIBERO Python environment:

```bash
cd /data3/yinmenghao/code/openpi/Realworld-Pi_dyn_libero
PYTHONPATH=/data3/yinmenghao/code/openpi/Realworld-Pi_dyn_libero/.libero_py38_deps:/data3/yinmenghao/code/openpi/third_party/libero:/data3/yinmenghao/code/openpi/RealWorld-Pi/packages/openpi-client/src \
  /data3/jikangye/tools/miniconda3/envs/libero/bin/python scripts/online_eval_libero_single_task.py \
  --host 127.0.0.1 --port 8000 \
  --task-suite-name libero_spatial --task-id 0 \
  --num-trials 1 --max-steps 40 --replan-steps 5
```

`--max-steps 40` is a chain smoke test, not a success-rate evaluation. Use `--max-steps 0` for the suite default horizon.

## LIBERO Offline Cache Pipeline

Prepare a small single-task hdf5 cache:

```bash
python scripts/00_prepare_raw_from_libero_hdf5.py --config configs/atg_vla.yaml
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

The default hdf5 is:

```text
/data1/vla-data/LIBERO-datasets/datasets/libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5
```

For a different single task, pass `--hdf5 /path/to/task_demo.hdf5` to `00_prepare_raw_from_libero_hdf5.py`.
