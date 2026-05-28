from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import pathlib
import sys
import types
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
OPENPI_ROOT = (ROOT / "../RealWorld-Pi").resolve()
THIRD_PARTY_LIBERO = (ROOT / "../third_party/libero").resolve()
OPENPI_CLIENT_SRC = OPENPI_ROOT / "packages/openpi-client/src"
for path in (THIRD_PARTY_LIBERO, OPENPI_CLIENT_SRC):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero_config"))


def _ensure_libero_config() -> None:
    config_dir = pathlib.Path(os.environ["LIBERO_CONFIG_PATH"])
    config_file = config_dir / "config.yaml"
    benchmark_root = THIRD_PARTY_LIBERO / "libero/libero"
    if config_file.exists():
        return
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "\n".join(
            [
                f"benchmark_root: {benchmark_root}",
                f"bddl_files: {benchmark_root / 'bddl_files'}",
                f"init_states: {benchmark_root / 'init_files'}",
                "datasets: /data1/vla-data/LIBERO-datasets/datasets",
                f"assets: {benchmark_root / 'assets'}",
                "",
            ]
        ),
        encoding="utf-8",
    )


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
DEFAULT_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=40, help="Short smoke-test horizon. Use 0 for suite default.")
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--video-out-path", default="outputs/libero_online/videos")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    summary = eval_single_task(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    sys.stdout.flush()
    sys.stderr.flush()
    # robosuite/mujoco can crash during interpreter teardown in some headless OSMesa builds after a successful rollout.
    os._exit(0)


def eval_single_task(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_libero_config()
    sys.modules.setdefault("robosuite.macros_private", types.ModuleType("robosuite.macros_private"))
    import robosuite.macros as _robosuite_macros

    _robosuite_macros.FILE_LOGGING_LEVEL = None

    from libero.libero import benchmark
    from openpi_client import websocket_client_policy as _websocket_client_policy

    np.random.seed(args.seed)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    max_steps = args.max_steps if args.max_steps > 0 else DEFAULT_MAX_STEPS[args.task_suite_name]
    video_dir = pathlib.Path(args.video_out_path)
    if not args.no_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logging.info("Connected to server metadata: %s", client.get_server_metadata())
    logging.info("LIBERO task %s/%s: %s", args.task_id, task_suite.n_tasks, task_description)

    successes = 0
    episodes = 0
    trial_summaries = []
    for episode_idx in range(args.num_trials):
        env.reset()
        init_state = initial_states[episode_idx % len(initial_states)]
        obs = env.set_init_state(init_state)
        action_plan: collections.deque[np.ndarray] = collections.deque()
        replay_images: list[np.ndarray] = []
        done = False
        steps_executed = 0
        chunks_queried = 0
        last_timing: dict[str, Any] = {}

        for t in range(max_steps + args.num_steps_wait):
            if t < args.num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                continue

            img, wrist_img = _preprocess_images(obs, args.resize_size)
            replay_images.append(img)

            if not action_plan:
                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                ).astype(np.float32)
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": state,
                    "prompt": str(task_description),
                }
                result = client.infer(element)
                action_chunk = np.asarray(result["actions"], dtype=np.float32)
                if len(action_chunk) < args.replan_steps:
                    raise RuntimeError(
                        f"Policy returned {len(action_chunk)} actions, shorter than replan_steps={args.replan_steps}"
                    )
                action_plan.extend(action_chunk[: args.replan_steps])
                chunks_queried += 1
                last_timing = result.get("server_timing", {})

            action = np.asarray(action_plan.popleft(), dtype=np.float32)
            obs, _, done, _ = env.step(action.tolist())
            steps_executed += 1
            if done:
                successes += 1
                break

        episodes += 1
        if not args.no_video and replay_images:
            _write_video(video_dir / f"task{args.task_id:02d}_trial{episode_idx:02d}_{'success' if done else 'failure'}.mp4", replay_images)
        trial_summaries.append(
            {
                "trial": episode_idx,
                "success": bool(done),
                "steps_executed": steps_executed,
                "chunks_queried": chunks_queried,
                "last_server_timing": last_timing,
            }
        )
        logging.info("trial=%s success=%s steps=%s chunks=%s", episode_idx, done, steps_executed, chunks_queried)

    env.close()
    return {
        "task_suite": args.task_suite_name,
        "task_id": args.task_id,
        "task_description": str(task_description),
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / max(1, episodes),
        "max_steps": max_steps,
        "num_steps_wait": args.num_steps_wait,
        "replan_steps": args.replan_steps,
        "trials": trial_summaries,
    }


def _get_libero_env(task, resolution: int, seed: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _preprocess_images(obs: dict[str, np.ndarray], resize_size: int) -> tuple[np.ndarray, np.ndarray]:
    from openpi_client import image_tools

    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))
    return img, wrist_img


def _write_video(path: pathlib.Path, images: list[np.ndarray]) -> None:
    import imageio

    imageio.mimwrite(path, [np.asarray(x) for x in images], fps=10)


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(float(quat[3]))) / den).astype(np.float32)


if __name__ == "__main__":
    main()
