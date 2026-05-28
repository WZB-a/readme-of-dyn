from __future__ import annotations

from libero_pi_dyn.config import ATGConfig


KNOWN_ACTION_SPACES = {
    "raw_8d_xyz_quat_gripper",
    "raw_7d_xyz_axisangle_gripper",
    "delta_7d_xyz_axisangle_gripper_abs_gripper",
    "libero_delta_7d_xyz_axisangle_gripper_state8",
    "normalized_8d_xyz_quat_gripper",
    "normalized_7d_xyz_axisangle_gripper",
}


def assert_action_space_ready(config: ATGConfig) -> None:
    action_space = config.data.action_space
    if not config.data.require_action_space_audit:
        return
    if action_space == "unknown" or action_space not in KNOWN_ACTION_SPACES:
        allowed = ", ".join(sorted(KNOWN_ACTION_SPACES))
        raise RuntimeError(
            "Action-space audit is not complete. Set data.action_space in the config after confirming that "
            "expert_action and base_action are in the same raw/normalized action space. "
            f"Allowed values: {allowed}"
        )
