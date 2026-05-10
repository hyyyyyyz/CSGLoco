"""SafeRecovery Benchmark — configuration variants."""

from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO


class SafeRecoveryCfg(A1RoughCfg):

    class env(A1RoughCfg.env):
        num_envs = 4096
        num_observations = 48
        episode_length_s = 20

    class terrain(A1RoughCfg.terrain):
        mesh_type = "plane"
        measure_heights = False

    class commands(A1RoughCfg.commands):
        heading_command = False
        resampling_time = 8.0
        class ranges(A1RoughCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-1.0, 1.0]

    class domain_rand(A1RoughCfg.domain_rand):
        push_robots = False
        randomize_friction = True
        friction_range = [0.5, 1.25]

    class safety:
        torque_limit = 25.0         # [Nm] lowered from 33.5 for meaningful activation
        contact_force_limit = 200.0 # [N] on base
        orientation_limit = 0.8     # [rad] ~46deg, tightened from 1.0
        enable_constraint_termination = False

    class perturbation:
        enabled = False
        force_range = [50.0, 200.0]
        duration_range = [0.1, 0.5]
        interval_range = [3.0, 8.0]
        direction = "random"

    class fall_detection:
        base_height_threshold = 0.15
        orientation_threshold = 1.25
        fall_confirm_duration = 0.1
        # Two-phase recovery
        recovery_height_threshold = 0.22
        recovery_orientation_threshold = 0.6
        selfright_stable_duration = 0.3   # Phase 1: sustained upright
        returntask_stable_duration = 0.2  # Phase 2: sustained velocity tracking
        recovery_vel_error_threshold = 0.5  # [m/s equiv] max tracking error for phase 2
        recovery_timeout = 5.0
        startup_grace_duration = 0.5

    class fallen_start:
        enabled = False  # off by default; recovery baseline enables it

    class rewards(A1RoughCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales(A1RoughCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0


class SafeRecoveryCfgPPO(A1RoughCfgPPO):
    class algorithm(A1RoughCfgPPO.algorithm):
        entropy_coef = 0.01
    class runner(A1RoughCfgPPO.runner):
        run_name = ""
        experiment_name = "safe_recovery_a1"
        max_iterations = 1500


# --- CaT variant ---
class SafeRecoveryCatCfg(SafeRecoveryCfg):
    class safety(SafeRecoveryCfg.safety):
        enable_constraint_termination = True


class SafeRecoveryCatCfgPPO(SafeRecoveryCfgPPO):
    class runner(SafeRecoveryCfgPPO.runner):
        experiment_name = "safe_recovery_a1_cat"


# --- Fallen-start recovery variant ---
class SafeRecoveryFallenStartCfg(SafeRecoveryCfg):
    class init_state(SafeRecoveryCfg.init_state):
        pos = [0.0, 0.0, 0.30]

    class fallen_start:
        enabled = True
        fraction = 0.5
        roll_range = [-2.5, 2.5]
        pitch_range = [-1.5, 1.5]
        height_range = [0.08, 0.15]


class SafeRecoveryFallenStartCfgPPO(SafeRecoveryCfgPPO):
    class runner(SafeRecoveryCfgPPO.runner):
        experiment_name = "safe_recovery_a1_fallen"
        max_iterations = 2000
