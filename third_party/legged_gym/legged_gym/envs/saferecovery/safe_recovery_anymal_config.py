"""SafeRecovery Benchmark — ANYmal-C configuration variants."""

from legged_gym.envs.anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO


class SafeRecoveryAnymalCfg(AnymalCRoughCfg):
    """ANYmal-C SafeRecovery base config (flat terrain, PD controller)."""

    class env(AnymalCRoughCfg.env):
        num_envs = 4096
        num_observations = 48
        episode_length_s = 20

    class terrain(AnymalCRoughCfg.terrain):
        mesh_type = "plane"
        measure_heights = False

    class control(AnymalCRoughCfg.control):
        use_actuator_network = False  # Use PD controller for SafeRecoveryEnv compatibility

    class commands(AnymalCRoughCfg.commands):
        heading_command = False
        resampling_time = 8.0
        class ranges(AnymalCRoughCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-1.0, 1.0]

    class domain_rand(AnymalCRoughCfg.domain_rand):
        push_robots = False
        randomize_friction = True
        friction_range = [0.5, 1.25]

    class safety:
        torque_limit = 40.0
        contact_force_limit = 250.0
        orientation_limit = 0.8
        enable_constraint_termination = False

    class perturbation:
        enabled = False
        force_range = [50.0, 200.0]
        duration_range = [0.1, 0.5]
        interval_range = [3.0, 8.0]
        direction = "random"

    class fall_detection:
        base_height_threshold = 0.20
        orientation_threshold = 1.25
        fall_confirm_duration = 0.1
        recovery_height_threshold = 0.30
        recovery_orientation_threshold = 0.6
        selfright_stable_duration = 0.3
        returntask_stable_duration = 0.2
        recovery_vel_error_threshold = 0.5
        recovery_timeout = 5.0
        startup_grace_duration = 0.5

    class fallen_start:
        enabled = False

    class asset(AnymalCRoughCfg.asset):
        self_collisions = 0


class SafeRecoveryAnymalCfgPPO(AnymalCRoughCfgPPO):
    class algorithm(AnymalCRoughCfgPPO.algorithm):
        entropy_coef = 0.01
    class runner(AnymalCRoughCfgPPO.runner):
        run_name = ""
        experiment_name = "safe_recovery_anymal"
        max_iterations = 1500


class SafeRecoveryAnymalCatCfg(SafeRecoveryAnymalCfg):
    class safety(SafeRecoveryAnymalCfg.safety):
        enable_constraint_termination = True


class SafeRecoveryAnymalCatCfgPPO(SafeRecoveryAnymalCfgPPO):
    class runner(SafeRecoveryAnymalCfgPPO.runner):
        experiment_name = "safe_recovery_anymal_cat"


class SafeRecoveryAnymalFallenStartCfg(SafeRecoveryAnymalCfg):
    class fallen_start:
        enabled = True
        fraction = 0.5
        roll_range = [-2.5, 2.5]
        pitch_range = [-1.5, 1.5]
        height_range = [0.10, 0.20]


class SafeRecoveryAnymalFallenStartCfgPPO(SafeRecoveryAnymalCfgPPO):
    class runner(SafeRecoveryAnymalCfgPPO.runner):
        experiment_name = "safe_recovery_anymal_fallen"
        max_iterations = 2000
