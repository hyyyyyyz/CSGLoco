"""SafeRecovery Benchmark environment.

Extends LeggedRobot with:
1. Safety constraint monitoring  (Axis 2)
2. External force perturbation   (Axis 3)
3. Fall detection with hysteresis + functional recovery (Axis 3)
4. Velocity tracking accumulation (Axis 1)
5. Optional fallen-start initialization for recovery training

Recovery is two-phase:
  Phase 1 (self-righting): height > threshold AND tilt < threshold, sustained 0.3s
  Phase 2 (return-to-task): velocity tracking error < threshold, sustained 0.2s
  Total recovery confirmed only when both phases complete.
"""

import torch
from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import quat_from_euler_xyz
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.safety_logger import SafetyLogger


class SafeRecoveryEnv(LeggedRobot):

    BASE_BODY_INDEX = 0

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._init_safety_buffers()
        self.safety_logger = SafetyLogger(self.num_envs, self.device)
        self._verify_base_body()

    def _verify_base_body(self):
        name = self.gym.get_actor_rigid_body_names(self.envs[0], 0)[self.BASE_BODY_INDEX]
        assert name == "base", f"Expected base, got {name}"

    def _init_safety_buffers(self):
        n = self.num_envs
        dev = self.device
        cfg_f = self.cfg.fall_detection

        # Axis 2
        self.torque_violation = torch.zeros(n, dtype=torch.bool, device=dev)
        self.contact_force_violation = torch.zeros(n, dtype=torch.bool, device=dev)
        self.orientation_violation = torch.zeros(n, dtype=torch.bool, device=dev)

        # Axis 3: perturbation
        self.perturb_active = torch.zeros(n, dtype=torch.bool, device=dev)
        self.perturb_force = torch.zeros(n, 3, device=dev)
        self.perturb_remaining = torch.zeros(n, device=dev)
        self.perturb_cooldown = torch.zeros(n, device=dev)
        self.rb_forces = torch.zeros((n * self.num_bodies, 3), dtype=torch.float32, device=dev)

        # Axis 3: fall detection with hysteresis
        self.is_fallen = torch.zeros(n, dtype=torch.bool, device=dev)
        self.fall_time = torch.zeros(n, device=dev)
        self.fall_confirm_counter = torch.zeros(n, dtype=torch.int32, device=dev)
        self.fall_confirm_steps = int(cfg_f.fall_confirm_duration / self.dt)

        # Two-phase recovery
        self.recovery_phase = torch.zeros(n, dtype=torch.int32, device=dev)  # 0=not recovering, 1=self-righting, 2=return-to-task
        self.recovery_stable_counter = torch.zeros(n, dtype=torch.int32, device=dev)
        self.selfright_steps = int(cfg_f.selfright_stable_duration / self.dt)
        self.returntask_steps = int(cfg_f.returntask_stable_duration / self.dt)

        # Grace period
        self.grace_steps = int(cfg_f.startup_grace_duration / self.dt)
        self.steps_since_reset = torch.zeros(n, dtype=torch.int32, device=dev)

        # Axis 1: velocity tracking
        self.vel_tracking_error_sum = torch.zeros(n, device=dev)
        self.vel_tracking_step_count = torch.zeros(n, dtype=torch.int64, device=dev)

        # Debug
        self._perturb_applied_count = 0
        self._safety_check_count = 0
        self._fall_detected_count = 0

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def post_physics_step(self):
        super().post_physics_step()
        self._accumulate_velocity_tracking()
        self._check_safety_constraints()
        self._update_perturbations()
        self._apply_perturbation_forces()
        self._check_falls()
        self.safety_logger.log_step(
            torque_violation=self.torque_violation,
            contact_force_violation=self.contact_force_violation,
            orientation_violation=self.orientation_violation,
            is_fallen=self.is_fallen,
            recovery_phase=self.recovery_phase,
        )
        self._safety_check_count += 1

    # ------------------------------------------------------------------
    # Axis 1
    # ------------------------------------------------------------------

    def _accumulate_velocity_tracking(self):
        lin_err = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        ang_err = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        total_err = lin_err + 0.5 * ang_err
        self.vel_tracking_error_sum += total_err
        self.vel_tracking_step_count += 1

    def _current_vel_tracking_error(self):
        """Instantaneous velocity tracking error for recovery phase 2."""
        lin_err = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        ang_err = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return lin_err + 0.5 * ang_err

    # ------------------------------------------------------------------
    # Axis 2
    # ------------------------------------------------------------------

    def _check_safety_constraints(self):
        cfg_s = self.cfg.safety
        max_torque = torch.abs(self.torques).max(dim=1).values
        self.torque_violation = max_torque > cfg_s.torque_limit
        base_contact = self.contact_forces[:, self.BASE_BODY_INDEX, :]
        self.contact_force_violation = torch.norm(base_contact, dim=1) > cfg_s.contact_force_limit
        tilt_angle = self._compute_tilt_angle()
        self.orientation_violation = tilt_angle > cfg_s.orientation_limit
        if cfg_s.enable_constraint_termination:
            any_v = self.torque_violation | self.contact_force_violation | self.orientation_violation
            self.reset_buf |= any_v

    # ------------------------------------------------------------------
    # Axis 3: Perturbation
    # ------------------------------------------------------------------

    def _update_perturbations(self):
        cfg_p = self.cfg.perturbation
        if not cfg_p.enabled:
            return
        dt = self.dt
        self.perturb_cooldown -= dt
        self.perturb_remaining -= dt
        expired = self.perturb_active & (self.perturb_remaining <= 0)
        self.perturb_active[expired] = False
        self.perturb_force[expired] = 0.0
        ready = (~self.perturb_active) & (self.perturb_cooldown <= 0)
        n_ready = ready.sum().item()
        if n_ready > 0:
            mag = torch.empty(n_ready, device=self.device).uniform_(*cfg_p.force_range)
            dur = torch.empty(n_ready, device=self.device).uniform_(*cfg_p.duration_range)
            cooldown = torch.empty(n_ready, device=self.device).uniform_(*cfg_p.interval_range)
            angle = torch.empty(n_ready, device=self.device).uniform_(0, 6.2832)
            fx = mag * torch.cos(angle)
            fy = mag * torch.sin(angle)
            fz = torch.zeros(n_ready, device=self.device)
            self.perturb_active[ready] = True
            self.perturb_force[ready] = torch.stack([fx, fy, fz], dim=1)
            self.perturb_remaining[ready] = dur
            self.perturb_cooldown[ready] = cooldown + dur
            self._perturb_applied_count += n_ready

    def _apply_perturbation_forces(self):
        cfg_p = self.cfg.perturbation
        if not cfg_p.enabled or not self.perturb_active.any():
            return
        self.rb_forces[:] = 0.0
        active_ids = self.perturb_active.nonzero(as_tuple=False).squeeze(-1)
        base_rb_idx = active_ids * self.num_bodies + self.BASE_BODY_INDEX
        self.rb_forces[base_rb_idx] = self.perturb_force[active_ids]
        self.gym.apply_rigid_body_force_tensors(
            self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.ENV_SPACE,
        )

    # ------------------------------------------------------------------
    # Axis 3: Fall detection + two-phase recovery
    # ------------------------------------------------------------------

    def _compute_tilt_angle(self):
        cos_tilt = torch.clamp(-self.projected_gravity[:, 2], -1.0, 1.0)
        return torch.acos(cos_tilt)

    def _check_falls(self):
        cfg_f = self.cfg.fall_detection
        base_height = self.root_states[:, 2]
        tilt_angle = self._compute_tilt_angle()

        # --- Fall detection with hysteresis ---
        in_fall_zone = (
            (base_height < cfg_f.base_height_threshold)
            | (tilt_angle > cfg_f.orientation_threshold)
        )
        self.steps_since_reset += 1
        not_fallen = (~self.is_fallen) & (self.steps_since_reset > self.grace_steps)
        self.fall_confirm_counter[not_fallen & in_fall_zone] += 1
        self.fall_confirm_counter[not_fallen & (~in_fall_zone)] = 0
        newly_fallen = not_fallen & (self.fall_confirm_counter >= self.fall_confirm_steps)
        n_new = newly_fallen.sum().item()
        if n_new > 0:
            self._fall_detected_count += n_new
            self.is_fallen[newly_fallen] = True
            self.fall_time[newly_fallen] = 0.0
            self.recovery_phase[newly_fallen] = 0
            self.recovery_stable_counter[newly_fallen] = 0
            self.fall_confirm_counter[newly_fallen] = 0

        # --- Recovery time tracking ---
        self.fall_time[self.is_fallen] += self.dt

        # --- Two-phase recovery ---
        # Phase 1: self-righting (postural)
        in_upright = (
            (base_height > cfg_f.recovery_height_threshold)
            & (tilt_angle < cfg_f.recovery_orientation_threshold)
        )
        # Phase 2: return-to-task (functional)
        vel_err = self._current_vel_tracking_error()
        in_tracking = vel_err < cfg_f.recovery_vel_error_threshold

        # Envs in phase 0 (just fell) -> check for upright
        phase0 = self.is_fallen & (self.recovery_phase == 0)
        entering_phase1 = phase0 & in_upright
        self.recovery_phase[entering_phase1] = 1
        self.recovery_stable_counter[entering_phase1] = 1

        # Envs in phase 1 (self-righting) -> accumulate upright steps
        phase1 = self.is_fallen & (self.recovery_phase == 1)
        phase1_ok = phase1 & in_upright
        phase1_fail = phase1 & (~in_upright)
        self.recovery_stable_counter[phase1_ok] += 1
        # Reset to phase 0 if falls back
        self.recovery_phase[phase1_fail] = 0
        self.recovery_stable_counter[phase1_fail] = 0

        # Transition phase 1 -> phase 2 after sustained upright
        phase1_done = phase1 & (self.recovery_stable_counter >= self.selfright_steps)
        self.recovery_phase[phase1_done] = 2
        self.recovery_stable_counter[phase1_done] = 0

        # Envs in phase 2 (return-to-task) -> accumulate tracking steps
        phase2 = self.is_fallen & (self.recovery_phase == 2)
        phase2_ok = phase2 & in_upright & in_tracking
        phase2_fail_posture = phase2 & (~in_upright)
        phase2_fail_tracking = phase2 & in_upright & (~in_tracking)
        self.recovery_stable_counter[phase2_ok] += 1
        # Falls back down -> reset to phase 0
        self.recovery_phase[phase2_fail_posture] = 0
        self.recovery_stable_counter[phase2_fail_posture] = 0
        # Upright but not tracking -> stay in phase 2, reset counter
        self.recovery_stable_counter[phase2_fail_tracking] = 0

        # Full recovery confirmed
        recovered = self.is_fallen & (self.recovery_phase == 2) & (
            self.recovery_stable_counter >= self.returntask_steps
        )
        if recovered.any():
            self.safety_logger.log_recovery(recovered, self.fall_time[recovered])
            self.is_fallen[recovered] = False
            self.fall_time[recovered] = 0.0
            self.recovery_phase[recovered] = 0
            self.recovery_stable_counter[recovered] = 0

        # Recovery timeout
        timed_out = self.is_fallen & (self.fall_time > cfg_f.recovery_timeout)
        if timed_out.any():
            self.safety_logger.log_recovery_failure(timed_out)
            self.is_fallen[timed_out] = False
            self.fall_time[timed_out] = 0.0
            self.recovery_phase[timed_out] = 0
            self.recovery_stable_counter[timed_out] = 0

    # ------------------------------------------------------------------
    # Fallen-start initialization
    # ------------------------------------------------------------------

    def _reset_root_states(self, env_ids):
        """Override to optionally start some envs in fallen poses."""
        super()._reset_root_states(env_ids)

        if not hasattr(self.cfg, "fallen_start") or not self.cfg.fallen_start.enabled:
            return

        cfg_fs = self.cfg.fallen_start
        n = len(env_ids)
        n_fallen = int(n * cfg_fs.fraction)
        if n_fallen == 0:
            return

        # Select random subset to start fallen
        perm = torch.randperm(n, device=self.device)[:n_fallen]
        fallen_ids = env_ids[perm]

        # Randomize height (near ground)
        h = torch.empty(n_fallen, device=self.device).uniform_(*cfg_fs.height_range)
        self.root_states[fallen_ids, 2] = h

        # Randomize orientation (roll, pitch)
        roll = torch.empty(n_fallen, device=self.device).uniform_(*cfg_fs.roll_range)
        pitch = torch.empty(n_fallen, device=self.device).uniform_(*cfg_fs.pitch_range)
        yaw = torch.zeros(n_fallen, device=self.device)
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        self.root_states[fallen_ids, 3:7] = quat

        # Zero velocities
        self.root_states[fallen_ids, 7:13] = 0.0

        # Write back
        env_ids_int32 = fallen_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) > 0:
            self.torque_violation[env_ids] = False
            self.contact_force_violation[env_ids] = False
            self.orientation_violation[env_ids] = False
            self.perturb_active[env_ids] = False
            self.perturb_force[env_ids] = 0.0
            self.perturb_remaining[env_ids] = 0.0
            self.perturb_cooldown[env_ids] = 0.0
            self.is_fallen[env_ids] = False
            self.fall_time[env_ids] = 0.0
            self.fall_confirm_counter[env_ids] = 0
            self.recovery_phase[env_ids] = 0
            self.recovery_stable_counter[env_ids] = 0
            self.steps_since_reset[env_ids] = 0
            self.vel_tracking_error_sum[env_ids] = 0.0
            self.vel_tracking_step_count[env_ids] = 0

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_safety_summary(self):
        summary = self.safety_logger.summarize()
        valid = self.vel_tracking_step_count > 0
        if valid.any():
            mean_err = (
                self.vel_tracking_error_sum[valid] / self.vel_tracking_step_count[valid].float()
            ).mean().item()
        else:
            mean_err = float("nan")
        summary["locomotion/mean_vel_tracking_error"] = mean_err
        summary["debug/perturb_applied_count"] = self._perturb_applied_count
        summary["debug/safety_check_count"] = self._safety_check_count
        summary["debug/fall_detected_count"] = self._fall_detected_count
        return summary
