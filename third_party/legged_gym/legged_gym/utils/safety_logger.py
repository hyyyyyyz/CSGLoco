"""Safety Logger v4 — fixes coupling metric to distinguish fallen vs active recovery."""

import torch


class SafetyLogger:

    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device
        self.reset()

    def reset(self):
        self.total_steps = 0
        self.torque_violation_count = 0
        self.contact_force_violation_count = 0
        self.orientation_violation_count = 0

        # Distinct violation events (transitions from no-viol to viol)
        self.torque_event_count = 0
        self.contact_event_count = 0
        self.orient_event_count = 0
        self._prev_torque_viol = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_contact_viol = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_orient_viol = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Violation occupancy (fraction of time in violation)
        self.torque_occupancy_steps = 0
        self.contact_occupancy_steps = 0
        self.orient_occupancy_steps = 0

        self.fall_count = 0
        self.recovery_success_count = 0
        self.recovery_failure_count = 0
        self.recovery_times = []

        # --- Coupling: during ANY fallen state ---
        self.violations_during_fallen = 0
        self.total_fallen_steps = 0
        self.torque_viol_during_fallen = 0
        self.contact_viol_during_fallen = 0
        self.orient_viol_during_fallen = 0

        # --- Coupling: during ACTIVE recovery only (phase > 0) ---
        self.violations_during_active_recovery = 0
        self.active_recovery_steps = 0
        self.torque_viol_during_active_rec = 0
        self.contact_viol_during_active_rec = 0
        self.orient_viol_during_active_rec = 0

        # --- Locomotion-only hazardous window: upright and not recovering ---
        self.loco_only_steps = 0
        self.torque_viol_during_loco_only = 0

        # Per-attempt violation tracking
        self._recovery_had_violation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._recovery_had_hazardous_viol = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.recoveries_with_violation = 0
        self.recoveries_with_hazardous_viol = 0
        # Track for successful vs all attempts separately
        self.successful_with_violation = 0
        self.successful_with_hazardous_viol = 0

        self._prev_is_fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def log_step(self, torque_violation, contact_force_violation, orientation_violation,
                 is_fallen, recovery_phase=None):
        self.total_steps += 1

        self.torque_violation_count += torque_violation.sum().item()
        self.contact_force_violation_count += contact_force_violation.sum().item()
        self.orientation_violation_count += orientation_violation.sum().item()

        # Distinct events
        new_torque = (~self._prev_torque_viol) & torque_violation
        new_contact = (~self._prev_contact_viol) & contact_force_violation
        new_orient = (~self._prev_orient_viol) & orientation_violation
        self.torque_event_count += new_torque.sum().item()
        self.contact_event_count += new_contact.sum().item()
        self.orient_event_count += new_orient.sum().item()
        self._prev_torque_viol = torque_violation.clone()
        self._prev_contact_viol = contact_force_violation.clone()
        self._prev_orient_viol = orientation_violation.clone()

        # Occupancy
        self.torque_occupancy_steps += torque_violation.sum().item()
        self.contact_occupancy_steps += contact_force_violation.sum().item()
        self.orient_occupancy_steps += orientation_violation.sum().item()

        # Falls
        new_falls = (~self._prev_is_fallen) & is_fallen
        self.fall_count += new_falls.sum().item()

        any_violation = torque_violation | contact_force_violation | orientation_violation
        hazardous_violation = torque_violation  # torque is the primary hazardous type

        # --- Coupling: during fallen state ---
        fallen_mask = is_fallen
        self.total_fallen_steps += fallen_mask.sum().item()
        self.violations_during_fallen += (any_violation & fallen_mask).sum().item()
        self.torque_viol_during_fallen += (torque_violation & fallen_mask).sum().item()
        self.contact_viol_during_fallen += (contact_force_violation & fallen_mask).sum().item()
        self.orient_viol_during_fallen += (orientation_violation & fallen_mask).sum().item()

        # --- Coupling: during active recovery (phase > 0) ---
        if recovery_phase is not None:
            active_rec = fallen_mask & (recovery_phase > 0)
            self.active_recovery_steps += active_rec.sum().item()
            self.violations_during_active_recovery += (any_violation & active_rec).sum().item()
            self.torque_viol_during_active_rec += (torque_violation & active_rec).sum().item()
            self.contact_viol_during_active_rec += (contact_force_violation & active_rec).sum().item()
            self.orient_viol_during_active_rec += (orientation_violation & active_rec).sum().item()

            loco_only = (~is_fallen) & (recovery_phase == 0)
            self.loco_only_steps += loco_only.sum().item()
            self.torque_viol_during_loco_only += (torque_violation & loco_only).sum().item()

        # Per-attempt violation tracking (reset on new falls)
        self._recovery_had_violation |= (any_violation & fallen_mask)
        self._recovery_had_hazardous_viol |= (hazardous_violation & fallen_mask)
        self._recovery_had_violation[new_falls] = False
        self._recovery_had_hazardous_viol[new_falls] = False

        self._prev_is_fallen = is_fallen.clone()

    def log_recovery(self, recovered_mask, recovery_times):
        n = recovered_mask.sum().item()
        self.recovery_success_count += n
        self.recovery_times.extend(recovery_times.cpu().tolist())
        self.successful_with_violation += self._recovery_had_violation[recovered_mask].sum().item()
        self.successful_with_hazardous_viol += self._recovery_had_hazardous_viol[recovered_mask].sum().item()
        self.recoveries_with_violation += self._recovery_had_violation[recovered_mask].sum().item()
        self.recoveries_with_hazardous_viol += self._recovery_had_hazardous_viol[recovered_mask].sum().item()
        self._recovery_had_violation[recovered_mask] = False
        self._recovery_had_hazardous_viol[recovered_mask] = False

    def log_recovery_failure(self, timed_out_mask):
        self.recovery_failure_count += timed_out_mask.sum().item()
        self.recoveries_with_violation += self._recovery_had_violation[timed_out_mask].sum().item()
        self.recoveries_with_hazardous_viol += self._recovery_had_hazardous_viol[timed_out_mask].sum().item()
        self._recovery_had_violation[timed_out_mask] = False
        self._recovery_had_hazardous_viol[timed_out_mask] = False

    def summarize(self):
        total_env_steps = max(self.total_steps * self.num_envs, 1)
        dt = 0.02
        total_time_sec = max(total_env_steps * dt, 0.001)
        total_falls = max(self.fall_count, 1)
        total_attempts = self.recovery_success_count + self.recovery_failure_count
        total_attempts_denom = max(total_attempts, 1)
        success_denom = max(self.recovery_success_count, 1)
        mean_recovery_time = (
            sum(self.recovery_times) / len(self.recovery_times)
            if self.recovery_times else float("nan")
        )

        # Coupling denominators
        total_fallen_sec = max(self.total_fallen_steps * 0.02, 0.001)
        active_rec_sec = max(self.active_recovery_steps * 0.02, 0.001)
        loco_only_sec = max(self.loco_only_steps * 0.02, 0.001)

        return {
            # Axis 2: Safety — rates per second
            "safety/torque_violation_rate": self.torque_violation_count / total_time_sec,
            "safety/contact_force_violation_rate": self.contact_force_violation_count / total_time_sec,
            "safety/orientation_violation_rate": self.orientation_violation_count / total_time_sec,
            "safety/total_violations": (
                self.torque_violation_count + self.contact_force_violation_count + self.orientation_violation_count
            ),
            # Axis 2: Safety — distinct events
            "safety/torque_events": self.torque_event_count,
            "safety/contact_events": self.contact_event_count,
            "safety/orient_events": self.orient_event_count,
            "safety/total_events": self.torque_event_count + self.contact_event_count + self.orient_event_count,
            # Axis 2: Safety — occupancy
            "safety/torque_occupancy": self.torque_occupancy_steps / total_env_steps,
            "safety/contact_occupancy": self.contact_occupancy_steps / total_env_steps,
            "safety/orient_occupancy": self.orient_occupancy_steps / total_env_steps,
            # Axis 3: Recovery
            "recovery/fall_count": self.fall_count,
            "recovery/success_rate": self.recovery_success_count / total_attempts_denom,
            "recovery/mean_time_to_upright": mean_recovery_time,
            "recovery/timeout_count": self.recovery_failure_count,
            "recovery/total_attempts": total_attempts,
            # Axis 4: Coupling (during fallen state — backward compatible)
            "coupling/violations_during_fallen": self.violations_during_fallen,
            "coupling/violations_per_fall": self.violations_during_fallen / total_falls,
            "coupling/viol_per_fallen_sec": self.violations_during_fallen / total_fallen_sec if self.total_fallen_steps > 0 else 0,
            # Axis 4: Coupling (during active recovery — NEW, correct metric)
            "coupling/viol_during_active_recovery": self.violations_during_active_recovery,
            "coupling/active_recovery_steps": self.active_recovery_steps,
            "coupling/viol_per_active_rec_sec": self.violations_during_active_recovery / active_rec_sec if self.active_recovery_steps > 0 else 0,
            # Axis 4: Per-type during fallen
            "coupling/torque_per_fallen_sec": self.torque_viol_during_fallen / total_fallen_sec if self.total_fallen_steps > 0 else 0,
            "coupling/contact_per_fallen_sec": self.contact_viol_during_fallen / total_fallen_sec if self.total_fallen_steps > 0 else 0,
            "coupling/orient_per_fallen_sec": self.orient_viol_during_fallen / total_fallen_sec if self.total_fallen_steps > 0 else 0,
            # Axis 4: Per-type during active recovery
            "coupling/torque_per_active_rec_sec": self.torque_viol_during_active_rec / active_rec_sec if self.active_recovery_steps > 0 else 0,
            "coupling/contact_per_active_rec_sec": self.contact_viol_during_active_rec / active_rec_sec if self.active_recovery_steps > 0 else 0,
            "coupling/orient_per_active_rec_sec": self.orient_viol_during_active_rec / active_rec_sec if self.active_recovery_steps > 0 else 0,
            # Locomotion-only hazardous torque rate
            "coupling/torque_per_loco_only_sec": self.torque_viol_during_loco_only / loco_only_sec if self.loco_only_steps > 0 else 0,
            # Axis 4: Per-attempt violation fractions
            "coupling/pct_attempts_with_violation": self.recoveries_with_violation / total_attempts_denom,
            "coupling/pct_attempts_with_hazardous": self.recoveries_with_hazardous_viol / total_attempts_denom,
            "coupling/pct_success_with_violation": self.successful_with_violation / success_denom if self.recovery_success_count > 0 else 0,
            "coupling/pct_success_with_hazardous": self.successful_with_hazardous_viol / success_denom if self.recovery_success_count > 0 else 0,
            # Raw denominators for transparency
            "raw/total_env_steps": total_env_steps,
            "raw/total_time_sec": total_time_sec,
            "raw/torque_violation_count": self.torque_violation_count,
            "raw/contact_force_violation_count": self.contact_force_violation_count,
            "raw/orientation_violation_count": self.orientation_violation_count,
            "raw/total_fallen_steps": self.total_fallen_steps,
            "raw/active_recovery_steps": self.active_recovery_steps,
            "raw/total_fallen_sec": self.total_fallen_steps * dt,
            "raw/active_recovery_sec": self.active_recovery_steps * dt,
        }
