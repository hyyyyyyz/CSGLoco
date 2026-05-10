"""SafeRecovery Benchmark — Comprehensive Audit Evaluation.

Fixes bugs found in code audit:
1. Episode resets now log recovery failures (was silently dropped)
2. Coupling metrics properly distinguish 'during active recovery' from 'during lying fallen'
3. Per-type coupling breakdown during recovery
4. Per-episode event-level logging
5. Raw counts for all metrics
6. Recovery definition sensitivity sweep
7. Failure taxonomy
8. Fair eval (CaT termination disabled) enforced

Usage:
    python eval_audit.py [--task TASK] [--run RUN_NAME]
"""

import os
import json
import sys
import math
import csv
from datetime import datetime
from collections import defaultdict

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch


class AuditLogger:
    """Detailed per-step and per-episode event logger for metric audit."""

    def __init__(self, num_envs, device, dt=0.02):
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.reset()

    def reset(self):
        n = self.num_envs
        dev = self.device

        # Per-step counters
        self.total_steps = 0

        # Violation counts (total and per-type)
        self.torque_viol_total = 0
        self.contact_viol_total = 0
        self.orient_viol_total = 0

        # Violation events (transitions)
        self.torque_events = 0
        self.contact_events = 0
        self.orient_events = 0
        self._prev_torque = torch.zeros(n, dtype=torch.bool, device=dev)
        self._prev_contact = torch.zeros(n, dtype=torch.bool, device=dev)
        self._prev_orient = torch.zeros(n, dtype=torch.bool, device=dev)

        # Falls
        self.fall_count = 0
        self._prev_is_fallen = torch.zeros(n, dtype=torch.bool, device=dev)

        # Recovery tracking
        self.recovery_success_count = 0
        self.recovery_failure_count = 0  # includes timeouts AND episode resets
        self.recovery_timeout_count = 0
        self.recovery_reset_count = 0  # falls dropped by episode reset
        self.recovery_times = []

        # Coupling: violations during ACTIVE recovery (is_fallen AND recovery_phase > 0)
        self.viol_during_active_recovery = 0
        self.active_recovery_steps = 0

        # Coupling: violations during ANY fallen state (including phase 0)
        self.viol_during_fallen = 0
        self.total_fallen_steps = 0

        # Per-type coupling during active recovery
        self.torque_viol_active_rec = 0
        self.contact_viol_active_rec = 0
        self.orient_viol_active_rec = 0

        # Per-type coupling during fallen
        self.torque_viol_fallen = 0
        self.contact_viol_fallen = 0
        self.orient_viol_fallen = 0

        # Per-recovery-attempt violation tracking
        self._attempt_had_viol = torch.zeros(n, dtype=torch.bool, device=dev)
        self._attempt_had_hazardous_viol = torch.zeros(n, dtype=torch.bool, device=dev)
        self.attempts_with_any_viol = 0
        self.attempts_with_hazardous_viol = 0
        self.successful_with_any_viol = 0
        self.successful_with_hazardous_viol = 0

        # Failure taxonomy
        self.fail_never_upright = 0  # never reached phase 1
        self.fail_upright_not_task = 0  # reached phase 1/2 but didn't complete
        self.fail_re_fell = 0  # reached upright then fell again

        # Event log (per-episode events for sample output)
        self.event_log = []  # list of dicts
        self._current_fall_events = {}  # env_id -> event dict

    def log_step(self, torque_viol, contact_viol, orient_viol,
                 is_fallen, recovery_phase, base_height, tilt_angle, step_idx):
        self.total_steps += 1

        # Violation totals
        self.torque_viol_total += torque_viol.sum().item()
        self.contact_viol_total += contact_viol.sum().item()
        self.orient_viol_total += orient_viol.sum().item()

        # Violation events (transitions)
        self.torque_events += ((~self._prev_torque) & torque_viol).sum().item()
        self.contact_events += ((~self._prev_contact) & contact_viol).sum().item()
        self.orient_events += ((~self._prev_orient) & orient_viol).sum().item()
        self._prev_torque = torque_viol.clone()
        self._prev_contact = contact_viol.clone()
        self._prev_orient = orient_viol.clone()

        # Fall detection (count transitions)
        new_falls = (~self._prev_is_fallen) & is_fallen
        n_new = new_falls.sum().item()
        if n_new > 0:
            self.fall_count += n_new
            # Reset per-attempt violation tracking
            self._attempt_had_viol[new_falls] = False
            self._attempt_had_hazardous_viol[new_falls] = False
            # Start event log entries
            for env_id in new_falls.nonzero(as_tuple=False).squeeze(-1).tolist():
                if isinstance(env_id, int):
                    self._current_fall_events[env_id] = {
                        "env_id": env_id,
                        "fall_step": step_idx,
                        "fall_time": step_idx * self.dt,
                        "first_upright_step": None,
                        "return_task_step": None,
                        "recovery_end_step": None,
                        "max_recovery_phase": 0,
                        "violations_during": [],
                        "success": False,
                        "failure_type": None,
                    }

        # Fallen state tracking
        any_viol = torque_viol | contact_viol | orient_viol
        hazardous_viol = torque_viol  # torque is the primary hazardous type

        # Violations during fallen
        fallen_mask = is_fallen
        self.total_fallen_steps += fallen_mask.sum().item()
        self.viol_during_fallen += (any_viol & fallen_mask).sum().item()
        self.torque_viol_fallen += (torque_viol & fallen_mask).sum().item()
        self.contact_viol_fallen += (contact_viol & fallen_mask).sum().item()
        self.orient_viol_fallen += (orient_viol & fallen_mask).sum().item()

        # Violations during active recovery (phase > 0)
        active_rec = fallen_mask & (recovery_phase > 0)
        self.active_recovery_steps += active_rec.sum().item()
        self.viol_during_active_recovery += (any_viol & active_rec).sum().item()
        self.torque_viol_active_rec += (torque_viol & active_rec).sum().item()
        self.contact_viol_active_rec += (contact_viol & active_rec).sum().item()
        self.orient_viol_active_rec += (orient_viol & active_rec).sum().item()

        # Per-attempt violation tracking
        self._attempt_had_viol |= (any_viol & fallen_mask)
        self._attempt_had_hazardous_viol |= (hazardous_viol & fallen_mask)

        # Track max recovery phase for event log
        for env_id in fallen_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
            if isinstance(env_id, int) and env_id in self._current_fall_events:
                ev = self._current_fall_events[env_id]
                phase = recovery_phase[env_id].item()
                if phase > ev["max_recovery_phase"]:
                    ev["max_recovery_phase"] = phase
                if phase >= 1 and ev["first_upright_step"] is None:
                    ev["first_upright_step"] = step_idx

        self._prev_is_fallen = is_fallen.clone()

    def log_recovery_success(self, recovered_mask, recovery_times, step_idx):
        n = recovered_mask.sum().item()
        self.recovery_success_count += n
        self.recovery_times.extend(recovery_times.cpu().tolist())

        self.attempts_with_any_viol += self._attempt_had_viol[recovered_mask].sum().item()
        self.attempts_with_hazardous_viol += self._attempt_had_hazardous_viol[recovered_mask].sum().item()
        self.successful_with_any_viol += self._attempt_had_viol[recovered_mask].sum().item()
        self.successful_with_hazardous_viol += self._attempt_had_hazardous_viol[recovered_mask].sum().item()

        self._attempt_had_viol[recovered_mask] = False
        self._attempt_had_hazardous_viol[recovered_mask] = False

        for env_id in recovered_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
            if isinstance(env_id, int) and env_id in self._current_fall_events:
                ev = self._current_fall_events[env_id]
                ev["success"] = True
                ev["recovery_end_step"] = step_idx
                ev["return_task_step"] = step_idx
                self.event_log.append(ev)
                del self._current_fall_events[env_id]

    def log_recovery_timeout(self, timed_out_mask, step_idx):
        n = timed_out_mask.sum().item()
        self.recovery_failure_count += n
        self.recovery_timeout_count += n

        self.attempts_with_any_viol += self._attempt_had_viol[timed_out_mask].sum().item()
        self.attempts_with_hazardous_viol += self._attempt_had_hazardous_viol[timed_out_mask].sum().item()

        self._attempt_had_viol[timed_out_mask] = False
        self._attempt_had_hazardous_viol[timed_out_mask] = False

        for env_id in timed_out_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
            if isinstance(env_id, int) and env_id in self._current_fall_events:
                ev = self._current_fall_events[env_id]
                ev["failure_type"] = "timeout"
                ev["recovery_end_step"] = step_idx
                if ev["max_recovery_phase"] == 0:
                    self.fail_never_upright += 1
                    ev["failure_type"] = "never_upright"
                else:
                    self.fail_upright_not_task += 1
                    ev["failure_type"] = "upright_not_task"
                self.event_log.append(ev)
                del self._current_fall_events[env_id]

    def log_episode_reset(self, reset_ids, step_idx):
        """Log falls that are dropped by episode reset (BUG FIX)."""
        for env_id in reset_ids:
            if env_id in self._current_fall_events:
                ev = self._current_fall_events[env_id]
                ev["failure_type"] = "episode_reset"
                ev["recovery_end_step"] = step_idx
                self.recovery_failure_count += 1
                self.recovery_reset_count += 1

                self.attempts_with_any_viol += int(self._attempt_had_viol[env_id].item())
                self.attempts_with_hazardous_viol += int(self._attempt_had_hazardous_viol[env_id].item())
                self._attempt_had_viol[env_id] = False
                self._attempt_had_hazardous_viol[env_id] = False

                if ev["max_recovery_phase"] == 0:
                    self.fail_never_upright += 1
                else:
                    self.fail_upright_not_task += 1
                self.event_log.append(ev)
                del self._current_fall_events[env_id]

    def summarize(self):
        total_env_steps = max(self.total_steps * self.num_envs, 1)
        total_attempts = self.recovery_success_count + self.recovery_failure_count
        total_fallen_sec = max(self.total_fallen_steps * self.dt, 0.001)
        active_rec_sec = max(self.active_recovery_steps * self.dt, 0.001)

        mean_rec_time = (
            sum(self.recovery_times) / len(self.recovery_times)
            if self.recovery_times else float("nan")
        )

        total_viols = self.torque_viol_total + self.contact_viol_total + self.orient_viol_total

        return {
            # Raw counts
            "raw/total_env_steps": total_env_steps,
            "raw/total_violations": total_viols,
            "raw/torque_violations": self.torque_viol_total,
            "raw/contact_violations": self.contact_viol_total,
            "raw/orient_violations": self.orient_viol_total,
            "raw/fall_count": self.fall_count,
            "raw/recovery_successes": self.recovery_success_count,
            "raw/recovery_failures": self.recovery_failure_count,
            "raw/recovery_timeouts": self.recovery_timeout_count,
            "raw/recovery_resets": self.recovery_reset_count,
            "raw/total_attempts": total_attempts,
            "raw/total_fallen_steps": self.total_fallen_steps,
            "raw/active_recovery_steps": self.active_recovery_steps,

            # Axis 2: Safety rates
            "safety/violations_per_sec": total_viols / (self.total_steps * self.dt * self.num_envs) if self.total_steps > 0 else 0,
            "safety/torque_rate": self.torque_viol_total / total_env_steps,
            "safety/contact_rate": self.contact_viol_total / total_env_steps,
            "safety/orient_rate": self.orient_viol_total / total_env_steps,
            "safety/hazardous_per_sec": self.torque_viol_total / (self.total_steps * self.dt * self.num_envs) if self.total_steps > 0 else 0,

            # Axis 2: Events
            "safety/torque_events": self.torque_events,
            "safety/contact_events": self.contact_events,
            "safety/orient_events": self.orient_events,

            # Axis 3: Recovery
            "recovery/fall_count": self.fall_count,
            "recovery/falls_per_min": self.fall_count / (self.total_steps * self.dt * self.num_envs / 60) if self.total_steps > 0 else 0,
            "recovery/success_rate": self.recovery_success_count / max(total_attempts, 1),
            "recovery/mean_time": mean_rec_time,
            "recovery/total_attempts": total_attempts,

            # Axis 4: Coupling (during fallen state)
            "coupling_fallen/viol_per_sec": self.viol_during_fallen / total_fallen_sec if self.total_fallen_steps > 0 else 0,
            "coupling_fallen/torque_per_sec": self.torque_viol_fallen / total_fallen_sec if self.total_fallen_steps > 0 else 0,
            "coupling_fallen/contact_per_sec": self.contact_viol_fallen / total_fallen_sec if self.total_fallen_steps > 0 else 0,
            "coupling_fallen/orient_per_sec": self.orient_viol_fallen / total_fallen_sec if self.total_fallen_steps > 0 else 0,

            # Axis 4: Coupling (during active recovery only)
            "coupling_active/viol_per_sec": self.viol_during_active_recovery / active_rec_sec if self.active_recovery_steps > 0 else 0,
            "coupling_active/torque_per_sec": self.torque_viol_active_rec / active_rec_sec if self.active_recovery_steps > 0 else 0,
            "coupling_active/contact_per_sec": self.contact_viol_active_rec / active_rec_sec if self.active_recovery_steps > 0 else 0,
            "coupling_active/orient_per_sec": self.orient_viol_active_rec / active_rec_sec if self.active_recovery_steps > 0 else 0,

            # Axis 4: Violation fraction of attempts
            "coupling/pct_attempts_any_viol": self.attempts_with_any_viol / max(total_attempts, 1),
            "coupling/pct_attempts_hazardous_viol": self.attempts_with_hazardous_viol / max(total_attempts, 1),
            "coupling/pct_success_any_viol": self.successful_with_any_viol / max(self.recovery_success_count, 1),
            "coupling/pct_success_hazardous_viol": self.successful_with_hazardous_viol / max(self.recovery_success_count, 1),

            # Failure taxonomy
            "taxonomy/fail_never_upright": self.fail_never_upright,
            "taxonomy/fail_upright_not_task": self.fail_upright_not_task,
            "taxonomy/fail_re_fell": self.fail_re_fell,
        }


def evaluate_audit(task_name, load_run, num_eval_steps=10000, num_envs=128,
                   force_range=None, interval_range=None):
    """Run comprehensive audit evaluation."""
    argv_backup = sys.argv
    sys.argv = ["eval", "--task=%s" % task_name, "--num_envs=%d" % num_envs, "--headless"]
    args = get_args()
    sys.argv = argv_backup

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = num_envs
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.perturbation.enabled = True
    if force_range is not None:
        env_cfg.perturbation.force_range = force_range
    if interval_range is not None:
        env_cfg.perturbation.interval_range = interval_range

    # FAIR EVALUATION: disable CaT termination
    env_cfg.safety.enable_constraint_termination = False

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    if load_run is not None:
        train_cfg.runner.load_run = load_run
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=task_name, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # Create audit logger
    audit = AuditLogger(num_envs, env.device, dt=env.dt)

    for step in range(num_eval_steps):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        # Log to audit logger
        audit.log_step(
            torque_viol=env.torque_violation,
            contact_viol=env.contact_force_violation,
            orient_viol=env.orientation_violation,
            is_fallen=env.is_fallen,
            recovery_phase=env.recovery_phase,
            base_height=env.root_states[:, 2],
            tilt_angle=env._compute_tilt_angle(),
            step_idx=step,
        )

        # Check for recoveries
        recovered = env.is_fallen & (env.recovery_phase == 2) & (
            env.recovery_stable_counter >= env.returntask_steps
        )
        if recovered.any():
            audit.log_recovery_success(recovered, env.fall_time[recovered], step)

        # Check for timeouts
        timed_out = env.is_fallen & (env.fall_time > env.cfg.fall_detection.recovery_timeout)
        if timed_out.any():
            audit.log_recovery_timeout(timed_out, step)

        # Check for episode resets (BUG FIX: log dropped recoveries)
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            fallen_done = done_ids[env.is_fallen[done_ids]]
            if len(fallen_done) > 0:
                audit.log_episode_reset(fallen_done.tolist(), step)

    # Also log any still-fallen envs at end of eval
    still_fallen = env.is_fallen.nonzero(as_tuple=False).squeeze(-1)
    if len(still_fallen) > 0:
        audit.log_episode_reset(still_fallen.tolist(), num_eval_steps)

    summary = audit.summarize()

    # Add velocity tracking
    valid = env.vel_tracking_step_count > 0
    if valid.any():
        mean_err = (
            env.vel_tracking_error_sum[valid] / env.vel_tracking_step_count[valid].float()
        ).mean().item()
    else:
        mean_err = float("nan")
    summary["locomotion/vel_tracking_error"] = mean_err
    summary["meta/task"] = task_name
    summary["meta/load_run"] = str(load_run)
    summary["meta/num_eval_steps"] = num_eval_steps
    summary["meta/num_envs"] = num_envs
    summary["meta/force_range"] = str(env_cfg.perturbation.force_range)
    summary["meta/interval_range"] = str(env_cfg.perturbation.interval_range)
    summary["meta/cat_termination_disabled"] = True

    # Event log (first 50 events)
    event_sample = audit.event_log[:50]

    env.gym.destroy_sim(env.sim)
    return summary, event_sample


def main():
    log_base = "/home/hurricane/RL/CSGLoco/legged_gym/logs"

    # Use original seed 0 checkpoints (verified)
    configs = {
        "Vanilla": ("safe_recovery_a1", "Mar16_23-37-58_"),
        "CaT": ("safe_recovery_a1_cat", "Mar17_00-12-03_"),
        "Recovery": ("safe_recovery_a1_fallen", "Mar17_12-09-35_"),
    }

    # Standard eval: medium perturbation
    eval_kwargs = dict(
        num_eval_steps=10000,
        num_envs=128,
        force_range=[80, 150],
        interval_range=[2.0, 5.0],
    )

    all_results = {}
    all_events = {}

    for method, (task, run) in configs.items():
        print("\n>>> AUDIT: %s (%s / %s)" % (method, task, run))
        summary, events = evaluate_audit(task, run, **eval_kwargs)
        all_results[method] = summary
        all_events[method] = events

    # Print audit table
    print("\n" + "=" * 120)
    print("METRIC AUDIT TABLE — SafeRecovery Benchmark (seed 0, fair eval)")
    print("=" * 120)

    sections = {
        "RAW COUNTS": [
            ("total_env_steps", "raw/total_env_steps"),
            ("total_violations", "raw/total_violations"),
            ("  torque_violations", "raw/torque_violations"),
            ("  contact_violations", "raw/contact_violations"),
            ("  orient_violations", "raw/orient_violations"),
            ("fall_count", "raw/fall_count"),
            ("recovery_successes", "raw/recovery_successes"),
            ("recovery_failures", "raw/recovery_failures"),
            ("  timeouts", "raw/recovery_timeouts"),
            ("  episode_resets", "raw/recovery_resets"),
            ("total_attempts", "raw/total_attempts"),
            ("fallen_steps", "raw/total_fallen_steps"),
            ("active_recovery_steps", "raw/active_recovery_steps"),
        ],
        "AXIS 2: SAFETY": [
            ("violations/sec (all)", "safety/violations_per_sec"),
            ("violations/sec (hazardous)", "safety/hazardous_per_sec"),
            ("torque_rate", "safety/torque_rate"),
            ("contact_rate", "safety/contact_rate"),
            ("orient_rate", "safety/orient_rate"),
        ],
        "AXIS 3: RECOVERY": [
            ("falls/min", "recovery/falls_per_min"),
            ("recovery_success_rate", "recovery/success_rate"),
            ("mean_recovery_time", "recovery/mean_time"),
            ("total_attempts", "recovery/total_attempts"),
        ],
        "AXIS 4: COUPLING (during fallen)": [
            ("viol/sec (all, fallen)", "coupling_fallen/viol_per_sec"),
            ("  torque/sec", "coupling_fallen/torque_per_sec"),
            ("  contact/sec", "coupling_fallen/contact_per_sec"),
            ("  orient/sec", "coupling_fallen/orient_per_sec"),
        ],
        "AXIS 4: COUPLING (active recovery)": [
            ("viol/sec (all, active)", "coupling_active/viol_per_sec"),
            ("  torque/sec", "coupling_active/torque_per_sec"),
            ("  contact/sec", "coupling_active/contact_per_sec"),
            ("  orient/sec", "coupling_active/orient_per_sec"),
        ],
        "AXIS 4: VIOLATION FRACTIONS": [
            ("% attempts w/ any viol", "coupling/pct_attempts_any_viol"),
            ("% attempts w/ hazardous", "coupling/pct_attempts_hazardous_viol"),
            ("% success w/ any viol", "coupling/pct_success_any_viol"),
            ("% success w/ hazardous", "coupling/pct_success_hazardous_viol"),
        ],
        "FAILURE TAXONOMY": [
            ("fail: never upright", "taxonomy/fail_never_upright"),
            ("fail: upright not task", "taxonomy/fail_upright_not_task"),
            ("fail: re-fell", "taxonomy/fail_re_fell"),
        ],
        "LOCOMOTION": [
            ("vel_tracking_error", "locomotion/vel_tracking_error"),
        ],
    }

    methods = list(all_results.keys())
    col_w = 18
    for section, keys in sections.items():
        print("\n  [%s]" % section)
        header = "  %-40s" % "Metric"
        for m in methods:
            header += ("  %" + str(col_w) + "s") % m
        print(header)
        print("  " + "-" * (40 + (col_w + 2) * len(methods)))
        for label, k in keys:
            row = "  %-40s" % label
            for m in methods:
                v = all_results[m].get(k, float("nan"))
                if isinstance(v, int):
                    row += ("  %" + str(col_w) + "d") % v
                elif isinstance(v, float) and math.isnan(v):
                    row += ("  %" + str(col_w) + "s") % "N/A"
                else:
                    row += ("  %" + str(col_w) + ".4f") % v
            print(row)

    # Print event log samples
    for method in methods:
        events = all_events[method]
        print("\n\n  EVENT LOG SAMPLE: %s (first 20)" % method)
        print("  " + "-" * 100)
        for ev in events[:20]:
            print("  env=%d fall_step=%d phase_max=%d success=%s type=%s" % (
                ev["env_id"], ev["fall_step"], ev["max_recovery_phase"],
                ev["success"], ev.get("failure_type", "N/A")
            ))

    # Save
    out_dir = os.path.join(log_base, "safe_recovery_eval")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary
    out_path = os.path.join(out_dir, "audit_%s.json" % ts)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\n\nAudit results saved to: %s" % out_path)

    # Save event logs
    event_path = os.path.join(out_dir, "audit_events_%s.json" % ts)
    with open(event_path, "w") as f:
        json.dump(all_events, f, indent=2, default=str)
    print("Event logs saved to: %s" % event_path)


if __name__ == "__main__":
    main()
